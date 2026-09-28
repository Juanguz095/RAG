from __future__ import annotations

import hashlib
import logging
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import defer

from src.config import get_settings
from src.api.deps import get_current_user, require_admin
from src.database import Document, Chunk, User, get_db
from src.schemas.document import DocumentListResponse, DocumentResponse

logger = logging.getLogger(__name__)
settings = get_settings()
router = APIRouter(prefix="/api/v1/documents", tags=["documents"])

UPLOAD_DIR = Path(os.environ.get("UPLOAD_DIR", "/app/uploads"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@router.post("", status_code=202, response_model=dict)
async def upload_document(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
    current_user: User | None = Depends(get_current_user),
):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files allowed")

    content = await file.read()
    if len(content) > 100 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large (max 100MB)")

    file_hash = hashlib.sha256(content).hexdigest()

    existing = await db.execute(
        select(Document).where(Document.file_hash == file_hash)
    )
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=409, detail="Document already uploaded")

    doc = Document(
        filename=file_hash + ".pdf",
        original_name=file.filename,
        file_hash=file_hash,
        mime_type=file.content_type or "application/pdf",
        file_size=len(content),
        status="pending",
    )
    db.add(doc)
    await db.commit()
    await db.refresh(doc)

    pdf_path = UPLOAD_DIR / f"{file_hash}.pdf"
    pdf_path.write_bytes(content)

    # Encolado con compensación: si la cola (Redis/Celery) está caída, no deja
    # ni documento huérfano ni PDF remanente (contrato PLAN-002 C4).
    from src.workers.ingestion_tasks import enqueue_process_document
    try:
        enqueue_process_document(str(doc.id), str(pdf_path))
    except Exception as exc:
        logger.error(f"Enqueue failed for {doc.id}: {exc}")
        await db.rollback()
        pdf_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=503,
            detail="Processing queue unavailable, try again later",
        ) from exc

    return {"id": str(doc.id), "status": "processing", "message": "Document uploaded, processing in background"}


def _doc_item(d: Document, include_meta: bool = False) -> DocumentResponse:
    meta = None
    if include_meta:
        # only light keys; avoid shipping word_boxes on list/detail
        raw = d.metadata_ or {}
        meta = {k: v for k, v in raw.items() if k not in ("word_boxes", "extracted_data")} or None
    return DocumentResponse(
        id=d.id,
        filename=d.filename,
        original_name=d.original_name,
        file_size=d.file_size,
        page_count=d.page_count,
        status=d.status,
        total_chunks=d.total_chunks or 0,
        created_at=d.created_at,
        processed_at=d.processed_at,
        metadata_=meta,
    )


@router.get("", response_model=DocumentListResponse)
async def list_documents(
    page: int = Query(1, ge=1),
    size: int = Query(20, ge=1, le=100),
    status_filter: Optional[str] = Query(None, alias="status"),
    q: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
    current_user: User | None = Depends(get_current_user),
):
    query = select(Document).options(defer(Document.metadata_))
    count_query = select(func.count(Document.id))

    if status_filter:
        query = query.where(Document.status == status_filter)
        count_query = count_query.where(Document.status == status_filter)
    if q:
        query = query.where(Document.original_name.ilike(f"%{q}%"))
        count_query = count_query.where(Document.original_name.ilike(f"%{q}%"))

    total = (await db.execute(count_query)).scalar() or 0
    pages = math.ceil(total / size) if total else 1

    query = query.order_by(Document.created_at.desc())
    query = query.offset((page - 1) * size).limit(size)
    result = await db.execute(query)
    docs = result.scalars().all()

    return DocumentListResponse(
        documents=[_doc_item(d) for d in docs],
        total=total,
        page=page,
        pages=pages,
    )


@router.get("/{doc_id}", response_model=DocumentResponse)
async def get_document(
    doc_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User | None = Depends(get_current_user),
):
    result = await db.execute(
        select(Document).where(Document.id == doc_id).options(defer(Document.metadata_))
    )
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return _doc_item(doc)


@router.get("/{doc_id}/pdf")
async def get_document_pdf(
    doc_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User | None = Depends(get_current_user),
):
    result = await db.execute(select(Document).where(Document.id == doc_id))
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    pdf_path = UPLOAD_DIR / doc.filename
    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail="PDF file not found on disk")

    return FileResponse(
        path=str(pdf_path),
        media_type="application/pdf",
        filename=doc.original_name,
        headers={"Cache-Control": "private, max-age=86400"},
    )


@router.get("/{doc_id}/chunks")
async def get_document_chunks(
    doc_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User | None = Depends(get_current_user),
):
    result = await db.execute(select(Document).where(Document.id == doc_id))
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    chunks_result = await db.execute(
        select(
            Chunk.id,
            Chunk.chunk_index,
            Chunk.content,
            Chunk.page_numbers,
            Chunk.token_count,
            Chunk.chunk_metadata,
        )
        .where(Chunk.document_id == doc_id)
        .order_by(Chunk.chunk_index)
    )
    rows = chunks_result.all()

    meta = doc.metadata_ or {}
    word_boxes = meta.get("word_boxes", {})

    return {
        "document_id": str(doc.id),
        "document_name": doc.original_name,
        "total_chunks": len(rows),
        "extracted_data": meta.get("extracted_data"),
        "word_boxes": word_boxes,
        "chunks": [
            {
                "id": str(r.id),
                "chunk_index": r.chunk_index,
                "content": r.content,
                "page_numbers": r.page_numbers or [],
                "token_count": r.token_count,
                "chunk_metadata": r.chunk_metadata or {},
            }
            for r in rows
        ],
    }


@router.delete("/{doc_id}", status_code=204)
async def delete_document(
    doc_id: UUID,
    db: AsyncSession = Depends(get_db),
    _admin: User | None = Depends(require_admin),
):
    result = await db.execute(select(Document).where(Document.id == doc_id))
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    if doc.status in ("processing", "pending"):
        raise HTTPException(status_code=409, detail="Cannot delete document while processing")

    pdf_path = UPLOAD_DIR / doc.filename
    if pdf_path.exists():
        pdf_path.unlink()

    await db.delete(doc)
    await db.commit()
