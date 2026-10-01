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

from fastapi import Request
from src.config import get_settings
from src.api.deps import get_current_user, require_admin
from src.core.permissions import require_permission
from src.database import Document, Chunk, User, get_db
from src.services.audit import audit
from src.schemas.document import DocumentListResponse, DocumentResponse

logger = logging.getLogger(__name__)
settings = get_settings()
router = APIRouter(prefix="/api/v1/documents", tags=["documents"])

UPLOAD_DIR = Path(os.environ.get("UPLOAD_DIR", "/app/uploads"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@router.post("/upload-multiple", status_code=207, response_model=dict)
async def upload_multiple(
    files: list[UploadFile] = File(...),
    db: AsyncSession = Depends(get_db),
    current_user: User | None = Depends(require_permission("documents:upload")),
):
    """CP-003: subida múltiple con resultado por archivo aislado (207).

    Un PDF corrupto no tira el resto: cada archivo se procesa en su propio
    try/except y el estado va en `results`.
    """
    results = []
    for f in files:
        try:
            res = await _upload_one(f, db, current_user)
            results.append({"filename": f.filename, "ok": True, **res})
        except HTTPException as e:
            results.append({"filename": f.filename, "ok": False, "error": e.detail,
                            "status_code": e.status_code})
        except Exception as e:
            logger.warning(f"upload-multiple fallo {f.filename}: {e}")
            results.append({"filename": f.filename, "ok": False, "error": str(e)[:200],
                            "status_code": 500})
    return {"results": results}


async def _upload_one(file: UploadFile, db: AsyncSession, current_user):
    """File 1:1 — validación, dedup por hash, persistencia y encolado con
    compensación (ni documento huérfano ni PDF remanente — PLAN-002 C4)."""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files allowed")

    content = await file.read()
    if len(content) > 100 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large (max 100MB)")

    # RAG-054 (CP-003): PDF inválido/corrupto → mensaje controlado; ya no se
    # depende solo de la extensión.
    if not content.lstrip().startswith(b"%PDF"):
        raise HTTPException(status_code=400, detail="Invalid PDF file (corrupt or not a PDF)")

    file_hash = hashlib.sha256(content).hexdigest()
    existing = await db.execute(select(Document).where(Document.file_hash == file_hash))
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

    from src.workers.ingestion_tasks import enqueue_process_document
    try:
        enqueue_process_document(str(doc.id), str(pdf_path))
    except Exception as exc:
        logger.error(f"Enqueue failed for {doc.id}: {exc}")
        await db.rollback()
        pdf_path.unlink(missing_ok=True)
        await audit(db, current_user, "error", resource_type="document",
                    resource_id=str(doc.id),
                    detail={"stage": "enqueue", "error": str(exc)[:200]})
        raise HTTPException(
            status_code=503,
            detail="Processing queue unavailable, try again later",
        ) from exc

    await audit(db, current_user, "upload", resource_type="document",
                resource_id=str(doc.id),
                detail={"filename": file.filename, "size": len(content)})
    await db.commit()
    return {"id": str(doc.id), "status": "processing",
            "message": "Document uploaded, processing in background"}


@router.post("", status_code=202, response_model=dict)
async def upload_document(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
    current_user: User | None = Depends(require_permission("documents:upload")),
):
    return await _upload_one(file, db, current_user)


@router.post("/{doc_id}/reprocess", status_code=202, response_model=dict)
async def reprocess_document(
    doc_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User | None = Depends(require_permission("documents:upload")),
):
    """RAG-013 (PLAN-006 Fase 4): re-chunk + re-embed + re-index + re-keywords.

    Idempotente: reusa el PDF original en disco (fuente de verdad) y pisa los
    chunks/keywords existentes del documento.
    """
    res = await db.execute(select(Document).where(Document.id == doc_id))
    doc = res.scalar_one_or_none()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    pdf_path = UPLOAD_DIR / f"{doc.file_hash}.pdf"
    if not pdf_path.exists():
        raise HTTPException(status_code=409, detail="Original PDF missing, re-upload required")
    doc.status = "pending"
    await db.commit()
    from src.workers.ingestion_tasks import enqueue_process_document
    try:
        enqueue_process_document(str(doc.id), str(pdf_path))
    except Exception as exc:
        logger.error(f"Reprocess enqueue failed for {doc.id}: {exc}")
        raise HTTPException(status_code=503, detail="Processing queue unavailable") from exc
    await audit(db, current_user, "update", resource_type="document", resource_id=str(doc.id),
                detail={"action": "reprocess"})
    await db.commit()
    return {"id": str(doc.id), "status": "processing", "message": "Reprocess enqueued"}


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
    current_user: User | None = Depends(require_permission("documents:delete")),
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
    await audit(db, current_user, "delete", resource_type="document", resource_id=str(doc_id),
                detail={"original_name": doc.original_name})
    await db.commit()
