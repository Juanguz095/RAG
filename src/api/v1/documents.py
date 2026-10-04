from __future__ import annotations

import copy
import hashlib
import logging
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import defer

from fastapi import Request
from src.config import get_settings
from src.api.deps import get_current_user, require_admin
from src.core.permissions import require_permission
from src.database import (Document, Chunk, ChunkKeyword, Keyword, Message,
                          Proposal, User, get_db)
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


def _doc_item(d: Document, include_meta: bool = False,
              timings: dict | None = None, domains: list | None = None) -> DocumentResponse:
    meta = None
    raw: Any = {}
    # METADATA no debe disparar carga perezosa: el listado/detalle hacen defer()
    # (JSONB con word_boxes es pesado). Si está diferido, no lo tocamos.
    try:
        from sqlalchemy import inspect as sa_inspect

        if "metadata_" not in sa_inspect(d).unloaded:
            raw = d.metadata_ or {}
    except Exception:
        # objeto no-ORM (fakes de tests): leer directo, no hay carga perezosa
        raw = getattr(d, "metadata_", None) or {}
    # WP1: exponer timings SIEMPRE (listado y detalle) — "Procesado en X s".
    if timings is None:
        timings = raw.get("timings")
    if include_meta:
        # only light keys; avoid shipping word_boxes on list/detail
        meta = {k: v for k, v in raw.items() if k not in ("word_boxes", "extracted_data", "timings")} or None
    if domains is None:
        domains = raw.get("domains") or None
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
        timings=timings,
        domains=domains,
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
    query = select(
        Document,
        Document.metadata_["timings"].label("timings"),
        Document.metadata_["domains"].label("domains"),
    ).options(defer(Document.metadata_))
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
    rows = result.all()

    return DocumentListResponse(
        documents=[_doc_item(d, timings=t, domains=dom) for (d, t, dom) in rows],
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
        select(Document,
               Document.metadata_["timings"].label("timings"),
               Document.metadata_["domains"].label("domains"))
        .where(Document.id == doc_id).options(defer(Document.metadata_))
    )
    row = result.first()
    if not row:
        raise HTTPException(status_code=404, detail="Document not found")
    doc, t, dom = row
    return _doc_item(doc, timings=t, domains=dom)


class PatientFieldsIn(BaseModel):
    nombre: str | None = None
    edad: str | None = None
    sexo: str | None = None
    dni: str | None = None
    historia_clinica: str | None = None
    fecha_atencion: str | None = None


@router.patch("/{doc_id}/patient", response_model=dict)
async def update_patient_fields(
    doc_id: UUID,
    body: PatientFieldsIn,
    db: AsyncSession = Depends(get_db),
    current_user: User | None = Depends(require_permission("documents:update")),
):
    """Corrección manual de los datos del paciente (se guardan en el documento)."""
    result = await db.execute(select(Document).where(Document.id == doc_id))
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    meta = dict(doc.metadata_ or {})
    data = dict(meta.get("extracted_data") or {})
    for k in ("nombre", "edad", "sexo", "dni", "historia_clinica", "fecha_atencion"):
        v = getattr(body, k)
        if v is not None:
            data[k] = v
    from src.services.patient_fields import sanitize_patient_fields

    data = sanitize_patient_fields(data)
    data["manual"] = True
    meta["extracted_data"] = data
    doc.metadata_ = meta  # rebinding marks JSONB dirty
    await db.commit()
    await audit(db, current_user, "update", resource_type="document",
                resource_id=str(doc.id), detail={"op": "patient_fields"})
    await db.commit()
    return data


class WordCorrectionIn(BaseModel):
    page: int
    box_index: int
    text: str


@router.patch("/{doc_id}/word", response_model=dict)
async def correct_word(
    doc_id: UUID,
    body: WordCorrectionIn,
    db: AsyncSession = Depends(get_db),
    current_user: User | None = Depends(require_permission("documents:update")),
):
    """Corrige manualmente una palabra OCR (manuscrito mal leído).

    Actualiza el word_box (lo usa el buscador inteligente) y, si el texto viejo
    aparece literal en algún chunk de esa página, lo reemplaza y recalcula su
    embedding (para que el chat/consulta también lo encuentre).
    """
    result = await db.execute(select(Document).where(Document.id == doc_id))
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    meta = dict(doc.metadata_ or {})
    wb = copy.deepcopy(meta.get("word_boxes") or {})
    page_boxes = wb.get(str(body.page))
    if not isinstance(page_boxes, list) or not (0 <= body.box_index < len(page_boxes)):
        raise HTTPException(status_code=404, detail="Word box not found")

    new_text = (body.text or "").strip()
    if not new_text:
        raise HTTPException(status_code=400, detail="Empty text")
    old_text = page_boxes[body.box_index].get("text", "")
    page_boxes[body.box_index] = {**page_boxes[body.box_index], "text": new_text}
    wb[str(body.page)] = page_boxes
    meta["word_boxes"] = wb
    doc.metadata_ = meta  # rebinding marks JSONB dirty

    reindexed = 0
    if old_text and old_text != new_text:
        from src.services.embeddings import encode_texts

        chunks = (
            await db.execute(select(Chunk).where(Chunk.document_id == doc.id))
        ).scalars().all()
        for ch in chunks:
            if body.page not in (ch.page_numbers or []):
                continue
            if old_text in (ch.content or ""):
                ch.content = ch.content.replace(old_text, new_text, 1)
                emb = encode_texts([ch.content])[0]
                ch.embedding = [float(x) for x in emb]
                reindexed += 1

    await db.commit()
    await audit(db, current_user, "update", resource_type="document",
                resource_id=str(doc.id),
                detail={"op": "word_correction", "page": body.page})
    await db.commit()
    return {
        "page": body.page,
        "box_index": body.box_index,
        "old_text": old_text,
        "text": new_text,
        "reindexed_chunks": reindexed,
    }


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

    # Saneado en lectura: no mostrar valores implausibles aunque vengan de una
    # extracción previa (p. ej. edad=210). Idempotente.
    extracted = meta.get("extracted_data")
    if extracted:
        from src.services.patient_fields import sanitize_patient_fields

        _was_manual = bool(extracted.get("manual"))
        extracted = sanitize_patient_fields(extracted)
        if _was_manual:
            extracted["manual"] = True

    # Dominios (categorias de keyword) por chunk, para el filtro por dominio.
    domains_map: dict[str, set[str]] = {}
    chunk_ids = [r.id for r in rows]
    if chunk_ids:
        ck = await db.execute(
            select(ChunkKeyword.chunk_id, Keyword.category)
            .join(Keyword, Keyword.id == ChunkKeyword.keyword_id)
            .where(ChunkKeyword.chunk_id.in_(chunk_ids))
        )
        for cid, cat in ck.all():
            if cat:
                domains_map.setdefault(str(cid), set()).add(cat)

    return {
        "document_id": str(doc.id),
        "document_name": doc.original_name,
        "total_chunks": len(rows),
        "extracted_data": extracted,
        "extract_status": meta.get("extract_status"),
        "extract_started_at": meta.get("extract_started_at"),
        "extract_finished_at": meta.get("extract_finished_at"),
        "extract_eta_s": 120,
        "word_boxes": word_boxes,
        "word_boxes_dpi": meta.get("word_boxes_dpi", 200),
        "ocr_low_confidence_pages": meta.get("ocr_low_confidence_pages", []),
        "chunks": [
            {
                "id": str(r.id),
                "chunk_index": r.chunk_index,
                "content": r.content,
                "page_numbers": r.page_numbers or [],
                "token_count": r.token_count,
                "chunk_metadata": r.chunk_metadata or {},
                "domains": sorted(domains_map.get(str(r.id), [])),
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
    # Solo bloquear si el PIPELINE está corriendo de verdad. Un 'pending'
    # (subido pero nunca procesado — task perdida u OCR no iniciado) debe
    # poder borrarse: es el caso de un upload atascado.
    if doc.status == "processing":
        raise HTTPException(status_code=409, detail="Cannot delete document while processing")

    pdf_path = UPLOAD_DIR / doc.filename

    # Limpieza FK-safe: chunk_keywords NO tiene ON DELETE CASCADE, hay que
    # limpiarla (y las propuestas quedaban huérfanas) antes de borrar chunks.
    chunk_ids = (
        select(Chunk.id).where(Chunk.document_id == doc_id)
    )
    await db.execute(
        delete(ChunkKeyword).where(ChunkKeyword.chunk_id.in_(chunk_ids))
    )
    await db.execute(
        delete(Proposal).where(
            Proposal.chunk_id.in_(chunk_ids),
            Proposal.status != "approved",
        )
    )

    if pdf_path.exists():
        pdf_path.unlink()

    await db.delete(doc)
    await db.commit()
    try:
        await audit(db, current_user, "delete", resource_type="document", resource_id=str(doc_id),
                    detail={"original_name": doc.original_name})
        await db.commit()
    except Exception:
        import logging; logging.getLogger(__name__).exception("delete doc audit fallo")
        raise
