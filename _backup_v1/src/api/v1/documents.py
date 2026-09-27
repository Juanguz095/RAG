import logging
from datetime import date
from math import ceil
from uuid import UUID

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from fastapi.responses import StreamingResponse
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.deps import get_current_user
from src.database import get_db
from src.models.chunk import DocumentChunk
from src.models.document import Document
from src.schemas.document import (
    ChunkResponse,
    DocumentListResponse,
    DocumentResponse,
    DocumentWithChunksResponse,
)
from src.services.ingestion import (
    DocumentTooLargeError,
    DuplicateDocumentError,
    InvalidPDFError,
    StorageError,
    get_document_file,
    get_page_with_ocr_overlay,
    validate_and_deduplicate,
)
from src.services.synonyms import expand_query, match_score

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("", status_code=202)
async def upload_document(
    file: UploadFile = File(...),
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    existing = await db.execute(select(Document.id).limit(1))
    if existing.scalar_one_or_none() is not None:
        raise HTTPException(
            status_code=409,
            detail="Solo se permite un PDF. Elimina el documento actual antes de subir otro.",
        )

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")

    content = await file.read()
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Empty file")

    try:
        doc = await validate_and_deduplicate(db, content, file.filename, file.content_type or "application/pdf")
    except DuplicateDocumentError as e:
        raise HTTPException(status_code=409, detail=f"Document already exists: {e.existing_id}") from e
    except DocumentTooLargeError as e:
        raise HTTPException(status_code=413, detail=str(e)) from e
    except InvalidPDFError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except StorageError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e

    from src.workers.ingestion_tasks import process_document_task
    try:
        process_document_task.delay(str(doc.id))
    except Exception as exc:
        doc.status = "failed"
        doc.document_metadata = {**(doc.document_metadata or {}), "error": f"No se pudo encolar: {exc}"}
        await db.commit()
        raise HTTPException(status_code=503, detail="La cola de procesamiento no está disponible") from exc

    return {
        "task_id": str(doc.id),
        "status": "processing",
        "filename": doc.original_name,
        "page_count": doc.page_count,
    }


@router.get("", response_model=DocumentListResponse)
async def list_documents(
    status: str | None = Query(None),
    doc_type: str | None = Query(None),
    date_from: date | None = Query(None),
    date_to: date | None = Query(None),
    q: str | None = Query(None),
    page: int = Query(1, ge=1),
    size: int = Query(20, ge=1, le=100),
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    query = select(Document)
    count_query = select(func.count(Document.id))

    if status:
        query = query.where(Document.status == status)
        count_query = count_query.where(Document.status == status)
    if doc_type:
        query = query.where(Document.doc_type == doc_type)
        count_query = count_query.where(Document.doc_type == doc_type)
    if date_from:
        query = query.where(Document.created_at >= date_from)
        count_query = count_query.where(Document.created_at >= date_from)
    if date_to:
        query = query.where(Document.created_at < date_to)
        count_query = count_query.where(Document.created_at < date_to)
    if q:
        query = query.where(Document.original_name.ilike(f"%{q}%"))
        count_query = count_query.where(Document.original_name.ilike(f"%{q}%"))

    total_result = await db.execute(count_query)
    total = total_result.scalar_one() or 0

    offset = (page - 1) * size
    query = query.order_by(Document.created_at.desc()).offset(offset).limit(size)
    result = await db.execute(query)
    docs = result.scalars().all()

    return DocumentListResponse(
        items=[DocumentResponse.model_validate(d) for d in docs],
        total=total,
        page=page,
        size=size,
        pages=ceil(total / size) if total > 0 else 1,
    )


@router.get("/{doc_id}", response_model=DocumentWithChunksResponse)
async def get_document(
    doc_id: UUID,
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(Document).where(Document.id == doc_id))
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    chunks_result = await db.execute(
        select(DocumentChunk)
        .where(DocumentChunk.document_id == doc_id)
        .order_by(DocumentChunk.chunk_index)
    )
    chunks = chunks_result.scalars().all()

    doc_dict = DocumentResponse.model_validate(doc).model_dump()
    doc_dict["chunks"] = [ChunkResponse.model_validate(c) for c in chunks]
    return DocumentWithChunksResponse(**doc_dict)


@router.get("/{doc_id}/file")
async def get_document_file_stream(
    doc_id: UUID,
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    content = await get_document_file(db, doc_id)
    if content is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return StreamingResponse(
        iter([content]),
        media_type="application/pdf",
        headers={"Content-Disposition": "inline"},
    )


@router.get("/{doc_id}/pages/{page_num}")
async def get_page_image(
    doc_id: UUID,
    page_num: int,
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    content = await get_page_with_ocr_overlay(db, doc_id, page_num)
    if content is None:
        raise HTTPException(status_code=404, detail="Página o documento no encontrado")
    return StreamingResponse(iter([content]), media_type="image/png")


@router.post("/{doc_id}/reprocess", status_code=202)
async def reprocess_document(
    doc_id: UUID,
    force: bool = Query(False, description="Permite recuperar una tarea atascada"),
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(Document).where(Document.id == doc_id))
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    if doc.status in {"processing", "ocr", "chunking", "embedding"} and not force:
        raise HTTPException(status_code=409, detail="El documento ya esta procesandose; usa force=true solo si quedo atascado")
    doc.status = "processing"
    doc.document_metadata = {**(doc.document_metadata or {}), "reprocess_requested": True}
    await db.commit()

    from src.workers.ingestion_tasks import process_document_task

    try:
        process_document_task.delay(str(doc.id))
    except Exception as exc:
        doc.status = "failed"
        doc.document_metadata = {**(doc.document_metadata or {}), "error": f"No se pudo encolar: {exc}"}
        await db.commit()
        raise HTTPException(status_code=503, detail="La cola de procesamiento no está disponible") from exc

    return {"document_id": str(doc.id), "status": "processing"}


@router.get("/{doc_id}/search")
async def search_in_document(
    doc_id: UUID,
    q: str = Query(..., min_length=1, max_length=200),
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Search for a term within a document's chunks with fuzzy matching and percentage scores."""
    result = await db.execute(select(Document).where(Document.id == doc_id))
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    chunks_result = await db.execute(
        select(DocumentChunk)
        .where(DocumentChunk.document_id == doc_id)
        .order_by(DocumentChunk.chunk_index)
    )
    chunks = chunks_result.scalars().all()

    expanded_terms = expand_query(q)
    matches = []

    for chunk in chunks:
        text = chunk.content or ""
        score = match_score(q, text)

        # Also check expanded terms
        for term in expanded_terms[1:]:
            term_score = match_score(term, text)
            score = max(score, term_score)

        if score < 0.15:
            continue

        # Find the position of the best matching snippet
        text_lower = text.lower()
        q_lower = q.lower().strip()
        best_pos = text_lower.find(q_lower)
        if best_pos < 0:
            for term in expanded_terms[1:]:
                best_pos = text_lower.find(term.lower())
                if best_pos >= 0:
                    break
        if best_pos < 0:
            best_pos = 0

        snippet_start = max(0, best_pos - 80)
        snippet_end = min(len(text), best_pos + len(q) + 80)
        snippet = text[snippet_start:snippet_end].strip()
        if snippet_start > 0:
            snippet = "..." + snippet
        if snippet_end < len(text):
            snippet = snippet + "..."

        pages = chunk.page_numbers or []

        matches.append({
            "chunk_id": str(chunk.id),
            "chunk_index": chunk.chunk_index,
            "page": pages[0] if pages else None,
            "pages": pages,
            "score": round(score * 100, 1),
            "snippet": snippet,
            "char_start": chunk.char_start,
            "char_end": chunk.char_end,
        })

    matches.sort(key=lambda m: m["score"], reverse=True)

    return {
        "query": q,
        "expanded_terms": expanded_terms,
        "total_matches": len(matches),
        "matches": matches[:50],
    }


@router.delete("/{doc_id}", status_code=204)
async def delete_document(
    doc_id: UUID,
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(Document).where(Document.id == doc_id))
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    await db.delete(doc)
    await db.commit()
