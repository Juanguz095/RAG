"""Document validation, storage and end-to-end ingestion pipeline."""

from __future__ import annotations

import gc
import io
import logging
from datetime import UTC, date, datetime
from uuid import UUID

import fitz
from minio import Minio
from sqlalchemy import delete, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.config import get_settings
from src.models.chunk import DocumentChunk
from src.models.document import Document
from src.schemas.document import (
    ChunkResponse,
    DocumentListResponse,
    DocumentResponse,
    DocumentWithChunksResponse,
)
from src.services.anonymization import anonymize_text
from src.services.chunking import semantic_medical_chunking as chunk_text
from src.services.embeddings import EMBEDDING_DIMENSION, encode_texts
from src.services.ner import extract_medical_entities
from src.services.ocr import ocr_page
from src.utils.hashing import compute_file_hash
from src.utils.pdf import extract_page_images, validate_pdf

logger = logging.getLogger(__name__)
settings = get_settings()

_minio_client: Minio | None = None


class DuplicateDocumentError(Exception):
    def __init__(self, existing_id: UUID):
        self.existing_id = existing_id
        super().__init__(f"Documento duplicado. ID existente: {existing_id}")


class InvalidPDFError(Exception):
    pass


class DocumentTooLargeError(Exception):
    pass


class StorageError(Exception):
    pass


async def update_progress(
    db: AsyncSession, doc: Document, pct: float, phase: str, message: str
) -> None:
    """Update document progress fields and commit to DB for real-time tracking."""
    doc.progress_pct = pct
    doc.progress_phase = phase
    doc.progress_message = message
    await db.commit()


def get_minio_client() -> Minio:
    global _minio_client
    if _minio_client is None:
        _minio_client = Minio(
            settings.MINIO_ENDPOINT,
            access_key=settings.MINIO_ACCESS_KEY,
            secret_key=settings.MINIO_SECRET_KEY,
            secure=settings.MINIO_SECURE,
        )
    return _minio_client


def _store_document(content: bytes, object_key: str, content_type: str) -> None:
    try:
        client = get_minio_client()
        if not client.bucket_exists(settings.MINIO_BUCKET):
            client.make_bucket(settings.MINIO_BUCKET)
        client.put_object(
            settings.MINIO_BUCKET,
            object_key,
            io.BytesIO(content),
            length=len(content),
            content_type=content_type,
        )
    except Exception as exc:
        raise StorageError(f"No se pudo guardar el documento: {exc}") from exc


async def validate_and_deduplicate(
    db: AsyncSession,
    content: bytes,
    filename: str,
    mime_type: str,
) -> Document:
    """Validate a PDF, reject duplicates and persist the original in MinIO."""
    if not filename.lower().endswith(".pdf") or mime_type not in {
        "application/pdf",
        "application/octet-stream",
    }:
        raise InvalidPDFError("Solo se aceptan archivos PDF")
    if not content:
        raise InvalidPDFError("El archivo está vacío")
    if len(content) > settings.MAX_UPLOAD_SIZE_BYTES:
        raise DocumentTooLargeError(
            f"El archivo supera el máximo de {settings.MAX_UPLOAD_SIZE_BYTES} bytes"
        )

    file_hash = compute_file_hash(content)
    existing = await db.execute(select(Document).where(Document.file_hash == file_hash))
    if existing_doc := existing.scalar_one_or_none():
        raise DuplicateDocumentError(existing_doc.id)

    try:
        pdf_info = validate_pdf(content)
    except Exception as exc:
        raise InvalidPDFError(f"PDF inválido: {exc}") from exc
    if pdf_info.page_count == 0:
        raise InvalidPDFError("El PDF no contiene páginas")
    if pdf_info.page_count > settings.MAX_PAGES_PER_DOC:
        raise DocumentTooLargeError(f"Máximo {settings.MAX_PAGES_PER_DOC} páginas")

    object_key = f"{file_hash[:2]}/{file_hash}.pdf"
    _store_document(content, object_key, "application/pdf")
    doc = Document(
        filename=object_key,
        original_name=filename[:512],
        file_hash=file_hash,
        mime_type="application/pdf",
        file_size=len(content),
        page_count=pdf_info.page_count,
        language="es",
        status="processing",
        document_metadata={"pdf_metadata": pdf_info.metadata, "is_scanned": pdf_info.is_scanned},
    )
    db.add(doc)
    await db.commit()
    await db.refresh(doc)
    return doc


def _extract_text_and_boundaries(content: bytes) -> tuple[str, list[tuple[int, int, int]], bool]:
    pdf = fitz.open(stream=content, filetype="pdf")
    page_texts: list[str] = []
    scanned_pages: list[int] = []
    for page in pdf:
        text = page.get_text("text").strip()
        page_texts.append(text)
        if len(text) < settings.MIN_TEXT_CHARS_PER_PAGE:
            scanned_pages.append(len(page_texts) - 1)
    pdf.close()

    scanned = bool(scanned_pages)
    if scanned_pages:
        images = extract_page_images(content, dpi=settings.OCR_DPI)
        from src.services.ocr import ocr_page
        for page_index in scanned_pages:
            try:
                page_texts[page_index] = ocr_page(images[page_index], page_num=page_index).text.strip()
            except Exception as exc:
                logger.warning("OCR failed on scanned page %d: %s", page_index + 1, exc)
            images[page_index] = None
            gc.collect()
        del images
        gc.collect()

    combined_parts: list[str] = []
    boundaries: list[tuple[int, int, int]] = []
    cursor = 0
    for page_number, page_text in enumerate(page_texts, start=1):
        if not page_text:
            continue
        if combined_parts:
            combined_parts.append("\n\n")
            cursor += 2
        start = cursor
        combined_parts.append(page_text)
        cursor += len(page_text)
        boundaries.append((page_number, start, cursor))
    return "".join(combined_parts), boundaries, scanned


async def marker_extract(pdf_path: str) -> tuple[str, dict]:
    """Extract Markdown/layout with Marker when available, with a PDF fallback."""
    try:
        from marker.convert import convert_single_pdf

        markdown, layout, _ = convert_single_pdf(pdf_path)
        return markdown, layout or {"pages": []}
    except Exception as exc:
        logger.info("Marker unavailable; using PyMuPDF fallback: %s", exc)
        path = fitz.open(pdf_path)
        pages = [{"page": i + 1, "text": page.get_text("text")} for i, page in enumerate(path)]
        markdown = "\n\n".join(page["text"].strip() for page in pages if page["text"].strip())
        path.close()
        return markdown, {"pages": pages, "fallback": "pymupdf"}


async def semantic_medical_chunking(
    markdown: str, layout: dict, doc_id: UUID
) -> list[dict]:
    page_boundaries = []
    cursor = 0
    for page in layout.get("pages", []):
        page_text = page.get("text", "")
        if not page_text:
            continue
        page_boundaries.append((int(page.get("page", 1)), cursor, cursor + len(page_text)))
        cursor += len(page_text) + 2
    chunk_items = chunk_text(
        markdown,
        document_id=str(doc_id),
        page_boundaries=page_boundaries,
    )
    return [
        {
            "document_id": doc_id,
            "content": item.content,
            "content_md": item.content,
            "token_count": item.token_count,
            "char_start": item.char_start,
            "char_end": item.char_end,
            "page_numbers": item.page_numbers,
            "chunk_metadata": item.chunk_metadata,
        }
        for item in chunk_items
    ]


async def generate_embeddings_batch(texts: list[str]) -> list[list[float]]:
    return encode_texts(texts, batch_size=settings.EMBEDDING_BATCH_SIZE)


async def save_chunks_with_embeddings(
    chunks: list[dict], embeddings: list[list[float]], db: AsyncSession
) -> list[DocumentChunk]:
    if len(chunks) != len(embeddings):
        raise ValueError("La cantidad de embeddings no coincide con la de chunks")
    if any(len(embedding) != EMBEDDING_DIMENSION for embedding in embeddings):
        raise ValueError(f"Cada embedding debe tener {EMBEDDING_DIMENSION} dimensiones")

    db_chunks = [
        DocumentChunk(
            document_id=chunk_data["document_id"],
            chunk_index=index,
            content=chunk_data["content"],
            content_md=chunk_data.get("content_md"),
            token_count=chunk_data.get("token_count"),
            char_start=chunk_data.get("char_start"),
            char_end=chunk_data.get("char_end"),
            page_numbers=chunk_data.get("page_numbers"),
            bbox=chunk_data.get("bbox"),
            chunk_metadata=chunk_data.get("chunk_metadata", {}),
            embedding=embedding,
        )
        for index, (chunk_data, embedding) in enumerate(zip(chunks, embeddings, strict=True))
    ]
    db.add_all(db_chunks)
    await db.flush()
    return db_chunks


async def process_document_full(
    db: AsyncSession, document_id: str, file_content: bytes
) -> Document:
    """Run extraction/OCR, chunking, embeddings and NER for one document."""
    doc = await db.get(Document, UUID(document_id))
    if doc is None:
        raise ValueError(f"Documento no encontrado: {document_id}")
    try:
        doc.status = "processing"
        await update_progress(db, doc, 0.0, "processing", "Iniciando procesamiento...")

        doc.status = "ocr"
        await update_progress(db, doc, 5.0, "ocr", "Extrayendo texto del PDF...")
        text, boundaries, scanned = _extract_text_and_boundaries(file_content)
        del file_content
        gc.collect()
        if not text.strip():
            raise InvalidPDFError("No se pudo extraer texto del PDF")
        await update_progress(db, doc, 50.0, "ocr", "Texto extraído correctamente")

        doc.status = "chunking"
        doc.document_metadata = {**(doc.document_metadata or {}), "is_scanned": scanned}
        anonymized = anonymize_text(text)
        text = anonymized.text
        doc.document_metadata = {
            **(doc.document_metadata or {}),
            "anonymized_replacements": anonymized.replacements,
            "anonymized_entity_types": list(anonymized.entity_types),
        }
        await update_progress(db, doc, 55.0, "chunking", "Dividiendo en fragmentos semánticos...")
        chunk_data = await semantic_medical_chunking(text, {"pages": [{"page": p, "text": text[s:e]} for p, s, e in boundaries]}, doc.id)
        if not chunk_data:
            raise ValueError("El documento no produjo chunks")
        await update_progress(db, doc, 65.0, "chunking", f"{len(chunk_data)} fragmentos creados")

        doc.status = "embedding"
        await update_progress(db, doc, 70.0, "embedding", "Generando embeddings con BGE-M3...")
        embeddings = await generate_embeddings_batch([item["content"] for item in chunk_data])
        await db.execute(delete(DocumentChunk).where(DocumentChunk.document_id == doc.id))
        db_chunks = await save_chunks_with_embeddings(chunk_data, embeddings, db)
        await update_progress(db, doc, 85.0, "embedding", "Embeddings generados. Extrayendo entidades médicas...")
        for db_chunk in db_chunks:
            entities = extract_medical_entities(db_chunk.content)
            entity_model = __import__("src.models.entity", fromlist=["MedicalEntity"]).MedicalEntity
            db.add_all(
                [
                    entity_model(
                        chunk_id=db_chunk.id,
                        entity_text=entity.text,
                        entity_type=entity.entity_type,
                        confidence=entity.confidence,
                        start_char=entity.start_char,
                        end_char=entity.end_char,
                        icd10_code=entity.icd10_code,
                        atc_code=entity.atc_code,
                    )
                    for entity in entities
                ]
            )

        doc.status = "completed"
        doc.processed_at = datetime.now(UTC)
        await update_progress(db, doc, 100.0, "completed", "Procesamiento completado")
        await db.refresh(doc)
        return doc
    except Exception as exc:
        await db.rollback()
        doc = await db.get(Document, UUID(document_id))
        if doc is not None:
            doc.status = "failed"
            doc.document_metadata = {**(doc.document_metadata or {}), "error": str(exc)[:2000]}
            await db.commit()
        raise


async def generate_embeddings_for_document(db: AsyncSession, document_id: str) -> int:
    """Generate embeddings for legacy chunks that do not have one."""
    doc_uuid = UUID(document_id)
    result = await db.execute(
        select(DocumentChunk).where(
            DocumentChunk.document_id == doc_uuid,
            DocumentChunk.embedding.is_(None),
        )
    )
    chunks = list(result.scalars().all())
    if not chunks:
        return 0
    embeddings = await generate_embeddings_batch([chunk.content for chunk in chunks])
    for chunk, embedding in zip(chunks, embeddings, strict=True):
        chunk.embedding = embedding
    await db.commit()
    return len(chunks)


async def search_documents(
    db: AsyncSession,
    status: str | None,
    doc_type: str | None,
    date_from: date | None,
    date_to: date | None,
    q: str | None,
    page: int,
    size: int,
) -> DocumentListResponse:
    query = select(Document)
    if status:
        query = query.where(Document.status == status)
    if doc_type:
        query = query.where(Document.doc_type == doc_type)
    if date_from:
        query = query.where(Document.created_at >= date_from)
    if date_to:
        query = query.where(Document.created_at < datetime.combine(date_to, datetime.min.time()).replace(tzinfo=UTC))
    if q:
        query = query.where(or_(Document.original_name.ilike(f"%{q}%"), Document.file_hash == q))
    total = (await db.execute(select(func.count()).select_from(query.subquery()))).scalar_one()
    result = await db.execute(
        query.order_by(Document.created_at.desc()).offset((page - 1) * size).limit(size)
    )
    items = result.scalars().all()
    return DocumentListResponse(
        items=[DocumentResponse.model_validate(item) for item in items],
        total=total,
        page=page,
        size=size,
        pages=(total + size - 1) // size if total else 1,
    )


async def get_document_with_chunks(
    db: AsyncSession, doc_id: UUID
) -> DocumentWithChunksResponse | None:
    doc = await db.get(Document, doc_id)
    if doc is None:
        return None
    result = await db.execute(
        select(DocumentChunk).where(DocumentChunk.document_id == doc_id).order_by(DocumentChunk.chunk_index)
    )
    return DocumentWithChunksResponse(
        **DocumentResponse.model_validate(doc).model_dump(),
        chunks=[ChunkResponse.model_validate(chunk) for chunk in result.scalars().all()],
    )


async def get_page_with_ocr_overlay(
    db: AsyncSession, doc_id: UUID, page_num: int
) -> bytes | None:
    if page_num < 1:
        return None
    doc = await db.get(Document, doc_id)
    if doc is None:
        return None
    client = get_minio_client()
    response = client.get_object(settings.MINIO_BUCKET, doc.filename)
    try:
        content = response.read()
    finally:
        response.close()
        response.release_conn()
    pdf = fitz.open(stream=content, filetype="pdf")
    if page_num > len(pdf):
        pdf.close()
        return None
    page = pdf[page_num - 1]
    pixmap = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False)
    from PIL import Image

    image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)

    # Render the page directly. OCR overlays are optional and must not prevent
    # the PDF preview from loading when chunk metadata is incomplete.
    output = io.BytesIO()
    image.save(output, format="PNG")
    pdf.close()
    return output.getvalue()


async def get_document_file(db: AsyncSession, doc_id: UUID) -> bytes | None:
    """Read the original PDF from object storage for the embedded viewer."""
    doc = await db.get(Document, doc_id)
    if doc is None:
        return None

    response = get_minio_client().get_object(settings.MINIO_BUCKET, doc.filename)
    try:
        return response.read()
    finally:
        response.close()
        response.release_conn()


async def delete_document(db: AsyncSession, doc_id: UUID) -> None:
    doc = await db.get(Document, doc_id)
    if doc is None:
        return
    try:
        get_minio_client().remove_object(settings.MINIO_BUCKET, doc.filename)
    except Exception as exc:
        logger.warning("No se pudo eliminar el objeto %s: %s", doc.filename, exc)
    await db.delete(doc)
    await db.commit()
