import asyncio
import logging
from uuid import UUID

from src.config import get_settings
from src.database import async_session_factory
from src.services.ingestion import generate_embeddings_for_document, process_document_full
from src.workers.celery_app import celery_app

logger = logging.getLogger(__name__)
settings = get_settings()
_worker_loop: asyncio.AbstractEventLoop | None = None


def _run_async(coro):
    global _worker_loop
    if _worker_loop is None or _worker_loop.is_closed():
        _worker_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(_worker_loop)
    return _worker_loop.run_until_complete(coro)


@celery_app.task(bind=True, max_retries=3, default_retry_delay=60)
def process_document_task(self, document_id: str, file_content_b64: str | None = None):
    import base64

    logger.info(f"Processing document {document_id}")

    if file_content_b64:
        file_content = base64.b64decode(file_content_b64)
    else:
        file_content = _fetch_file_from_minio(document_id)

    async def _process():
        async with async_session_factory() as db:
            try:
                doc = await process_document_full(db, document_id, file_content)
                return {"status": doc.status, "document_id": document_id}
            except Exception as e:
                logger.error(f"Error processing {document_id}: {e}")
                raise

    try:
        return _run_async(_process())
    except Exception as exc:
        logger.error(f"Task failed for {document_id}: {exc}")
        raise self.retry(exc=exc) from None


@celery_app.task(bind=True, max_retries=3)
def generate_embeddings_task(self, document_id: str):
    logger.info(f"Generating embeddings for document {document_id}")

    async def _generate():
        async with async_session_factory() as db:
            try:
                count = await generate_embeddings_for_document(db, document_id)
                return {"embeddings_generated": count, "document_id": document_id}
            except Exception as e:
                logger.error(f"Error generating embeddings for {document_id}: {e}")
                raise

    try:
        return _run_async(_generate())
    except Exception as exc:
        logger.error(f"Embedding task failed for {document_id}: {exc}")
        raise self.retry(exc=exc) from None


def _fetch_file_from_minio(document_id: str) -> bytes:

    from minio import Minio
    from sqlalchemy import select

    from src.models.document import Document

    async def _fetch():
        async with async_session_factory() as db:
            result = await db.execute(
                select(Document).where(Document.id == UUID(document_id))
            )
            doc = result.scalar_one_or_none()
            if not doc:
                raise ValueError(f"Document not found: {document_id}")

            client = Minio(
                settings.MINIO_ENDPOINT,
                access_key=settings.MINIO_ACCESS_KEY,
                secret_key=settings.MINIO_SECRET_KEY,
                secure=settings.MINIO_SECURE,
            )
            response = client.get_object(settings.MINIO_BUCKET, doc.filename)
            try:
                return response.read()
            finally:
                response.close()
                response.release_conn()

    return _run_async(_fetch())
