import asyncio
import logging

from src.config import get_settings
from src.database import async_session_factory
from src.services.ingestion import generate_embeddings_for_document
from src.workers.celery_app import celery_app

logger = logging.getLogger(__name__)
settings = get_settings()


def _run_async(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


@celery_app.task(bind=True, max_retries=3, default_retry_delay=30)
def generate_embeddings_batch(self, document_id: str):
    logger.info(f"Embeddings batch task for document {document_id}")

    async def _generate():
        async with async_session_factory() as db:
            try:
                count = await generate_embeddings_for_document(db, document_id)
                logger.info(f"Generated {count} embeddings for {document_id}")
                return {"status": "completed", "count": count, "document_id": document_id}
            except Exception as e:
                logger.error(f"Embedding batch error for {document_id}: {e}")
                raise

    try:
        return _run_async(_generate())
    except Exception as exc:
        raise self.retry(exc=exc) from None


@celery_app.task
def preload_models():
    logger.info("Preloading models...")
    try:
        from src.services.embeddings import preload as preload_emb
        preload_emb()
    except Exception as e:
        logger.error(f"Failed to preload embeddings: {e}")

    try:
        from src.services.llm import preload as preload_llm
        preload_llm()
    except Exception as e:
        logger.error(f"Failed to preload LLM: {e}")

    logger.info("Models preloaded")
