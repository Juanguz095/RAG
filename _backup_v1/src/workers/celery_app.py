from celery import Celery

from src.config import get_settings

settings = get_settings()

celery_app = Celery(
    "rag_medical",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=100,
    task_routes={
        "src.workers.ingestion_tasks.*": {"queue": "ingestion"},
        "src.workers.embedding_tasks.*": {"queue": "embeddings"},
    },
    task_default_queue="default",
)

celery_app.autodiscover_tasks(["src.workers"])
