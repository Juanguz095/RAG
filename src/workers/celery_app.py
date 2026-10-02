from __future__ import annotations

import logging
import os

# PASSIVE avoids OpenMP spin-wait when idle (was ~350% CPU), BUT it makes
# tesseract ~15x slower — ocr.py forces ACTIVE only during OCR and restores this.
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
os.environ.setdefault("MKL_NUM_THREADS", str(os.cpu_count() or 4))

from celery import Celery
from celery.signals import worker_process_init

redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
celery_app = Celery(
    "rag_worker",
    broker=redis_url,
    backend=redis_url,
    include=["src.workers.ingestion_tasks"],
)
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="America/Lima",
    enable_utc=True,
    task_track_started=True,
    # None = do not recycle on task count (models stay warm)
    worker_max_tasks_per_child=None,
    # recycle if child RSS exceeds ~3.5GB (protects 4g mem_limit / 7.8GB host)
    worker_max_memory_per_child=3500000,
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    broker_connection_retry_on_startup=True,
    # Ingest must never wait behind 2–3 min LLM extract on a -c 1 worker
    task_routes={
        "src.workers.ingestion_tasks.process_document_task": {"queue": "ingest"},
        "src.workers.ingestion_tasks.extract_patient_fields_task": {"queue": "extract"},
    },
)


@worker_process_init.connect
def preload_models(**kwargs):
    """Load only what this worker's queues need (saves RAM + startup)."""
    queues = os.getenv("WORKER_QUEUES", "ingest,extract,celery")
    log = logging.getLogger(__name__)
    log.info(f"Preloading models for queues={queues}...")
    if "ingest" in queues or "celery" in queues:
        from src.services.embeddings import preload
        preload()
    # El LLM (~1 GB) solo lo usa la cola extract. La cola `celery` (tareas
    # periódicas BSC/limpieza) NO lo necesita: precargarlo desperdicia RAM y
    # compite con el LLM de la API durante las consultas. Se carga perezosamente
    # si alguna tarea extract lo requiere.
    if "extract" in queues:
        from src.services.llm import _get_llm
        _get_llm()
    log.info("All models preloaded")

