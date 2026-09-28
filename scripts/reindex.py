"""Re-embed de chunks existentes tras migrar a BGE-M3 1024-d (PLAN-001 Fase 2).

Se ejecuta DENTRO del contenedor del worker (tiene sentence-transformers y
acceso a la BD). Por defecto re-embeddea chunks existentes con el modelo
actual de config; con --full elimina chunks y re-encola el documento entero.

Uso:
    python scripts/reindex.py            # re-embed en sitio
    python scripts/reindex.py --full     # reproceso completo por Celery
"""
from __future__ import annotations

import argparse
import sys
import time

sys.path.insert(0, "/app")

from src.config import get_settings  # noqa: E402
from src.database import Chunk, Document  # noqa: E402


def reembed_in_place(batch_size: int = 16) -> None:
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session
    from src.services.embeddings import encode_texts

    settings = get_settings()
    sync_url = settings.DATABASE_URL.replace("+asyncpg", "+psycopg2")
    engine = create_engine(sync_url, echo=False)

    t0 = time.time()
    with Session(engine) as db:
        total = (
            db.query(Chunk)
            .filter(Chunk.embedding.is_(None))
            .count()
        )
        print(f"chunks sin embedding: {total}")
        done = 0
        while True:
            rows = (
                db.query(Chunk)
                .filter(Chunk.embedding.is_(None))
                .limit(batch_size)
                .all()
            )
            if not rows:
                break
            embs = encode_texts([r.content for r in rows], batch_size=batch_size)
            for chunk, emb in zip(rows, embs):
                chunk.embedding = [float(x) for x in emb]
            db.commit()
            done += len(rows)
            rate = done / max(1e-9, time.time() - t0)
            print(f"  {done}/{total} ({rate:.1f} chunks/s)", flush=True)
    print(f"OK en {time.time()-t0:.1f}s")


def reprocess_all() -> None:
    from pathlib import Path

    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session
    from src.api.v1.documents import UPLOAD_DIR
    from src.workers.ingestion_tasks import enqueue_process_document

    settings = get_settings()
    sync_url = settings.DATABASE_URL.replace("+asyncpg", "+psycopg2")
    engine = create_engine(sync_url, echo=False)
    with Session(engine) as db:
        for doc in db.query(Document).all():
            pdf_path = UPLOAD_DIR / doc.filename
            if pdf_path.exists():
                doc.status = "pending"
                doc.total_chunks = 0
                db.commit()
                enqueue_process_document(str(doc.id), str(pdf_path))
                print(f"encolado: {doc.original_name}")
            else:
                print(f"sin PDF en disco para {doc.original_name}; omitido")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="reproceso completo via Celery")
    args = parser.parse_args()
    if args.full:
        reprocess_all()
    else:
        reembed_in_place()
