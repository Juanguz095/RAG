"""Migración: embeddings BGE-M3 1024-d → MiniLM multilingüe 384-d.

Alinea `chunks.embedding` con el nuevo `EMBED_MODEL_NAME` por defecto
(MiniLM multilingüe 384-d, ~35x más rápido que BGE-M3 en CPU).

Los embeddings existentes quedan inválidos: se anulan y hay que reprocesar
los documentos (`scripts/reindex.py` o `POST /api/v1/documents/{id}/reprocess`).
"""
from __future__ import annotations

from alembic import op

revision = "008_embed_384d"
down_revision = "007_chat"


def _rebuild(dim: int) -> None:
    op.execute("UPDATE chunks SET embedding = NULL")
    op.execute("DROP INDEX IF EXISTS idx_chunks_embedding_hnsw")
    op.execute(f"ALTER TABLE chunks ALTER COLUMN embedding TYPE vector({dim})")
    op.execute(
        """
        CREATE INDEX idx_chunks_embedding_hnsw ON chunks
        USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64)
        """
    )


def upgrade() -> None:
    _rebuild(384)


def downgrade() -> None:
    _rebuild(1024)
