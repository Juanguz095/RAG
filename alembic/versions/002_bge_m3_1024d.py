"""Migración PLAN-001 Fase 2: embeddings MiniLM 384-d → BGE-M3 1024-d.

Altera chunks.embedding de 384 a 1024 dimensiones y reconstruye el índice
HNSW con los mismos parámetros (m=16, ef_construction=64). Los embeddings
existentes quedan inválidos: hay que ejecutar scripts/reindex.py después
(o reprocesar los documentos). Downgrade vuelve a 384 (la recolumnar da NULL).
"""
from __future__ import annotations

from alembic import op

revision = "002_bge_m3_1024d"
down_revision = "001"


def upgrade() -> None:
    # Los embeddings 384-d antiguos no son reutilizables: bórralos y re-embédelos.
    op.execute("UPDATE chunks SET embedding = NULL")
    op.execute("DROP INDEX IF EXISTS idx_chunks_embedding_hnsw")
    op.execute("ALTER TABLE chunks ALTER COLUMN embedding TYPE vector(1024)")
    op.execute("""
        CREATE INDEX idx_chunks_embedding_hnsw ON chunks
        USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64)
    """)


def downgrade() -> None:
    op.execute("UPDATE chunks SET embedding = NULL")
    op.execute("DROP INDEX IF EXISTS idx_chunks_embedding_hnsw")
    op.execute("ALTER TABLE chunks ALTER COLUMN embedding TYPE vector(384)")
    op.execute("""
        CREATE INDEX idx_chunks_embedding_hnsw ON chunks
        USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64)
    """)
