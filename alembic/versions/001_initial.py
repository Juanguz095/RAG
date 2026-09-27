"""Initial schema

Revision ID: 001
Revises: 
Create Date: 2025-01-01
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID, JSONB, ARRAY

revision = "001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")

    op.create_table(
        "users",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("username", sa.String(50), unique=True, nullable=False),
        sa.Column("email", sa.String(255), unique=True, nullable=False),
        sa.Column("hashed_password", sa.String(255), nullable=False),
        sa.Column("role", sa.String(20), server_default="assistant"),
        sa.Column("is_active", sa.Integer, server_default="1"),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.now()),
    )

    op.create_table(
        "documents",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("filename", sa.String(255), nullable=False),
        sa.Column("original_name", sa.String(500), nullable=False),
        sa.Column("file_hash", sa.String(64), nullable=False),
        sa.Column("mime_type", sa.String(100)),
        sa.Column("file_size", sa.Integer),
        sa.Column("page_count", sa.Integer),
        sa.Column("status", sa.String(20), server_default="pending"),
        sa.Column("total_chunks", sa.Integer, server_default="0"),
        sa.Column("metadata", JSONB, server_default="{}"),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.now()),
        sa.Column("processed_at", sa.DateTime),
    )
    op.create_index("idx_documents_file_hash", "documents", ["file_hash"])

    op.create_table(
        "chunks",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("document_id", UUID(as_uuid=True), sa.ForeignKey("documents.id", ondelete="CASCADE"), nullable=False),
        sa.Column("chunk_index", sa.Integer, nullable=False),
        sa.Column("content", sa.Text, nullable=False),
        sa.Column("page_numbers", ARRAY(sa.Integer), server_default="{}"),
        sa.Column("bbox", JSONB),
        sa.Column("token_count", sa.Integer),
        sa.Column("chunk_metadata", JSONB, server_default="{}"),
        sa.Column("embedding", sa.String),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.now()),
    )
    op.create_index("idx_chunks_doc_id", "chunks", ["document_id"])

    op.execute("""
        CREATE INDEX idx_chunks_embedding_hnsw ON chunks
        USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64)
    """)

    op.execute("""
        CREATE INDEX idx_chunks_content_trgm ON chunks
        USING gin (content gin_trgm_ops)
    """)

    op.create_table(
        "medical_entities",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("chunk_id", UUID(as_uuid=True), sa.ForeignKey("chunks.id", ondelete="CASCADE"), nullable=False),
        sa.Column("entity_text", sa.Text, nullable=False),
        sa.Column("entity_type", sa.String(50), nullable=False),
        sa.Column("icd10_code", sa.String(20)),
        sa.Column("confidence", sa.Float),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.now()),
    )
    op.create_index("idx_entities_chunk", "medical_entities", ["chunk_id"])

    op.create_table(
        "medical_synonyms",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("canonical", sa.String(200), nullable=False),
        sa.Column("synonym", sa.String(200), nullable=False),
        sa.Column("category", sa.String(50)),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.now()),
    )
    op.create_index("idx_synonyms_canonical", "medical_synonyms", ["canonical"])
    op.create_index("idx_synonyms_synonym", "medical_synonyms", ["synonym"])


def downgrade() -> None:
    op.drop_table("medical_synonyms")
    op.drop_table("medical_entities")
    op.execute("DROP INDEX IF EXISTS idx_chunks_embedding_hnsw")
    op.execute("DROP INDEX IF EXISTS idx_chunks_content_trgm")
    op.drop_table("chunks")
    op.drop_table("documents")
    op.drop_table("users")
    op.execute("DROP EXTENSION IF EXISTS pg_trgm")
    op.execute("DROP EXTENSION IF EXISTS vector")
