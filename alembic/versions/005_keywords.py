"""005: keywords + chunk_keywords (PLAN-006 Fase 4 — RAG-013/015-019).

Tablas del catálogo de keywords y sus matches por chunk. Solo CREATE.
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID

revision = "005_keywords"
down_revision = "004_bsc_kpis"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "keywords",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("term", sa.String(200), nullable=False),
        sa.Column("category", sa.String(100), nullable=True),
        sa.Column("description", sa.String(500), nullable=True),
        sa.Column("is_active", sa.Boolean, nullable=False, server_default=sa.text("true")),
        sa.Column("created_by", UUID(as_uuid=True), nullable=True),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.now()),
    )
    op.create_index("ix_keywords_term", "keywords", ["term"], unique=True)
    op.create_table(
        "chunk_keywords",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("chunk_id", UUID(as_uuid=True),
                  sa.ForeignKey("chunks.id", ondelete="CASCADE"), nullable=False),
        sa.Column("keyword_id", UUID(as_uuid=True),
                  sa.ForeignKey("keywords.id", ondelete="CASCADE"), nullable=False),
        sa.Column("match_count", sa.Integer, nullable=False, server_default="1"),
        sa.UniqueConstraint("chunk_id", "keyword_id", name="uq_chunk_keyword"),
    )
    op.create_index("ix_ck_chunk", "chunk_keywords", ["chunk_id"])
    op.create_index("ix_ck_keyword", "chunk_keywords", ["keyword_id"])


def downgrade() -> None:
    op.drop_table("chunk_keywords")
    op.drop_table("keywords")
