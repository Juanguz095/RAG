"""006: proposals — validación humana (PLAN-006 Fase 5, RAG-032)."""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID

revision = "006_proposals"
down_revision = "005_keywords"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "proposals",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("chunk_id", UUID(as_uuid=True),
                  sa.ForeignKey("chunks.id", ondelete="CASCADE"), nullable=False),
        sa.Column("proposer", UUID(as_uuid=True), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("content", sa.Text, nullable=False),
        sa.Column("status", sa.String(20), nullable=False, server_default="proposed"),
        sa.Column("reviewer", UUID(as_uuid=True), sa.ForeignKey("users.id"), nullable=True),
        sa.Column("reviewed_at", sa.DateTime, nullable=True),
        sa.Column("comment", sa.String(500), nullable=True),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.now()),
    )
    op.create_index("ix_proposals_chunk", "proposals", ["chunk_id"])
    op.create_index("ix_proposals_status", "proposals", ["status"])


def downgrade() -> None:
    op.drop_table("proposals")
