"""Migración: tabla de candidatos a keyword (conceptos emergentes)."""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "010_keyword_candidates"
down_revision = "009_keyword_priority"


def upgrade() -> None:
    op.create_table(
        "keyword_candidates",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("term", sa.String(200), nullable=False, unique=True, index=True),
        sa.Column("count", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("sample_document", sa.String(500)),
        sa.Column("status", sa.String(20), nullable=False, server_default="proposed"),
        sa.Column("created_at", sa.DateTime()),
    )


def downgrade() -> None:
    op.drop_table("keyword_candidates")
