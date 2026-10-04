"""Migración: feedback del usuario en mensajes del chat (spec §49)."""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "011_message_feedback"
down_revision = "010_keyword_candidates"


def upgrade() -> None:
    op.add_column("messages", sa.Column("feedback", sa.String(20), nullable=True))
    op.add_column("messages", sa.Column("feedback_at", sa.DateTime(), nullable=True))


def downgrade() -> None:
    op.drop_column("messages", "feedback_at")
    op.drop_column("messages", "feedback")
