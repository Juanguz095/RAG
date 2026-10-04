"""Migración: prioridad en el catálogo de keywords (1=baja, 2=media, 3=alta)."""
from __future__ import annotations

from alembic import op

revision = "009_keyword_priority"
down_revision = "008_embed_384d"


def upgrade() -> None:
    op.execute("ALTER TABLE keywords ADD COLUMN IF NOT EXISTS priority integer NOT NULL DEFAULT 1")


def downgrade() -> None:
    op.execute("ALTER TABLE keywords DROP COLUMN IF EXISTS priority")
