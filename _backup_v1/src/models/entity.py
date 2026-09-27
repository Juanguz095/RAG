from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, Float, ForeignKey, Index, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.database import Base

if TYPE_CHECKING:
    from src.models.chunk import DocumentChunk


class MedicalEntity(Base):
    __tablename__ = "medical_entities"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    chunk_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("document_chunks.id", ondelete="CASCADE"), nullable=False
    )
    entity_text: Mapped[str] = mapped_column(Text, nullable=False)
    entity_type: Mapped[str] = mapped_column(String(50), nullable=False)
    icd10_code: Mapped[str | None] = mapped_column(String(20))
    atc_code: Mapped[str | None] = mapped_column(String(20))
    confidence: Mapped[float | None] = mapped_column(Float)
    start_char: Mapped[int] = mapped_column(Integer, nullable=False)
    end_char: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    chunk: Mapped[DocumentChunk] = relationship("DocumentChunk", back_populates="entities")

    __table_args__ = (
        Index("idx_entities_chunk", "chunk_id"),
        Index("idx_entities_type", "entity_type"),
        Index("idx_entities_icd10", "icd10_code"),
        Index("idx_entities_atc", "atc_code"),
    )

    def __repr__(self) -> str:
        return f"<MedicalEntity(text={self.entity_text[:30]}, type={self.entity_type})>"
