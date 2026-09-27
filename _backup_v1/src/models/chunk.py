from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from pgvector.sqlalchemy import Vector
from sqlalchemy import ARRAY, DateTime, ForeignKey, Index, Integer, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.database import Base

if TYPE_CHECKING:
    from src.models.document import Document
    from src.models.entity import MedicalEntity


class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    content_md: Mapped[str | None] = mapped_column(Text)
    token_count: Mapped[int | None] = mapped_column(Integer)
    char_start: Mapped[int | None] = mapped_column(Integer)
    char_end: Mapped[int | None] = mapped_column(Integer)
    page_numbers: Mapped[list[int] | None] = mapped_column(ARRAY(Integer))
    bbox: Mapped[dict | None] = mapped_column(JSONB)
    chunk_metadata: Mapped[dict] = mapped_column(JSONB, default=dict)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(1024))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    document: Mapped[Document] = relationship("Document", back_populates="chunks")
    entities: Mapped[list[MedicalEntity]] = relationship(
        "MedicalEntity", back_populates="chunk", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("idx_chunks_doc_id", "document_id"),
        Index("idx_chunks_embedding_hnsw", "embedding", postgresql_using="hnsw",
              postgresql_with={"m": 16, "ef_construction": 64},
              postgresql_ops={"embedding": "vector_cosine_ops"}),
        Index("idx_chunks_embedding_ivfflat", "embedding", postgresql_using="ivfflat",
              postgresql_with={"lists": 100},
              postgresql_ops={"embedding": "vector_cosine_ops"}),
        Index("idx_chunks_content_trgm", "content", postgresql_using="gin",
              postgresql_ops={"content": "gin_trgm_ops"}),
        Index("idx_chunks_metadata", "chunk_metadata", postgresql_using="gin"),
    )

    def __repr__(self) -> str:
        return f"<DocumentChunk(id={self.id}, doc_id={self.document_id}, idx={self.chunk_index})>"
