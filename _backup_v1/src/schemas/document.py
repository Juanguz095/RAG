from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class DocumentCreate(BaseModel):
    model_config = ConfigDict(from_attributes=True)


class DocumentResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    filename: str
    original_name: str
    file_hash: str
    mime_type: str | None = None
    file_size: int | None = None
    page_count: int | None = None
    language: str = "es"
    doc_type: str | None = None
    status: str
    metadata: dict = Field(
        default_factory=dict,
        validation_alias="document_metadata",
        serialization_alias="metadata",
    )
    created_at: datetime
    updated_at: datetime
    processed_at: datetime | None = None
    progress_pct: float = 0.0
    progress_phase: str | None = None
    progress_message: str | None = None


class DocumentListResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    items: list[DocumentResponse]
    total: int
    page: int
    size: int
    pages: int


class ChunkResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    document_id: UUID
    chunk_index: int
    content: str
    content_md: str | None = None
    token_count: int | None = None
    page_numbers: list[int] | None = None
    chunk_metadata: dict = Field(default_factory=dict)
    created_at: datetime


class DocumentWithChunksResponse(DocumentResponse):
    chunks: list[ChunkResponse] = []
