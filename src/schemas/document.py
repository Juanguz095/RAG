from __future__ import annotations

from datetime import datetime
from typing import Any, Optional
from uuid import UUID

from pydantic import BaseModel


class DocumentResponse(BaseModel):
    id: UUID
    filename: str
    original_name: str
    file_size: Optional[int]
    page_count: Optional[int]
    status: str
    total_chunks: int
    created_at: datetime
    processed_at: Optional[datetime]
    metadata_: Optional[dict[str, Any]] = None
    timings: Optional[dict[str, Any]] = None
    domains: Optional[list[str]] = None

    class Config:
        from_attributes = True


class DocumentListResponse(BaseModel):
    documents: list[DocumentResponse]
    total: int
    page: int
    pages: int
