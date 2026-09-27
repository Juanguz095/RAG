from __future__ import annotations

from typing import Any, Optional
from uuid import UUID

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    max_chunks: int = Field(default=10, ge=1, le=30)
    doc_filter: Optional[UUID] = None


class ChunkResult(BaseModel):
    id: UUID
    document_id: UUID
    content: str
    snippet: str = ""
    page_numbers: list[int]
    score: float
    document_name: str
    chunk_metadata: Optional[dict[str, Any]] = None
    chunk_index: int = 0
    relevance: int = 0
    matched_terms: list[str] = []


class QueryResponse(BaseModel):
    answer: str
    sources: list[ChunkResult]
    query: str
    processing_time_ms: float


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(default=10, ge=1, le=30)
    doc_filter: Optional[UUID] = None
    page_filter: Optional[int] = Field(default=None, ge=0)


class SearchResponse(BaseModel):
    results: list[ChunkResult]
    query: str
    total: int
