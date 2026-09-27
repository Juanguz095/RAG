from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class QueryRequest(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    query: str = Field(..., min_length=1, max_length=2000)
    filters: dict = Field(default_factory=dict)
    top_k: int = Field(default=20, ge=1, le=50)
    stream: bool = True


class SearchRequest(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    query: str = Field(..., min_length=1, max_length=2000)
    filters: dict = Field(default_factory=dict)
    top_k: int = Field(default=20, ge=1, le=50)
    search_type: str = Field(default="hybrid", pattern="^(vector|keyword|hybrid)$")


class ChunkSearchResult(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    document_id: UUID
    chunk_index: int
    content: str
    content_md: str | None = None
    page_numbers: list[int] | None = None
    chunk_metadata: dict = Field(default_factory=dict)
    score: float
    search_type: str


class SearchResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    results: list[ChunkSearchResult]
    total: int
    query_time_ms: int


class ExtractRequest(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    query: str = Field(..., min_length=1, max_length=2000)
    schema_: dict = Field(..., alias="schema", description="JSON Schema para extracción estructurada")
    filters: dict = Field(default_factory=dict)


class ExtractResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    data: dict
    citations: list[dict]
    confidence: float


class QueryResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    answer: str
    citations: list[dict]
    chunks_used: list[UUID]
    latency_ms: int
