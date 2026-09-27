import time

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.deps import get_current_user
from src.database import get_db
from src.schemas.query import ChunkSearchResult, SearchRequest, SearchResponse

router = APIRouter()


@router.post("/vector", response_model=SearchResponse)
async def vector_search(
    request: SearchRequest,
    db: AsyncSession = Depends(get_db),
    user: dict = Depends(get_current_user),
):
    """Búsqueda vectorial pura."""
    from src.services.embeddings import encode_single
    from src.services.retrieval import vector_search as vs

    started = time.perf_counter()
    query_emb = encode_single(request.query)
    chunks = await vs(db, query_emb, request.top_k, request.filters)

    return SearchResponse(
        results=[
            ChunkSearchResult(
                id=c["id"],
                document_id=c["document_id"],
                chunk_index=c["chunk_index"],
                content=c["content"],
                content_md=c.get("content_md"),
                page_numbers=c.get("page_numbers"),
                chunk_metadata=c.get("chunk_metadata", {}),
                score=c.get("score", 0.0),
                search_type="vector"
            )
            for c in chunks
        ],
        total=len(chunks),
        query_time_ms=round((time.perf_counter() - started) * 1000)
    )


@router.post("/keyword", response_model=SearchResponse)
async def keyword_search(
    request: SearchRequest,
    db: AsyncSession = Depends(get_db),
    user: dict = Depends(get_current_user),
):
    """Búsqueda full-text (pg_trgm)."""
    from src.services.retrieval import keyword_search as ks
    started = time.perf_counter()
    chunks = await ks(db, request.query, request.top_k, request.filters)

    return SearchResponse(
        results=[
            ChunkSearchResult(
                id=c["id"],
                document_id=c["document_id"],
                chunk_index=c["chunk_index"],
                content=c["content"],
                content_md=c.get("content_md"),
                page_numbers=c.get("page_numbers"),
                chunk_metadata=c.get("chunk_metadata", {}),
                score=c.get("score", 0.0),
                search_type="keyword"
            )
            for c in chunks
        ],
        total=len(chunks),
        query_time_ms=round((time.perf_counter() - started) * 1000)
    )


@router.post("/hybrid", response_model=SearchResponse)
async def hybrid_search(
    request: SearchRequest,
    db: AsyncSession = Depends(get_db),
    user: dict = Depends(get_current_user),
):
    """Búsqueda híbrida + rerank."""
    from src.services.retrieval import hybrid_search as hs
    started = time.perf_counter()
    chunks = await hs(db, request.query, request.filters, request.top_k)

    return SearchResponse(
        results=[
            ChunkSearchResult(
                id=c["id"],
                document_id=c["document_id"],
                chunk_index=c["chunk_index"],
                content=c["content"],
                content_md=c.get("content_md"),
                page_numbers=c.get("page_numbers"),
                chunk_metadata=c.get("chunk_metadata", {}),
                score=c.get("rerank_score", c.get("score", 0.0)),
                search_type="hybrid"
            )
            for c in chunks
        ],
        total=len(chunks),
        query_time_ms=round((time.perf_counter() - started) * 1000)
    )
