import asyncio
import time

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.deps import get_current_user
from src.database import get_db
from src.schemas.query import ExtractRequest, ExtractResponse, QueryRequest, QueryResponse

router = APIRouter()


@router.post("")
async def query_rag(
    request: QueryRequest,
    db: AsyncSession = Depends(get_db),
    user: dict = Depends(get_current_user),
):
    """Consulta RAG completa con streaming."""
    from src.services.context import build_citations
    from src.services.llm import generate_answer_stream
    from src.services.retrieval import hybrid_search

    started = time.perf_counter()
    chunks = await hybrid_search(db, request.query, request.filters, request.top_k)

    if request.stream:
        return StreamingResponse(
            generate_answer_stream(request.query, chunks),
            media_type="text/plain",
            headers={"X-Chunks-Used": ",".join(str(c["id"]) for c in chunks)}
        )
    else:
        async def _collect():
            answer = ""
            async for token in generate_answer_stream(request.query, chunks):
                answer += token
            return answer

        answer = await asyncio.wait_for(_collect(), timeout=120)
        return QueryResponse(
            answer=answer,
            citations=build_citations(chunks),
            chunks_used=[c["id"] for c in chunks],
            latency_ms=round((time.perf_counter() - started) * 1000),
        )


@router.post("/search")
async def search_only(
    request: QueryRequest,
    db: AsyncSession = Depends(get_db),
    user: dict = Depends(get_current_user),
):
    """Solo retrieval (sin LLM) para debugging."""
    from src.services.retrieval import hybrid_search
    chunks = await hybrid_search(db, request.query, request.filters, request.top_k)
    from src.schemas.query import ChunkSearchResult
    return [
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
    ]


@router.post("/extract", response_model=ExtractResponse)
async def structured_extraction(
    request: ExtractRequest,
    db: AsyncSession = Depends(get_db),
    user: dict = Depends(get_current_user),
):
    """Extracción estructurada guiada por JSON Schema."""
    from src.services.context import build_context
    from src.services.llm import extract_structured
    from src.services.retrieval import hybrid_search

    chunks = await hybrid_search(db, request.query, request.filters, 10)
    context = build_context(chunks, max_tokens=2500)

    result = await extract_structured(request.query, request.schema_, context)
    return result
