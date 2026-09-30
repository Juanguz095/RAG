from __future__ import annotations

import json
import logging
import time
from typing import AsyncGenerator
from uuid import UUID

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.deps import get_current_user
from src.config import get_settings
from src.database import User, get_db
from src.schemas.query import (
    ChunkResult,
    QueryRequest,
    QueryResponse,
    SearchRequest,
    SearchResponse,
)
from src.services.context import build_context
from src.services.llm import generate_answer_stream, generate_answer_timed
from src.services.retrieval import hybrid_search

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["query"])
settings = get_settings()


def _make_snippet(content: str, matched_terms: list[str], radius: int = 120) -> str:
    if not matched_terms:
        return content[:radius * 2]
    text_l = content.lower()
    best_pos = -1
    for t in matched_terms:
        pos = text_l.find(t.lower())
        if pos >= 0 and (best_pos < 0 or pos < best_pos):
            best_pos = pos
    if best_pos < 0:
        return content[:radius * 2]
    start = max(0, best_pos - radius)
    end = min(len(content), best_pos + radius)
    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(content) else ""
    return prefix + content[start:end] + suffix


def _results_to_sources(results) -> list[ChunkResult]:
    return [
        ChunkResult(
            id=r.chunk_id,
            document_id=r.document_id,
            content=r.content[:500],
            snippet=_make_snippet(r.content, r.matched_terms),
            page_numbers=r.page_numbers,
            score=round(r.score, 4),
            document_name=r.document_name,
            chunk_metadata=r.chunk_metadata,
            chunk_index=r.chunk_index,
            relevance=r.relevance,
            matched_terms=r.matched_terms,
        )
        for r in results
    ]


@router.post("/query", response_model=QueryResponse)
async def query_rag(
    req: QueryRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User | None = Depends(get_current_user),
):
    from src.services import reranker as rerank_svc

    t_total = time.time()
    doc_filter = str(req.doc_filter) if req.doc_filter else None

    # Retrival amplio: pool de candidatos para el re-ranking
    t0 = time.time()
    results = await hybrid_search(req.query, db, top_k=req.max_chunks, doc_filter=doc_filter)
    retrieval_ms = (time.time() - t0) * 1000

    if not results:
        total_ms = (time.time() - t_total) * 1000
        return QueryResponse(
            answer="No encontre fragmentos relevantes para esa consulta.",
            sources=[],
            query=req.query,
            processing_time_ms=round(total_ms, 1),
            retrieval_ms=round(retrieval_ms, 1),
            rerank_ms=0.0,
            llm_ms=0.0,
            total_ms=round(total_ms, 1),
        )

    # Re-ranking cross-encoder: recorta a RERANK_CANDIDATES antes del modelo
    # (PLAN-003 P5) y queda con RERANK_TOP_K finales.
    t0 = time.time()
    results = rerank_svc.rerank(
        req.query,
        results[: max(1, settings.RERANK_CANDIDATES)],
        top_k=max(1, settings.RERANK_TOP_K),
    )
    rerank_ms = (time.time() - t0) * 1000

    context = build_context(results)
    t0 = time.time()
    timed = generate_answer_timed(context, req.query)
    llm_ms = timed["prompt_eval_ms"] + timed["generation_ms"]
    total_ms = (time.time() - t_total) * 1000
    return QueryResponse(
        answer=timed["text"],
        sources=_results_to_sources(results),
        query=req.query,
        processing_time_ms=round(total_ms, 1),
        retrieval_ms=round(retrieval_ms, 1),
        rerank_ms=round(rerank_ms, 1),
        llm_ms=round(llm_ms, 1),
        total_ms=round(total_ms, 1),
        prompt_eval_ms=timed["prompt_eval_ms"],
        generation_ms=timed["generation_ms"],
    )


@router.post("/query/stream")
async def query_rag_stream(
    req: QueryRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User | None = Depends(get_current_user),
):
    doc_filter = str(req.doc_filter) if req.doc_filter else None
    results = await hybrid_search(req.query, db, top_k=req.max_chunks, doc_filter=doc_filter)

    async def event_stream():
        try:
            sources_data = [s.model_dump(mode="json") for s in _results_to_sources(results)]
            yield f"data: {json.dumps({'sources': sources_data})}\n\n"
            if not results:
                yield f"data: {json.dumps({'token': 'No encontre fragmentos relevantes para esa consulta.'})}\n\n"
            else:
                context = build_context(results)
                async for token in generate_answer_stream(context, req.query):
                    yield f"data: {json.dumps({'token': token})}\n\n"
        except Exception as e:
            logger.error(f"Stream error: {e}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.post("/search", response_model=SearchResponse)
async def search_only(
    req: SearchRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User | None = Depends(get_current_user),
):
    doc_filter = str(req.doc_filter) if req.doc_filter else None
    # Fetch a wider pool so page_filter still has candidates after MIN_RELEVANCE
    pool = req.top_k * 5 if req.page_filter is not None else req.top_k * 3
    results = await hybrid_search(req.query, db, top_k=pool, doc_filter=doc_filter)

    filtered = results
    if req.page_filter is not None:
        filtered = [r for r in filtered if req.page_filter in (r.page_numbers or [])]

    return SearchResponse(
        results=_results_to_sources(filtered[:req.top_k]),
        query=req.query,
        total=len(filtered),
    )
