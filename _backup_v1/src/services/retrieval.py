import asyncio
import logging
from pathlib import Path

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from src.config import get_settings
from src.services.embeddings import encode_single

logger = logging.getLogger(__name__)
settings = get_settings()

_reranker_model = None


def _get_reranker():
    global _reranker_model
    if _reranker_model is None:
        try:
            from FlagEmbedding import FlagReranker
            logger.info(f"Loading reranker: {settings.RERANKER_MODEL_NAME}")
            _reranker_model = FlagReranker(settings.RERANKER_MODEL_NAME, use_fp16=False)
            logger.info("Reranker loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load reranker: {e}")
    return _reranker_model


async def vector_search(
    db: AsyncSession,
    query_embedding: list[float],
    limit: int = 50,
    filters: dict | None = None,
) -> list[dict]:
    filter_clauses = []
    params: dict = {"embedding": str(query_embedding), "limit": limit}

    if filters:
        if "document_id" in filters:
            filter_clauses.append("dc.document_id = :doc_id")
            params["doc_id"] = filters["document_id"]
        if "page_numbers" in filters:
            filter_clauses.append("dc.page_numbers && CAST(:pages AS integer[])")
            params["pages"] = filters["page_numbers"]

    where_clause = ""
    if filter_clauses:
        where_clause = "WHERE " + " AND ".join(filter_clauses)

    query = text(f"""
        SELECT dc.id, dc.document_id, dc.chunk_index, dc.content, dc.content_md,
               dc.page_numbers, dc.chunk_metadata, dc.token_count,
               1 - (dc.embedding <=> CAST(:embedding AS vector)) as score
        FROM document_chunks dc
        {where_clause}{' AND' if where_clause else 'WHERE'} dc.embedding IS NOT NULL
        ORDER BY dc.embedding <=> CAST(:embedding AS vector)
        LIMIT :limit
    """)

    result = await db.execute(query, params)
    rows = result.mappings().all()

    return [
        {
            "id": str(row["id"]),
            "document_id": str(row["document_id"]),
            "chunk_index": row["chunk_index"],
            "content": row["content"],
            "content_md": row["content_md"],
            "page_numbers": row["page_numbers"],
            "chunk_metadata": row["chunk_metadata"],
            "token_count": row["token_count"],
            "score": float(row["score"]),
            "search_type": "vector",
        }
        for row in rows
    ]


async def keyword_search(
    db: AsyncSession,
    query: str,
    limit: int = 50,
    filters: dict | None = None,
) -> list[dict]:
    filter_clauses = []
    params: dict = {"query": query, "exact_query": f"%{query}%", "limit": limit}

    if filters:
        if "document_id" in filters:
            filter_clauses.append("dc.document_id = :doc_id")
            params["doc_id"] = filters["document_id"]

    where_clause = ""
    if filter_clauses:
        where_clause = "WHERE " + " AND ".join(filter_clauses)

    search_query = text(f"""
        SELECT dc.id, dc.document_id, dc.chunk_index, dc.content, dc.content_md,
               dc.page_numbers, dc.chunk_metadata, dc.token_count,
               GREATEST(similarity(dc.content, :query),
                        CASE WHEN lower(dc.content) LIKE lower(:exact_query) THEN 1.0 ELSE 0.0 END) as score
        FROM document_chunks dc
        {where_clause}
        ORDER BY GREATEST(similarity(dc.content, :query),
                          CASE WHEN lower(dc.content) LIKE lower(:exact_query) THEN 1.0 ELSE 0.0 END) DESC
        LIMIT :limit
    """)

    result = await db.execute(search_query, params)
    rows = result.mappings().all()

    return [
        {
            "id": str(row["id"]),
            "document_id": str(row["document_id"]),
            "chunk_index": row["chunk_index"],
            "content": row["content"],
            "content_md": row["content_md"],
            "page_numbers": row["page_numbers"],
            "chunk_metadata": row["chunk_metadata"],
            "token_count": row["token_count"],
            "score": float(row["score"]),
            "search_type": "keyword",
        }
        for row in rows
    ]


def reciprocal_rank_fusion(
    vector_results: list[dict],
    keyword_results: list[dict],
    k: int = 60,
) -> list[dict]:
    scores: dict[str, float] = {}
    docs: dict[str, dict] = {}

    for rank, result in enumerate(vector_results):
        rid = result["id"]
        scores[rid] = scores.get(rid, 0) + 1.0 / (k + rank + 1)
        docs[rid] = result

    for rank, result in enumerate(keyword_results):
        rid = result["id"]
        scores[rid] = scores.get(rid, 0) + 1.0 / (k + rank + 1)
        if rid not in docs:
            docs[rid] = result

    sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    fused = []
    for rid in sorted_ids:
        entry = docs[rid].copy()
        entry["score"] = scores[rid]
        entry["search_type"] = "hybrid"
        fused.append(entry)

    return fused


async def rerank(
    query: str, chunks: list[dict], top_k: int | None = None
) -> list[dict]:
    top_k = top_k or settings.RERANK_TOP_K
    if not settings.RERANKER_ENABLED:
        return sorted(chunks, key=lambda item: item.get("score", 0), reverse=True)[:top_k]
    reranker = _get_reranker()

    if not chunks:
        return []
    if reranker is None:
        return sorted(chunks, key=lambda item: item.get("score", 0), reverse=True)[:top_k]

    try:
        pairs = [[query, c["content"]] for c in chunks]
        scores = reranker.compute_score(pairs, normalize=True)
        if isinstance(scores, float):
            scores = [scores]

        for chunk, score in zip(chunks, scores, strict=True):
            chunk["rerank_score"] = score

        reranked = sorted(chunks, key=lambda x: x.get("rerank_score", 0), reverse=True)
        return reranked[:top_k]
    except Exception as e:
        logger.error(f"Reranking failed: {e}")
        return sorted(chunks, key=lambda item: item.get("score", 0), reverse=True)[:top_k]


async def hybrid_search(
    db: AsyncSession,
    query: str,
    filters: dict | None = None,
    top_k: int | None = None,
) -> list[dict]:
    top_k = top_k or settings.RETRIEVAL_TOP_K

    has_embeddings = False
    vector_results = []
    try:
        from src.services.embeddings import is_loaded, encode_single
        if is_loaded():
            query_emb = encode_single(query)
            vector_results = await vector_search(db, query_emb, limit=top_k, filters=filters)
            has_embeddings = bool(vector_results)
    except Exception as e:
        logger.debug("Vector search unavailable: %s", e)

    keyword_results = await keyword_search(db, query, limit=top_k, filters=filters)

    if has_embeddings and keyword_results:
        fused = reciprocal_rank_fusion(vector_results, keyword_results, k=60)
    elif has_embeddings:
        fused = vector_results
    else:
        fused = keyword_results

    return fused[:top_k]
