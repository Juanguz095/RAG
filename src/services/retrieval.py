from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from src.database import Chunk, MedicalSynonym, async_session
from src.config import get_settings

logger = logging.getLogger(__name__)

settings = get_settings()

_synonym_cache: list[MedicalSynonym] = []
_synonym_pairs: list[tuple[str, str, str, str]] = []  # (canon_l, syn_l, canon, syn)
_synonym_cache_time: float = 0
SYNONYM_CACHE_TTL = 300
MIN_RELEVANCE = 12  # keep pure-semantic MiniLM hits (cos*60 alone ≈ 18-30)
# Short/common Spanish tokens that pollute keyword ranking (keep dni, ia, tx…)
_STOP = {
    "de", "la", "el", "en", "con", "por", "para", "que", "se", "su", "al",
    "es", "lo", "como", "mas", "o", "e", "a", "y", "u", "no", "si", "un",
    "una", "unos", "unas", "del", "las", "los", "os", "as", "me", "te",
    "le", "lo", "ni", "ya", "ha", "he", "sa", "pa", "da", "re", "ve",
}


def _keep_term(t: str) -> bool:
    if len(t) < 2 or t in _STOP:
        return False
    if len(t) == 2 and t.isalpha() and t not in {"ia", "tx", "hr", "pc", "mg", "iv", "im", "sc", "po"}:
        return False
    return True


@dataclass
class RetrievalResult:
    chunk_id: str
    content: str
    document_id: str
    document_name: str
    page_numbers: list[int]
    score: float
    chunk_metadata: dict
    chunk_index: int = 0
    relevance: int = 0
    matched_terms: list[str] = field(default_factory=list)


async def _expand_synonyms(query: str, db: AsyncSession) -> str:
    global _synonym_cache, _synonym_pairs, _synonym_cache_time
    now = time.time()
    if not _synonym_pairs or (now - _synonym_cache_time) > SYNONYM_CACHE_TTL:
        result = await db.execute(select(MedicalSynonym))
        _synonym_cache = result.scalars().all()
        _synonym_pairs = [
            (s.canonical.lower(), s.synonym.lower(), s.canonical, s.synonym)
            for s in _synonym_cache
        ]
        _synonym_cache_time = now
    q_low = query.lower()
    extras: list[str] = []
    for canon_l, syn_l, canon, syn in _synonym_pairs:
        if canon_l in q_low:
            extras.append(syn)
        elif syn_l in q_low:
            extras.append(canon)
    if extras:
        query = query + " " + " ".join(extras)
    return query


async def vector_search(
    query_emb: list[float], db: AsyncSession, limit: int = 50, doc_filter: str | None = None
) -> list[tuple[str, float]]:
    emb_str = "[" + ",".join(str(x) for x in query_emb) + "]"
    where_clause = ""
    params = {"emb": emb_str, "limit": limit}
    if doc_filter:
        where_clause = "AND c.document_id = CAST(:doc_id AS uuid)"
        params["doc_id"] = doc_filter

    sql = text(f"""
        SELECT c.id::text, c.content, c.document_id::text, c.page_numbers,
               c.chunk_metadata, d.original_name, c.chunk_index,
                1 - (c.embedding <=> CAST(:emb AS vector)) AS cos_sim
        FROM chunks c
        JOIN documents d ON d.id = c.document_id
        WHERE c.embedding IS NOT NULL {where_clause}
        ORDER BY c.embedding <=> CAST(:emb AS vector)
        LIMIT :limit
    """)
    result = await db.execute(sql, params)
    rows = result.fetchall()
    return rows


async def keyword_search(
    query: str, db: AsyncSession, limit: int = 50, doc_filter: str | None = None
) -> list[tuple]:
    sanitized = re.sub(r'[^\w\sáéíóúñü]', '', query)
    terms = [t for t in sanitized.split() if _keep_term(t)]
    if not terms:
        return []

    # OR (no AND): la consulta ya viene expandida con sinonimos; exigir todos
    # los terminos reduciria demasiado el recall. El re-ranking ordena luego.
    tsquery = " | ".join(f"{t}:*" for t in terms)
    search_cond = (
        "(to_tsvector('spanish', c.content) @@ to_tsquery('spanish', :tsquery) "
        "OR c.content %> :q)"
    )
    params = {"tsquery": tsquery, "q": query, "limit": limit}
    if doc_filter:
        search_cond += " AND c.document_id = CAST(:doc_id AS uuid)"
        params["doc_id"] = doc_filter

    sql = text(f"""
        SELECT c.id::text, c.content, c.document_id::text, c.page_numbers,
               c.chunk_metadata, d.original_name, c.chunk_index,
               ts_rank_cd(to_tsvector('spanish', c.content), to_tsquery('spanish', :tsquery)) AS rank,
               similarity(c.content, :q) AS sml
        FROM chunks c
        JOIN documents d ON d.id = c.document_id
        WHERE {search_cond}
        ORDER BY sml DESC, rank DESC
        LIMIT :limit
    """)
    result = await db.execute(sql, params)
    return result.fetchall()


def _query_terms(query: str) -> list[str]:
    return [t for t in re.sub(r'[^\w\sáéíóúñü]', '', query.lower()).split() if _keep_term(t)]


def _term_hits(content: str, terms: list[str]) -> list[str]:
    text_l = content.lower()
    return [t for t in terms if t in text_l]


def reciprocal_rank_fusion(
    vector_results: list, keyword_results: list, query: str = "", k: int = 60
) -> list[RetrievalResult]:
    scores: dict[str, float] = {}
    items: dict[str, dict] = {}
    terms = _query_terms(query)

    for rank, row in enumerate(vector_results):
        cid = row[0]
        scores[cid] = scores.get(cid, 0) + 1.0 / (k + rank + 1)
        cos_sim = float(row[7]) if len(row) > 7 else 0.0
        items[cid] = {
            "chunk_id": cid,
            "content": row[1],
            "document_id": row[2],
            "page_numbers": row[3] or [],
            "chunk_metadata": row[4] or {},
            "document_name": row[5],
            "chunk_index": row[6] or 0,
            "cos_sim": cos_sim,
            "rank": rank,
        }

    for rank, row in enumerate(keyword_results):
        cid = row[0]
        scores[cid] = scores.get(cid, 0) + 1.0 / (k + rank + 1)
        if cid not in items:
            items[cid] = {
                "chunk_id": cid,
                "content": row[1],
                "document_id": row[2],
                "page_numbers": row[3] or [],
                "chunk_metadata": row[4] or {},
                "document_name": row[5],
                "chunk_index": row[6] or 0,
                "cos_sim": 0.0,
                "rank": rank,
            }

    results = []
    for cid, data in sorted(items.items(), key=lambda kv: scores[kv[0]]):
        content = data["content"]

        hits = _term_hits(content, terms)
        cos_sim = data.get("cos_sim", 0.0)

        # Relevancia combinada: coseno (0-100) + bono por coincidencia de términos
        base = max(0.0, cos_sim) * 60.0
        kw_bonus = min(40.0, len(hits) * 15.0 + (12.0 if hits and len(terms) > 0 and len(hits) == len(terms) else 0.0))
        relevance = int(min(100, base + kw_bonus))

        results.append(RetrievalResult(
            chunk_id=data["chunk_id"],
            content=content,
            document_id=data["document_id"],
            document_name=data["document_name"],
            page_numbers=data["page_numbers"],
            score=scores[cid],
            chunk_metadata=data["chunk_metadata"],
            chunk_index=data["chunk_index"],
            relevance=relevance,
            matched_terms=hits,
        ))
    return results


async def hybrid_search(
    query: str,
    db: AsyncSession,
    top_k: int | None = None,
    doc_filter: str | None = None,
) -> list[RetrievalResult]:
    top_k = top_k or settings.RETRIEVAL_TOP_K

    from src.services.embeddings import encode_query
    expanded_query = await _expand_synonyms(query, db)
    query_emb = encode_query(expanded_query)

    # Separate sessions: AsyncSession is not safe for concurrent execute()
    async with async_session() as db_vec, async_session() as db_kw:
        vector_rows, keyword_rows = await asyncio.gather(
            vector_search(query_emb, db_vec, limit=top_k * 3, doc_filter=doc_filter),
            keyword_search(expanded_query, db_kw, limit=top_k * 3, doc_filter=doc_filter),
        )

    fused = reciprocal_rank_fusion(vector_rows, keyword_rows, query=expanded_query)

    # Solo resultados realmente relevantes
    relevant = [r for r in fused if r.relevance >= MIN_RELEVANCE]

    seen = set()
    deduped = []
    for r in relevant:
        if r.chunk_id not in seen:
            seen.add(r.chunk_id)
            deduped.append(r)

    return deduped[:top_k]



