from __future__ import annotations

import json
import logging
import time

from src.services.anonymizer import anonymize_text
from typing import AsyncGenerator
from uuid import UUID

from fastapi import APIRouter, Depends, Request, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.deps import get_current_user
from src.config import get_settings
from src.core.permissions import require_permission
from src.database import User, get_db
from src.schemas.query import (
    ChunkResult,
    QueryRequest,
    QueryResponse,
    SearchRequest,
    SearchResponse,
)
from src.services.audit import audit as audit_event
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
            section=(r.chunk_metadata or {}).get("section") if r.chunk_metadata else None,
            fragment=(r.chunk_metadata or {}).get("fragment") if r.chunk_metadata else None,
        )
        for r in results
    ]


@router.post("/query", response_model=QueryResponse)
async def query_rag(
    req: QueryRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
    current_user: User | None = Depends(require_permission("query")),
):
    from src.services import reranker as rerank_svc

    t_total = time.time()
    doc_filter = str(req.doc_filter) if req.doc_filter else None

    # Retrival amplio: pool de candidatos para el re-ranking
    t0 = time.time()
    results = await hybrid_search(req.query, db, top_k=req.max_chunks, doc_filter=doc_filter)
    retrieval_ms = (time.time() - t0) * 1000

    # C7 (PLAN-004): los chunks de documentos restricted no se entregan a
    # roles sin entrada en document_acl (regla §6.4 del diseño).
    async def _can_see(r) -> bool:
        doc = getattr(r, "document", None) or getattr(getattr(r, "chunk", None), "document", None)
        vis = getattr(doc, "visibility", "public") if doc is not None else "public"
        if vis != "restricted":
            return True
        if current_user is None or current_user.role == "admin":
            return True
        from sqlalchemy import select as _select

        from src.database import DocumentACL

        acl = None
        try:
            result_acl = await db.execute(
                _select(DocumentACL).where(
                    DocumentACL.document_id == doc.id, DocumentACL.role == current_user.role
                )
            )
            acl = result_acl.scalar_one_or_none()
        except Exception:
            acl = None
        return acl is not None

    if results:
        visible = []
        for r in results:
            if await _can_see(r):
                visible.append(r)
        results = visible

    if not results:
        total_ms = (time.time() - t_total) * 1000
        return QueryResponse(
            # CP-006 (literal): declara insuficiencia de información y no inventa.
            answer=(
                "No dispongo de información suficiente en los documentos indexados "
                "para responder esa consulta; no voy a inventar una respuesta."
            ),
            sources=[],
            query=req.query,
            processing_time_ms=round(total_ms, 1),
            retrieval_ms=round(retrieval_ms, 1),
            rerank_ms=0.0,
            llm_ms=0.0,
            total_ms=round(total_ms, 1),
            abstained=True,
            grounded=False,
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

    # RAG-025 (CP-006): umbral de evidencia. Sin fuentes suficiente → abstención.
    def _evidence_score(r):
        # Preferir el score del cross-encoder (sigmoid 0-1, seteado por
        # rerank()); fallback: score híbrido escalado (RRF ~0-0.03 → x10).
        rs = getattr(r, "rerank_score", None)
        if rs is not None:
            return float(rs)
        return float(getattr(r, "score", 0.0) or 0.0) * 10.0

    strong = [r for r in results if _evidence_score(r) >= settings.EVIDENCE_MIN_SCORE]
    if len(strong) < max(1, settings.EVIDENCE_MIN_SOURCES):
        total_ms = (time.time() - t_total) * 1000
        await audit_event(db, current_user, "query", resource_type="query",
                          detail={"question": req.query, "abstained": True,
                                  "best_score": max([_evidence_score(r) for r in results], default=0.0),
                                  "threshold": settings.EVIDENCE_MIN_SCORE})
        await db.commit()
        return QueryResponse(
            answer=(
                "No dispongo de información suficiente en los documentos indexados "
                "para responder esa consulta; no voy a inventar una respuesta."
            ),
            sources=[],
            query=req.query,
            processing_time_ms=round(total_ms, 1),
            retrieval_ms=round(retrieval_ms, 1),
            rerank_ms=round(rerank_ms, 1),
            llm_ms=0.0,
            total_ms=round(total_ms, 1),
            abstained=True,
            grounded=False,
        )
    results = strong

    context = build_context(results)
    t0 = time.time()
    timed = generate_answer_timed(context, req.query)
    llm_ms = timed["prompt_eval_ms"] + timed["generation_ms"]
    total_ms = (time.time() - t_total) * 1000

    # M4 (PLAN-004): trazabilidad completa usuario→consulta→chunks→respuesta
    # →fuentes en un solo evento `query` de audit_log (CP-009, RAG-040).
    sources = _results_to_sources(results)

    # RAG-024 (CP-007): verificación de grounding — toda cita [n] del texto
    # debe mapear a una fuente real; sin citas válidas → grounded=False.
    import re as _re

    answer_text = timed["text"]
    cited = {int(n) for n in _re.findall(r"\[(\d+)\]", answer_text)}
    grounded = bool(cited) and all(1 <= n <= len(sources) for n in cited)
    # Reintento único con recordatorio de citar (PLAN-006 Fase 2):
    if not grounded:
        timed2 = generate_answer_timed(
            context + "\n\nIMPORTANTE: cita las fuentes con [1], [2]... correspondientes.",
            req.query,
        )
        llm_ms += timed2["prompt_eval_ms"] + timed2["generation_ms"]
        total_ms = (time.time() - t_total) * 1000
        answer_text = timed2["text"]
        cited = {int(n) for n in _re.findall(r"\[(\d+)\]", answer_text)}
        grounded = bool(cited) and all(1 <= n <= len(sources) for n in cited)
        timed = timed2

    # CP-007 (RAG-026): la respuesta SIEMPRE incluye la referencia al
    # documento y ubicación de las fuentes usadas; si el LLM no citó inline
    # ([n] ausente/inválido), se adjunta el bloque "Fuentes" verificable.
    if not grounded and sources:
        fparts = []
        for s in sources[:3]:
            fparts.append(
                f"- {s.document_name}, pág. {s.page_numbers}"
                + (f' (sección: {s.section})' if s.section else "")
            )
        answer_text += "\n\nFuentes consultadas:\n" + "\n".join(fparts)

    # RAG-038 (CP-008): política de salida — si el LLM reintroduce un dato
    # sensible (alucinación), se enmascara antes de devolver la respuesta.
    anonymized_answer, _n_masked = anonymize_text(answer_text)
    answer_text = anonymized_answer
    await audit_event(
        db,
        current_user,
        "query",
        resource_type="query",
        detail={
            "question": req.query,
            "chunks": [
                {
                    "chunk_id": str(getattr(r, "chunk_id", None) or getattr(getattr(r, "chunk", None), "id", None)),
                    "document_id": str(getattr(r, "document_id", None) or getattr(getattr(r, "chunk", None), "document_id", None)),
                    "pages": list(getattr(r, "page_numbers", None) or getattr(getattr(r, "chunk", None), "page_numbers", None) or []),
                }
                for r in results
            ],
            "answer": timed["text"],
            "sources": [s.model_dump(mode="json") for s in sources],
            "model": settings.__dict__.get("LLM_MODEL_PATH", "") or "default",
            "prompt_eval_ms": timed["prompt_eval_ms"],
            "generation_ms": timed["generation_ms"],
        },
    )
    await db.commit()

    return QueryResponse(
        answer=answer_text,
        sources=sources,
        query=req.query,
        processing_time_ms=round(total_ms, 1),
        retrieval_ms=round(retrieval_ms, 1),
        rerank_ms=round(rerank_ms, 1),
        llm_ms=round(llm_ms, 1),
        total_ms=round(total_ms, 1),
        prompt_eval_ms=timed["prompt_eval_ms"],
        generation_ms=timed["generation_ms"],
        abstained=False,
        grounded=grounded,
    )


@router.post("/query/stream")
async def query_rag_stream(
    req: QueryRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
    current_user: User | None = Depends(require_permission("query")),
):
    doc_filter = str(req.doc_filter) if req.doc_filter else None
    results = await hybrid_search(req.query, db, top_k=req.max_chunks, doc_filter=doc_filter)

    async def event_stream():
        try:
            sources_data = [s.model_dump(mode="json") for s in _results_to_sources(results)]
            yield f"data: {json.dumps({'sources': sources_data})}\n\n"
            if not results:
                yield f"data: {json.dumps({'token': 'No dispongo de información suficiente en los documentos indexados para responder esa consulta; no voy a inventar una respuesta.'})}\n\n"
            else:
                context = build_context(results)
                async for token in generate_answer_stream(context, req.query):
                    yield f"data: {json.dumps({'token': token})}\n\n"
        except Exception as e:
            logger.error(f"Stream error: {e}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.get("/query/export/history")
async def export_history(request: Request, fmt: str = "json",
                         format_: str | None = Query(None, alias="format"),
                         db: AsyncSession = Depends(get_db),
                         current_user: User | None = Depends(require_permission("export"))):
    """RAG-033/034 (PLAN-006 Fase 6): exporta el historial de consultas del usuario
    desde audit_log (JSON | TXT | CSV) — 100% trazable, sin dashboard extra."""
    import json as _json
    res = await db.execute(_select_audit_query_events())
    rows = list(res.scalars().all()) if hasattr(res, "scalars") else list(res or [])
    items = [
        {"ts": str(r.created_at), "question": (r.detail or {}).get("question", ""),
         "answer": (r.detail or {}).get("answer", ""),
         "sources": (r.detail or {}).get("sources", [])}
        for r in rows
    ]
    from fastapi.responses import Response
    fmt = format_ or fmt
    if fmt == "json":
        payload = _json.dumps(items, ensure_ascii=False, default=str)
        return Response(content=payload, media_type="application/json",
                        headers={"Content-Disposition": 'attachment; filename="historial.json"'})
    lines = []
    for it2 in items:
        lines.append(f"=== {it2['ts']} ===\nQ: {it2['question']}\nA: {it2['answer']}\nFuentes: {len(it2['sources'])}\n")
    if fmt == "csv":
        import csv as _csv, io as _io
        buf = _io.StringIO()
        w = _csv.writer(buf)
        w.writerow(["ts", "question", "answer", "n_sources"])
        for it2 in items:
            w.writerow([it2["ts"], it2["question"], it2["answer"], len(it2["sources"])])
        return Response(content=buf.getvalue(), media_type="text/csv",
                        headers={"Content-Disposition": 'attachment; filename="historial.csv"'})
    return Response(content="\n".join(lines), media_type="text/plain",
                    headers={"Content-Disposition": 'attachment; filename="historial.txt"'})


def _select_audit_query_events():
    from sqlalchemy import select
    from src.database import AuditLog

    return select(AuditLog).where(AuditLog.action == "query").order_by(AuditLog.created_at.desc()).limit(100)


@router.post("/search", response_model=SearchResponse)
async def search_only(
    req: SearchRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
    current_user: User | None = Depends(require_permission("search")),
):
    doc_filter = str(req.doc_filter) if req.doc_filter else None
    # Fetch a wider pool so page_filter still has candidates after MIN_RELEVANCE
    pool = req.top_k * 5 if req.page_filter is not None else req.top_k * 3
    results = await hybrid_search(req.query, db, top_k=pool, doc_filter=doc_filter)
    if results:
        visible = []
        for r in results:
            doc = getattr(r, "document", None) or getattr(getattr(r, "chunk", None), "document", None)
            vis = getattr(doc, "visibility", "public") if doc is not None else "public"
            if vis == "restricted" and current_user is not None and current_user.role != "admin":
                from sqlalchemy import select as _select

                from src.database import DocumentACL

                try:
                    acl = (
                        await db.execute(
                            _select(DocumentACL).where(
                                DocumentACL.document_id == doc.id,
                                DocumentACL.role == current_user.role,
                            )
                        )
                    ).scalar_one_or_none()
                except Exception:
                    acl = None
                if acl is None:
                    continue
            visible.append(r)
        results = visible
    await audit_event(db, current_user, "search", resource_type="search",
                      detail={"query": req.query, "results": len(results)})
    await db.commit()

    filtered = results
    if req.page_filter is not None:
        filtered = [r for r in filtered if req.page_filter in (r.page_numbers or [])]

    return SearchResponse(
        results=_results_to_sources(filtered[:req.top_k]),
        query=req.query,
        total=len(filtered),
    )
