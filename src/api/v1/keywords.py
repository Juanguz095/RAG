"""API /api/v1/keywords (PLAN-006 Fase 4 — RAG-013/015-019, CP-004)."""
from __future__ import annotations

import csv
import io
import uuid

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from sqlalchemy import delete, select

from src.core.permissions import require_permission
from src.database import Chunk, ChunkKeyword, Document, Keyword, KeywordCandidate, get_db
from src.services.audit import audit
from src.services.keywords import extract_keyword_matches

router = APIRouter(prefix="/api/v1/keywords", tags=["keywords"])


class KeywordIn(BaseModel):
    term: str
    category: str | None = None
    domain: str | None = None  # WP2: alias de "category" (dominio clínico)
    description: str | None = None
    priority: int | None = None  # 1=baja 2=media 3=alta


class KeywordUpdate(BaseModel):
    term: str | None = None
    category: str | None = None
    domain: str | None = None
    description: str | None = None
    priority: int | None = None
    is_active: bool | None = None


def _kw_public(k) -> dict:
    return {"id": str(k.id), "term": k.term, "category": k.category,
            "domain": k.category, "description": k.description,
            "priority": int(getattr(k, "priority", 1) or 1),
            "is_active": bool(k.is_active)}


@router.get("")
async def list_keywords(
    category: str | None = None,
    include_inactive: bool = False,
    db=Depends(get_db),
    _u=Depends(require_permission("keywords:read")),
):
    q = select(Keyword)
    if not include_inactive:
        q = q.where(Keyword.is_active == True)  # noqa: E712
    if category:
        q = q.where(Keyword.category == category)
    q = q.order_by(Keyword.category, Keyword.term)
    res = await db.execute(q)
    rows = res.scalars().all() if hasattr(res, "scalars") else list(res)
    return [_kw_public(k) for k in rows]


@router.get("/domains")
async def list_domains(db=Depends(get_db), _u=Depends(require_permission("keywords:read"))):
    """WP2: dominios distintos del catálogo (agrupación por dominio)."""
    res = await db.execute(
        select(Keyword.category)
        .where(Keyword.is_active == True, Keyword.category.isnot(None))  # noqa: E712
        .distinct()
        .order_by(Keyword.category)
    )
    rows = res.scalars().all() if hasattr(res, "scalars") else list(res)
    return [r for r in rows if r]


@router.post("", status_code=201)
async def create_keyword(body: KeywordIn, db=Depends(get_db), _u=Depends(require_permission("keywords:write"))):
    term = body.term.strip()
    if not term:
        raise HTTPException(422, "term vacío")
    category = body.category or body.domain  # WP2: domain es alias de category
    res = await db.execute(select(Keyword).where(Keyword.term == term))
    existing = res.scalar_one_or_none()
    if existing and existing.is_active:
        raise HTTPException(409, "keyword duplicada")
    if existing:
        existing.is_active = True
        existing.category = category or existing.category
        if body.priority is not None:
            existing.priority = body.priority
        await db.commit()
        await db.refresh(existing)
        return _kw_public(existing)
    kw = Keyword(term=term, category=category, description=body.description,
                 is_active=True, priority=body.priority or 1)
    db.add(kw)
    await db.commit()
    await db.refresh(kw)
    return _kw_public(kw)


@router.delete("/{kid}")
async def delete_keyword(kid: uuid.UUID, db=Depends(get_db), _u=Depends(require_permission("keywords:write"))):
    res = await db.execute(select(Keyword).where(Keyword.id == kid))
    kw = res.scalar_one_or_none()
    if not kw:
        raise HTTPException(404, "no existe")
    kw.is_active = False  # delete lógico (RAG-016)
    await db.commit()
    return {"ok": True}


@router.patch("/{kid}")
async def update_keyword(
    kid: uuid.UUID,
    body: KeywordUpdate,
    db=Depends(get_db),
    _u=Depends(require_permission("keywords:write")),
):
    """Edita term/dominio/descripción o reactiva (is_active). RAG-016."""
    res = await db.execute(select(Keyword).where(Keyword.id == kid))
    kw = res.scalar_one_or_none()
    if not kw:
        raise HTTPException(404, "no existe")
    changes: dict = {}
    if body.term is not None:
        new_term = body.term.strip()
        if not new_term:
            raise HTTPException(422, "term vacío")
        if new_term != kw.term:
            dup = (
                await db.execute(
                    select(Keyword).where(Keyword.term == new_term, Keyword.id != kid)
                )
            ).scalar_one_or_none()
            if dup:
                raise HTTPException(409, "keyword duplicada")
            kw.term = new_term
            changes["term"] = new_term
    cat = body.category if body.category is not None else body.domain
    if cat is not None:
        kw.category = cat.strip() or None
        changes["category"] = kw.category
    if body.description is not None:
        kw.description = body.description
        changes["description"] = body.description
    if body.priority is not None:
        kw.priority = body.priority
        changes["priority"] = body.priority
    if body.is_active is not None:
        kw.is_active = body.is_active
        changes["is_active"] = body.is_active
    if not changes:
        raise HTTPException(422, "nada que actualizar")
    await db.commit()
    await db.refresh(kw)
    await audit(db, _u, "admin_action", resource_type="keyword", resource_id=str(kw.id),
                detail={"op": "update", "changes": changes})
    await db.commit()
    return _kw_public(kw)


@router.post("/import")
async def import_keywords(request: Request, db=Depends(get_db), _u=Depends(require_permission("keywords:write"))):
    """CSV (term,category[,description]) — RAG-015. Import idempotente."""
    async with request.form() as form:
        up = form.get("file")
        if up is None:
            raise HTTPException(422, "falta file")
        raw = await up.read()
    text = raw.decode("utf-8-sig", errors="replace")
    imported = 0
    for i, row in enumerate(csv.reader(io.StringIO(text))):
        if not row or (i == 0 and row[0].strip().lower() == "term"):
            continue
        term = (row[0] or "").strip()
        if not term:
            continue
        res = await db.execute(select(Keyword).where(Keyword.term == term))
        if res.scalar_one_or_none():
            continue
        db.add(Keyword(term=term, category=(row[1].strip() if len(row) > 1 else None)))
        imported += 1
    await db.commit()
    return {"imported": imported}


@router.get("/export")
async def export_keywords(request: Request, db=Depends(get_db), _u=Depends(require_permission("keywords:read"))):
    fmt = (request.query_params.get("format") or "csv").lower()
    res = await db.execute(select(Keyword).where(Keyword.is_active == True))  # noqa: E712
    rows = res.scalars().all() if hasattr(res, "scalars") else list(res)
    if fmt == "xlsx":
        import openpyxl

        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "keywords"
        ws.append(["term", "category", "description"])
        for k in rows:
            ws.append([k.term, k.category, k.description])
        buf = io.BytesIO()
        wb.save(buf)
        buf.seek(0)
        from fastapi.responses import Response

        return Response(content=buf.getvalue(), media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        headers={"Content-Disposition": 'attachment; filename="keywords.xlsx"'})
    out = io.StringIO()
    w = csv.writer(out)
    w.writerow(["term", "category", "description"])
    for k in rows:
        w.writerow([k.term, k.category or "", k.description or ""])
    from fastapi.responses import Response

    return Response(content=out.getvalue(), media_type="text/csv",
                    headers={"Content-Disposition": 'attachment; filename="keywords.csv"'})


@router.get("/search")
async def search_keyword(request: Request, db=Depends(get_db), _u=Depends(require_permission("keywords:read"))):
    """Búsqueda exacta por término del catálogo → documento, páginas, fragmento (CP-004)."""
    term = (request.query_params.get("term") or "").strip().lower()
    if not term:
        raise HTTPException(422, "falta term")
    res = await db.execute(select(Keyword).where(Keyword.term == term, Keyword.is_active == True))  # noqa: E712
    kw = res.scalar_one_or_none()
    if not kw:
        return []
    res = await db.execute(select(ChunkKeyword).where(ChunkKeyword.keyword_id == kw.id))
    cks = list(res.scalars().all())
    out = []
    for ck in cks:
        ch_res = await db.execute(select(Chunk).where(Chunk.id == ck.chunk_id))
        ch = ch_res.scalar_one_or_none()
        doc_id = getattr(ch, "document_id", None)
        content = getattr(ch, "content", "") or ""
        meta = getattr(ch, "chunk_metadata", None) or {}
        frag = meta.get("fragment") or content[:240]
        pages = getattr(ch, "page_numbers", None) or []
        doc_name = None
        if doc_id:
            dres = await db.execute(select(Document).where(Document.id == doc_id))
            d = dres.scalar_one_or_none()
            if d:
                doc_name = getattr(d, "original_name", None) or getattr(d, "filename", None)
        out.append({
            "term": kw.term,
            "document": doc_name or "",
            "document_id": str(doc_id) if doc_id else None,
            "pages": list(pages),
            "fragment": frag,
            "section": meta.get("section"),
            "match_count": getattr(ck, "match_count", 1) if ck is not None else 1,
        })
    return out


# ── Candidatos (conceptos emergentes) → validación humana ────────────────

@router.get("/candidates")
async def list_candidates(
    status: str = "proposed",
    db=Depends(get_db),
    _u=Depends(require_permission("keywords:read")),
):
    q = select(KeywordCandidate)
    if status and status != "all":
        q = q.where(KeywordCandidate.status == status)
    q = q.order_by(KeywordCandidate.count.desc())
    rows = (await db.execute(q)).scalars().all()
    return [
        {"id": str(c.id), "term": c.term, "count": c.count,
         "sample_document": c.sample_document, "status": c.status}
        for c in rows
    ]


@router.post("/candidates/{cid}/approve", status_code=201)
async def approve_candidate(
    cid: uuid.UUID,
    db=Depends(get_db),
    _u=Depends(require_permission("keywords:write")),
):
    c = (await db.execute(select(KeywordCandidate).where(KeywordCandidate.id == cid))).scalar_one_or_none()
    if not c:
        raise HTTPException(404, "candidato no existe")
    existing = (await db.execute(select(Keyword).where(Keyword.term == c.term))).scalar_one_or_none()
    if existing:
        existing.is_active = True
    else:
        db.add(Keyword(term=c.term, is_active=True, priority=1))
    c.status = "approved"
    await db.commit()
    return {"ok": True, "term": c.term}


@router.post("/candidates/{cid}/reject")
async def reject_candidate(
    cid: uuid.UUID,
    db=Depends(get_db),
    _u=Depends(require_permission("keywords:write")),
):
    c = (await db.execute(select(KeywordCandidate).where(KeywordCandidate.id == cid))).scalar_one_or_none()
    if not c:
        raise HTTPException(404, "candidato no existe")
    c.status = "rejected"
    await db.commit()
    return {"ok": True, "term": c.term}
