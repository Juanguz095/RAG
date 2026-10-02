"""API /api/v1/keywords (PLAN-006 Fase 4 — RAG-013/015-019, CP-004)."""
from __future__ import annotations

import csv
import io
import uuid

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from sqlalchemy import delete, select

from src.core.permissions import require_permission
from src.database import Chunk, ChunkKeyword, Document, Keyword, get_db
from src.services.keywords import extract_keyword_matches

router = APIRouter(prefix="/api/v1/keywords", tags=["keywords"])


class KeywordIn(BaseModel):
    term: str
    category: str | None = None
    domain: str | None = None  # WP2: alias de "category" (dominio clínico)
    description: str | None = None


@router.get("")
async def list_keywords(
    category: str | None = None,
    db=Depends(get_db),
    _u=Depends(require_permission("keywords:read")),
):
    q = select(Keyword).where(Keyword.is_active == True)  # noqa: E712
    if category:
        q = q.where(Keyword.category == category)
    res = await db.execute(q)
    rows = res.scalars().all() if hasattr(res, "scalars") else list(res)
    return [
        {"id": str(k.id), "term": k.term, "category": k.category,
         "domain": k.category, "description": k.description, "is_active": bool(k.is_active)}
        for k in rows
    ]


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
        await db.commit()
        await db.refresh(existing)
        return {"id": str(existing.id), "term": existing.term, "category": existing.category,
                "domain": existing.category, "description": existing.description, "is_active": True}
    kw = Keyword(term=term, category=category, description=body.description, is_active=True)
    db.add(kw)
    await db.commit()
    await db.refresh(kw)
    return {"id": str(kw.id), "term": kw.term, "category": kw.category,
            "domain": kw.category, "description": kw.description, "is_active": True}


@router.delete("/{kid}")
async def delete_keyword(kid: uuid.UUID, db=Depends(get_db), _u=Depends(require_permission("keywords:write"))):
    res = await db.execute(select(Keyword).where(Keyword.id == kid))
    kw = res.scalar_one_or_none()
    if not kw:
        raise HTTPException(404, "no existe")
    kw.is_active = False  # delete lógico (RAG-016)
    await db.commit()
    return {"ok": True}


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
