from __future__ import annotations

import csv
import io
import logging

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.permissions import require_permission
from src.database import MedicalSynonym, User, get_db
from src.services.audit import audit
from src.services.synonyms import expand_term, get_synonym_groups

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["synonyms"])


class SynonymGroupIn(BaseModel):
    canonical: str
    category: str | None = None
    variants: list[str] = []


class SynonymGroupUpdate(BaseModel):
    category: str | None = None
    variants: list[str] | None = None


def _group_public(canonical: str, category: str | None, variants: list[str]) -> dict:
    return {"canonical": canonical, "category": category or "general", "variants": variants}


def _clean_variants(canonical: str, variants: list[str] | None) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for v in [canonical, *(variants or [])]:
        s = (v or "").strip()
        k = s.lower()
        if s and k not in seen:
            seen.add(k)
            out.append(s)
    return out


@router.get("/synonyms")
async def list_synonyms(
    db: AsyncSession = Depends(get_db),
    current_user: User | None = Depends(require_permission("keywords:read")),
):
    groups = await get_synonym_groups(db)
    return {"groups": groups, "count": len(groups)}


@router.get("/synonyms/expand")
async def expand(
    q: str,
    db: AsyncSession = Depends(get_db),
    current_user: User | None = Depends(require_permission("keywords:read")),
):
    groups = await get_synonym_groups(db)
    variants = expand_term(q, groups)
    return {"query": q, "variants": variants}


@router.post("/synonyms", status_code=201)
async def upsert_group(
    body: SynonymGroupIn,
    db: AsyncSession = Depends(get_db),
    user: User | None = Depends(require_permission("keywords:write")),
):
    canonical = (body.canonical or "").strip()
    if not canonical:
        raise HTTPException(status_code=422, detail="canonical vacío")
    variants = _clean_variants(canonical, body.variants)
    await db.execute(delete(MedicalSynonym).where(MedicalSynonym.canonical == canonical))
    for v in variants:
        db.add(MedicalSynonym(canonical=canonical, synonym=v, category=body.category))
    await db.commit()
    await audit(db, user, "admin_action", resource_type="synonym", resource_id=canonical,
                detail={"op": "upsert", "n_variants": len(variants)})
    await db.commit()
    return _group_public(canonical, body.category, variants)


@router.patch("/synonyms/{canonical}")
async def update_group(
    canonical: str,
    body: SynonymGroupUpdate,
    db: AsyncSession = Depends(get_db),
    user: User | None = Depends(require_permission("keywords:write")),
):
    rows = (
        await db.execute(select(MedicalSynonym).where(MedicalSynonym.canonical == canonical))
    ).scalars().all()
    if not rows:
        raise HTTPException(status_code=404, detail="no existe")
    cat = body.category if body.category is not None else (rows[0].category if rows else None)
    if body.variants is not None:
        variants = _clean_variants(canonical, body.variants)
        await db.execute(delete(MedicalSynonym).where(MedicalSynonym.canonical == canonical))
        for v in variants:
            db.add(MedicalSynonym(canonical=canonical, synonym=v, category=cat))
    else:
        for r in rows:
            r.category = cat
    await db.commit()
    await audit(db, user, "admin_action", resource_type="synonym", resource_id=canonical,
                detail={"op": "update"})
    await db.commit()
    return {"ok": True, "canonical": canonical}


@router.delete("/synonyms/{canonical}", status_code=204)
async def delete_group(
    canonical: str,
    db: AsyncSession = Depends(get_db),
    user: User | None = Depends(require_permission("keywords:write")),
):
    res = await db.execute(delete(MedicalSynonym).where(MedicalSynonym.canonical == canonical))
    await db.commit()
    if res.rowcount == 0:
        raise HTTPException(status_code=404, detail="no existe")
    await audit(db, user, "admin_action", resource_type="synonym", resource_id=canonical,
                detail={"op": "delete"})
    await db.commit()
    return Response(status_code=204)


@router.post("/synonyms/import")
async def import_synonyms(
    request: Request,
    db: AsyncSession = Depends(get_db),
    user: User | None = Depends(require_permission("keywords:write")),
):
    """CSV (canonical, synonym[, category]) — idempotente por (canonical, synonym)."""
    async with request.form() as form:
        up = form.get("file")
        if up is None:
            raise HTTPException(status_code=422, detail="falta file")
        raw = await up.read()
    text = raw.decode("utf-8-sig", errors="replace")
    existing = {
        (r.canonical.lower(), r.synonym.lower())
        for r in (await db.execute(select(MedicalSynonym))).scalars().all()
    }
    imported = 0
    for i, row in enumerate(csv.reader(io.StringIO(text))):
        if not row or (i == 0 and row[0].strip().lower() in ("canonical", "canonico")):
            continue
        canonical = (row[0] or "").strip()
        synonym = (row[1] or "").strip() if len(row) > 1 else canonical
        category = (row[2].strip() if len(row) > 2 and row[2] else None)
        if not canonical:
            continue
        if (canonical.lower(), synonym.lower()) in existing:
            continue
        db.add(MedicalSynonym(canonical=canonical, synonym=synonym or canonical, category=category))
        existing.add((canonical.lower(), (synonym or canonical).lower()))
        imported += 1
    await db.commit()
    await audit(db, user, "admin_action", resource_type="synonym", detail={"op": "import", "n": imported})
    await db.commit()
    return {"imported": imported}


@router.get("/synonyms/export")
async def export_synonyms(
    request: Request,
    db: AsyncSession = Depends(get_db),
    _u: User | None = Depends(require_permission("keywords:read")),
):
    fmt = (request.query_params.get("format") or "csv").lower()
    rows = (await db.execute(select(MedicalSynonym).order_by(MedicalSynonym.canonical))).scalars().all()
    items = [(r.canonical, r.synonym, r.category or "") for r in rows]
    if fmt == "xlsx":
        import openpyxl

        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "sinonimos"
        ws.append(["canonical", "synonym", "category"])
        for it in items:
            ws.append(list(it))
        buf = io.BytesIO()
        wb.save(buf)
        buf.seek(0)
        return Response(
            content=buf.getvalue(),
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": 'attachment; filename="sinonimos.xlsx"'},
        )
    out = io.StringIO()
    w = csv.writer(out)
    w.writerow(["canonical", "synonym", "category"])
    for it in items:
        w.writerow(list(it))
    return Response(content=out.getvalue(), media_type="text/csv",
                    headers={"Content-Disposition": 'attachment; filename="sinonimos.csv"'})
