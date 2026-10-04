"""Métricas de conocimiento sobre el CATÁLOGO DE KEYWORDS (SQL agregado).

Reemplaza el cómputo en Python que cargaba todos los chunks en memoria
(`compute_keyword_coverage`/`compute_gaps` sobre sinónimos). Usa
`chunk_keywords` (doc×keyword, ya indexado) + `keywords`.

Todo es SQL agregado → rápido y sin traer el corpus a memoria.
"""
from __future__ import annotations

from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession


async def keyword_coverage(db: AsyncSession, weight_priority: bool = False) -> dict[str, Any]:
    """Cobertura = keywords activas con evidencia / total de keywords activas.

    Si `weight_priority`, pondera por `priority` (keywords más importantes pesan más).
    """
    if weight_priority:
        row = (
            await db.execute(
                text(
                    """
                    SELECT
                      COALESCE(SUM(COALESCE(k.priority,1)),0) AS total_w,
                      COALESCE(SUM(CASE WHEN f.keyword_id IS NOT NULL THEN COALESCE(k.priority,1) ELSE 0 END),0) AS found_w,
                      COUNT(*) AS total,
                      COUNT(f.keyword_id) AS found
                    FROM keywords k
                    LEFT JOIN (SELECT DISTINCT keyword_id FROM chunk_keywords) f
                      ON f.keyword_id = k.id
                    WHERE k.is_active = true
                    """
                )
            )
        ).mappings().first()
        total_w = float(row["total_w"] or 0)
        found_w = float(row["found_w"] or 0)
        pct = round(found_w / total_w * 100.0, 2) if total_w else 100.0
        return {"pct": pct, "found": int(row["found"] or 0), "total": int(row["total"] or 0)}
    row = (
        await db.execute(
            text(
                """
                SELECT
                  COUNT(*) AS total,
                  COUNT(f.keyword_id) AS found
                FROM keywords k
                LEFT JOIN (SELECT DISTINCT keyword_id FROM chunk_keywords) f
                  ON f.keyword_id = k.id
                WHERE k.is_active = true
                """
            )
        )
    ).mappings().first()
    total = int(row["total"] or 0)
    found = int(row["found"] or 0)
    pct = round(found / total * 100.0, 2) if total else 100.0
    return {"pct": pct, "found": found, "total": total}


async def keyword_gaps(db: AsyncSession) -> dict[str, Any]:
    """Keywords activas SIN evidencia (brechas), global y por dominio."""
    rows = (
        await db.execute(
            text(
                """
                SELECT k.term, COALESCE(k.category,'Sin dominio') AS category
                FROM keywords k
                WHERE k.is_active = true
                  AND NOT EXISTS (SELECT 1 FROM chunk_keywords ck WHERE ck.keyword_id = k.id)
                ORDER BY k.term
                """
            )
        )
    ).mappings().all()
    brechas = [r["term"] for r in rows]
    by_domain: dict[str, list[str]] = {}
    for r in rows:
        by_domain.setdefault(r["category"], []).append(r["term"])
    return {"brechas": brechas, "gaps_by_domain": by_domain, "count": len(brechas)}


async def keyword_matches(db: AsyncSession) -> list[dict[str, Any]]:
    """Coincidencias por keyword (suma de match_count)."""
    rows = (
        await db.execute(
            text(
                """
                SELECT k.term, COALESCE(k.category,'Sin dominio') AS category,
                       SUM(ck.match_count) AS matches
                FROM chunk_keywords ck
                JOIN keywords k ON k.id = ck.keyword_id
                WHERE k.is_active = true
                GROUP BY k.term, k.category
                ORDER BY matches DESC
                """
            )
        )
    ).mappings().all()
    return [{"term": r["term"], "category": r["category"], "matches": int(r["matches"] or 0)} for r in rows]


async def keyword_heatmap(db: AsyncSession, domain: str | None = None) -> list[dict[str, Any]]:
    """Matriz documento × keyword (frecuencia), opcionalmente filtrada por dominio."""
    where = "WHERE k.is_active = true"
    params: dict[str, Any] = {}
    if domain:
        where += " AND k.category = :domain"
        params["domain"] = domain
    rows = (
        await db.execute(
            text(
                f"""
                SELECT d.original_name AS document, k.term AS term,
                       SUM(ck.match_count) AS n
                FROM chunk_keywords ck
                JOIN chunks c ON c.id = ck.chunk_id
                JOIN documents d ON d.id = c.document_id
                JOIN keywords k ON k.id = ck.keyword_id
                {where}
                GROUP BY d.original_name, k.term
                """
            ),
            params,
        )
    ).mappings().all()
    by_doc: dict[str, dict[str, Any]] = {}
    for r in rows:
        d = by_doc.setdefault(r["document"], {"document": r["document"]})
        d[r["term"]] = int(r["n"] or 0)
    return list(by_doc.values())


async def document_domains(db: AsyncSession) -> dict[str, list[str]]:
    """Documento (nombre) → dominios (categorías de keywords detectadas)."""
    rows = (
        await db.execute(
            text(
                """
                SELECT DISTINCT d.original_name AS document, k.category AS category
                FROM chunk_keywords ck
                JOIN chunks c ON c.id = ck.chunk_id
                JOIN documents d ON d.id = c.document_id
                JOIN keywords k ON k.id = ck.keyword_id
                WHERE k.is_active = true AND k.category IS NOT NULL
                ORDER BY d.original_name, k.category
                """
            )
        )
    ).mappings().all()
    out: dict[str, list[str]] = {}
    for r in rows:
        out.setdefault(r["document"], [])
        if r["category"] not in out[r["document"]]:
            out[r["document"]].append(r["category"])
    return out


async def domains_in_corpus(db: AsyncSession) -> list[str]:
    rows = (
        await db.execute(
            text(
                """
                SELECT DISTINCT k.category AS category
                FROM chunk_keywords ck
                JOIN keywords k ON k.id = ck.keyword_id
                WHERE k.is_active = true AND k.category IS NOT NULL
                ORDER BY k.category
                """
            )
        )
    ).mappings().all()
    return [r["category"] for r in rows]


async def knowledge_summary(db: AsyncSession, domain: str | None = None) -> dict[str, Any]:
    cov = await keyword_coverage(db)
    wcov = await keyword_coverage(db, weight_priority=True)
    gaps = await keyword_gaps(db)
    matches = await keyword_matches(db)
    heatmap = await keyword_heatmap(db, domain=domain)
    doc_domains = await document_domains(db)
    return {
        "coverage": cov,
        "weighted_coverage": wcov,
        "matches": matches,
        "gaps": gaps,
        "heatmap": heatmap,
        "document_domains": doc_domains,
        "domains": await domains_in_corpus(db),
        "period": None,
    }
