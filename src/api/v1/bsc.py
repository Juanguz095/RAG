"""Router BSC: dashboard RAG Intelligence BSC (PLAN-005, RAG-042..052).

Todo el módulo BSC es SOLO para el rol admin: lecturas, reportes y mutaciones.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.permissions import require_permission
from src.database import ActionPlan, Alert, Kpi, KpiSnapshot, User, get_db
from src.services import bsc as bsc_svc
from src.services import bsc_knowledge as bsc_kw
from src.services.audit import audit

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/bsc", tags=["bsc"])


# --------------------------------------------------------------------- schemas

class ActionPlanIn(BaseModel):
    title: str
    kpi_code: str | None = None
    description: str | None = None
    owner: str | None = None
    due_date: str | None = None       # ISO date
    status: str = "open"


class ActionPlanUpdate(BaseModel):
    status: str | None = None
    description: str | None = None


# --------------------------------------------------------------------- reads

@router.get("/kpis")
async def list_kpis(
    db: AsyncSession = Depends(get_db),
    user: User | None = Depends(require_permission("bsc:read")),
):
    result = await db.execute(select(Kpi).where(Kpi.is_active == 1))
    kpis = result.scalars().all()
    out = []
    for k in kpis:
        # Últimos 2 snapshots → tendencia (spec §55: "Tendencia" por perspectiva).
        snaps = (
            await db.execute(
                select(KpiSnapshot).where(KpiSnapshot.kpi_id == k.id)
                .order_by(desc(KpiSnapshot.computed_at)).limit(2)
            )
        ).scalars().all()
        snap = snaps[0] if snaps else None
        value = snap.value if snap is not None else None
        prev = snaps[1].value if len(snaps) > 1 else None
        trend = None
        if value is not None and prev is not None:
            delta = round(float(value) - float(prev), 2)
            direction = (k.direction or "gte").lower()
            improved = (delta > 0) if direction == "gte" else (delta < 0)
            trend = {
                "delta": delta,
                "arrow": "up" if delta > 0 else ("down" if delta < 0 else "flat"),
                "improved": None if delta == 0 else improved,
            }
        out.append({
            "code": k.code, "name": k.name, "perspective": k.perspective,
            "direction": k.direction, "unit": k.unit, "target": k.target,
            "thresholds": k.thresholds,
            "last_value": value,
            "state": bsc_svc.state_for(k, value) if value is not None else "gray",
            "emoji": bsc_svc.state_emoji(bsc_svc.state_for(k, value) if value is not None else "gray"),
            "computed_at": snap.computed_at.isoformat(timespec="seconds") if snap is not None else None,
            "trend": trend,
        })
    return {"items": out, "count": len(out)}


@router.get("/gaps")
async def gaps(
    domain: str | None = None,
    db: AsyncSession = Depends(get_db),
    user: User | None = Depends(require_permission("bsc:read")),
):
    """Brechas + heatmap sobre el CATÁLOGO DE KEYWORDS (SQL agregado)."""
    kgaps = await bsc_kw.keyword_gaps(db)
    heatmap = await bsc_kw.keyword_heatmap(db, domain=domain)
    await audit(db, user, "keyword", resource_type="bsc",
                detail={"gaps": len(kgaps["brechas"]), "domain": domain})
    await db.commit()
    return {
        "brechas": kgaps["brechas"],
        "gaps_by_domain": kgaps["gaps_by_domain"],
        "heatmap": heatmap,
        "emergentes": [],
        "emergentes_detalle": [],
    }


@router.get("/knowledge")
async def knowledge(
    domain: str | None = None,
    db: AsyncSession = Depends(get_db),
    user: User | None = Depends(require_permission("bsc:read")),
):
    """Resumen de conocimiento del catálogo de keywords (cobertura, coincidencias,
    brechas por dominio, heatmap, dominios por documento)."""
    return await bsc_kw.knowledge_summary(db, domain=domain)


@router.get("/support")
async def support(
    db: AsyncSession = Depends(get_db),
    user: User | None = Depends(require_permission("bsc:read")),
):
    # B3 (RAG-045): índice de respaldo sobre las últimas consultas auditadas
    from src.database import AuditLog

    arows = (
        await db.execute(
            select(AuditLog).where(AuditLog.action == "query")
            .order_by(desc(AuditLog.created_at)).limit(50)
        )
    ).scalars().all()
    total_ev = 0
    total_pages = 0
    weights_used = []
    for r in arows:
        d = r.detail or {}
        sources = d.get("sources") or []
        total_ev += len(sources)
        pages = 0
        for s in sources:
            pn = s.get("page_numbers") or []
            pages += len(pn)
        total_pages += pages
        weights_used.append(d)
    evidence_per_query = total_ev / max(1, len(arows)) if arows else 0.0
    idx = bsc_svc.compute_support_index(
        evidence=round(evidence_per_query, 2), keyword_hits=0, avg_score=0.5,
        pages=1 if arows else 0,
    )
    idx["queries_considered"] = len(arows)
    idx["kpi"] = "SUPPORT_INDEX"
    return idx


@router.get("/score")
async def get_score(
    db: AsyncSession = Depends(get_db),
    user: User | None = Depends(require_permission("bsc:read")),
):
    kpis = (await db.execute(select(Kpi).where(Kpi.is_active == 1))).scalars().all()
    pairs = []
    for k in kpis:
        snap = (
            await db.execute(
                select(KpiSnapshot).where(KpiSnapshot.kpi_id == k.id)
                .order_by(desc(KpiSnapshot.computed_at)).limit(1)
            )
        ).scalar_one_or_none()
        if snap is not None:
            pairs.append((k, snap.value))
    score, formula = bsc_svc.compute_score(pairs)
    return {"value": score, "formula": formula,
            "nota": "Indicador de gestión; NO es probabilidad de verdad de la IA"}


@router.get("/usage")
async def usage(
    db: AsyncSession = Depends(get_db),
    user: User | None = Depends(require_permission("bsc:read")),
):
    """WP11 (PLAN-010): métricas de uso por usuario/cliente desde audit_log."""
    from src.database import AuditLog

    rows = (
        await db.execute(
            select(AuditLog).where(AuditLog.action.in_(["query", "upload"]))
            .order_by(desc(AuditLog.created_at)).limit(1000)
        )
    ).scalars().all()
    return bsc_svc.compute_usage_stats(rows)


# ------------------------------------------------------------------ mutations

@router.post("/kpis/{code}/compute")
async def compute_kpi(
    code: str,
    db: AsyncSession = Depends(get_db),
    user: User | None = Depends(require_permission("admin:panel")),  # mutaciones de KPI: admin
):
    k = (
        await db.execute(select(Kpi).where(Kpi.code == code))
    ).scalar_one_or_none()
    if k is None:
        raise HTTPException(status_code=404, detail="KPI not found")
    out = await bsc_svc.compute_and_snapshot(db, k)
    await audit(db, user, "admin_action", resource_type="kpi", resource_id=k.code,
                detail={**out})
    await db.commit()
    return out
    # (lógica movida a services/bsc.compute_and_snapshot — PLAN-007 F7)


async def _compute_kpi_legacy(db, user=None, code=None):
    data = await _collect_inputs(db)
    value: float
    formula = ""
    if k.query_type == "keyword_coverage":
        value, formula = bsc_svc.compute_keyword_coverage(data["synonyms"], data["chunks"])
    elif k.query_type == "query_stats":
        stats = bsc_svc.compute_query_stats(data["audit_rows"])
        value = stats["with_evidence"] if k.code == "QUERY_WITH_EVIDENCE" else (
            stats["without_evidence"] if k.code == "QUERY_NO_EVIDENCE" else stats["total"]
        )
        if k.code == "QUERY_WITH_EVIDENCE":
            value = stats["pct_with_evidence"]
        formula = "con/sin evidencia desde audit_log"
    else:
        raise HTTPException(status_code=422, detail=f"query_type no soportado: {k.query_type}")

    period = bsc_svc._period()
    snap = KpiSnapshot(
        kpi_id=k.id, value=value,
        period_start=datetime.fromisoformat(period["start"]),
        period_end=datetime.fromisoformat(period["end"]),
    )
    db.add(snap)
    state = bsc_svc.state_for(k, value)
    # B6: alerta si el estado es amber/red (dedup contra alertas abiertas en BD)
    open_alerts = (
        await db.execute(
            select(Alert).where(Alert.kpi_id == k.id, Alert.status == "open")
        )
    ).scalars().all()
    created = []
    bsc_svc.ensure_alert(k, state, value, created)  # type: ignore[arg-type]
    for a in created:
        if not open_alerts:
            db.add(a)  # dedup simple: si ya hay alerta open para el kpi, no duplicar
    await db.commit()
    await audit(db, user, "admin_action", resource_type="kpi", resource_id=k.code,
                detail={"value": value, "state": state, "formula": formula})
    await db.commit()
    return {"code": k.code, "value": value, "state": state, "formula": formula}


@router.post("/action-plans", status_code=201)
async def create_plan(
    body: ActionPlanIn,
    db: AsyncSession = Depends(get_db),
    user: User | None = Depends(require_permission("admin:panel")),
):
    kpi_id = None
    if body.kpi_code:
        k = (await db.execute(select(Kpi).where(Kpi.code == body.kpi_code))).scalar_one_or_none()
        if k is not None:
            kpi_id = k.id
    due = datetime.fromisoformat(body.due_date) if body.due_date else None
    plan = ActionPlan(
        title=body.title, kpi_id=kpi_id, description=body.description,
        owner=body.owner, due_date=due, status=body.status if body.status in
        ("open", "in_progress", "done", "cancelled") else "open",
    )
    db.add(plan)
    await db.commit()
    await db.refresh(plan)
    await audit(db, user, "admin_action", resource_type="action_plan",
                resource_id=str(plan.id), detail={"title": body.title})
    await db.commit()
    return {"id": str(plan.id), "title": plan.title, "status": plan.status}


@router.get("/action-plans")
async def list_plans(
    db: AsyncSession = Depends(get_db),
    user: User | None = Depends(require_permission("bsc:read")),
):
    rows = (await db.execute(select(ActionPlan).order_by(desc(ActionPlan.created_at)))).scalars().all()
    return {"items": [{
        "id": str(p.id), "title": p.title, "owner": p.owner, "status": p.status,
        "due_date": p.due_date.isoformat() if p.due_date else None,
        "kpi_id": str(p.kpi_id) if p.kpi_id else None,
    } for p in rows], "count": len(rows)}


@router.patch("/action-plans/{plan_id}")
async def update_plan(
    plan_id: str,
    body: ActionPlanUpdate,
    db: AsyncSession = Depends(get_db),
    user: User | None = Depends(require_permission("admin:panel")),
):
    p = None
    try:
        from uuid import UUID as _U

        p = (
            await db.execute(select(ActionPlan).where(ActionPlan.id == _U(plan_id)))
        ).scalar_one_or_none()
    except Exception:
        p = None
    if p is None:
        raise HTTPException(status_code=404, detail="Plan not found")
    if body.status and body.status in ("open", "in_progress", "done", "cancelled"):
        p.status = body.status
    if body.description is not None:
        p.description = body.description
    await db.commit()
    return {"id": str(p.id), "status": p.status}


# ---------------------------------------------------------------- colección de datos

async def _collect_inputs(db: AsyncSession) -> dict:
    """Trae el insumo de los cómputos: synonyms activos, chunks (con doc) y auditoría."""
    from src.database import Chunk, Document, MedicalSynonym

    syns = (await db.execute(select(MedicalSynonym))).scalars().all()
    cmap = {}
    try:
        crows = (
            await db.execute(select(Chunk))
        ).scalars().all()
        drows = (
            await db.execute(select(Document))
        ).scalars().all()
        cmap = {str(d.id): d for d in drows}
    except Exception:
        crows, cmap = [], {}
    chunks = []
    for chunk in crows:
        chunk.document = cmap.get(str(getattr(chunk, "document_id", None)))
        chunks.append(chunk)
    from src.database import AuditLog

    arows = (
        await db.execute(
            select(AuditLog).where(AuditLog.action.in_(["query", "upload"]))
            .order_by(desc(AuditLog.created_at)).limit(500)
        )
    ).scalars().all()
    return {"synonyms": syns, "chunks": chunks, "audit_rows": arows, "alerts": []}


# ---------------------------------------------------------------- reportes (B9)

def _dashboard_payload(data: dict, driver: str = "db") -> dict:
    """Compone el dashboard desde las colecciones de _collect_inputs."""
    kpis_snapshot = []
    gaps = bsc_svc.compute_gaps(data["synonyms"], data["chunks"])
    qstats = bsc_svc.compute_query_stats(data["audit_rows"])
    sup = bsc_svc.compute_support_index(
        evidence=qstats["with_evidence"] or 1, keyword_hits=0,
        avg_score=0.6, pages=2,
    )
    score, formula = bsc_svc.compute_score([])
    return {
        "title": "RAG INTELLIGENCE BSC — Del documento a la decisión",
        "generated_at": bsc_svc._period()["end"],
        "kpis": kpis_snapshot,
        "gaps": gaps,
        "queries": qstats,
        "support": sup,
        "score": {"value": score, "formula": formula},
    }


@router.get("/report.pdf")
async def report_pdf(
    db: AsyncSession = Depends(get_db),
    user: User | None = Depends(require_permission("bsc:read")),
):
    from fastapi.responses import Response

    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas
    except ImportError:
        raise HTTPException(status_code=503, detail="reportlab no disponible en este entorno")

    data = await _collect_inputs(db)
    payload = _dashboard_payload(data)
    import io

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    w, h = A4
    # Portada
    c.setFont("Helvetica-Bold", 22)
    c.drawCentredString(w / 2, h - 100, payload["title"])
    c.setFont("Helvetica", 11)
    c.drawCentredString(w / 2, h - 130, payload["generated_at"])
    c.showPage()
    # Perspectiva/KPIs
    y = h - 60
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y, "Resumen ejecutivo")
    y -= 26
    c.setFont("Helvetica", 10)
    lines = [
        f"Consultas: {payload['queries']['total']} (con evidencia {payload['queries']['with_evidence']}, "
        f"sin {payload['queries']['without_evidence']})",
        f"Cobertura keywords/brechas: {len(payload['gaps']['brechas'])} sin evidencia",
        f"Conceptos emergentes: {', '.join(payload['gaps']['emergentes'][:5]) or '—'}",
        f"Índice de respaldo: {payload['support']['level']} ({payload['support']['score']}/100)",
        f"Score consolidado: {payload['score']['value']}/100 — formula: {payload['score']['formula'][:90]}",
    ]
    for line in lines:
        c.drawString(60, y, line)
        y -= 18
    # Brechas / acciones
    y -= 12
    c.setFont("Helvetica-Bold", 13)
    c.drawString(50, y, "Brechas y planes de acción")
    y -= 20
    c.setFont("Helvetica", 10)
    plans = (await db.execute(select(ActionPlan).order_by(desc(ActionPlan.created_at)).limit(20))).scalars().all()
    for bch in payload["gaps"]["brechas"][:15]:
        c.drawString(60, y, f"- Sin evidencia: {bch}")
        y -= 14
    for p in plans[:10]:
        c.drawString(60, y, f"[P] {p.title} — {p.status} ({p.owner or 'sin owner'})")
        y -= 14
        if y < 60:
            c.showPage()
            y = h - 60
    c.save()
    await audit(db, user, "export", resource_type="bsc_report", resource_id="pdf")
    await db.commit()
    return Response(
        content=buf.getvalue(),
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=reporte_bsc.pdf"},
    )


@router.get("/report.xlsx")
async def report_xlsx(
    db: AsyncSession = Depends(get_db),
    user: User | None = Depends(require_permission("bsc:read")),
):
    from fastapi.responses import Response

    try:
        import openpyxl
    except ImportError:
        raise HTTPException(status_code=503, detail="openpyxl no disponible en este entorno")

    data = await _collect_inputs(db)
    payload = _dashboard_payload(data)
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "KPIs"
    ws.append(["KPI", "Valor/estado", "Detalle"])
    ws.append(["Consultas", payload["queries"]["total"], "total"])
    ws.append(["Con evidencia", payload["queries"]["with_evidence"], f"{payload['queries']['pct_with_evidence']}%"])
    ws.append(["Sin evidencia", payload["queries"]["without_evidence"], ""])
    ws.append(["Índice respaldo", payload["support"]["level"], f"{payload['support']['score']}/100"])
    ws.append(["Score", payload["score"]["value"], payload["score"]["formula"][:80]])
    ws.append(["Brechas", len(payload["gaps"]["brechas"]), "; ".join(payload["gaps"]["brechas"][:10])])
    for t in payload["gaps"]["emergentes"][:10]:
        ws.append(["Emergente", t, ""])
    import io

    buf = io.BytesIO()
    wb.save(buf)
    await audit(db, user, "export", resource_type="bsc_report", resource_id="xlsx")
    await db.commit()
    return Response(
        content=buf.getvalue(),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": "attachment; filename=reporte_bsc.xlsx"},
    )

