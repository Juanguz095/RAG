"""Consulta de auditoría — read-only (C6); auditor/admin (RAG-039)."""
from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends, Query, Request
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.permissions import require_permission
from src.database import AuditLog, get_db

router = APIRouter(prefix="/api/v1/audit", tags=["audit"])


def _row_to_dict(r: AuditLog) -> dict:
    return {
        "id": str(r.id),
        "user_id": str(r.user_id) if r.user_id else None,
        "username": r.username,
        "action": r.action,
        "resource_type": r.resource_type,
        "resource_id": r.resource_id,
        "detail": r.detail if isinstance(r.detail, dict) else {},
        "ip": r.ip,
        "created_at": r.created_at.isoformat() if r.created_at else None,
    }


@router.get("")
async def list_audit(
    db: AsyncSession = Depends(get_db),
    user: object = Depends(require_permission("audit:read")),
    action: str | None = Query(None),
    username: str | None = Query(None),
    since: datetime | None = Query(None),
    until: datetime | None = Query(None),
    limit: int = Query(100, le=500),
    offset: int = Query(0, ge=0),
):
    """Listado read-only con filtros. Sin UPDATE/DELETE por diseño (C6)."""
    q = select(AuditLog).order_by(AuditLog.created_at.desc())
    if action:
        q = q.where(AuditLog.action == action)
    if username:
        q = q.where(AuditLog.username == username)
    if since:
        q = q.where(AuditLog.created_at >= since)
    if until:
        q = q.where(AuditLog.created_at <= until)
    q = q.limit(limit).offset(offset)
    result = await db.execute(q)
    rows = result.scalars().all()
    return {"items": [_row_to_dict(r) for r in rows], "count": len(rows)}


@router.get("/export")
async def export_audit(
    request,
    db: AsyncSession = Depends(get_db),
    user: object = Depends(require_permission("audit:read")),
    action: str | None = Query(None),
    username: str | None = Query(None),
    since: datetime | None = Query(None),
    until: datetime | None = Query(None),
):
    """WP6 (PLAN-010): exporta la bitácora de auditoría (CSV | XLSX | JSON).

    La bitácora (tabla audit_log, 11 eventos) ya existe — aquí solo se añade
    la exportación, la recomendación más pedida por los grupos (03, 05, 01).
    Filtros: rango de fechas, usuario y tipo de evento.
    """
    from fastapi import Request
    from fastapi.responses import Response

    fmt = (request.query_params.get("format") or "csv").lower()

    q = select(AuditLog).order_by(AuditLog.created_at.desc())
    if action:
        q = q.where(AuditLog.action == action)
    if username:
        q = q.where(AuditLog.username == username)
    if since:
        q = q.where(AuditLog.created_at >= since)
    if until:
        q = q.where(AuditLog.created_at <= until)
    q = q.limit(5000)
    result = await db.execute(q)
    rows = result.scalars().all()

    items = [
        {
            "fecha": r.created_at.isoformat() if r.created_at else "",
            "usuario": r.username or "",
            "accion": r.action,
            "tipo": r.resource_type or "",
            "recurso": r.resource_id or "",
            "detalle": r.detail if isinstance(r.detail, dict) else {},
            "ip": r.ip or "",
        }
        for r in rows
    ]

    if fmt == "json":
        import json as _json
        payload = _json.dumps(items, ensure_ascii=False, default=str)
        return Response(
            content=payload, media_type="application/json",
            headers={"Content-Disposition": 'attachment; filename="bitacora.json"'},
        )

    if fmt == "xlsx":
        import io as _io

        import openpyxl

        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "bitacora"
        ws.append(["fecha", "usuario", "accion", "tipo", "recurso", "detalle", "ip"])
        for it in items:
            ws.append([
                it["fecha"], it["usuario"], it["accion"], it["tipo"],
                it["recurso"],
                _json_dumps_detail(it["detalle"]),
                it["ip"],
            ])
        buf = _io.BytesIO()
        wb.save(buf)
        buf.seek(0)
        return Response(
            content=buf.getvalue(),
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": 'attachment; filename="bitacora.xlsx"'},
        )

    # default CSV
    import csv as _csv
    import io as _io

    buf = _io.StringIO()
    w = _csv.writer(buf)
    w.writerow(["fecha", "usuario", "accion", "tipo", "recurso", "detalle", "ip"])
    for it in items:
        w.writerow([
            it["fecha"], it["usuario"], it["accion"], it["tipo"], it["recurso"],
            _json_dumps_detail(it["detalle"]),
            it["ip"],
        ])
    return Response(
        content=buf.getvalue(), media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="bitacora.csv"'},
    )


def _json_dumps_detail(detail) -> str:
    import json as _json

    if not detail:
        return ""
    if isinstance(detail, dict):
        return _json.dumps(detail, ensure_ascii=False, default=str)
    return str(detail)
