"""Consulta de auditoría — read-only (C6); auditor/admin (RAG-039)."""
from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends, Query
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
