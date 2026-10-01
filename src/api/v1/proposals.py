"""API /api/v1/proposals — validación humana (RAG-032, PLAN-006 Fase 5)."""
from __future__ import annotations

import uuid as _uuid
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select

from src.api.deps import get_db
from src.core.permissions import require_permission
from src.database import Proposal, User
from src.services.audit import audit

router = APIRouter(prefix="/api/v1/proposals", tags=["proposals"])


class ProposalIn(BaseModel):
    chunk_id: _uuid.UUID
    content: str


@router.post("", status_code=201)
async def propose(body: ProposalIn, db=Depends(get_db),
                  user: User | None = Depends(require_permission("documents:read"))):
    """assistant+ propone una corrección sobre un chunk (RAG-032)."""
    if user is None:
        raise HTTPException(401, "Not authenticated")
    if user.role not in ("admin", "editor", "assistant"):
        raise HTTPException(403, "Rol sin permiso de proponer")
    if not body.content.strip():
        raise HTTPException(422, "content vacío")
    p = Proposal(chunk_id=body.chunk_id, proposer=user.id, content=body.content, status="proposed")
    db.add(p)
    await db.commit()
    await db.refresh(p)
    await audit(db, user, "validation", resource_type="proposal", resource_id=str(p.id),
                detail={"action": "propose", "chunk_id": str(body.chunk_id)})
    await db.commit()
    return {"id": str(p.id), "status": p.status, "chunk_id": str(p.chunk_id)}


@router.get("")
async def list_proposals(db=Depends(get_db),
                         _u: User | None = Depends(require_permission("documents:update"))):
    res = await db.execute(select(Proposal).order_by(Proposal.created_at.desc()))
    rows = list(res.scalars().all()) if hasattr(res, "scalars") else list(res)
    return [
        {"id": str(p.id), "chunk_id": str(p.chunk_id), "content": p.content,
         "status": p.status, "comment": p.comment}
        for p in rows
    ]


async def _review(p: Proposal, db, user, status: str, comment: str | None):
    p.status = status
    p.reviewer = user.id
    p.reviewed_at = datetime.utcnow()
    p.comment = comment
    await db.commit()
    await audit(db, user, "validation", resource_type="proposal", resource_id=str(p.id),
                detail={"action": status, "comment": comment})
    await db.commit()
    return {"id": str(p.id), "status": p.status}


@router.post("/{pid}/approve")
async def approve(pid: _uuid.UUID, db=Depends(get_db),
                  user: User | None = Depends(require_permission("documents:update"))):
    res = await db.execute(select(Proposal).where(Proposal.id == pid))
    p = res.scalar_one_or_none()
    if not p:
        raise HTTPException(404, "no existe")
    return await _review(p, db, user, "approved", None)


@router.post("/{pid}/reject")
async def reject(pid: _uuid.UUID, body=None, db=Depends(get_db),
                 user: User | None = Depends(require_permission("documents:update"))):
    res = await db.execute(select(Proposal).where(Proposal.id == pid))
    p = res.scalar_one_or_none()
    if not p:
        raise HTTPException(404, "no existe")
    comment = (body or {}).get("comment") if isinstance(body, dict) else None
    return await _review(p, db, user, "rejected", comment)
