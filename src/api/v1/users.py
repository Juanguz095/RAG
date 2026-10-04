"""Gestión de usuarios: alta/listado/cambio con rol — admin-only (C3/M1)."""
from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Response
from sqlalchemy import delete, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.deps import get_current_user, hash_password
from src.core.permissions import require_permission
from src.database import Conversation, Keyword, Message, Proposal, User, get_db
from src.schemas.auth import UserCreateByAdmin, UserOut, UserUpdate
from src.services.audit import audit

router = APIRouter(prefix="/api/v1/users", tags=["users"])


@router.post("", status_code=201, response_model=UserOut)
async def create_user(
    req: UserCreateByAdmin,
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(require_permission("users:create")),
):
    existing = await db.execute(select(User).where(User.username == req.username))
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=409, detail="Username exists")
    existing_email = await db.execute(select(User).where(User.email == req.email))
    if existing_email.scalar_one_or_none():
        raise HTTPException(status_code=409, detail="Email exists")

    user = User(
        username=req.username,
        email=req.email,
        hashed_password=hash_password(req.password),
        role=req.role,
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)
    await audit(db, admin, "admin_action", resource_type="user", resource_id=str(user.id),
                detail={"op": "create", "new_role": req.role})
    await db.commit()
    return UserOut(id=str(user.id), username=user.username, email=user.email,
                   role=user.role, is_active=bool(user.is_active))


@router.get("", response_model=list[UserOut])
async def list_users(
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(require_permission("users:read")),
):
    result = await db.execute(select(User).order_by(User.created_at))
    users = result.scalars().all()
    return [
        UserOut(id=str(u.id), username=u.username, email=u.email,
                role=u.role, is_active=bool(u.is_active))
        for u in users
    ]


@router.patch("/{user_id}", response_model=UserOut)
async def update_user(
    user_id: UUID,
    req: UserUpdate,
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(require_permission("users:update")),
):
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    changes = {}
    if req.is_active is not None:
        user.is_active = 1 if req.is_active else 0
        changes["is_active"] = req.is_active
    if req.role is not None:
        user.role = req.role
        changes["role"] = req.role
    if not changes:
        raise HTTPException(status_code=422, detail="Nada que actualizar")

    await db.commit()
    await db.refresh(user)
    await audit(db, admin, "admin_action", resource_type="user", resource_id=str(user_id),
                detail={"op": "update", "changes": changes})
    await db.commit()
    return UserOut(id=str(user.id), username=user.username, email=user.email,
                   role=user.role, is_active=bool(user.is_active))


@router.delete("/{user_id}", status_code=204)
async def delete_user(
    user_id: UUID,
    db: AsyncSession = Depends(get_db),
    admin: User | None = Depends(require_permission("users:update")),
):
    """Elimina un usuario (admin). No permite auto-eliminarse.

    Antes de borrar, limpia las referencias para respetar las FK:
    borra sus conversaciones/mensajes, anula `keywords.created_by` y
    `proposals.reviewer`, y reasigna `proposals.proposer` (NOT NULL) al admin.
    `audit_log.user_id` y `alerts.resolved_by` pasan a NULL por ON DELETE SET NULL.
    """
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    if admin is not None and str(user.id) == str(admin.id):
        raise HTTPException(status_code=400, detail="No puedes eliminar tu propia cuenta")

    conv_ids = select(Conversation.id).where(Conversation.user_id == user.id)
    await db.execute(delete(Message).where(Message.conversation_id.in_(conv_ids)))
    await db.execute(delete(Conversation).where(Conversation.user_id == user.id))
    await db.execute(update(Keyword).where(Keyword.created_by == user.id).values(created_by=None))
    await db.execute(update(Proposal).where(Proposal.reviewer == user.id).values(reviewer=None))
    if admin is not None:
        await db.execute(update(Proposal).where(Proposal.proposer == user.id).values(proposer=admin.id))

    username = user.username
    await db.delete(user)
    await db.commit()
    await audit(db, admin, "admin_action", resource_type="user", resource_id=str(user_id),
                detail={"op": "delete", "username": username})
    await db.commit()
    return Response(status_code=204)
