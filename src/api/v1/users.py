"""Gestión de usuarios: alta/listado/cambio con rol — admin-only (C3/M1)."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.deps import get_current_user, hash_password
from src.core.permissions import require_permission
from src.database import User, get_db
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
    user_id: str,
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
    await audit(db, admin, "admin_action", resource_type="user", resource_id=user_id,
                detail={"op": "update", "changes": changes})
    await db.commit()
    return UserOut(id=str(user.id), username=user.username, email=user.email,
                   role=user.role, is_active=bool(user.is_active))
