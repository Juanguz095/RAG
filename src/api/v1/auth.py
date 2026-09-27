from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.deps import create_token, get_current_user, hash_password
from src.database import User, get_db
from src.schemas.auth import LoginRequest, LoginResponse, UserCreate

router = APIRouter(prefix="/api/v1/auth", tags=["auth"])


@router.post("/login", response_model=LoginResponse)
async def login(req: LoginRequest, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(User).where(User.username == req.username))
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    from src.api.deps import verify_password
    if not verify_password(req.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = create_token(str(user.id), user.role, user.username)
    return LoginResponse(
        access_token=token,
        role=user.role,
        username=user.username,
    )


@router.post("/register", response_model=LoginResponse)
async def register(
    req: UserCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User | None = Depends(get_current_user),
):
    from src.config import get_settings
    settings = get_settings()
    if settings.AUTH_REQUIRED and (not current_user or current_user.role != "admin"):
        raise HTTPException(status_code=403, detail="Admin required")

    existing = await db.execute(select(User).where(User.username == req.username))
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=409, detail="Username exists")

    user = User(
        username=req.username,
        email=req.email,
        hashed_password=hash_password(req.password),
        role=req.role,
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)

    token = create_token(str(user.id), user.role, user.username)
    return LoginResponse(access_token=token, role=user.role, username=user.username)
