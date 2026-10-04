from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.deps import (
    create_token,
    get_current_user,
    get_current_user_soft,
    hash_password,
    needs_rehash,
    verify_password,
)
from src.database import User, get_db
from src.schemas.auth import LoginRequest, LoginResponse, UserCreate
from src.services.audit import audit

router = APIRouter(prefix="/api/v1/auth", tags=["auth"])


def _client_ip(request: Request | None) -> str | None:
    if request is not None and request.client:
        return request.client.host
    return None


@router.post("/login", response_model=LoginResponse)
async def login(
    req: LoginRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(User).where(User.username == req.username))
    user = result.scalar_one_or_none()
    if not user or not verify_password(req.password, user.hashed_password):
        await audit(db, None, "login_failed", resource_type="auth",
                    resource_id=req.username, ip=_client_ip(request))
        await db.commit()
        raise HTTPException(status_code=401, detail="Invalid credentials")

    # Usuario desactivado por un admin: no se emite token (C3/M1).
    if not user.is_active:
        await audit(db, None, "login_failed", resource_type="auth",
                    resource_id=req.username, ip=_client_ip(request))
        await db.commit()
        raise HTTPException(status_code=401, detail="Usuario inactivo")

    token = create_token(str(user.id), user.role, user.username)

    # M6: re-hash transparente del SHA-256 heredado tras login exitoso
    if needs_rehash(user.hashed_password):
        user.hashed_password = hash_password(req.password)
        await db.commit()

    await audit(db, user, "login", resource_type="auth",
                resource_id=str(user.id), ip=_client_ip(request))
    await db.commit()
    return LoginResponse(access_token=token, role=user.role, username=user.username)


@router.post("/logout", status_code=204)
async def logout(
    request: Request,
    response: Response,
    db: AsyncSession = Depends(get_db),
    current_user: User | None = Depends(get_current_user_soft),
):
    """M5: logout simple — evento auditado + 204.

    El JWT es stateless: la invalidación efectiva es la expiración corta
    (8 h, M6). No se construye blocklist en Redis (decisión PLAN-004 M5).
    """
    if current_user is not None:
        await audit(db, current_user, "logout", resource_type="auth",
                    resource_id=str(current_user.id), ip=_client_ip(request))
        await db.commit()
    return None


@router.post("/register", response_model=LoginResponse)
async def register(
    req: UserCreate,
    request: Request,
    db: AsyncSession = Depends(get_db),
    current_user: User | None = Depends(get_current_user_soft),
):
    from src.config import get_settings
    settings = get_settings()

    # M1 — bootstrap server-side: si users está vacía, el primer usuario
    # recibe role='admin' FORZADO por el servidor (ignora el body). Después,
    # solo admin crea cuentas y con rol de menor privilegio (422 para admin:
    # contrato PLAN-002 intacto). El resto de roles pasa por POST /users.
    # Bootstrap solo con gate ON: con AUTH_REQUIRED=false se preserva el
    # comportamiento legado exacto de PLAN-002 (contrato test_fase1).
    # scalar_one_or_none: el fake de los tests legados (test_fase1) solo
    # implementa scalar_one_or_none; None se interpreta como tabla vacía (M1).
    # Si la BD no está disponible no se puede verificar el bootstrap: gate
    # 401 (safe default, QA RAG-006).
    try:
        count_result = await db.execute(select(func.count()).select_from(User))
        total_users = count_result.scalar_one_or_none() or 0
    except Exception:
        if settings.AUTH_REQUIRED:
            raise HTTPException(status_code=401, detail="Not authenticated")
        total_users = 0

    if settings.AUTH_REQUIRED and total_users == 0:
        role = "admin"  # bootstrap M1
    elif settings.AUTH_REQUIRED:
        if not current_user:
            raise HTTPException(status_code=401, detail="Not authenticated")
        if current_user.role != "admin":
            raise HTTPException(status_code=403, detail="Admin required")
        if req.role != "assistant":
            raise HTTPException(status_code=422, detail="role must be assistant")
        role = req.role
    else:
        if req.role != "assistant":
            raise HTTPException(status_code=422, detail="role must be assistant")
        role = req.role

    existing = await db.execute(select(User).where(User.username == req.username))
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=409, detail="Username exists")

    user = User(
        username=req.username,
        email=req.email,
        hashed_password=hash_password(req.password),
        role=role,
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)

    await audit(db, current_user or user, "register", resource_type="auth",
                resource_id=str(user.id),
                detail={"role": role, "bootstrap": total_users == 0},
                ip=_client_ip(request))
    await db.commit()

    token = create_token(str(user.id), user.role, user.username)
    return LoginResponse(access_token=token, role=user.role, username=user.username)
