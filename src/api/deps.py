from __future__ import annotations

import hashlib
import hmac
import os
import time
from typing import Optional

import jwt
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.config import get_settings
from src.database import User, get_db

settings = get_settings()
security = HTTPBearer(auto_error=False)


_PBKDF2_ITERS = 600_000  # PLAN-004 M6


def hash_password(password: str) -> str:
    """pbkdf2-hmac con salt aleatorio, formato `pbkdf2$iters$salt$hash` (M6)."""
    salt = os.urandom(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, _PBKDF2_ITERS)
    return f"pbkdf2${_PBKDF2_ITERS}${salt.hex()}${dk.hex()}"


def verify_password(password: str, hashed: str) -> bool:
    """Acepta el formato nuevo pbkdf2 y el legacy SHA-256 sin salt (M6)."""
    if hashed.startswith("pbkdf2$"):
        try:
            _, iters, salt_hex, hash_hex = hashed.split("$")
            dk = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt_hex), int(iters))
            return hmac.compare_digest(dk.hex(), hash_hex)
        except (ValueError, TypeError):
            return False
    legacy = hashlib.sha256(password.encode()).hexdigest()
    return hmac.compare_digest(legacy, hashed)


def needs_rehash(hashed: str) -> bool:
    """True si el hash es legacy (re-hashear tras login exitoso, M6)."""
    return not hashed.startswith("pbkdf2$")


def create_token(user_id: str, role: str, username: str) -> str:
    payload = {
        "sub": user_id,
        "role": role,
        "username": username,
        "exp": time.time() + settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    }
    return jwt.encode(payload, settings.SECRET_KEY, algorithm=settings.JWT_ALGORITHM)


def decode_token(token: str) -> dict:
    try:
        return jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.JWT_ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")


async def get_current_user(
    cred: Optional[HTTPAuthorizationCredentials] = Depends(security),
    db: AsyncSession = Depends(get_db),
) -> Optional[User]:
    if not settings.AUTH_REQUIRED:
        return None
    if not cred:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
    data = decode_token(cred.credentials)
    result = await db.execute(select(User).where(User.id == data["sub"]))
    user = result.scalar_one_or_none()
    if not user or not user.is_active:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
    return user


def require_admin(user: Optional[User] = Depends(get_current_user)) -> Optional[User]:
    if settings.AUTH_REQUIRED and (not user or user.role != "admin"):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin required")
    return user


async def get_current_user_soft(
    cred: Optional[HTTPAuthorizationCredentials] = Depends(security),
    db: AsyncSession = Depends(get_db),
) -> Optional[User]:
    """Como get_current_user pero devuelve None sin cred (bootstrap M1).

    Solo lo usan endpoints que deben funcionar sin token en el bootstrap
    (el PRIMER usuario de un sistema vacío). Cualquier token inválido
    sigue dando 401; cred válida resuelve el usuario.
    """
    if not settings.AUTH_REQUIRED:
        return None
    if not cred:
        return None
    data = decode_token(cred.credentials)
    result = await db.execute(select(User).where(User.id == data["sub"]))
    user = result.scalar_one_or_none()
    if not user or not user.is_active:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
    return user
