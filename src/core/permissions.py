"""Matriz central de permisos (D2 del diseño, PLAN-004 §3) + factory FastAPI.

Sin tabla `roles`: los 5 roles son fijos. Un solo dict gobierna todo.
Permisos: "modulo:accion".
"""
from __future__ import annotations

from typing import Optional

from fastapi import Depends, HTTPException, status

from src.api.deps import get_current_user
from src.database import User

ROLE_PERMISSIONS: dict[str, set[str]] = {
    # admin: todo (incluida gestión de usuarios y configuración)
    "admin": {
        "users:create", "users:read", "users:update",
        "documents:upload", "documents:update", "documents:delete", "documents:read",
        "keywords:write", "keywords:read",
        "query", "search", "export",
        "audit:read", "bsc:read",
        "alerts:resolve", "action_plans:write", "kpis:write",
        "chat:read", "chat:write",
        "admin:panel",
    },
    # editor (Especialista): gestiona documentos y catálogo
    "editor": {
        "documents:upload", "documents:update", "documents:delete", "documents:read",
        "keywords:write", "keywords:read",
        "query", "search", "export",
        "alerts:resolve", "action_plans:write", "kpis:write",
        "bsc:read", "chat:read", "chat:write", "chat:read", "chat:write",
    },
    # assistant (Usuario autorizado): consulta + exporta; el contrato C1
    # (CP-002) exige que también pueda subir documentos al pipeline.
    "assistant": {
        "documents:read", "documents:upload",
        "keywords:read",
        "query", "search", "export",
        "bsc:read", "chat:read", "chat:write",
    },
    # viewer (Usuario de consulta): solo lectura
    "viewer": {
        "documents:read",
        "keywords:read",
        "query", "search", "chat:read", "chat:write",
        "bsc:read",
    },
    # auditor: lectura de auditoría y tablero
    "auditor": {
        "audit:read", "bsc:read",
        "documents:read", "keywords:read", "query", "search",
        "chat:read", "chat:write",
    },
}


def has_permission(user: Optional[User], permission: str) -> bool:
    """True si `user` (con gate ON) tiene el permiso dado.

    Con AUTH_REQUIRED apagado (desarrollo local), se comporta como admin
    para no romper el arnés ni el smoke sin token.
    """
    from src.config import get_settings

    if not get_settings().AUTH_REQUIRED:
        return True
    if not user:
        return False
    return permission in ROLE_PERMISSIONS.get(user.role, set())


def require_permission(permission: str):
    """Dependency factory: 403 si el rol no tiene el permiso."""

    async def checker(
        user: Optional[User] = Depends(get_current_user),
    ) -> Optional[User]:
        if not has_permission(user, permission):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission required: {permission}",
            )
        return user

    return checker
