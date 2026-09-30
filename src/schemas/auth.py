from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class LoginRequest(BaseModel):
    username: str
    password: str


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    role: str
    username: str


class UserCreate(BaseModel):
    username: str
    password: str
    email: str
    # Registro público: SOLO rol de menor privilegio. `role:"admin"` en el
    # body responde 422 SIEMPRE (contrato PLAN-002) — la validación del
    # literal vive en el handler para que el gate 401 (sin token) preceda al
    # 422 (QA RAG-006). El bootstrap del primer admin es server-side (M1).
    role: str = "assistant"


class UserCreateByAdmin(BaseModel):
    """Alta de usuarios con rol arbitrario; admin-only vía POST /users (C3)."""

    username: str
    password: str
    email: str
    role: Literal["admin", "editor", "assistant", "viewer", "auditor"] = "assistant"


class UserUpdate(BaseModel):
    """PATCH /users/{id}: activar/desactivar y cambiar rol (C3)."""

    is_active: bool | None = None
    role: Literal["admin", "editor", "assistant", "viewer", "auditor"] | None = None


class UserOut(BaseModel):
    id: str
    username: str
    email: str
    role: str
    is_active: bool
