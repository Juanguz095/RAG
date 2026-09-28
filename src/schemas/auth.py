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
    # Solo roles de menor privilegio por JSON público; admin se asigna por
    # otros canales (ver docs/DISENO_RBAC_AUDITORIA_BSC.md D1/D2).
    role: Literal["assistant"] = "assistant"
