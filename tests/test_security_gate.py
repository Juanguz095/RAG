"""Gate de autenticacion: endpoints criticos deben rechazar peticiones sin token.

Evidencia original (QA, RAG-006): get_current_user solo se usaba en auth.py y
AUTH_REQUIRED=false por defecto -> cualquier endpoint pasaba sin credenciales.
Estos tests fijan el contrato: con el gate ON y sin token => 401.
"""
from __future__ import annotations

import httpx


def test_register_sin_token_rechazado(client):
    resp = client.post(
        "/api/v1/auth/register",
        json={"username": "x", "email": "x@y.z", "password": "p", "role": "admin"},
    )
    assert resp.status_code == httpx.codes.UNAUTHORIZED


def test_documents_sin_token_rechazado(client):
    resp = client.get("/api/v1/documents")
    assert resp.status_code == httpx.codes.UNAUTHORIZED


def test_query_sin_token_rechazado(client):
    resp = client.post("/api/v1/query", json={"query": "test"})
    assert resp.status_code == httpx.codes.UNAUTHORIZED


def test_token_falso_rechazado(client):
    resp = client.get(
        "/api/v1/documents",
        headers={"Authorization": "Bearer no-es-un-jwt"},
    )
    assert resp.status_code == httpx.codes.UNAUTHORIZED
