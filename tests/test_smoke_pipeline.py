"""Humo sin infraestructra: routers montados, health y gate apagado.

Corre sin Postgres/Redis/MinIO: verifica el arnes para QA (compose caido) y
detecta que el gate OFF deja pasar (estado actual: parcial RAG-006).
"""
from __future__ import annotations

import httpx


def test_openapi_expone_los_cuatro_routers(client):
    resp = client.get("/openapi.json")
    assert resp.status_code == httpx.codes.OK
    paths = resp.json()["paths"]
    for path in (
        "/api/v1/auth/login",
        "/api/v1/auth/register",
        "/api/v1/documents",
        "/api/v1/query",
    ):
        assert path in paths, f"router ausente del contrato: {path}"


def test_health_arranca_sin_infraes(client):
    resp = client.get("/api/v1/health")
    assert resp.status_code == httpx.codes.OK
    body = resp.json()
    assert body["checks"]["api"] is True
    # Sin postgres/redis la app degrada en vez de caer.
    assert body["status"] in ("healthy", "degraded")


def test_gate_apagado_no_bloquea(auth_off):
    # Estado del MVP (AUTH_REQUIRED=false): la ruta llega al handler.
    resp = auth_off.get("/api/v1/documents")
    assert resp.status_code not in (httpx.codes.UNAUTHORIZED, httpx.codes.FORBIDDEN)
