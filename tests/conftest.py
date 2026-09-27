"""Fixtures compartidas de la suite.

Estado intencional de la suite:
- Sin infraexterna (Postgres/Redis/MinIO/modelos): todo lo que falla en el
  lifespan de la app esta envuelto en try/except degrades, y aqui stubbeamos
  el preload de embeddings para no cargar MiniLM en cada test.
- El gate de seguridad se fuerza ON (AUTH_REQUIRED=true) al importar, para
  verificar el camino autenticado (regresion RAG-006) por defecto.
"""
from __future__ import annotations

import os

# Debe ejecutarse ANTES de cualquier import de src.* (deps.py cachea settings).
os.environ["AUTH_REQUIRED"] = "true"
os.environ.setdefault("LOG_LEVEL", "WARNING")

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

# Stub del preload de embeddings: los tests de humo no deben cargar MiniLM.
try:
    import src.services.embeddings as _emb  # noqa: E402

    _emb.preload = lambda: None
except Exception:  # pragma: no cover
    pass

from src.main import app  # noqa: E402


@pytest.fixture()
def client():
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


@pytest.fixture()
def auth_off(client, monkeypatch):
    """Misma app, con el gate de auth apagado (estado actual del MVP)."""
    from src.api import deps

    monkeypatch.setattr(deps.settings, "AUTH_REQUIRED", False)
    return client
