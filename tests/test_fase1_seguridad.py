"""Fase 1 de seguridad (orden @jefe-de-desarrollo).

Contratos que fallan HOY (RED) y deben pasar despues del arreglo (GREEN),
mapeados a los hallazgos confirmados por QA:

1. `/api/v1/synonyms` y `/synonyms/expand` sin token => 401.
   Hoy: sin `Depends(get_current_user)` el handler toca BD y da 500.
2. `POST /auth/register` no crea admins: `role="admin"` en el body => 422
   (Literal de roles). Hoy: aceptado y persistido (escalada de privilegios).
3. Alta de usuario valida (role assistant) sigue funcionando y persiste
   exactamente ese rol.
4. Subida con Redis caido => 503 controlado (HTTPException), rollback del
   documento y sin PDF remanente en disco. Hoy: 500 + doc en pending.

La BD se reemplaza por un fake (override de la dependencia `get_db`) para
que la suite siga corriendo sin Postgres/Redis, igual que el resto del
arnes ya existente en conftest.py.
"""
from __future__ import annotations

import sys
from unittest.mock import MagicMock

import httpx
import pytest

# `src.workers.ingestion_tasks` crea un sync_engine con el dialecto
# psycopg2 al importar. En host no hay psycopg2 (solo dentro del
# contenedor del worker); para probar el contrato de encolado hacemos
# stub del modulo: nunca se conecta a BD en este test.
if sys.modules.get("psycopg2") is None:
    try:
        import psycopg2  # noqa: F401
    except ModuleNotFoundError:
        sys.modules["psycopg2"] = MagicMock()


# ---------------------------------------------------------------- #1 synonyms

def test_synonyms_sin_token_401(client):
    resp = client.get("/api/v1/synonyms")
    assert resp.status_code == httpx.codes.UNAUTHORIZED


def test_synonyms_expand_sin_token_401(client):
    resp = client.get("/api/v1/synonyms/expand?q=dolor")
    assert resp.status_code == httpx.codes.UNAUTHORIZED


# --------------------------------------------------- #2/#3 registro de roles

def test_register_role_admin_rechazado(auth_off, fake_db):
    """Body con role='admin' no pasa: rechazado por validacion, no persistido."""
    resp = auth_off.post(
        "/api/v1/auth/register",
        json={
            "username": "vuln_admin",
            "email": "vuln@rag.local",
            "password": "T3st-clav3-larga",
            "role": "admin",
        },
    )
    assert resp.status_code == httpx.codes.UNPROCESSABLE_ENTITY
    assert fake_db.added == []  # nada persistido


def test_register_role_assistant_valido(auth_off, fake_db):
    resp = auth_off.post(
        "/api/v1/auth/register",
        json={
            "username": "medico_test",
            "email": "medico@rag.local",
            "password": "T3st-clav3-larga",
            "role": "assistant",
        },
    )
    assert resp.status_code == httpx.codes.OK
    assert len(fake_db.added) == 1
    assert fake_db.added[0].role == "assistant"


# --------------------------------------------- #4 Redis caido durante upload


class _FakeDb:
    """Reemplazo minimo del contrato que usa upload_document."""

    def __init__(self) -> None:
        self.added: list = []
        self.commits = 0
        self.rollbacks = 0
        self.persisted = False

    async def execute(self, *_a, **_k):
        class _Result:
            def scalar_one_or_none(self):
                return None

        return _Result()

    def add(self, obj) -> None:
        self.added.append(obj)
        self.persisted = True

    async def commit(self) -> None:
        self.commits += 1
        self.persisted = True

    async def refresh(self, _obj) -> None:
        pass

    async def rollback(self) -> None:
        self.rollbacks += 1
        self.persisted = False


@pytest.fixture()
def fake_db(client):
    """Override de get_db sobre el client del conftest (aislado por test)."""
    from src.database import get_db

    db = _FakeDb()
    client.app.dependency_overrides[get_db] = lambda: db
    yield db
    client.app.dependency_overrides.pop(get_db, None)


def test_upload_redis_caido_503_con_rollback(auth_off, fake_db, tmp_path, monkeypatch):
    import src.workers.ingestion_tasks as worker_module

    def enqueue_roto(document_id: str, pdf_path: str) -> None:
        raise ConnectionError("Redis caido")

    monkeypatch.setattr(worker_module, "enqueue_process_document", enqueue_roto)
    monkeypatch.setattr(
        "src.api.v1.documents.UPLOAD_DIR", tmp_path, raising=True
    )

    resp = auth_off.post(
        "/api/v1/documents",
        files={"file": ("demo.pdf", b"%PDF-1.4 demo", "application/pdf")},
    )

    assert resp.status_code == 503
    assert fake_db.rollbacks >= 1
    assert fake_db.persisted is False  # sin huerfano en pending
    assert list(tmp_path.iterdir()) == []  # sin PDF remanente en disco
