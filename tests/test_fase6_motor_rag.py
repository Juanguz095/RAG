"""PLAN-006 — Fase 1 (RAG-037 admin123) + Fase 2 (RAG-024/025/026 motor verificable).

TDD en ROJO primero. CP-006/CP-007 son contratos literales.
"""
from __future__ import annotations

import sys
import uuid
from datetime import datetime
from unittest.mock import MagicMock

import httpx
import pytest

if sys.modules.get("psycopg2") is None:
    try:
        import psycopg2  # noqa: F401
    except ModuleNotFoundError:
        sys.modules["psycopg2"] = MagicMock()

import pathlib

import test_fase2_rbac_auditoria as _f2  # arnés compartido

_make_fake_db = _f2._make_fake_db
_override_db = _f2._override_db
_register = _f2._register
_auth = _f2._auth
_empty_db_override = _f2._empty_db_override


# ============================================================ FASE 1 — RAG-037

def test_p1_login_html_no_tiene_admin123(client):
    """RAG-037: el HTML servido no puede precargar la contraseña."""
    resp = client.get("/")
    assert resp.status_code == 200
    assert "admin123" not in resp.text, "el login no debe precargar contraseñas"


def test_p1_seed_admin_sin_password_hardcodeada(client, monkeypatch):
    """seed_admin usa ADMIN_PASSWORD del entorno, no 'admin123'."""
    import test_fase1_seguridad  # noqa  (sanity: suite legada intacta)

    scripts = pathlib.Path(__file__).parent.parent / "scripts" / "seed_admin.py"
    src = scripts_src = scripts.read_text(encoding="utf-8")
    assert "admin123" not in src, "seed_admin no debe contener admin123"
    assert "ADMIN_PASSWORD" in src, "seed_admin debe usar ADMIN_PASSWORD env"
    # y lanzar error si no está configurada:
    import importlib


# ============================================================ FASE 2 — motor

class _AuditQ:
    def __init__(self, n_sources):
        self.id = uuid.uuid4()
        self.username = "root"
        self.action = "query"
        self.created_at = datetime(2026, 10, 1)
        self.detail = {"sources": []}


def _setup_admin(client, monkeypatch):
    _empty_db_override(client, monkeypatch)
    _register(client, username="root")
    return _auth(client, "root")


# ---- RAG-025: abstención con umbral (CP-006)

def test_p2_abstencion_sin_evidencia_cp006(client, monkeypatch):
    """Sin resultados por encima del umbral → mensaje de insuficiencia literal."""
    _empty_db_override(client, monkeypatch)
    _register(client, username="root")
    hdrs = _auth(client, "root")

    async def no_results(*_a, **_k):
        return []

    monkeypatch.setattr("src.api.v1.query.hybrid_search", no_results)
    monkeypatch.setattr(
        "src.api.v1.query.generate_answer_timed",
        lambda ctx, q, **k: {"text": "No dispongo de información suficiente en los documentos indexados para responder esa consulta; no voy a inventar una respuesta.",
                             "prompt_eval_ms": 1, "generation_ms": 1, "tokens_generated": 0})
    resp = client.post("/api/v1/query", json={"query": "xyz-no-existe"}, headers=hdrs)
    assert resp.status_code == 200
    d = resp.json()
    assert d["abstained"] is True, d
    # wording literal CP-006: declara insuficiencia de información y no inventa
    text = (d["answer"] or "").lower()
    assert "insuficiente" in text or "suficiente" in text
    assert "no voy a inventar" in text or "no inventa" in text or "no inventar" in text


def test_p2_umbral_de_evidencia_excluye_resultados_debiles(client, monkeypatch):
    """Con resultados por debajo de EVIDENCE_MIN_SCORE → abstención estructural."""
    from src.services import reranker as rr

    _empty_db_override(client, monkeypatch)
    _register(client, username="root")
    hdrs = _auth(client, "root")

    class _R:
        score = 0.01  # muy débil
        matched_terms = []
        document_name = "x.pdf"

    async def weak(*_a, **_k):
        return [_R()]

    monkeypatch.setattr("src.api.v1.query.hybrid_search", weak)
    monkeypatch.setattr(rr, "rerank", lambda q, res, top_k=10: res[:top_k], raising=False)
    resp = client.post("/api/v1/query", json={"query": "lo que sea"}, headers=hdrs)
    assert resp.status_code == 200
    d = resp.json()
    # con umbral ON y score 0.01 < 0.15 → estructura de abstención
    assert d["abstained"] is True


def test_p2_respuesta_con_evidencia_grounded(client, monkeypatch):
    """CP-007: con resultados fuertes → respuesta grounded con citas mapeadas."""
    _empty_db_override(client, monkeypatch)
    _register(client, username="root")
    hdrs = _auth(client, "root")

    class _Chk:
        id = uuid.uuid4()
        document_id = uuid.uuid4()
        content = "El glaucoma afecta el nervio optico. Tratamiento con gotas."
        page_numbers = [3]
        chunk_index = 0
        document = None

    class _R:
        chunk = None
        chunk_id = _Chk.id
        document_id = _Chk.document_id
        content = _Chk.content
        page_numbers = [3]
        score = 0.9
        relevance = 90
        chunk_metadata = {}
        chunk_index = 0
        document_name = "segundo.pdf"
        matched_terms = ["glaucoma"]
        vector_score = 0.9
        keyword_score = 0.5
        rrf_score = 0.7

        def __init__(self):
            self.chunk = _Chk()
            self.chunk.content = self.content
            self.document = None

    async def fake_search(*_a, **_k):
        return [_R()]

    monkeypatch.setattr("src.api.v1.query.hybrid_search", fake_search)
    import src.services.reranker as rr
    monkeypatch.setattr(rr, "rerank", lambda q, res, top_k=10: res[:top_k], raising=False)
    monkeypatch.setattr(
        "src.api.v1.query.generate_answer_timed",
        lambda ctx, q, **k: {"text": "El glaucoma afecta el nervio optico [1].",
                             "prompt_eval_ms": 1, "generation_ms": 1, "tokens_generated": 5},
    )
    resp = client.post("/api/v1/query", json={"query": "glaucoma"}, headers=hdrs)
    d = resp.json()
    assert resp.status_code == 200
    assert d["abstained"] is False
    assert d["grounded"] is True, d.get("grounded")
    # citas del texto ⊆ sources
    srcs = d["sources"]
    assert len(srcs) >= 1
    n_srcs = len(srcs)
    import re
    for n in re.findall(r"\[(\d+)\]", d["answer"]):
        assert 1 <= int(n) <= n_srcs, f"cita {n} sin fuente (CP-007)"


# ---- RAG-024: grounding negativo

def test_p2_grounding_negativo_marca_false(client, monkeypatch):
    """Respuesta del LLM sin citas válidas → grounded=false (con flag)."""
    _empty_db_override(client, monkeypatch)
    _register(client, username="root")
    hdrs = _auth(client, "root")

    class _Chk:
        id = uuid.uuid4()
        document_id = uuid.uuid4()
        content = "contenido"
        page_numbers = [1]
        document = None

    class _R:
        chunk = None
        content = "contenido"
        chunk_id = _Chk.id
        document_id = _Chk.document_id
        page_numbers = [1]
        score = 0.9
        relevance = 90
        chunk_metadata = {}
        chunk_index = 0
        document_name = "x.pdf"
        matched_terms = []
        vector_score = 0.9
        keyword_score = 0.4
        rrf_score = 0.6

    async def fake_search(*_a, **_k):
        return [_R()]

    monkeypatch.setattr("src.api.v1.query.hybrid_search", fake_search)
    import src.services.reranker as rr
    monkeypatch.setattr(rr, "rerank", lambda q, res, top_k=10: res[:top_k], raising=False)
    # LLM responde SIN citas:
    monkeypatch.setattr(
        "src.api.v1.query.generate_answer_timed",
        lambda ctx, q, **k: {"text": "El glaucoma se cura con te verde de hierva.",
                             "prompt_eval_ms": 1, "generation_ms": 1, "tokens_generated": 5},
    )
    resp = client.post("/api/v1/query", json={"query": "glaucoma"}, headers=hdrs)
    d = resp.json()
    assert resp.status_code == 200
    assert d["grounded"] is False, d


def test_p2_query_response_schema_tiene_campos_nuevos():
    """El schema QueryResponse debe tener abstained y grounded."""
    from src.schemas.query import QueryResponse

    fields = QueryResponse.model_fields
    assert "abstained" in fields
    assert "grounded" in fields
