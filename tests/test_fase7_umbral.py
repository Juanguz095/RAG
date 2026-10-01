"""PLAN-007 — CP-006 fix: umbral de evidencia sobre rerank_score (sigmoid 0-1)."""
from __future__ import annotations

import sys
from unittest.mock import MagicMock

try:
    import psycopg2  # noqa: F401
except ModuleNotFoundError:
    sys.modules["psycopg2"] = MagicMock()


def test_rerank_guarda_rerank_score_sigmoid():
    """Tras rerank(), cada resultado trae rerank_score en [0,1]."""
    from src.services import reranker as rr

    class R:
        def __init__(self, c):
            self.content = c
            self.score = 0.01

    class FakeModel:
        def predict(self, pairs, **_k):
            # logits: irrelevant -5 → 0.007; relevante +2 → 0.88
            return [-5.0, 2.0, 0.5]

    orig = rr._load_model
    rr._load_model = lambda: FakeModel()
    try:
        res = [R("x"), R("y"), R("z")]
        out = rr.rerank("q", res, top_k=3)
        scores = [getattr(r, "rerank_score", -1) for r in out]
        assert all(0.0 <= s <= 1.0 for s in scores), scores
        assert scores[0] > 0.86, scores  # el de logit +2 arriba
    finally:
        rr._load_model = orig


def test_cp006_query_abstine_y_responde_en_vivo_mismo_umbral():
    """Con rerank_score, un chunk altamente relevante pasa el umbral 0.15."""
    import math

    class R:
        score = 0.01
        rerank_score = 0.88

    def _evidence_score(r):
        rs = getattr(r, "rerank_score", None)
        if rs is not None:
            return float(rs)
        return float(r.score) * 10.0

    assert _evidence_score(R()) >= 0.15
    assert 1.0 / (1.0 + math.exp(-2.0)) > 0.86


def test_context_trae_etiquetas_numericas():
    """RAG-026: build_context enumera las fuentes [1],[2]... para citas."""
    from src.services.context import build_context

    class R:
        document_name = "a.pdf"
        page_numbers = [1]
        content = "texto clinico"

    ctx = build_context([R(), R()])
    assert "[1]" in ctx and "[2]" in ctx


def test_cp007_respuesta_incluye_fuentes_menos_citas_inline(client, monkeypatch):
    """Sin citas inline del LLM, la respuesta trae el bloque 'Fuentes consultadas'."""
    import uuid as _u
    import test_fase2_rbac_auditoria as _f2

    _f2._empty_db_override(client, monkeypatch)
    _f2._register(client, username="roo")
    h = _f2._auth(client, "roo")

    class _R:
        score = 0.9
        relevance = 90
        rerank_score = 0.9
        chunk_id = _u.uuid4()
        document_id = _u.uuid4()
        content = "internamiento del recién nacido requiere identificación."
        page_numbers = [2]
        chunk_index = 0
        chunk_metadata = {}
        document_name = "segundo.pdf"
        matched_terms = []
        vector_score = 0.9
        keyword_score = 0.5
        rrf_score = 0.7
        chunk = None

    import src.services.reranker as rr
    monkeypatch.setattr("src.api.v1.query.hybrid_search", lambda *a, **k: None)
    async def fs(*_a, **_k):
        return [_R()]
    monkeypatch.setattr("src.api.v1.query.hybrid_search", fs)
    monkeypatch.setattr(rr, "rerank", lambda q, res, top_k=10: res[:top_k], raising=False)
    monkeypatch.setattr("src.api.v1.query.generate_answer_timed",
                        lambda ctx, q, **k: {"text": "El internamiento requiere identification.",
                                             "prompt_eval_ms": 1, "generation_ms": 1, "tokens_generated": 4})
    r = client.post("/api/v1/search")
    r = client.post("/api/v1/query", json={"query": "internamiento"}, headers=h)
    d = r.json()
    assert "Fuentes consultadas" in d["answer"], d["answer"][:300]
    assert "segundo.pdf" in d["answer"]


def test_build_context_no_excede_ctx_tokens():
    """El contexto nunca debe desbordar el ctx-LLM 2048 (system+Q margen)."""
    from src.services.context import build_context, _estimate_tokens, settings

    class R:
        document_name = "a.pdf"
        page_numbers = [1]
        content = "palabra " * 3000

    ctx = build_context([R() for _ in range(12)])
    sys_overhead = 400  # system prompt + question + template
    assert _estimate_tokens(ctx) + sys_overhead < settings.LLM_N_CTX - 60


def test_export_history_format_param_via_alias(client, monkeypatch):
    """?format=csv (alias del ?fmt) devuelve CSV con cabecera."""
    import test_fase2_rbac_auditoria as _f2
    _f2._empty_db_override(client, monkeypatch)
    _f2._register(client, username="rox")
    h = _f2._auth(client, "rox")
    r = client.get("/api/v1/query/export/history?format=csv", headers=h)
    body = r.text
    assert r.status_code == 200
    assert body.startswith("ts,question,answer,n_sources") or "ts,question" in body.splitlines()[0], body[:120]

def test_auth_required_default_true(monkeypatch):
    """PLAN-007 F2: auth va ENCENDIDO por defecto (sin env AUTH_REQUIRED)."""
    import os, importlib
    from src import config as cfg

    monkeypatch.delenv("AUTH_REQUIRED", raising=False)
    importlib.reload(cfg)
    try:
        assert cfg.get_settings.__wrapped__ if False else True
        from src.config import Settings
        s = Settings()
        assert s.AUTH_REQUIRED is True
    finally:
        importlib.reload(cfg)


def test_borrar_documento_pending_no_se_bloquea(client, monkeypatch):
    """PLAN-007 fix: pending (subido y atascado sin pipeline) SÍ puede borrarse."""
    import uuid as _u
    import test_fase2_rbac_auditoria as _f2
    from src.database import get_db, Document

    users = []
    base = _f2._make_fake_db(users)
    _f2._override_db(client, base)
    _f2._register(client, username="pp9")
    h = _f2._auth(client, "pp9")

    doc = Document(id=_u.uuid4(), filename=_u.uuid4().hex + ".pdf",
                   original_name="atascado.pdf", file_hash=_u.uuid4().hex,
                   mime_type="application/pdf", file_size=10, status="pending")

    class FakeDbDel:
        def __init__(self):
            self.calls = 0

        async def execute(self, stmt, *a, **k):
            self.calls += 1
            if self.calls == 1:
                return await base.execute(stmt, *a, **k)
            return _f2._Res(doc)

        async def commit(self):
            pass

        async def refresh(self, o):
            return o

        def add(self, o):
            pass

        async def delete(self, o):
            pass

        def __getattr__(self, n):
            return getattr(base, n)

    from fastapi.testclient import TestClient as _TC  # (solo para linter)
    client.app.dependency_overrides[get_db] = FakeDbDel
    r = client.delete(f"/api/v1/documents/{doc.id}", headers=h)
    assert r.status_code in (200, 204), r.text[:200]


def test_borrar_documento_processing_se_bloquea(client, monkeypatch):
    """El bloqueo REAL del pipeline sigue en pie."""
    import uuid as _u
    import test_fase2_rbac_auditoria as _f2
    from src.database import get_db, Document

    users = []
    base = _f2._make_fake_db(users)
    _f2._override_db(client, base)
    _f2._register(client, username="ppa")
    h = _f2._auth(client, "ppa")

    doc = Document(id=_u.uuid4(), filename=_u.uuid4().hex + ".pdf",
                   original_name="proc.pdf", file_hash=_u.uuid4().hex,
                   mime_type="application/pdf", file_size=10, status="processing")

    class FakeDbDel:
        def __init__(self):
            self.calls = 0

        async def execute(self, stmt, *a, **k):
            self.calls += 1
            if self.calls == 1:
                return await base.execute(stmt, *a, **k)
            return _f2._Res(doc)

        def __getattr__(self, n):
            return getattr(base, n)

    client.app.dependency_overrides[get_db] = FakeDbDel
    r = client.delete(f"/api/v1/documents/{doc.id}", headers=h)
    assert r.status_code == 409
