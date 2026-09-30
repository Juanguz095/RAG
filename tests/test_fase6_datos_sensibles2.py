"""PLAN-006 — Fase 3 (continuación): enmascaramiento por rol en respuestas.

CP-008 literal: "Consultar con usuario sin autorización → el dato se
oculta/restringe según política". Con el anonimizador de ingesta los datos
nunca llegan al índice; aquí se garantiza además que respuestas del LLM
no re-introduzcan patrones de datos sensibles y se registra el evento.
"""
from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest

if sys.modules.get("psycopg2") is None:
    try:
        import psycopg2  # noqa: F401
    except ModuleNotFoundError:
        sys.modules["psycopg2"] = MagicMock()

import test_fase2_rbac_auditoria as _f2  # arnés compartido

_register = _f2._register
_auth = _f2._auth
_empty_db_override = _f2._empty_db_override


def test_m_respuesta_enmascara_datos_sensibles_residuales(client, monkeypatch):
    """Si el LLM "alucina" un DNI/teléfono, la respuesta saldrá enmascarada."""
    _empty_db_override(client, monkeypatch)
    _register(client, username="root")
    hdrs = _auth(client, "root")

    class _R:
        score = 0.9
        relevance = 90
        chunk_id = _uuid.uuid4()
        document_id = _uuid.uuid4()
        content = "contenido clinico"
        page_numbers = [1]
        chunk_index = 0
        chunk_metadata = {}
        document_name = "x.pdf"
        matched_terms = ["x"]
        vector_score = 0.9
        keyword_score = 0.4
        rrf_score = 0.6
        chunk = None

    async def fake_search(*_a, **_k):
        return [_R()]

    monkeypatch.setattr("src.api.v1.query.hybrid_search", fake_search)
    import src.services.reranker as rr
    monkeypatch.setattr(rr, "rerank", lambda q, res, top_k=10: res[:top_k], raising=False)
    monkeypatch.setattr(
        "src.api.v1.query.generate_answer_timed",
        lambda ctx, q, **k: {
            "text": "El paciente con DNI 12345678A debe seguir el tratamiento [1].",
            "prompt_eval_ms": 1, "generation_ms": 1, "tokens_generated": 5,
        },
    )
    resp = client.post("/api/v1/query", json={"query": "tratamiento"}, headers=hdrs)
    d = resp.json()
    assert resp.status_code == 200
    assert "12345678A" not in d["answer"], d["answer"]
    assert "[DNI]" in d["answer"]


def test_ingesta_anonimiza_antes_de_chunkiar(monkeypatch):
    """_StreamingEmbedder.add_page llama anonymize_text (ingesta limpia)."""
    captured = {}

    class FakeEmbedder:
        pass

    from src.workers import ingestion_tasks as it

    # parchear el chunker real para capturar el texto que le llega:
    def fake_chunk(text, pages, idx):
        captured["text"] = text
        return []

    monkeypatch.setattr(it, "_CHUNK_FN_ORIG", fake_chunk, raising=False)
    e = it._StreamingEmbedder()
    e._chunk_text = fake_chunk
    e._count = lambda t: len(t.split())
    e.add_page(1, "Paciente DNI 12345678A con glaucoma.")
    e._flush()
    assert "12345678A" not in captured["text"]
    assert "[DNI]" in captured["text"]


import uuid as _uuid  # noqa: E402  (al final para no ensuciar el arnés)
