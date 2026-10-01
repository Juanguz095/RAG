"""PLAN-006 — Fase 4: keywords persistentes + sección (RAG-013/015-019 — CP-004).

TDD ROJO primero. Arnés compartido con fase2 (mismos helpers).
"""
from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

import pytest

try:
    import psycopg2  # noqa: F401
except ModuleNotFoundError:
    sys.modules["psycopg2"] = MagicMock()

import test_fase2_rbac_auditoria as _f2  # arnés compartido

_make_fake_db = _f2._make_fake_db
_override_db = _f2._override_db
_register = _f2._register
_auth = _f2._auth
_empty_db_override = _f2._empty_db_override
_make_user = _f2._make_user


def _kw_db():
    """Fake DB con soporte keywords/chunk_keywords/documents/chunks/audit."""
    from src.database import AuditLog, Chunk, ChunkKeyword, Document, Keyword

    store = {"keywords": [], "chunk_keywords": [], "documents": [], "chunks": [], "audit": [], "users": []}

    class _R:
        def __init__(self, rows):
            self._rows = rows
        def scalars(self):
            return self
        def all(self):
            return self._rows
        def scalar_one_or_none(self):
            return self._rows[0] if self._rows else None
        def scalar(self):
            return self._rows[0] if self._rows else None
        def first(self):
            return self._rows[0] if self._rows else None

    class Db:
        async def execute(self, stmt, *_a, **_k):
            sql = str(stmt).lower()
            try:
                params = dict(stmt.compile().params or {})
            except Exception:
                params = {}
            if "count(" in sql:
                n = len(store["users"]) if "users" in sql else 0
                return _R([n])
            if "chunk_keywords" in sql:
                table = "chunk_keywords"
            elif "medical_synonyms" in sql:
                table = "medical_synonyms"  # vacío
            elif "keywords" in sql:
                table = "keywords"
            elif "audit_log" in sql:
                table = "audit"
            elif "documents" in sql:
                table = "documents"
            elif "chunks" in sql:
                table = "chunks"
            else:
                table = "users"
            rows = store.get(table, [])
            key = params.get("id_1") or params.get("id")
            if key is not None:
                rows = [x for x in rows if str(getattr(x, "id", "")) == str(key)]
            tmatch = params.get("term_1")
            if tmatch is not None and rows and hasattr(rows[0], "term"):
                rows = [x for x in rows if getattr(x, "term", "").lower() == str(tmatch).lower() and x.is_active]
            umatch = params.get("username_1")
            if umatch is not None and rows and hasattr(rows[0], "username"):
                rows = [x for x in rows if getattr(x, "username", "") == str(umatch)]
            ematch = params.get("email_1")
            if ematch is not None and rows and hasattr(rows[0], "email"):
                rows = [x for x in rows if getattr(x, "email", "") == str(ematch)]
            if table == "keywords" and params.get("id_1") is None:
                rows = [x for x in rows if x.is_active]
            if table == "chunk_keywords":
                rows = [
                    x for x in rows
                    if (params.get("keyword_id_1") is None
                        or str(x.keyword_id) == str(params.get("keyword_id_1")))
                    and (params.get("chunk_id_1") is None
                         or str(x.chunk_id) == str(params.get("chunk_id_1")))
                ]
            return _R(list(rows))

        def add(self, obj):
            t = type(obj).__name__
            m = {"Keyword": "keywords", "ChunkKeyword": "chunk_keywords",
                 "Document": "documents", "Chunk": "chunks", "AuditLog": "audit"}
            if t == "User" and getattr(obj, "is_active", None) is None:
                obj.is_active = True
            if t in m:
                if not getattr(obj, "id", None):
                    obj.id = __import__("uuid").uuid4()
                if t == "Keyword" and getattr(obj, "is_active", None) is None:
                    obj.is_active = True
                store[m[t]].append(obj)
            else:
                store["users"].append(obj)

        async def commit(self): pass
        async def refresh(self, obj):
            if getattr(obj, "id", None) is None:
                obj.id = __import__("uuid").uuid4()
            return obj
        async def rollback(self): pass
    return Db(), store


# ---------------------------------------------------------------- modelos

def test_kw_modelos_existen():
    from src.database import ChunkKeyword, Keyword

    assert Keyword.__tablename__ == "keywords"
    assert ChunkKeyword.__tablename__ == "chunk_keywords"


# ---------------------------------------------------------------- CRUD (RAG-016)

def test_kw_crud_requiere_roles_escritura(client, monkeypatch):
    # viewer (sin keywords:write) no puede crear:
    _empty_db_override(client, monkeypatch)
    _make_user(client, monkeypatch, "vw", "viewer")
    h = _auth(client, "vw")
    r = client.post("/api/v1/keywords", json={"term": "glaucoma"},
                    headers=h)
    assert r.status_code in (403, 404)


def test_kw_crud_crear_y_listar_y_duplicado_409(client, monkeypatch):
    db, _store = _kw_db()
    _override_db(client, db)
    _register(client, username="roo")
    h = _auth(client, "roo")
    r = client.post("/api/v1/keywords", json={"term": "glaucoma", "category": "oftalmologia"}, headers=h)
    assert r.status_code in (200, 201), r.text[:200]
    # duplicado → 409
    r2 = client.post("/api/v1/keywords", json={"term": "glaucoma"}, headers=h)
    assert r2.status_code == 409
    # listar contiene la creada:
    r3 = client.get("/api/v1/keywords", headers=h)
    assert r3.status_code == 200
    assert any(k["term"] == "glaucoma" for k in r3.json())


def test_kw_delete_logico(client, monkeypatch):
    db, _store = _kw_db()
    _override_db(client, db)
    _register(client, username="roo")
    h = _auth(client, "roo")
    r = client.post("/api/v1/keywords", json={"term": "hipertension"}, headers=h)
    kid = r.json()["id"]
    rd = client.delete(f"/api/v1/keywords/{kid}", headers=h)
    assert rd.status_code in (200, 204)
    # tras delete lógico ya no aparece en la lista:
    r3 = client.get("/api/v1/keywords", headers=h)
    assert not any(k["term"] == "hipertension" for k in r3.json())


# ---------------------------------------------------------------- import/export (RAG-015)

def test_kw_import_csv_y_export_csv(client, monkeypatch):
    db, _store = _kw_db()
    _override_db(client, db)
    _register(client, username="roo")
    h = _auth(client, "roo")
    csv_body = "term,category\nglaucoma,oftalmologia\nretina,oftalmologia\n"
    ri = client.post("/api/v1/keywords/import",
                     files={"file": ("kw.csv", csv_body.encode(), "text/csv")},
                     headers=h)
    assert ri.status_code in (200, 201), ri.text[:200]
    assert (ri.json().get("imported") or ri.json().get("count", 0)) >= 2

    re_ = client.get("/api/v1/keywords/export", headers=h)
    assert re_.status_code == 200
    assert "glaucoma" in re_.text


# ---------------------------------------------------------------- búsqueda exacta (RAG-017 — CP-004)

def test_kw_busqueda_exacta_devuelve_doc_pagina_fragmento(client, monkeypatch):
    """CP-004 literal: muestra documento, ubicación y fragmento correcto."""
    db, store = _kw_db()
    _override_db(client, db)
    _register(client, username="roo")
    h = _auth(client, "roo")
    # seed directo del vocabulario: keyword 'glaucoma' + chunk que la contiene
    from src.database import Chunk, ChunkKeyword, Document, Keyword
    import uuid as _u

    import uuid as _u
    doc = Document(id=_u.uuid4(), filename="segundo.pdf", file_hash="kwhash1",
                   status="ready", page_count=5, visibility="public")
    # (fake add al final)
    ch = Chunk(id=_u.uuid4(), document_id=doc.id, chunk_index=0, page_numbers=[3],
               content="El glaucoma se trata con gotas de timolol.", chunk_metadata={})
    kw = Keyword(id=_u.uuid4(), term="glaucoma", is_active=True)
    ck = ChunkKeyword(chunk_id=ch.id, keyword_id=kw.id, match_count=1)
    for x in (doc, ch, kw, ck):
        db.add(x)

    r = client.get("/api/v1/keywords/search?term=glaucoma", headers=h)
    assert r.status_code == 200, r.text[:200]
    hits = r.json()
    assert len(hits) >= 1
    h0 = hits[0]
    assert h0["document"] == "segundo.pdf"
    assert 3 in (h0["pages"] or [])
    assert "glaucoma" in (h0["fragment"] or "")


# ---------------------------------------------------------------- sección persistida (RAG-019)

def test_kw_sources_incluyen_seccion_y_fragmento(client, monkeypatch):
    import uuid as _u
    _empty_db_override(client, monkeypatch)
    _register(client, username="roo")
    h = _auth(client, "roo")

    class _R:
        score = 0.9
        relevance = 90
        chunk_id = _u.uuid4()
        document_id = _u.uuid4()
        content = "Diagnostico diferencial del glaucoma."
        page_numbers = [2]
        chunk_index = 0
        chunk_metadata = {"section": "Diagnóstico", "fragment": "…Diagnostico diferencial del glaucoma…"}
        document_name = "segundo.pdf"
        matched_terms = ["glaucoma"]
        vector_score = 0.9
        keyword_score = 0.5
        rrf_score = 0.7
        chunk = None

    async def fake_search(*_a, **_k):
        return [_R()]

    monkeypatch.setattr("src.api.v1.query.hybrid_search", fake_search)
    import src.services.reranker as rr
    monkeypatch.setattr(rr, "rerank", lambda q, res, top_k=10: res[:top_k], raising=False)
    monkeypatch.setattr(
        "src.api.v1.query.generate_answer_timed",
        lambda ctx, q, **k: {"text": "Respuesta con cita [1].",
                             "prompt_eval_ms": 1, "generation_ms": 1, "tokens_generated": 3},
    )
    r = client.post("/api/v1/query", json={"query": "glaucoma"}, headers=h)
    d = r.json()
    assert r.status_code == 200
    s0 = d["sources"][0]
    assert s0.get("section") == "Diagnóstico", s0
    assert "fragment" in s0 and s0["fragment"]


# ---------------------------------------------------------------- pipeline ingesta (RAG-013)

def test_pipeline_pobla_chunk_keywords_y_seccion(client, monkeypatch):
    """La etapa keywords del pipeline persiste matches y sección en chunk_metadata."""
    from src.services import chunking

    # preparar chunk con término de catálogo y encabezado:
    text = "TRATAMIENTO\nEl glaucoma responde a hipotensores topicos.\n"
    chunks = chunking.chunk_text(text, [1], 0)
    assert chunks
    from src.services.keywords import extract_keyword_matches, enrich_section

    en = enrich_section(chunks[0])
    assert en.get("section"), en
    # catálogo en memoria: {term: [variantes]}
    counts = extract_keyword_matches(chunks[0], {"glaucoma": ["glaucoma"]})
    assert "glaucoma" in counts


def test_reprocess_endpoint_existe(client, monkeypatch):
    """RAG-013: POST /documents/{id}/reprocess encolará de nuevo (idempotente)."""
    db, store = _kw_db()
    _override_db(client, db)
    import uuid as _u
    from src.database import Document

    doc = Document(id=_u.uuid4(), filename="a.pdf", file_hash="rephash9",
                   status="ready", page_count=2, visibility="public")
    db.add(doc)
    from src.api.v1.documents import UPLOAD_DIR
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    (UPLOAD_DIR / f"{doc.file_hash}.pdf").write_bytes(b"stub")
    _register(client, username="roo")
    h = _auth(client, "roo")

    called = {}
    import src.workers.ingestion_tasks as ing
    monkeypatch.setattr(ing, "enqueue_process_document",
                        lambda did, path: called.setdefault("id", did), raising=False)
    r = client.post(f"/api/v1/documents/{doc.id}/reprocess", headers=h)
    assert r.status_code in (200, 202), r.text[:200]
    assert called.get("id") == str(doc.id)
    try:
        (UPLOAD_DIR / f"{doc.file_hash}.pdf").unlink(missing_ok=True)
    except Exception:
        pass
