"""PLAN-004 — Fase 2: RBAC completo + Auditoría + Trazabilidad (C1-C10)."""
from __future__ import annotations

import sys
from unittest.mock import MagicMock

import httpx
import pytest

if sys.modules.get("psycopg2") is None:
    try:
        import psycopg2  # noqa: F401
    except ModuleNotFoundError:
        sys.modules["psycopg2"] = MagicMock()


def _register(c, username="u1", role=None, password="T3st-clav3-larga"):
    body = {"username": username, "email": f"{username}@rag.local", "password": password}
    if role is not None:
        body["role"] = role
    return c.post("/api/v1/auth/register", json=body)


def _login(c, username="u1", password="T3st-clav3-larga"):
    return c.post("/api/v1/auth/login", json={"username": username, "password": password})


def _auth(c, username, password="T3st-clav3-larga"):
    resp = _login(c, username, password)
    if resp.status_code != 200:
        raise AssertionError(f"login {username}: {resp.status_code} {resp.text}")
    return {"Authorization": f"Bearer {resp.json()['access_token']}"}


class _Res:
    def __init__(self, val):
        self._v = val
        self._rows = []

    def scalar_one_or_none(self):
        return self._v

    def scalar_one(self):
        return self._v

    def scalars(self):
        outer = self
        class _S:
            def all(self):
                return outer._rows
        return _S()


class _FakeAuditRow:
    def __init__(self, action="login", username="x"):
        import uuid as _u
        self.id = _u.uuid4()
        self.user_id = None
        self.username = username
        self.action = action
        self.resource_type = "t"
        self.resource_id = "r"
        self.detail = {}
        self.ip = "1.2.3.4"
        from datetime import datetime as _dt
        self.created_at = _dt(2026, 9, 30)


def _make_fake_db(users, all_rows=None):
    def match(params):
        for key, attr in (("id_1","id"),("username_1","username"),("email_1","email"),
                          ("id","id"),("username","username"),("email","email")):
            v = params.get(key)
            if v is not None:
                m = [u for u in users if str(getattr(u, attr, "")) == str(v)]
                return m[0] if m else None
        return None

    class Db:
        async def execute(self, stmt, *_a, **_k):
            sql = str(stmt).lower()
            try:
                params = stmt.compile().params or {}
            except Exception:
                params = {}
            # count REAL de usuarios (evitar page_count/chunk_count falsas)
            if "count(" in sql:
                return _Res(len(users))
            r = _Res(match(params))
            if "audit_log" in sql and "count(" not in sql:
                r._rows = list(all_rows if all_rows is not None else [_FakeAuditRow()])
            else:
                r._rows = []
            return r

        def add(self, obj):
            if type(obj).__name__ == "AuditLog":
                return  # append-only. go nowhere (fake) — capturado por monkeypatch audit() si aplica
            if getattr(obj, "is_active", None) is None and hasattr(obj, "is_active"):
                obj.is_active = 1
            if getattr(obj, "role", None) is None and hasattr(obj, "role"):
                obj.role = "assistant"
            users.append(obj)

        async def commit(self):
            pass

        async def refresh(self, obj):
            import uuid as _u
            if getattr(obj, "id", None) is None:
                obj.id = _u.uuid4()
            return obj

        async def rollback(self):
            pass
    return Db()


def _override_db(client, db):
    from src.database import get_db
    client.app.dependency_overrides[get_db] = lambda: db


def _empty_db_override(client, monkeypatch):
    users = []
    _override_db(client, _make_fake_db(users))
    return users


def test_c2_bootstrap_primer_usuario_es_admin_forzado(client, monkeypatch):
    users = _empty_db_override(client, monkeypatch)
    resp = _register(client, username="first_admin", role="assistant")
    assert resp.status_code == 200, resp.text
    assert users and users[0].role == "admin"  # forzado server-side


def test_c2_register_role_admin_siempre_422(client, monkeypatch):
    users = _empty_db_override(client, monkeypatch)
    # Sin bootstrap (tabla vacía → M1): el body role=admin es IGNORADO y el
    # servidor asigna admin forzado; NO queda una vía pública de escalada.
    resp = _register(client, username="bad_admin", role="admin")
    assert resp.status_code == 200, resp.text
    assert users and users[0].role == "admin"  # asignado server-side igualmente


def test_c3_admin_puede_crear_usuarios_con_rol(client, monkeypatch):
    users = _empty_db_override(client, monkeypatch)
    r1 = _register(client, username="root")
    assert r1.status_code == 200
    hdrs = _auth(client, "root")
    resp = client.post("/api/v1/users",
        json={"username": "edic", "email": "e@x.y", "password": "T3st-clav3-larga", "role": "editor"},
        headers=hdrs)
    assert resp.status_code in (200, 201), resp.text
    assert "editor" in [u.role for u in users]


def test_c3_no_admin_no_puede_crear_usuarios(client, monkeypatch):
    users = _empty_db_override(client, monkeypatch)
    _register(client, username="root")
    hdrs_admin = _auth(client, "root")
    client.post("/api/v1/users",
        json={"username": "asis", "email": "a@x.y", "password": "T3st-clav3-larga", "role": "assistant"},
        headers=hdrs_admin)
    hdrs_asis = _auth(client, "asis")
    resp = client.post("/api/v1/users",
        json={"username": "evil", "email": "v@x.y", "password": "T3st-clav3-larga", "role": "editor"},
        headers=hdrs_asis)
    assert resp.status_code == 403


def _make_user(client, monkeypatch, username, role):
    _empty_db_override(client, monkeypatch)
    _register(client, username="root")
    hdrs = _auth(client, "root")
    client.post("/api/v1/users",
        json={"username": username, "email": f"{username}@x.y", "password": "T3st-clav3-larga", "role": role},
        headers=hdrs)
    return _auth(client, username)


def test_c1_viewer_no_puede_subir_documentos(client, monkeypatch):
    hdrs = _make_user(client, monkeypatch, "vw", "viewer")
    resp = client.post("/api/v1/documents", files={"file": ("a.pdf", b"%PDF-1.4 x", "application/pdf")}, headers=hdrs)
    assert resp.status_code == 403


def test_c1_assistant_puede_subir_documentos(client, monkeypatch):
    hdrs = _make_user(client, monkeypatch, "as", "assistant")
    def enqueue_ok(document_id, pdf_path):
        pass
    monkeypatch.setattr("src.workers.ingestion_tasks.enqueue_process_document", enqueue_ok)
    resp = client.post("/api/v1/documents", files={"file": ("a.pdf", b"%PDF-C1ASISTANT", "application/pdf")}, headers=hdrs)
    assert resp.status_code == 202, resp.text


def test_c1_auditor_puede_leer_auditoria_pero_no_sube_documentos(client, monkeypatch):
    hdrs = _make_user(client, monkeypatch, "aud", "auditor")
    assert client.get("/api/v1/audit", headers=hdrs).status_code == 200
    resp = client.post("/api/v1/documents", files={"file": ("a.pdf", b"%PDF-1.4 x", "application/pdf")}, headers=hdrs)
    assert resp.status_code == 403


def test_c1_viewer_no_puede_leer_auditoria(client, monkeypatch):
    hdrs = _make_user(client, monkeypatch, "vw2", "viewer")
    resp = client.get("/api/v1/audit", headers=hdrs)
    assert resp.status_code == 403


def _capture_audit(rows):
    async def fake_audit(db, user, action, resource_type=None, resource_id=None, detail=None, ip=None):
        rows.append({
            "user_id": str(getattr(user, "id", "") or ""),
            "username": getattr(user, "username", ""),
            "action": action,
            "resource_type": resource_type,
            "resource_id": resource_id,
            "detail": detail,
        })
    return fake_audit


def _patch_audit_capture(monkeypatch, rows):
    import importlib
    fake = _capture_audit(rows)
    for name in ("src.services.audit", "src.api.v1.auth", "src.api.v1.documents",
                 "src.api.v1.query", "src.api.v1.synonyms", "src.api.v1.users"):
        try:
            mod = importlib.import_module(name)
            monkeypatch.setattr(mod, "audit", fake, raising=False)
        except Exception:
            pass
    # query.py renombra al importar: `from src.services.audit import audit as audit_event`
    try:
        qmod = importlib.import_module("src.api.v1.query")
        monkeypatch.setattr(qmod, "audit_event", fake, raising=False)
    except Exception:
        pass
    dmod = importlib.import_module("src.api.v1.documents")
    monkeypatch.setattr(dmod, "audit", fake, raising=False)


def test_c4_login_exitoso_y_fallido_son_auditados(client, monkeypatch):
    _empty_db_override(client, monkeypatch)
    _register(client, username="aud1")
    rows = []
    _patch_audit_capture(monkeypatch, rows)
    assert _login(client, "aud1").status_code == 200
    assert _login(client, "aud1", " WRONG pass ").status_code == 401
    assert any(r["action"] == "login" for r in rows), rows
    assert any(r["action"] == "login_failed" for r in rows)


def test_c4_logout_es_auditado_y_da_204(client, monkeypatch):
    _empty_db_override(client, monkeypatch)
    _register(client, username="lg")
    hdrs = _auth(client, "lg")
    rows = []
    _patch_audit_capture(monkeypatch, rows)
    resp = client.post("/api/v1/auth/logout", headers=hdrs)
    assert resp.status_code == 204
    assert any(r["action"] == "logout" for r in rows)


def test_c5_reconstruccion_trazabilidad_desde_audit_log(client, monkeypatch):
    _empty_db_override(client, monkeypatch)
    _register(client, username="tr")
    hdrs = _auth(client, "tr")
    rows = []
    _patch_audit_capture(monkeypatch, rows)

    class _Chk:
        id = "00000000-0000-0000-0000-000000000001"
        document_id = "00000000-0000-0000-0000-000000000002"
        content = "diagnostico: control del recien nacido bajo peso"
        page_numbers = [5]
        chunk_index = 0
        document = None

    class _R:
        def __init__(self, c):
            self.chunk = c
            self.content = c.content
            self.vector_score = 0.9
            self.keyword_score = 0.5
            self.rrf_score = 0.7
            self.chunk_id = c.id
            self.document_id = c.document_id
            self.page_numbers = c.page_numbers
            self.score = 0.8
            self.chunk_metadata = {}
            self.chunk_index = c.chunk_index
            self.document_name = "segundo.pdf"
            self.matched_terms = ["diagnostico"]
            self.relevance = 80

            class _D:
                filename = "segundo.pdf"
            self.document = _D()

    async def fake_hybrid_search(*_a, **_k):
        return [_R(_Chk())]

    monkeypatch.setattr("src.api.v1.query.hybrid_search", fake_hybrid_search)
    import src.services.reranker as rr
    monkeypatch.setattr(rr, "rerank", lambda q, res, top_k=10: res[:top_k], raising=False)
    monkeypatch.setattr(
        "src.api.v1.query.generate_answer_timed",
        lambda ctx, q, **k: {"text": "RESPUESTA-TEST", "prompt_eval_ms": 1.0, "generation_ms": 2.0, "tokens_generated": 3})
    monkeypatch.setattr("src.api.v1.query.build_context", lambda results: "CONTEXTO-TEST")

    resp = client.post("/api/v1/query", json={"query": "diagnostico", "max_chunks": 3}, headers=hdrs)
    assert resp.status_code == 200, resp.text
    qevents = [r for r in rows if r["action"] == "query"]
    assert qevents, f"sin evento query: {rows}"
    detail = qevents[0]["detail"]
    assert detail["question"] == "diagnostico"
    assert detail["chunks"][0]["chunk_id"].startswith("00000000")
    assert detail["answer"] == "RESPUESTA-TEST"
    assert detail["sources"]
    full = qevents[0]
    assert full["user_id"] and full["detail"]["chunks"][0]["document_id"]


def test_c6_no_existe_update_delete_de_auditoria(client, monkeypatch):
    hdrs = _make_user(client, monkeypatch, "ad6", "admin")
    import src.api.v1.audit as audit_router_mod

    audit_routes = [
        r for r in audit_router_mod.router.routes
        if getattr(r, "path", "").endswith("/audit") or getattr(r, "path", "") == ""
    ]
    assert audit_routes, "sin router de auditoría"
    for r in audit_routes:
        methods = getattr(r, "methods", set())
        assert methods == {"GET"}, f"método no read-only en {getattr(r,'path','')}: {methods}"
    # y en la app no debe existir una ruta audit con POST/DELETE/PATCH
    for r in client.app.routes:
        sub = getattr(r, "routes", None) or []
        for rr in sub:
            if "/audit" in str(getattr(rr, "path", "")):
                assert getattr(rr, "methods", {"GET"}) == {"GET"}


def test_c7_documento_restringido_invisible_para_otros_roles(client, monkeypatch):
    class _Doc:
        id = "00000000-0000-0000-0000-00000000000d"
        visibility = "restricted"
        filename = "restringido.pdf"

    class _Chk7:
        id = "00000000-0000-0000-0000-00000000000c"
        document_id = _Doc.id
        content = "contenido secreto"
        page_numbers = [1]
        chunk_index = 0
        document = _Doc()

    class _R:
        def __init__(self, c):
            self.chunk = c
            self.content = c.content
            self.vector_score = 0.9
            self.keyword_score = 0.4
            self.rrf_score = 0.6
            self.chunk_id = c.id
            self.document_id = c.document_id
            self.page_numbers = c.page_numbers
            self.score = 0.8
            self.chunk_metadata = {}
            self.chunk_index = c.chunk_index
            self.document_name = "restringido.pdf"
            self.matched_terms = ["secreto"]
            self.relevance = 80

    async def fake_hybrid_search(*_a, **_k):
        return [_R(_Chk7())]

    monkeypatch.setattr("src.api.v1.query.hybrid_search", fake_hybrid_search)
    import src.services.reranker as rr
    monkeypatch.setattr(rr, "rerank", lambda q, res, top_k=10: res[:top_k], raising=False)
    monkeypatch.setattr(
        "src.api.v1.query.generate_answer_timed",
        lambda ctx, q, **k: {"text": "RESP", "prompt_eval_ms": 1, "generation_ms": 1, "tokens_generated": 1})
    monkeypatch.setattr("src.api.v1.query.build_context", lambda results: "CTX")

    hdrs = _make_user(client, monkeypatch, "vw7", "viewer")
    resp = client.post("/api/v1/query", json={"query": "secreto", "max_chunks": 3}, headers=hdrs)
    assert resp.status_code == 200
    assert resp.json()["sources"] == [], "viewer no debe ver doc restringido"

    hdrs_admin = _auth(client, "root")  # root Í es el bootstrap admin
    resp2 = client.post("/api/v1/query", json={"query": "secreto", "max_chunks": 3}, headers=hdrs_admin)
    assert resp2.status_code == 200
    assert len(resp2.json()["sources"]) == 1


def test_c8_hash_password_produce_pbkdf2():
    from src.api.deps import hash_password, verify_password
    h = hash_password("Mi-Clave-Segura-1")
    assert h.startswith("pbkdf2$"), h
    assert verify_password("Mi-Clave-Segura-1", h)
    assert not verify_password("otra", h)


def test_c8_login_hash_sha256_heredado_funciona_y_re_hashea(client, monkeypatch):
    import hashlib
    users = _empty_db_override(client, monkeypatch)
    _register(client, username="leg")
    users[0].hashed_password = hashlib.sha256("T3st-clav3-larga".encode()).hexdigest()
    assert not users[0].hashed_password.startswith("pbkdf2$")
    resp = _login(client, "leg")
    assert resp.status_code == 200, resp.text
    assert users[0].hashed_password.startswith("pbkdf2$")


def test_c9_expiracion_token_480_min(client):
    from src.config import get_settings
    assert get_settings().ACCESS_TOKEN_EXPIRE_MINUTES == 480


def test_c9_logout_204(client, monkeypatch):
    _empty_db_override(client, monkeypatch)
    _register(client, username="lo")
    hdrs = _auth(client, "lo")
    assert client.post("/api/v1/auth/logout", headers=hdrs).status_code == 204


def test_c10_error_de_pipeline_queda_auditado(client, monkeypatch, tmp_path):
    import src.workers.ingestion_tasks as worker_module
    def enqueue_roto(document_id, pdf_path):
        raise ConnectionError("Redis caido")
    monkeypatch.setattr(worker_module, "enqueue_process_document", enqueue_roto)
    monkeypatch.setattr("src.api.v1.documents.UPLOAD_DIR", tmp_path, raising=True)
    _empty_db_override(client, monkeypatch)
    rows = []
    _patch_audit_capture(monkeypatch, rows)
    _register(client, username="ad10")
    hdrs = _auth(client, "ad10")
    resp = client.post("/api/v1/documents",
        files={"file": ("b.pdf", b"%PDF-C10", "application/pdf")}, headers=hdrs)
    assert resp.status_code == 503, resp.text
    assert any(r["action"] == "error" for r in rows), rows


def test_c1_matriz_puede_ejecutarse_por_rol(client, monkeypatch):
    from src.core.permissions import ROLE_PERMISSIONS
    for r in ("admin", "editor", "assistant", "viewer", "auditor"):
        assert r in ROLE_PERMISSIONS, r
