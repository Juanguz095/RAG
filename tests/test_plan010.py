"""PLAN-010 — WP1 (velocidad + tiempos visibles) y WP6 (exportar bitácora).

Tests autocontenidos (sin infraestructura externa), siguiendo el patrón de
fakes de test_fase2_rbac_auditoria.py.
"""
from __future__ import annotations

import sys
from datetime import datetime
from unittest.mock import MagicMock

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
    assert resp.status_code == 200, resp.text
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
        self.resource_type = "document"
        self.resource_id = "r1"
        self.detail = {"filename": "a.pdf"}
        self.ip = "1.2.3.4"
        self.created_at = datetime(2026, 9, 30)


def _make_fake_db(users, all_rows=None):
    def match(params):
        for key, attr in (("id_1", "id"), ("username_1", "username"), ("email_1", "email"),
                          ("id", "id"), ("username", "username"), ("email", "email")):
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
            if "count(" in sql:
                return _Res(len(users))
            r = _Res(match(params))
            if "audit_log" in sql:
                r._rows = list(all_rows if all_rows is not None else [_FakeAuditRow()])
            else:
                r._rows = []
            return r

        def add(self, obj):
            if type(obj).__name__ == "AuditLog":
                return
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


def _empty_db(client, all_rows=None):
    users = []
    _override_db(client, _make_fake_db(users, all_rows))
    return users


def _make_auditor(client):
    _empty_db(client)
    _register(client, username="root")
    hdrs_admin = _auth(client, "root")
    client.post("/api/v1/users",
                json={"username": "aud", "email": "aud@x.y",
                      "password": "T3st-clav3-larga", "role": "auditor"},
                headers=hdrs_admin)
    return _auth(client, "aud")


# ── WP6: exportar la bitácora ──────────────────────────────────────────────


def test_wp6_export_bitacora_csv(client):
    hdrs = _make_auditor(client)
    resp = client.get("/api/v1/audit/export?format=csv", headers=hdrs)
    assert resp.status_code == 200, resp.text
    assert resp.headers["content-type"].startswith("text/csv")
    body = resp.text
    assert "fecha,usuario,accion" in body


def test_wp6_export_bitacora_json(client):
    hdrs = _make_auditor(client)
    resp = client.get("/api/v1/audit/export?format=json", headers=hdrs)
    assert resp.status_code == 200, resp.text
    assert resp.headers["content-type"].startswith("application/json")
    data = resp.json()
    assert isinstance(data, list)
    assert data and data[0]["accion"] in ("login", "upload", "query", "delete")


def test_wp6_export_bitacora_con_filtro(client):
    hdrs = _make_auditor(client)
    resp = client.get("/api/v1/audit/export?format=csv&action=login&username=root", headers=hdrs)
    assert resp.status_code == 200, resp.text
    assert "login" in resp.text


def test_wp6_viewer_no_exporta_bitacora(client):
    _empty_db(client)
    _register(client, username="root")
    hdrs_admin = _auth(client, "root")
    client.post("/api/v1/users",
                json={"username": "vw", "email": "vw@x.y",
                      "password": "T3st-clav3-larga", "role": "viewer"},
                headers=hdrs_admin)
    hdrs = _auth(client, "vw")
    resp = client.get("/api/v1/audit/export?format=csv", headers=hdrs)
    assert resp.status_code == 403


def test_wp6_export_solo_lectura_no_rompe_router(client):
    import src.api.v1.audit as audit_router_mod

    for r in audit_router_mod.router.routes:
        if "/audit" in getattr(r, "path", ""):
            assert getattr(r, "methods", {"GET"}) == {"GET"}


# ── WP1: timings visibles ──────────────────────────────────────────────────


def test_wp1_doc_item_expone_timings():
    from src.api.v1.documents import _doc_item

    class _D:
        id = "00000000-0000-0000-0000-000000000001"
        filename = "x.pdf"
        original_name = "x.pdf"
        file_size = 100
        page_count = 2
        status = "completed"
        total_chunks = 5
        created_at = datetime(2026, 9, 30)
        processed_at = datetime(2026, 9, 30)
        metadata_ = {
            "timings": {"ocr_s": 10.5, "embed_s": 3.2, "total_s": 15.0},
            "word_boxes": {"0": []},
        }

    item = _doc_item(_D())
    assert item.timings == {"ocr_s": 10.5, "embed_s": 3.2, "total_s": 15.0}
    # list view (include_meta=False) no expone word_boxes ni timings duplicados
    assert item.metadata_ is None


def test_wp1_doc_item_sin_timings_devuelve_none():
    from src.api.v1.documents import _doc_item

    class _D:
        id = "00000000-0000-0000-0000-000000000002"
        filename = "y.pdf"
        original_name = "y.pdf"
        file_size = 0
        page_count = None
        status = "pending"
        total_chunks = 0
        created_at = datetime(2026, 9, 30)
        processed_at = None
        metadata_ = {}

    item = _doc_item(_D())
    assert item.timings is None


def test_wp1_ocr_dpi_resuelve_desde_config():
    from src.services import ocr

    assert ocr.OCR_DPI >= 100, "el DPI debe venir de config (150), no hardcodear 200"


# ── WP8: edad desde fecha de nacimiento (antes de anonimizar) ───────────────


def test_wp8_extract_birth_date_y_compute_age():
    from src.services.anonymizer import compute_age, extract_birth_date

    assert extract_birth_date("Paciente nacido: 14/03/1985, ingresa por dolor") is not None
    assert extract_birth_date("sin datos de nacimiento aqui") is None
    age = compute_age("nacido: 14/03/1985")
    assert age is not None and 0 < age < 120


def test_wp8_compute_age_anio_2_digitos():
    from src.services.anonymizer import compute_age

    # año 2 dígitos 85 → 1985 (heurística)
    assert compute_age("nacido el 14/03/85") is not None


def test_wp8_compute_age_fecha_invalida_devuelve_none():
    from src.services.anonymizer import compute_age

    assert compute_age("fecha rota 99/99/9999") is None
    assert compute_age("no hay fecha") is None


def test_wp8_streaming_embedder_extrae_edad_y_anonimiza():
    from src.workers.ingestion_tasks import _StreamingEmbedder
    from src.services.anonymizer import compute_age

    e = _StreamingEmbedder()
    try:
        e.add_page(0, "Paciente nacido: 14/03/1985 ingresa por dolor toracico.")
        assert e._edad is not None
        assert e._edad == compute_age("14/03/1985")
        # la fecha queda anonimizada (RAG-038 intacto)
        assert "14/03/1985" not in e._current_text
        assert "[FECHA_NACIMIENTO]" in e._current_text
    finally:
        e._pool.shutdown(wait=True)


def test_wp8_build_context_inyecta_edad():
    from types import SimpleNamespace

    from src.services.context import build_context

    r = SimpleNamespace(
        document_name="hc.pdf",
        page_numbers=[1],
        content="diagnostico de neumonia",
        chunk_metadata={"edad": 40},
    )
    ctx = build_context([r])
    assert "Edad: 40 anios" in ctx

    r2 = SimpleNamespace(
        document_name="hc2.pdf",
        page_numbers=[1],
        content="sin edad",
        chunk_metadata={},
    )
    ctx2 = build_context([r2])
    assert "Edad:" not in ctx2


# ── WP11: métricas de uso por usuario (BSC) ────────────────────────────────


def test_wp11_compute_usage_stats_desglosa_por_usuario():
    from types import SimpleNamespace

    from src.services.bsc import compute_usage_stats

    rows = [
        SimpleNamespace(username="admin", action="query", detail={"sources": [1]}),
        SimpleNamespace(username="admin", action="query", detail={"sources": []}),
        SimpleNamespace(username="medico", action="query", detail={"sources": [1]}),
        SimpleNamespace(username="admin", action="upload", detail=None),
        SimpleNamespace(username="medico", action="upload", detail=None),
    ]
    s = compute_usage_stats(rows)
    assert s["active_users"] == 2
    assert s["total_queries"] == 3
    assert s["total_uploads"] == 2
    assert s["queries_per_user"] == {"admin": 2, "medico": 1}
    assert s["docs_per_user"] == {"admin": 1, "medico": 1}
    assert s["queries_per_user_avg"] == 1.5


def test_wp11_compute_usage_stats_sin_eventos():
    from src.services.bsc import compute_usage_stats

    s = compute_usage_stats([])
    assert s["active_users"] == 0
    assert s["total_queries"] == 0
    assert s["queries_per_user_avg"] == 0.0


# ── WP9: filtros por metadatos en la búsqueda ──────────────────────────────


def test_wp9_search_request_acepta_filtros_metadata():
    from src.schemas.query import SearchRequest

    r = SearchRequest(query="dolor toracico", age_min=30, age_max=50, domain="cardiologia")
    assert r.age_min == 30
    assert r.age_max == 50
    assert r.domain == "cardiologia"

    r2 = SearchRequest(query="x")
    assert r2.age_min is None and r2.age_max is None and r2.domain is None


def test_wp9_search_request_valida_rango_edad():
    import pytest
    from pydantic import ValidationError

    from src.schemas.query import SearchRequest

    with pytest.raises(ValidationError):
        SearchRequest(query="x", age_min=-1)
    with pytest.raises(ValidationError):
        SearchRequest(query="x", age_max=200)


# ── WP3 (PLAN-009 Fase A): rotación robusta sin Paddle ─────────────────────


def _fake_pytesseract_osd_falla(monkeypatch):
    import sys
    import types

    fake = types.ModuleType("pytesseract")

    def _raise(*_a, **_k):
        raise RuntimeError("OSD no disponible")

    fake.image_to_osd = _raise
    fake.Output = types.SimpleNamespace(DICT="DICT")
    monkeypatch.setitem(sys.modules, "pytesseract", fake)


def test_wp3_rotation_fallback_elige_180(monkeypatch):
    from PIL import Image

    from src.services import ocr

    _fake_pytesseract_osd_falla(monkeypatch)
    calls = {"n": 0}

    def fake_score(_img):
        calls["n"] += 1
        return 10.0 if calls["n"] == 1 else 60.0  # 0° malo, 180° bueno

    monkeypatch.setattr(ocr, "_quick_ocr_score", fake_score)

    img = Image.new("RGB", (200, 100), "white")
    angle, fixed = ocr.detect_and_fix_rotation(img)
    assert angle == 180
    assert fixed.size == (200, 100)
    assert calls["n"] == 2  # comparó ambas orientaciones


def test_wp3_rotation_fallback_conserva_0_si_no_mejora(monkeypatch):
    from PIL import Image

    from src.services import ocr

    _fake_pytesseract_osd_falla(monkeypatch)
    monkeypatch.setattr(ocr, "_quick_ocr_score", lambda _img: 50.0)

    img = Image.new("RGB", (200, 100), "white")
    angle, fixed = ocr.detect_and_fix_rotation(img)
    assert angle == 0
    assert fixed.size == (200, 100)


def test_wp3_quick_ocr_score_es_float_sin_tesseract():
    from PIL import Image

    from src.services import ocr

    s = ocr._quick_ocr_score(Image.new("RGB", (50, 50), "white"))
    assert isinstance(s, float)
