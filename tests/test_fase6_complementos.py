"""PLAN-006 Fase 6 (barata): export historial + nav sin muertos (RAG-033/035)."""
from __future__ import annotations

import sys
from unittest.mock import MagicMock

try:
    import psycopg2  # noqa: F401
except ModuleNotFoundError:
    sys.modules["psycopg2"] = MagicMock()

import test_fase2_rbac_auditoria as _f2

_register = _f2._register
_auth = _f2._auth
_empty_db_override = _f2._empty_db_override


def test_exp_export_historial_txt_csv_json(client, monkeypatch):
    """GET /query/export/history devuelve el historial en los 3 formatos."""
    _empty_db_override(client, monkeypatch)
    _register(client, username="roo")
    h = _auth(client, "roo")

    # _empty_db_override ya dejó el fake de fase2 (users OK); parcheamos SOLO
    # el select de audit events del endpoint (audit rows QA):
    import src.api.v1.query as qmod
    import src.database as dbm

    real_select_q = None

    class _Row:
        created_at = "2026-10-01 12:00:00"
        detail = {"question": "q1", "answer": "a1 [1]", "sources": [1]}

    class _FR:
        def scalars(self):
            return self
        def all(self):
            return [_Row()]
        def scalar_one_or_none(self):
            return None

    class AuditProxy:
        def __init__(self, base):
            self._base = base
        async def execute(self, stmt, *a, **k):
            from src.database import AuditLog
            if "audit_log" in str(stmt).lower() and "count(" not in str(stmt).lower():
                return _FR()
            return await self._base.execute(stmt, *a, **k)
        def __getattr__(self, name):
            return getattr(self._base, name)

    db_inst = None
    from src.database import get_db

    # capturar el fake actual:
    orig_dep = client.app.dependency_overrides[get_db]

    def proxy_dep():
        return AuditProxy(orig_dep())

    client.app.dependency_overrides[get_db] = proxy_dep

    for fmt, needle in (("json", "q1"), ("txt", "Q: q1"), ("csv", "q1")):
        r = client.get(f"/api/v1/query/export/history?fmt={fmt}", headers=h)
        assert r.status_code == 200, (fmt, r.status_code, r.text[:160])
        assert needle in r.text, (fmt, r.text[:160])


def test_nav_sin_items_muertos():
    """RAG-035: los ítems 'Historial'/'Config' muertos no se muestran."""
    import pathlib

    idx = (pathlib.Path(__file__).parent.parent / "frontend" / "index.html").read_text(encoding="utf-8")
    assert "setView(this,'historial')" not in idx
    assert "setView(this,'config')" not in idx
    # el BSC sí debe estar:
    assert "setView(this,'bsc')" in idx
