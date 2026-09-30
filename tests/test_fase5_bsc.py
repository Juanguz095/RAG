"""PLAN-005 — Fase 3: BSC (RAG Intelligence BSC) — contratos B1-B10.

TDD: ROJO primero (convención del repo). Arnés compartido con test_fase2
(gate ON, BD fake por override). Los cómputos del servicio (`src.services.bsc`)
se prueban con listas en memoria; los endpoints sobre un cliente con BD override.
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

import test_fase2_rbac_auditoria as _f2  # arnés compartido (mismo dir)

_make_fake_db = _f2._make_fake_db
_override_db = _f2._override_db
_register = _f2._register
_auth = _f2._auth


# ---------------------------------------------------------------- modelos fake

class _Kpi:
    def __init__(self, code, name="KPI", perspective="conocimiento", direction="gte",
                 target=None, thresholds=None, query_type="manual", unit="%"):
        self.id = uuid.uuid4()
        self.code = code
        self.name = name
        self.perspective = perspective
        self.direction = direction
        self.target = target
        self.thresholds = thresholds or {"amber": 85.0, "red": 70.0}
        self.query_type = query_type
        self.unit = unit
        self.is_active = True


class _Snap:
    def __init__(self, kpi_id, value):
        self.id = uuid.uuid4()
        self.kpi_id = kpi_id
        self.value = value
        self.period_start = datetime(2026, 9, 1)
        self.period_end = datetime(2026, 9, 30)
        self.computed_at = datetime(2026, 9, 30, 12, 0, 0)


class _Alert:
    def __init__(self, kpi_id, severity, message):
        self.id = uuid.uuid4()
        self.kpi_id = kpi_id
        self.severity = severity
        self.message = message
        self.status = "open"
        self.created_at = datetime(2026, 9, 30, 12, 0, 0)


class _Plan:
    def __init__(self, title, kpi_id=None, description=None, owner=None,
                 due_date=None, status="open"):
        self.id = uuid.uuid4()
        self.title = title
        self.kpi_id = kpi_id
        self.description = description
        self.owner = owner
        self.due_date = due_date
        self.status = status
        self.created_at = datetime(2026, 9, 30, 12, 0, 0)


class _Syn:
    """MedicalSynonym fake para coverage."""

    def __init__(self, syn, canonical=None):
        self.id = uuid.uuid4()
        self.synonym = syn
        self.canonical = canonical or syn.split()[0]
        self.category = "signos"


class _Chk:
    def __init__(self, content, document_id=None, document=None):
        self.id = uuid.uuid4()
        self.document_id = document_id or uuid.uuid4()
        self.content = content
        self.document = document


class _Doc:
    def __init__(self, name):
        self.id = uuid.uuid4()
        self.original_name = name
        self.filename = name


class _AuditQ:
    """audit row fake de tipo query con detail.sources."""

    def __init__(self, n_sources, question="q"):
        self.id = uuid.uuid4()
        self.username = "root"
        self.action = "query"
        self.created_at = datetime(2026, 9, 30, 12, 0, 0)
        self.detail = {"sources": [{"id": i} for i in range(n_sources)]}


# ---------------------------------------------------------------- BD fake BSC

class _BscDb:
    """BD fake mínima para los cómputos: listas en memoria en vez de SQL."""

    def __init__(self, users=None, synonyms=None, chunks=None, audit_rows=None):
        self.users = users or []
        self.synonyms = synonyms or []
        self.chunks = chunks or []
        self.audit_rows = audit_rows or []

    def add(self, obj):
        pass

    async def commit(self):
        pass

    async def rollback(self):
        pass

    async def refresh(self, obj):
        if getattr(obj, "id", None) is None:
            obj.id = uuid.uuid4()
        return obj


# ---------------------------------------------------------------- setup API

def _setup_cliente_admin(client):
    users = []
    _override_db(client, _make_fake_db(users))
    _register(client, username="root")
    return _auth(client, "root"), users


# ---------------------------------------------------------------- B1 cobertura

def test_b1_cobertura_keywords(client):
    from src.services.bsc import compute_keyword_coverage

    chunks = [_Chk("glaucoma y cribado neonatal"), _Chk("hipertension pulmonar")]
    syns = [_Syn("glaucoma"), _Syn("cribado neonatal"), _Syn("hipertension")]
    v, f = compute_keyword_coverage(syns, chunks)  # type: ignore
    assert v == pytest.approx(100.0), (v, f)  # 3/3
    assert f  # fórmula documentada no vacía

    syns2 = syns + [_Syn("anemia")]
    v2, _ = compute_keyword_coverage(syns2, chunks)  # type: ignore
    assert v2 == pytest.approx(75.0)  # 3/4

    v3, f3 = compute_keyword_coverage([], chunks)  # type: ignore
    assert v3 == pytest.approx(100.0)  # catálogo vacío → 100, documentado en f3


# ---------------------------------------------------------------- B2 consultas

def test_b2_consultas_con_sin_evidencia():
    from src.services.bsc import compute_query_stats

    rows = [_AuditQ(3), _AuditQ(0), _AuditQ(2)]
    stats = compute_query_stats(rows)  # type: ignore
    assert stats["total"] == 3
    assert stats["with_evidence"] == 2
    assert stats["without_evidence"] == 1
    assert stats["pct_with_evidence"] == pytest.approx(66.67, abs=0.05)


def test_b2_vacio():
    from src.services.bsc import compute_query_stats

    stats = compute_query_stats([])  # type: ignore
    assert stats["total"] == 0 and stats["pct_with_evidence"] == 0.0


# ---------------------------------------------------------------- B3 respaldo

def test_b3_indice_respaldo_niveles():
    from src.services.bsc import compute_support_index

    alto = compute_support_index(evidence=10, keyword_hits=5, avg_score=0.9, pages=3)
    bajo = compute_support_index(evidence=0, keyword_hits=0, avg_score=0.1, pages=0)
    medio = compute_support_index(evidence=2, keyword_hits=1, avg_score=0.5, pages=1)
    assert alto["level"] in ("Alto", "MEDIO", "Bajo")
    assert bajo["level"].lower() == "bajo"
    assert 0.0 <= alto["score"] <= 100.0
    assert "formula" in alto and "peso" in alto["formula"].lower()
    assert medio["score"] != alto["score"]


# ---------------------------------------------------------------- B4 semáforo

def test_b4_semaforo_configurable():
    from src.services.bsc import state_for

    k = _Kpi("K", thresholds={"amber": 90.0, "red": 75.0}, direction="gte")
    assert state_for(k, 95.0) == "green"
    assert state_for(k, 85.0) == "amber"
    assert state_for(k, 60.0) == "red"
    k2 = _Kpi("K2", thresholds={"amber": 10.0, "red": 5.0}, direction="lte")
    assert state_for(k2, 4.0) == "green"
    assert state_for(k2, 8.0) == "amber"
    assert state_for(k2, 20.0) == "red"
    # cambiar umbral cambia estado (RAG-048):
    k3 = _Kpi("K3", thresholds={"amber": 60.0, "red": 40.0})
    assert state_for(k3, 70.0) == "green"
    k3.thresholds = {"amber": 80.0, "red": 60.0}
    assert state_for(k3, 70.0) == "amber"


# ---------------------------------------------------------------- B5 gaps/heat

def test_b5_gaps_heatmap_emergentes():
    from src.services.bsc import compute_gaps

    doc1 = _Doc("segundo.pdf")
    chunks = [
        _Chk("glaucoma cronico avanzado", doc1.id, doc1),
        _Chk("glaucoma congenito", doc1.id, doc1),
    ]
    syns = [_Syn("glaucoma"), _Syn("anemia falciforme")]
    gaps = compute_gaps(syns, chunks)  # type: ignore
    assert gaps["brechas"] == ["anemia falciforme"]
    assert gaps["heatmap"][0]["document"] == "segundo.pdf"
    assert gaps["heatmap"][0]["glaucoma"] == 2
    assert gaps["emergentes"], "conceptos fuera del catálogo esperados"


def test_b5_endpoint_gaps_requiere_auth(client):
    resp = client.get("/api/v1/bsc/gaps")
    assert resp.status_code in (401, 403)  # gate


# ---------------------------------------------------------------- B6 alertas

def test_b6_alerta_kpi_critico():
    from src.services.bsc import ensure_alert

    alerts = []
    k = _Kpi("K", thresholds={"amber": 85.0, "red": 70.0})
    ensure_alert(k, "amber", value=78.0, alerts=alerts)  # type: ignore
    assert len(alerts) == 1 and alerts[0].severity == "warning"
    ensure_alert(k, "red", value=50.0, alerts=alerts)  # type: ignore
    assert alerts[-1].severity == "critical"
    # dedup: mismo estado + kpi abierto NO duplica:
    n = len(alerts)
    ensure_alert(k, "red", value=50.0, alerts=alerts)  # type: ignore
    assert len(alerts) == n


# ---------------------------------------------------------------- B7 planes

def test_b7_planes_de_accion_crud(client, monkeypatch):
    hdrs, users = _setup_cliente_admin(client)
    resp = client.post("/api/v1/bsc/action-plans", json={
        "title": "Cerrar brecha anemia", "owner": "Especialista",
        "kpi_code": "KEYWORD_COVERAGE", "status": "open",
    }, headers=hdrs)
    assert resp.status_code in (200, 201), resp.text
    # 422 sin title:
    resp2 = client.post("/api/v1/bsc/action-plans", json={"owner": "x"}, headers=hdrs)
    assert resp2.status_code == 422
    resp3 = client.get("/api/v1/bsc/action-plans", headers=hdrs)
    assert resp3.status_code == 200


# ---------------------------------------------------------------- B8 score

def test_b8_score_consolidado():
    from src.services.bsc import compute_score

    k1 = _Kpi("K1", thresholds={"amber": 85.0, "red": 70.0})
    k2 = _Kpi("K2", direction="lte", thresholds={"amber": 10.0, "red": 5.0})
    score, formula = compute_score([(k1, 90.0), (k2, 6.0)])  # type: ignore
    assert 0.0 <= score <= 100.0
    assert "peso" in formula.lower()
    # sin KPIs → 0 y fórmula igual:
    s0, f0 = compute_score([])
    assert s0 == 0.0


# ---------------------------------------------------------------- B9 reporte

def test_b9_reporte_pdf_xlsx(client, monkeypatch):
    pytest.importorskip("reportlab", reason="reportlab requiere instalacion local")
    pytest.importorskip("openpyxl")
    hdrs, users = _setup_cliente_admin(client)
    monkeypatch.setattr(
        "src.services.bsc.build_dashboard",
        lambda db, *a, **k: {"kpis": [], "gaps": {"brechas": []},
                             "score": {"value": 0.0, "formula": "x"}},
    )
    resp = client.get("/api/v1/bsc/report.pdf", headers=hdrs)
    assert resp.status_code == 200, resp.text[:200]
    assert resp.headers["content-type"].startswith("application/pdf")
    assert len(resp.content) > 1000
    resp2 = client.get("/api/v1/bsc/report.xlsx", headers=hdrs)
    assert resp2.status_code == 200 and len(resp2.content) > 1000


# ---------------------------------------------------------------- B10 permisos

def test_b10_compute_requiere_kpis_write(client, monkeypatch):
    # editor (NO admin) → POST compute 403; el kpi aún no existe → admin 404
    users = []
    _override_db(client, _make_fake_db(users))
    _register(client, username="root")
    hdrs_admin = _auth(client, "root")
    client.post("/api/v1/users", json={
        "username": "ed1", "email": "ed1@x.y", "password": "T3st-clav3-larga", "role": "editor"},
        headers=hdrs_admin)
    hdrs_ed = _auth(client, "ed1")
    resp = client.post("/api/v1/bsc/kpis/KEYWORD_COVERAGE/compute", headers=hdrs_ed)
    assert resp.status_code == 403
    resp2 = client.post("/api/v1/bsc/kpis/KEYWORD_COVERAGE/compute", headers=hdrs_admin)
    assert resp2.status_code == 404
