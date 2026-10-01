"""PLAN-006 — Fase 5: validación humana (RAG-032) + dataset sintético (RAG-058)."""
from __future__ import annotations

import sys
from unittest.mock import MagicMock

try:
    import psycopg2  # noqa: F401
except ModuleNotFoundError:
    sys.modules["psycopg2"] = MagicMock()

import test_fase2_rbac_auditoria as _f2  # arnés

_register = _f2._register
_auth = _f2._auth
_empty_db_override = _f2._empty_db_override
_make_user = _f2._make_user


def test_val_modelo_proposal_existe():
    from src.database import Proposal

    assert Proposal.__tablename__ == "proposals"


def test_val_assistant_propone_y_editor_aprueba(client, monkeypatch):
    _empty_db_override(client, monkeypatch)
    _register(client, username="boss1")  # admin (M1)
    ha = _auth(client, "boss1")
    # crear assistant y editor via /users:
    import uuid as _u
    r = client.post("/api/v1/users", json={"username": "asist1", "email": "a1@x.local",
                                           "password": "T3st-clav3-larga", "role": "assistant"}, headers=ha)
    assert r.status_code in (200, 201), r.text[:120]
    r = client.post("/api/v1/users", json={"username": "edit1", "email": "e1@x.local",
                                           "password": "T3st-clav3-larga", "role": "editor"}, headers=ha)
    assert r.status_code in (200, 201), r.text[:120]
    h_asist = _auth(client, "asist1")
    h_edit = _auth(client, "edit1")

    r = client.post("/api/v1/proposals", headers=h_asist,
                    json={"chunk_id": str(_u.uuid4()),
                          "content": "El glaucoma se trata con timolol y prostaglandinas."})
    assert r.status_code in (200, 201), r.text[:200]
    pid = r.json()["id"]

    r = client.post("/api/v1/proposals", headers=h_asist,
                    json={"chunk_id": str(_u.uuid4()), "content": "otra propuesta"})
    pid2 = r.json()["id"]

    # aprobar con editor:
    r = client.post(f"/api/v1/proposals/{pid}/approve", headers=h_edit)
    assert r.status_code in (200, 204), r.text[:200]
    # rechazar con comentario:
    r = client.post(f"/api/v1/proposals/{pid2}/reject", headers=h_edit,
                    json={"comment": "fuera de contexto"})
    assert r.status_code in (200, 204)


def test_val_viewer_no_puede_proponer(client, monkeypatch):
    _empty_db_override(client, monkeypatch)
    _register(client, username="boss2")
    ha = _auth(client, "boss2")
    client.post("/api/v1/users", json={"username": "vw2", "email": "v2@x.local",
                                       "password": "T3st-clav3-larga", "role": "viewer"}, headers=ha)
    h = _auth(client, "vw2")
    import uuid as _u
    r = client.post("/api/v1/proposals", headers=h,
                    json={"chunk_id": str(_u.uuid4()), "content": "x"})
    assert r.status_code == 403


def test_val_dataset_sintetico_se_genera(tmp_path):
    """RAG-058: PDFs ficticios generables sin RRSS ni datos reales."""
    import importlib
    script = __import__("pathlib").Path(__file__).parent.parent / "scripts" / "generate_synthetic_dataset.py"
    assert script.exists()
    sys.path.insert(0, str(script.parent))
    mod = importlib.import_module("generate_synthetic_dataset")
    pdfs = mod.generate(out_dir=tmp_path, count=2)
    assert len(pdfs) == 2
    for p in pdfs:
        assert p.exists() and p.stat().st_size > 1000
        bytes_head = p.read_bytes()[:5]
        assert bytes_head.startswith(b"%PDF")
