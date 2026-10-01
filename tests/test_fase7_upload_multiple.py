"""PLAN-007 F3 — CP-003: subida múltiple con resultados por archivo aislados."""
from __future__ import annotations

import io


def _pdf(name: str, body: bytes = b"%PDF-1.4 Muestra") -> tuple:
    return ("files", (name, io.BytesIO(body), "application/pdf"))


def test_upload_multiple_3archivos_uno_corrupto(client, monkeypatch):
    import test_fase2_rbac_auditoria as _f2
    _f2._empty_db_override(client, monkeypatch)
    _f2._register(client, username="mul1")
    h = _f2._auth(client, "mul1")

    called = []
    import src.api.v1.documents as dm
    monkeypatch.setattr(dm, "_enqueue", lambda *a, **k: called.append(1) or True, raising=False)

    files = [_pdf("a.pdf"), _pdf("corrupto.pdf", b"no-es-pdf"), _pdf("c.pdf")]
    r = client.post("/api/v1/documents/upload-multiple", files=files, headers=h)
    d = r.json()
    assert r.status_code == 207, d
    res = d["results"]
    assert len(res) == 3
    assert [x["ok"] for x in res] == [True, False, True], res
    assert any("PDF" in x.get("error", "") for x in res if not x["ok"])


def test_upload_multiple_requires_auth(client):
    r = client.post("/api/v1/documents/upload-multiple", files=[_pdf("x.pdf")])
    assert r.status_code in (401, 403)
