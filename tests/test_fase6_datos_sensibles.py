"""PLAN-006 — Fase 3: datos sensibles (RAG-038 — CP-008).

Anonimizador en ingesta + enmascaramiento por rol en respuestas. TDD ROJO.
"""
from __future__ import annotations

import sys
import uuid
from unittest.mock import MagicMock

import pytest

if sys.modules.get("psycopg2") is None:
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


# ------------------------------------------------------------ detección

def test_d_deteccion_patrones():
    from src.services.anonymizer import detect_sensitive

    text = "Paciente Juan Pérez, DNI 12345678A, teléfono +34 612 345 678, HC 2026-00145."
    findings = detect_sensitive(text)
    kinds = {f["kind"] for f in findings}
    assert "dni" in kinds, findings
    assert "telefono" in kinds, findings
    assert "historia_clinica" in kinds, findings


def test_d_anonimizar_reemplaza_los_patrones():
    from src.services.anonymizer import anonymize_text

    text = "DNI 12345678A y tel +34 612 345 678."
    out, n = anonymize_text(text)
    assert n >= 2
    assert "12345678A" not in out
    assert "612 345 678" not in out
    assert "[DNI]" in out and "[TELEFONO]" in out


def test_d_texto_limpio_no_cambia():
    from src.services.anonymizer import anonymize_text

    text = "El glaucoma es una neuropatía óptica progresiva."
    out, n = anonymize_text(text)
    assert n == 0 and out == text
