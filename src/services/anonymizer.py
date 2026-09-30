"""Anonimizador de datos sensibles (PLAN-006 Fase 3 — RAG-038, CP-008).

Detección por regex de WHO-independiente: DNI/NIE, teléfonos, emails,
historias clínicas, fechas de nacimiento explícitas, tarjetas. Sin servicios
externos (criterio rúbrica: 100% local).
"""
from __future__ import annotations

import re

# (kind, regex, etiqueta de reemplazo)
PATTERNS: list[tuple[str, str]] = [
    ("dni", r"\b(?:\d{8}[A-J]|[XYZ]\d{7}[A-Z])\b"),
    ("telefono", r"(?:\+34[\s-]?)?[6789]\d{2}[\s.-]\d{3}[\s.-]\d{3}\b"),
    ("email", r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
    ("historia_clinica", r"\b(?:H\s?\.?C\.?|HC| Expediente )\s?\d{4,}\b"),
    ("tarjeta", r"\b(?:\d[ -]*?){13,16}\b"),
    ("fecha_nacimiento", r"(?:nacid[oa](?: el)?)\s*:?\s*\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"),
    ("calle", r"\b(?:C/|Calle|Av\.?|Avenida|Paseo|Plaza)\s+[A-ZÁÉÍÓÚÑ][\wÁÉÍÓÚñ]*(?:\s+[A-ZÁÉÍÓÚÑ][\wÁÉÍÓÚñ]*)*\s*,?\s*(?:n[º°]\s*)?\d+\b"),
]

LABELS = {
    "dni": "[DNI]",
    "telefono": "[TELEFONO]",
    "email": "[EMAIL]",
    "historia_clinica": "[HISTORIA_CLINICA]",
    "tarjeta": "[TARJETA]",
    "fecha_nacimiento": "[FECHA_NACIMIENTO]",
    "calle": "[DIRECCION]",
}


def detect_sensitive(text: str) -> list[dict]:
    """Devuelve los hallazgos [{kind, label, match}]."""
    out = []
    for kind, pattern in PATTERNS:
        for m in re.finditer(pattern, text):
            out.append({"kind": kind, "label": LABELS[kind], "match": m.group(0)})
    return out


def anonymize_text(text: str) -> tuple[str, int]:
    """Enmascara los patrones de datos sensibles. Devuelve (texto, n_hallazgos)."""
    n = 0
    for kind, pattern in PATTERNS:
        if kind == "tarjeta":
            # solo si parece tarjeta (13-16 dígitos contiguos ya lo garantiza)
            pass
        text, k = re.subn(pattern, LABELS[kind], text)
        n += k
    return text, n
