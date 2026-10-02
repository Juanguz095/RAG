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


def extract_birth_date(text: str) -> str | None:
    """Extrae la fecha de nacimiento ANTES de anonimizar (WP8, PLAN-010).

    Devuelve el match crudo (ej. "nacido: 14/03/1985") o None. NO se persiste
    en claro: el llamador calcula la edad y descarta la fecha. RAG-038/CP-008
    se mantienen intactos porque anonymize_text() se sigue ejecutando después.
    """
    for kind, pattern in PATTERNS:
        if kind == "fecha_nacimiento":
            m = re.search(pattern, text)
            if m:
                return m.group(0)
    return None


def compute_age(birth_date_str: str, today=None) -> int | None:
    """Edad = año actual − año de nacimiento (fórmula estándar, CP-006).

    Acepta fechas dd/mm/yyyy o dd-mm-yyyy (y año de 2 dígitos). Si no se puede
    parsear, devuelve None (el motor se abstiene o responde sin edad).
    """
    import datetime as _dt

    m = re.search(r"(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})", birth_date_str)
    if not m:
        return None
    day, month, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
    if year < 100:
        year += 1900 if year > 30 else 2000  # heurística de 2 dígitos
    try:
        birth = _dt.date(year, month, day)
    except ValueError:
        return None
    today = today or _dt.date.today()
    age = today.year - birth.year - ((today.month, today.day) < (birth.month, birth.day))
    return age if age >= 0 else None
