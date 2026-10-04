"""Saneado de los campos del paciente extraídos (best-effort).

El OCR de formularios escaneados es ruidoso y el LLM local puede devolver
valores imposibles (p.ej. edad=210). Aquí se valida plausibilidad por campo
para no mostrar datos sin sentido en el panel.
"""
from __future__ import annotations

import re
from datetime import datetime

# Palabras que NO son un nombre/historia válidos (etiquetas de formulario).
_FORBIDDEN = re.compile(
    r"(mestizo|mestiza|femenino|femenina|masculino|hombre|mujer|parto|seguro|"
    r"fallecimiento|natimuerto|salud|materna|paterno|apellido|asegurado|dni|"
    r"historia|fecha|documento|atencion|atención|responsable|paciente|nombre)",
    re.IGNORECASE,
)


def _clean(v) -> str | None:
    if v is None:
        return None
    s = str(v).strip()
    return s or None


def valid_edad(v) -> str | None:
    digits = re.sub(r"\D", "", str(v or ""))
    if not digits:
        return None
    n = int(digits)
    return str(n) if 0 <= n <= 120 else None


def valid_dni(v) -> str | None:
    s = re.sub(r"[^0-9A-Za-z]", "", str(v or ""))
    return s if 6 <= len(s) <= 12 else None


def valid_fecha(v) -> str | None:
    s = str(v or "").strip()
    if not s:
        return None
    for fmt in ("%d/%m/%Y", "%d-%m-%Y", "%Y-%m-%d", "%d.%m.%Y"):
        try:
            return datetime.strptime(s, fmt).strftime("%d/%m/%Y")
        except ValueError:
            continue
    return None


def valid_sexo(v) -> str | None:
    s = re.sub(r"[^a-z]", "", str(v or "").lower())
    if not s:
        return None
    if s.startswith("f") or "femenin" in s or "mujer" in s:
        return "F"
    if s.startswith("m") or s.startswith("h") or "masculin" in s or "hombre" in s:
        return "M"
    return None


def valid_nombre(v) -> str | None:
    s = _clean(v)
    if not s or len(s) < 3 or len(s) > 80:
        return None
    if _FORBIDDEN.search(s):
        return None
    if not re.search(r"[A-Za-zÁÉÍÓÚÑáéíóúñ]", s):
        return None
    return s


def valid_hc(v) -> str | None:
    s = _clean(v)
    if not s or len(s) > 40 or len(s) < 4 or _FORBIDDEN.search(s):
        return None
    if re.fullmatch(r"[\d\s\-]+", s):  # solo numeros/cortos → no es una HC
        return None
    return s


def _clean_list(items, cap: int = 8) -> list[str]:
    """Deduplica (case-insensitive) y limita la lista."""
    out: list[str] = []
    seen: set[str] = set()
    for x in items or []:
        s = _clean(x)
        if not s:
            continue
        k = s.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(s)
        if len(out) >= cap:
            break
    return out


def sanitize_patient_fields(data: dict) -> dict:
    """Valida cada campo; devuelve None en los implausibles."""
    data = data or {}
    sv = data.get("signos_vitales") or {}
    secundarios = data.get("diagnosticos_secundarios") or []
    if not isinstance(secundarios, list):
        secundarios = [secundarios] if secundarios else []
    return {
        "nombre": valid_nombre(data.get("nombre")),
        "edad": valid_edad(data.get("edad")),
        "sexo": valid_sexo(data.get("sexo")),
        "dni": valid_dni(data.get("dni")),
        "historia_clinica": valid_hc(data.get("historia_clinica")),
        "fecha_atencion": valid_fecha(data.get("fecha_atencion")),
        "diagnostico_principal": _clean(data.get("diagnostico_principal")),
        "diagnosticos_secundarios": _clean_list(secundarios, cap=8),
        "signos_vitales": {
            "pa": _clean(sv.get("pa")),
            "fc": _clean(sv.get("fc")),
            "temp": _clean(sv.get("temp")),
            "sato2": _clean(sv.get("sato2")),
        },
    }
