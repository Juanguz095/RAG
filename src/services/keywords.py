"""Gestión de keywords (PLAN-006 Fase 4 — RAG-013/015-019).

Catálogo en BD (tabla keywords), matches por chunk (chunk_keywords),
enriquecimiento de sección/fragmento en chunk_metadata. 100% local.
"""
from __future__ import annotations

import unicodedata
import re


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", (s or "")).encode("ascii", "ignore").decode()
    return re.sub(r"\s+", " ", s).strip().lower()


def enrich_section(chunk) -> dict:
    """Extrae la sección (encabezado) y el fragmento contextual del chunk.

    Encabezado = primera línea en MAYÚSCULAS/numerales del contenido; si no
    hay, primera oración. Fragmento = el propio content truncado con elipsis.
    Devuelve chunk_metadata actualizado.
    """
    content = (getattr(chunk, "content", "") or "")
    meta = dict(getattr(chunk, "chunk_metadata", None) or {})
    section = meta.get("section")
    if not section:
        m = re.search(r"^(?:\d+[.)]\s*)?([A-ZÁÉÍÓÚÑ][A-ZÁÉÍÓÚÑ\s]{3,60})$", content, re.M)
        if not m:
            first_line = content.split("\n", 1)[0].strip()
            section = first_line[:80] if first_line else ""
        else:
            section = m.group(1).title()
        meta["section"] = section or "General"
    if "fragment" not in meta:
        frag = content.strip()
        meta["fragment"] = frag if len(frag) <= 240 else frag[:237] + "..."
    try:
        chunk.chunk_metadata = meta
    except Exception:
        pass
    return meta


def extract_keyword_matches(chunk, catalog: dict[str, list[str]]) -> dict[str, int]:
    """Cuenta matches de keyword (por variantes) en el contenido del chunk.

    catalog = {term: [variantes normalizadas]}. Devuelve {term: n_matches}.
    """
    content = _norm(getattr(chunk, "content", ""))
    if not content:
        return {}
    counts: dict[str, int] = {}
    for term, variants in (catalog or {}).items():
        n = 0
        for v in {term.lower(), *(_norm(x) for x in (variants or []))}:
            if not v:
                continue
            n += len(re.findall(r"\b" + re.escape(v) + r"\b", content))
        if n:
            counts[term] = counts.get(term, 0) + n
    return counts
