"""Servicio BSC: cómputos del tablero RAG Intelligence BSC (PLAN-005).

Cada `compute_*` trabaja sobre colecciones en memoria (testable sin BD) y
documenta su fórmula (RAG-042/045/049). El semáforo es configurable por
`thresholds` JSONB del KPI (RAG-048).
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import Any, Iterable, Optional


# ------------------------------------------------------------------- semáforo

def state_for(kpi, value: float) -> str:
    """Estado verde/amarillo/rojo según thresholds y direction del KPI.

    amber = umbral de atención, red = umbral crítico (§51). Ambos viven en
    `kpis.thresholds` (JSONB), no hardcodeados.
    """
    th = getattr(kpi, "thresholds", None) or {}
    amber = th.get("amber")
    red = th.get("red")
    direction = (getattr(kpi, "direction", None) or "gte").lower()
    if value is None or amber is None or red is None:
        return "gray"
    if direction == "lte":  # menor es mejor (p.ej. % sin evidencia)
        if value <= red:
            return "green"
        if value <= amber:
            return "amber"
        return "red"
    # gte: mayor es mejor
    if value >= amber:
        return "green"
    if value >= red:
        return "amber"
    return "red"


def state_emoji(state: str) -> str:
    return {"green": "🟢", "amber": "🟡", "red": "🔴", "gray": "⚪"}.get(state, "⚪")


# ------------------------------------------------- B1 coberturas / B2 consultas

def compute_keyword_coverage(synonyms: Iterable[Any], chunks: Iterable[Any]) -> tuple[float, str]:
    """B1 (RAG-042): % de keywords del catálogo con evidencia en chunks.

    Fórmula: `cobertura = keywords_encontradas / keywords_del_catalogo * 100`
    (100% si el catálogo está vacío). Búsqueda literal case-insensitive.
    """
    terms = {s.synonym for s in synonyms}
    if not terms:
        formula = "cobertura = encontradas / total_catalogo * 100 (100% si catalogo vacio)"
        return 100.0, formula
    corpus = "\n".join((c.content or "").lower() for c in chunks)
    found = sum(1 for t in terms if t.lower() in corpus)
    total = len(terms)
    coverage = found / total * 100.0
    formula = f"cobertura = {found}/{total} * 100 = {round(coverage, 1)}%"
    return round(coverage, 2), formula


def compute_query_stats(audit_rows: Iterable[Any]) -> dict[str, Any]:
    """B2 (RAG-044): consultas con y sin evidencia desde audit_log (action='query').

    Con evidencia = el detalle del evento trae >=1 source (documento/página).
    """
    rows = [r for r in audit_rows if getattr(r, "action", "") == "query"]
    total = len(rows)
    with_ev = 0
    for r in rows:
        detail = getattr(r, "detail", None) or {}
        sources = detail.get("sources") or []
        if sources:
            with_ev += 1
    without_ev = total - with_ev
    pct = round(with_ev / total * 100.0, 2) if total else 0.0
    return {
        "total": total,
        "with_evidence": with_ev,
        "without_evidence": without_ev,
        "pct_with_evidence": pct,
        "period": _period(),
    }


def _period() -> dict[str, str]:
    end = datetime.utcnow()
    start = end - timedelta(days=30)
    return {
        "start": start.isoformat(timespec="seconds"),
        "end": end.isoformat(timespec="seconds"),
    }


# --------------------------------------------------------------- B3 respaldo

DEFAULT_WEIGHTS = {"evidencia": 0.4, "keywords": 0.2, "relevancia": 0.2, "paginas": 0.2}


def compute_support_index(
    evidence: int,
    keyword_hits: int,
    avg_score: float,
    pages: int,
    max_evidence: int = 10,
    max_keywords: int = 10,
    max_pages: int = 5,
    weights: Optional[dict[str, float]] = None,
) -> dict[str, Any]:
    """B3 (RAG-045 §47): índice de respaldo documental Alto/Medio/Bajo.

    Fórmula documentada (pesos por componente, 0-100):
      score = 40*pct_evidencia + 20*pct_keywords + 20*pct_relevancia + 20*pct_paginas
    NOTA: mide respaldo en evidencia, NO es probabilidad de que la IA diga verdad.
    """
    w = {**DEFAULT_WEIGHTS, **(weights or {})}
    pe = min(1.0, evidence / max(1, max_evidence))
    pk = min(1.0, keyword_hits / max(1, max_keywords))
    pr = max(0.0, min(1.0, avg_score))
    pp = min(1.0, pages / max(1, max_pages))
    score = round(100.0 * (w["evidencia"] * pe + w["keywords"] * pk
                           + w["relevancia"] * pr + w["paginas"] * pp), 1)
    level = "Alto" if score >= 70 else ("Medio" if score >= 40 else "Bajo")
    formula = (
        "work_score = pesoe*min(1, evidencia/max_e) + pesok*min(1, keywords/max_k) "
        "+ pesor*avg_score + pesop*min(1, paginas/max_p); pesos %s" % w
    )
    return {"score": score, "level": level, "formula": formula,
            "nota": "Mide respaldo en evidencia; NO es probabilidad subjetiva del LLM"}


# ------------------------------------------------------------------- B4/B6 extras

def ensure_alert(kpi, state: str, value: float, alerts: list) -> None:
    """B6 (RAG-051): crea alerta warning/critical; dedup por (kpi, estado) abierto."""
    if state not in ("amber", "red"):
        return
    sev = "warning" if state == "amber" else "critical"
    open_same = [
        a for a in alerts
        if getattr(a, "kpi_id", None) == getattr(kpi, "id", None)
        and getattr(a, "severity", "") in (sev,)
        and getattr(a, "status", "open") == "open"
    ]
    if open_same:
        return
    alerts.append(_Alert(kpi.id, sev, f"KPI {kpi.code} ({kpi.name}): {value} → estado {state}"))


def _Alert(kpi_id, severity, message):  # construimos la clase local (fake o modelo real)
    try:
        from src.database import Alert as _A

        return _A(kpi_id=kpi_id, severity=severity, message=message, status="open")
    except Exception:
        class _Self:
            pass

        a = _Self()
        a.id = uuid4()
        a.kpi_id, a.severity, a.message, a.status = kpi_id, severity, message, "open"
        return a


from uuid import uuid4  # noqa: E402  (usado arriba)


# ------------------------------------------------------------------- B5 gaps

WORD_RE = re.compile(r"[a-záéíóúñü]{4,}", re.IGNORECASE)
STOP = {
    "para", "con", "del", "los", "las", "por", "una", "que", "este", "esta",
    "donde", "como", "sobre", "entre", "años", "mais", "cada", "desde", "hasta",
    "global", "médico", "documento", "documentos", "paciente",
}


def compute_gaps(synonyms: Iterable[Any], chunks: Iterable[Any]) -> dict[str, Any]:
    """B5 (RAG-042/043 §44-45): brechas + heatmap doc×keyword + emergentes.

    brechas: keywords del catálogo sin ninguna coincidencia en chunks.
    heatmap: tope por documento →地表 frecuencia del keyword.
    emergentes: terminos frecuentes fuera del catálogo (top 10).
    """
    syn_list = list(synonyms)
    chk_list = list(chunks)
    terms = [s.synonym for s in syn_list]

    # documento → término → recuento
    heatmap = []
    docs = {}
    for c in chk_list:
        doc = getattr(c, "document", None)
        dname = getattr(doc, "original_name", None) or getattr(doc, "filename", None) if doc is not None else "sin_documento"
        d = docs.setdefault(dname, {"document": dname})
        content = (getattr(c, "content", "") or "").lower()
        for t in terms:
            n = content.count(t.lower())
            if n:
                d[t] = d.get(t, 0) + n
    for dname, d in docs.items():
        heatmap.append(d)

    corpus = "\n".join((c.content or "").lower() for c in chk_list)
    from collections import Counter

    counts = Counter(WORD_RE.findall(corpus))
    brechas = [t for t in terms if counts.get(t.lower(), 0) == 0]
    catalog_set = {t.lower() for t in terms}
    catalog_set |= {getattr(s, "canonical", "").lower() for s in syn_list}
    emergentes = [
        {"term": term, "count": n}
        for term, n in counts.most_common(50)
        if term.lower() not in catalog_set
    ][:10]

    return {
        "brechas": brechas,
        "heatmap": heatmap,
        "emergentes": [e["term"] for e in emergentes],
        "emergentes_detalle": emergentes,
        "period": _period(),
    }


# ------------------------------------------------------------------ B8 score

def compute_score(kpi_values: list[tuple[Any, float]],
                  weights: Optional[dict[str, float]] = None) -> tuple[float, str]:
    """B8 (RAG-049 §57): score consolidado 0-100.

    Fórmula: promedio ponderado de (100 - normalize(value)) por KPI según
    direction y umbrales; pesos por perspectiva configurables en settings.
    NO es probabilidad de verdad de la IA.
    """
    w = weights or {"conocimiento": 1.0, "rag": 1.0, "usuario": 1.0, "seguridad": 1.0}
    rows = []
    for k, v in kpi_values:
        s = _normalize(k, v)  # 0-100 estandarizado a "mayor es mejor"
        persp = getattr(k, "perspective", "conocimiento") or "conocimiento"
        pw = w.get(persp, 1.0)
        rows.append((k.code, s, pw))
    if not rows:
        return 0.0, "promedio_ponderado(normalize(kpi, metas/umbrales)) * pesos_perspectiva (0 KPIs → 0)"
    total_w = sum(pw for _, _, pw in rows)
    score = round(sum(s * pw for _, s, pw in rows) / max(total_w, 1e-9), 1)
    formula = (
        "score = Σ(normalize(kpi)*peso_perspectiva)/Σ(pesos). %s" % rows
    )
    return score, formula


def _normalize(kpi, value: float) -> float:
    """Normaliza un valor de KPI a 0-100 (mayor es mejor)."""
    direction = (getattr(kpi, "direction", None) or "gte").lower()
    th = getattr(kpi, "thresholds", None) or {}
    amber = th.get("amber") if th else None
    red = th.get("red") if th else None
    target = getattr(kpi, "target", None)
    # Si hay meta: normalizar respecto a meta; si no, respecto a umbrales.
    ref_amber = amber if amber is not None else target if direction == "gte" else None
    ref_red = red if red is not None else (0 if direction == "gte" else None)
    if ref_amber is None or ref_red is None or ref_amber == ref_red:
        return max(0.0, min(100.0, value))
    if direction == "gte":
        raw = (value - ref_red) / (ref_amber - ref_red) * 100.0
    else:
        raw = (ref_red - value) / (ref_red - ref_amber) * 100.0
    return max(0.0, min(100.0, raw))


# ----------------------------------------------------------------- dashboard

def build_dashboard(db, driver: str = "http") -> dict[str, Any]:
    """Levanta KPIs desde la BD y compone el dashboard completo (B1-B8).

    En tests se monkeypatea; en producción lee con SQL async.
    """
    raise NotImplementedError("build_dashboard se implemente on wire (ver tests B9)")
