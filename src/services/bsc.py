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


def compute_usage_stats(audit_rows: Iterable[Any]) -> dict[str, Any]:
    """WP11 (PLAN-010): métricas de uso por usuario/cliente desde audit_log.

    Sin tablas nuevas: agrupa por `username` los eventos `query` (consultas)
    y `upload` (documentos) de la bitácora. Actividad por periodo = 30 días.
    """
    from collections import Counter

    queries: Counter = Counter()
    docs: Counter = Counter()
    active_users: set[str] = set()
    for r in audit_rows:
        u = getattr(r, "username", None) or "anonimo"
        a = getattr(r, "action", "")
        if a == "query":
            queries[u] += 1
            active_users.add(u)
        elif a == "upload":
            docs[u] += 1
            active_users.add(u)
    total_q = sum(queries.values())
    total_d = sum(docs.values())
    n_users = max(1, len(active_users))
    return {
        "queries_per_user": dict(queries),
        "docs_per_user": dict(docs),
        "active_users": len(active_users),
        "total_queries": total_q,
        "total_uploads": total_d,
        "queries_per_user_avg": round(total_q / n_users, 2),
        "docs_per_user_avg": round(total_d / n_users, 2),
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


async def compute_and_snapshot(db, k) -> dict:
    """Compute + snapshot + alerta dedup para un KPI (usado por el endpoint
    /bsc/kpis/{code}/compute y por la tareaCelery kpi.snapshot_all)."""
    from datetime import datetime
    from sqlalchemy import func, select
    from src.database import Kpi, KpiSnapshot, Alert

    from src.api.v1 import bsc as bsc_api  # _collect_inputs vive ahí

    value: float
    formula = ""
    if k.query_type == "keyword_coverage":
        # Cobertura sobre el CATÁLOGO DE KEYWORDS (SQL agregado, sin cargar chunks).
        from src.services import bsc_knowledge as _bkw
        cov = await _bkw.keyword_coverage(db)
        value = cov["pct"]
        formula = f"cobertura = {cov['found']}/{cov['total']} * 100 = {cov['pct']}%"
    elif k.query_type == "keyword_gaps":
        from src.services import bsc_knowledge as _bkw
        kg = await _bkw.keyword_gaps(db)
        value = float(kg["count"])
        formula = "keywords activas sin evidencia (brechas) = " + str(kg["count"])
    elif k.query_type == "feedback":
        # Feedback útil/no útil sobre respuestas (spec §48/§49).
        from src.database import Message

        rows = (
            await db.execute(select(Message.feedback).where(Message.feedback.isnot(None)))
        ).all()
        vals = [r[0] for r in rows]
        total = len(vals)
        useful = sum(1 for v in vals if v == "useful")
        pct = round(useful / total * 100, 2) if total else 0.0
        if k.code == "FEEDBACK_USEFUL":
            value = pct
        else:
            value = round(100.0 - pct, 2) if total else 0.0
        formula = f"{useful}/{total} respuestas marcadas utiles"
    elif k.query_type == "doc_total":
        from src.database import Document

        value = float((await db.execute(select(func.count(Document.id)))).scalar() or 0)
        formula = "documentos cargados"
    elif k.query_type == "doc_processed":
        from src.database import Document

        total = (await db.execute(select(func.count(Document.id)))).scalar() or 0
        done = (
            await db.execute(select(func.count(Document.id)).where(Document.status == "completed"))
        ).scalar() or 0
        value = round(done / total * 100, 2) if total else 0.0
        formula = f"{done}/{total} documentos procesados"
    elif k.query_type == "users_total":
        from src.database import User

        value = float((await db.execute(select(func.count(User.id)))).scalar() or 0)
        formula = "usuarios registrados"
    elif k.query_type == "audit_operations":
        from src.database import AuditLog

        value = float((await db.execute(select(func.count(AuditLog.id)))).scalar() or 0)
        formula = "operaciones auditadas"
    elif k.query_type == "failed_logins":
        from src.database import AuditLog

        value = float(
            (
                await db.execute(
                    select(func.count(AuditLog.id)).where(AuditLog.action == "login_failed")
                )
            ).scalar()
            or 0
        )
        formula = "accesos rechazados (login_failed)"
    else:
        # query_stats / usage_stats: necesitan audit_log (no cargamos chunks).
        data = await bsc_api._collect_inputs(db)
        if k.query_type == "query_stats":
            stats = compute_query_stats(data["audit_rows"])
            if k.code == "QUERY_WITH_EVIDENCE":
                value = stats["pct_with_evidence"]
            elif k.code == "QUERY_NO_EVIDENCE":
                value = stats["without_evidence"]
            else:
                value = stats["total"]
            formula = "con/sin evidencia desde audit_log"
        elif k.query_type == "usage_stats":
            stats = compute_usage_stats(data["audit_rows"])
            if k.code == "USAGE_ACTIVE_USERS":
                value = float(stats["active_users"])
            elif k.code == "USAGE_QUERIES_PER_USER":
                value = float(stats["queries_per_user_avg"])
            elif k.code == "USAGE_DOCS_PER_USER":
                value = float(stats["docs_per_user_avg"])
            else:
                value = float(stats["total_queries"])
            formula = "uso por usuario/cliente desde audit_log"
        else:
            raise ValueError(f"query_type no soportado: {k.query_type}")

    period = _period()
    snap = KpiSnapshot(
        kpi_id=k.id, value=value,
        period_start=datetime.fromisoformat(period["start"]),
        period_end=datetime.fromisoformat(period["end"]),
    )
    db.add(snap)
    state = state_for(k, value)
    open_alerts = (
        await db.execute(select(Alert).where(Alert.kpi_id == k.id, Alert.status == "open"))
    ).scalars().all()
    created = []
    ensure_alert(k, state, value, created)
    for a in created:
        if not open_alerts:
            db.add(a)
    await db.commit()
    return {"code": k.code, "value": value, "state": state, "formula": formula}
