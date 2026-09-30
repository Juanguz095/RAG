"""Seed de KPIs por defecto del BSC (PLAN-005; definidos según spec §43-§50)."""
from __future__ import annotations

import logging

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.database import Kpi

logger = logging.getLogger(__name__)

DEFAULT_KPIS = [
    # P1 Conocimiento documental
    dict(code="DOC_TOTAL", name="Documentos cargados", perspective="conocimiento",
         direction="gte", unit="docs", target=None, thresholds={"amber": 1, "red": 0},
         query_type="doc_total"),
    dict(code="DOC_PROCESSED", name="Documentos procesados", perspective="conocimiento",
         direction="gte", unit="%", target=98.0, thresholds={"amber": 95.0, "red": 85.0},
         query_type="doc_processed"),
    dict(code="KEYWORD_COVERAGE", name="Cobertura de palabras clave", perspective="conocimiento",
         direction="gte", unit="%", target=90.0, thresholds={"amber": 90.0, "red": 70.0},
         query_type="keyword_coverage"),
    dict(code="KEYWORD_GAPS", name="Keywords sin evidencia (brechas)", perspective="conocimiento",
         direction="lte", unit="kw", target=None, thresholds={"amber": 5, "red": 10},
         query_type="keyword_gaps"),
    # P2 Inteligencia RAG
    dict(code="QUERY_TOTAL", name="Consultas realizadas", perspective="rag",
         direction="gte", unit="q", target=None, thresholds={"amber": 1, "red": 0},
         query_type="query_stats"),
    dict(code="QUERY_WITH_EVIDENCE", name="Consultas con evidencia", perspective="rag",
         direction="gte", unit="%", target=90.0, thresholds={"amber": 90.0, "red": 70.0},
         query_type="query_stats"),
    dict(code="QUERY_NO_EVIDENCE", name="Consultas sin evidencia", perspective="rag",
         direction="lte", unit="%", target=10.0, thresholds={"amber": 10.0, "red": 30.0},
         query_type="query_stats"),
    # P3 Usuario y servicio
    dict(code="USERS_TOTAL", name="Usuarios registrados", perspective="usuario",
         direction="gte", unit="users", target=None, thresholds={"amber": 1, "red": 0},
         query_type="users_total"),
    # P4 Seguridad y gobierno
    dict(code="AUDIT_OPERATIONS", name="Operaciones auditadas", perspective="seguridad",
         direction="gte", unit="ops", target=None, thresholds={"amber": 1, "red": 0},
         query_type="audit_operations"),
    dict(code="FAILED_LOGINS", name="Accesos rechazados", perspective="seguridad",
         direction="lte", unit="ops", target=None, thresholds={"amber": 10, "red": 50},
         query_type="failed_logins"),
]


async def seed_kpis(db: AsyncSession) -> None:
    """Inserta los KPIs default que falten (idempotente por code)."""
    rows = (await db.execute(select(Kpi))).scalars().all()
    existing = {k.code for k in rows}
    created = 0
    for spec in DEFAULT_KPIS:
        if spec["code"] in existing:
            continue
        db.add(Kpi(**spec))
        created += 1
    if created:
        await db.commit()
    logger.info("seed_kpis: %d creados de %d", created, len(DEFAULT_KPIS))
