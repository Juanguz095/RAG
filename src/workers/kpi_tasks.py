"""PLAN-007 F7 — snapshots automáticos de KPIs (RAG-048).

Celery-beat periodic task: diariamente corre el compute de todos los KPIs
activos y persiste la fila en `kpi_snapshots` (con alertas si corresponde).
El endpoint manual POST /bsc/kpis/{code}/compute se conserva.
"""
from __future__ import annotations

import asyncio
import logging

from src.workers.celery_app import celery_app

logger = logging.getLogger(__name__)


@celery_app.task(name="kpi.snapshot_all")
def snapshot_all_kpis():
    """Compute + snapshot de todos los KPIs activos. Devuelve resumen."""
    from sqlalchemy import select
    from sqlalchemy.ext.asyncio import async_sessionmaker

    from src.database import async_session, Kpi
    from src.services.bsc import compute_and_snapshot

    async def _run():
        rows = 0
        async with async_session() as db:
            try:
                kpis = (await db.execute(select(Kpi))).scalars().all()
            except Exception:
                logger.exception("kpi snapshot: no se pudo listar KPIs")
                return {"ok": False, "snapshots": 0}
            for k in kpis:
                try:
                    await compute_and_snapshot(db, k)
                    rows += 1
                except Exception:
                    logger.warning("kpi %s snapshot fallo", getattr(k, "code", "?"))
        return {"ok": True, "snapshots": rows}

    try:
        res = asyncio.new_event_loop().run_until_complete(_run())
        logger.info("KPI snapshot_all: %s", res)
        return res
    except Exception:
        logger.exception("kpi snapshot_all fallo")
        return {"ok": False, "snapshots": 0}


@celery_app.on_after_configure.connect
def _setup_periodic(sender, **_k):
    try:
        from celery.schedules import crontab
        sender.add_periodic_task(
            crontab(hour=6, minute=0),  # diario 06:00
            snapshot_all_kpis.s(),
            name="kpi snapshot diario",
        )
    except Exception:
        logger.exception("no se pudo registrar beat de KPI snapshots")
