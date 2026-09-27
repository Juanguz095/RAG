from __future__ import annotations

import logging
import os
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from src.config import get_settings
from src.database import init_db

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting RAG Medical API...")
    try:
        await init_db()
        logger.info("Database initialized")
    except Exception as e:
        logger.error(f"Database init failed: {e}")
        logger.warning("Starting without database (health will report unhealthy)")

    try:
        from src.services.synonyms import seed_synonyms, load_synonyms_from_xlsx
        from src.database import async_session
        async with async_session() as db:
            await seed_synonyms(db)
            await load_synonyms_from_xlsx(db)
    except Exception as e:
        logger.warning(f"Synonym load failed: {e}")

    try:
        from src.services.embeddings import preload
        preload()
        logger.info("Embeddings preloaded")
    except Exception as e:
        logger.warning(f"Embedding preload failed: {e}")

    yield
    logger.info("Shutting down...")


app = FastAPI(
    title="RAG Medical",
    description="Sistema RAG local para documentos médicos escaneados",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000", "http://127.0.0.1:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from src.api.v1.auth import router as auth_router
from src.api.v1.documents import router as documents_router
from src.api.v1.query import router as query_router
from src.api.v1.synonyms import router as synonyms_router

app.include_router(auth_router)
app.include_router(documents_router)
app.include_router(query_router)
app.include_router(synonyms_router)


@app.get("/api/v1/health")
async def health():
    checks = {"api": True, "database": False, "redis": False}
    try:
        from src.database import engine
        async with engine.begin() as conn:
            from sqlalchemy import text
            await conn.execute(text("SELECT 1"))
        checks["database"] = True
    except Exception as e:
        logger.warning(f"DB health check failed: {e}")

    try:
        import redis as redis_lib
        r = redis_lib.from_url(settings.REDIS_URL)
        r.ping()
        checks["redis"] = True
    except Exception as e:
        logger.warning(f"Redis health check failed: {e}")

    healthy = all(checks.values())
    return {"status": "healthy" if healthy else "degraded", "checks": checks}


app.mount("/static", StaticFiles(directory="frontend", check_dir=False), name="static")


@app.get("/{full_path:path}")
async def serve_frontend(full_path: str):
    from fastapi.responses import FileResponse
    from pathlib import Path
    frontend_dir = Path("frontend").resolve()
    file_path = (frontend_dir / full_path).resolve()
    if file_path.is_file() and str(file_path).startswith(str(frontend_dir)):
        return FileResponse(str(file_path))
    return FileResponse(str(frontend_dir / "index.html"))
