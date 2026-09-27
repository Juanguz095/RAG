from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.v1 import auth, documents, entities, query, search
from src.config import get_settings
from src.database import close_db, get_db

settings = get_settings()

# Prometheus metrics
REQUEST_COUNT = Counter("http_requests_total", "Total HTTP requests", ["method", "endpoint", "status"])
REQUEST_LATENCY = Histogram("http_request_duration_seconds", "HTTP request latency", ["method", "endpoint"])


@asynccontextmanager
async def lifespan(app: FastAPI):
    import logging

    logger = logging.getLogger(__name__)
    logger.info("API started - hybrid search (vectorial + keyword + RRF)")
    yield
    await close_db()


app = FastAPI(
    title="RAG Médico Local",
    description="Sistema RAG 100% local para documentos médicos con OCR avanzado",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in settings.CORS_ORIGINS.split(",") if origin.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Middleware para métricas
@app.middleware("http")
async def metrics_middleware(request, call_next):
    import time
    start = time.time()
    response = await call_next(request)
    duration = time.time() - start

    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code
    ).inc()
    REQUEST_LATENCY.labels(
        method=request.method,
        endpoint=request.url.path
    ).observe(duration)
    return response


# Health check
@app.get("/api/v1/health")
async def health_check(db: AsyncSession = Depends(get_db)):
    import redis.asyncio as redis
    from minio import Minio
    from sqlalchemy import text

    checks = {"db": False, "redis": False, "minio": False}

    # DB check
    try:
        await db.execute(text("SELECT 1"))
        checks["db"] = True
    except Exception:
        pass

    # Redis check
    try:
        r = redis.from_url(settings.REDIS_URL)
        await r.ping()
        await r.close()
        checks["redis"] = True
    except Exception:
        pass

    # MinIO check
    try:
        client = Minio(
            settings.MINIO_ENDPOINT,
            access_key=settings.MINIO_ACCESS_KEY,
            secret_key=settings.MINIO_SECRET_KEY,
            secure=settings.MINIO_SECURE,
        )
        checks["minio"] = client.bucket_exists(settings.MINIO_BUCKET)
    except Exception:
        pass

    all_healthy = all(checks.values())
    return {
        "status": "ok" if all_healthy else "degraded",
        "dependencies": checks,
        "llm_enabled": settings.LLM_ENABLED,
    }


# Prometheus metrics endpoint
@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


# Routers
app.include_router(auth.router, prefix="/api/v1", tags=["auth"])
app.include_router(documents.router, prefix="/api/v1/documents", tags=["documents"])
app.include_router(query.router, prefix="/api/v1/query", tags=["query"])
app.include_router(search.router, prefix="/api/v1/search", tags=["search"])
app.include_router(entities.router, prefix="/api/v1/entities", tags=["entities"])

# Static files (CSS, JS)
frontend_dir = Path(__file__).resolve().parent.parent / "frontend"
static_dir = frontend_dir / "static"
if static_dir.is_dir():
    from fastapi.staticfiles import StaticFiles

    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/login")
async def login_page():
    login_html = frontend_dir / "login.html"
    if login_html.is_file():
        return FileResponse(login_html, media_type="text/html")
    return {"error": "Login page not found"}


@app.get("/")
async def root():
    frontend = Path(__file__).resolve().parent.parent / "frontend" / "index.html"
    if frontend.is_file():
        return FileResponse(frontend, media_type="text/html")
    return {
        "name": "RAG Médico Local",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/api/v1/health",
    }