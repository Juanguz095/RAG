from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL",
        "postgresql+asyncpg://rag_user:rag_secure_pass@localhost:5433/rag_medical",
    )
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    MINIO_ENDPOINT: str = os.getenv("MINIO_ENDPOINT", "localhost:9000")
    MINIO_ACCESS_KEY: str = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
    MINIO_SECRET_KEY: str = os.getenv("MINIO_SECRET_KEY", "minioadmin")
    MINIO_BUCKET: str = "documents"
    MINIO_SECURE: bool = False

    SECRET_KEY: str = os.getenv("SECRET_KEY", "dev-secret-key-change-in-production")
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 480  # 8 h (PLAN-004 M6)

    AUTH_REQUIRED: bool = os.getenv("AUTH_REQUIRED", "false").lower() == "true"

    MODEL_DIR: str = "/models"
    QWEN_MODEL_PATH: str = os.getenv(
        "QWEN_MODEL_PATH",
        "/models/qwen2.5-1.5b-instruct-q4_k_m.gguf",
    )
    LLM_MODEL_PATH: str = os.getenv("LLM_MODEL_PATH", "")
    EMBED_MODEL_NAME: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    # ── OCR en cascada (PLAN-001 Fase 1) ──
    OCR_ENGINE: str = os.getenv("OCR_ENGINE", "tesseract")  # tesseract|paddle|surya
    OCR_DPI: int = 200
    OCR_MAX_PAGES_PADDLE: int = 0  # 0 = sin límite

    # ── Re-ranking (PLAN-001 Fase 3) ──
    RERANK_MODEL_NAME: str = "BAAI/bge-reranker-base"

    # ── LLM (PLAN-003: P2/P3 recortes + tuning de CPU) ──
    LLM_N_CTX: int = 2048
    LLM_N_THREADS: int = 6
    LLM_N_BATCH: int = 512
    LLM_MAX_TOKENS: int = 256
    LLM_TEMPERATURE: float = 0.1

    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    RETRIEVAL_TOP_K: int = 30
    RERANK_TOP_K: int = 10
    RERANK_CANDIDATES: int = 15
    CONTEXT_MAX_CHARS: int = 4000

    # ── Perfil de eficiencia (PLAN-003 P6) ──
    PERFIL: str = os.getenv("PERFIL", "calidad")  # rapido|calidad

    # ── Umbral de evidencia (PLAN-006 Fase 2 — RAG-025, CP-006) ──
    # Si tras el filtro de visibilidad no hay >= EVIDENCE_MIN_SOURCES fuentes
    # con score >= EVIDENCE_MIN_SCORE, el motor SE ABSTIENE (no inventa).
    EVIDENCE_MIN_SOURCES: int = 1
    EVIDENCE_MIN_SCORE: float = 0.15

    class Config:
        extra = "ignore"

    def __init__(self, **data):
        super().__init__(**data)
        # El perfil aplica defaults si los variables hijas no fueron fijadas
        # de forma explícita (kwargs > perfil > env).
        if self.PERFIL == "rapido":
            if self.CONTEXT_MAX_CHARS > 2500:
                self.CONTEXT_MAX_CHARS = 2500
            if self.RERANK_TOP_K > 5:
                self.RERANK_TOP_K = 5
            if self.RERANK_CANDIDATES > 8:
                self.RERANK_CANDIDATES = 8
        # perfil calidad: defaults ya declarados arriba


@lru_cache
def get_settings() -> Settings:
    return Settings()
