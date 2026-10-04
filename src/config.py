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

    AUTH_REQUIRED: bool = os.getenv("AUTH_REQUIRED", "true").lower() == "true"

    # Chat con memoria (RAG-028, PLAN-007 F5)
    CHAT_MEMORY_N: int = int(os.getenv("CHAT_MEMORY_N", "4"))
    # Chat conversacional: saludos/cortesia se responden sin RAG ni citas.
    CHAT_CONVERSATIONAL: bool = os.getenv("CHAT_CONVERSATIONAL", "true").lower() == "true"

    MODEL_DIR: str = "/models"
    QWEN_MODEL_PATH: str = os.getenv(
        "QWEN_MODEL_PATH",
        "/models/qwen2.5-1.5b-instruct-q4_k_m.gguf",
    )
    LLM_MODEL_PATH: str = os.getenv("LLM_MODEL_PATH", "")
    # MiniLM multilingüe 384-d: ~35x más rápido que BGE-M3 en CPU (misma familia
    # de recuperación; el re-ranking por cross-encoder se mantiene). Para volver
    # a BGE-M3: EMBED_MODEL_NAME=BAAI/bge-m3 + migrar la columna a vector(1024).
    EMBED_MODEL_NAME: str = os.getenv(
        "EMBED_MODEL_NAME", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    # ── OCR en cascada (PLAN-001 Fase 1 / PLAN-009 Fase B) ──
    # auto = Tesseract primero y, si la página sale con baja confianza (manuscrita
    # o degradada), re-OCR con RapidOCR (PP-OCRv5 latin). tesseract|paddle|rapidocr
    # fuerzan un único motor (rollback). Default: auto.
    OCR_ENGINE: str = os.getenv("OCR_ENGINE", "auto")  # auto|tesseract|paddle|rapidocr
    OCR_DPI: int = 150
    # Idioma de Tesseract. "spa" es ~2x más rápido que "spa+eng" (carga y corre
    # ambos traineddata). Subir a "spa+eng" si hay documentos en inglés.
    OCR_LANG: str = os.getenv("OCR_LANG", "spa")
    # Detección de rotación (OSD + fallback 0/180). Cuesta pasadas extra de OCR;
    # poner "false" acelera cuando se sabe que los PDFs vienen derechos.
    OCR_DETECT_ROTATION: bool = os.getenv("OCR_DETECT_ROTATION", "true").lower() == "true"
    OCR_MAX_PAGES_PADDLE: int = 0  # 0 = sin límite
    # Cascada `auto`: si la confianza media de Tesseract (0-100) cae por debajo
    # de este umbral, se re-OCR la página con RapidOCR (manuscritos).
    OCR_AUTO_MIN_SCORE: float = float(os.getenv("OCR_AUTO_MIN_SCORE", "60"))
    # Chunks con confianza OCR por debajo de este umbral se marcan low_confidence.
    OCR_LOW_CONF_THRESHOLD: float = float(os.getenv("OCR_LOW_CONF_THRESHOLD", "45"))
    # Rotación (fallback 0/180): solo girar si la versión girada se lee bien de
    # verdad (score absoluto). Evita falsos positivos en páginas con poco texto.
    OCR_ROTATION_MIN_SCORE: float = float(os.getenv("OCR_ROTATION_MIN_SCORE", "60"))

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
            # El cross-encoder es el segundo mayor coste en CPU (~4-5 s/par):
            # 5 candidatos bastan para las citas y recortan ~40% del rerank.
            if self.RERANK_CANDIDATES > 5:
                self.RERANK_CANDIDATES = 5
        # perfil calidad: defaults ya declarados arriba


@lru_cache
def get_settings() -> Settings:
    return Settings()
