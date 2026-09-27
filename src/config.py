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
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7

    AUTH_REQUIRED: bool = os.getenv("AUTH_REQUIRED", "false").lower() == "true"

    MODEL_DIR: str = "/models"
    QWEN_MODEL_PATH: str = ""
    EMBED_MODEL_NAME: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    LLM_N_CTX: int = 4096
    LLM_N_THREADS: int = 4
    LLM_N_BATCH: int = 256
    LLM_MAX_TOKENS: int = 1024
    LLM_TEMPERATURE: float = 0.1

    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    RETRIEVAL_TOP_K: int = 30
    RERANK_TOP_K: int = 10
    CONTEXT_MAX_CHARS: int = 6000

    class Config:
        extra = "ignore"


@lru_cache
def get_settings() -> Settings:
    return Settings()
