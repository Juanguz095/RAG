from functools import lru_cache

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # Database
    DATABASE_URL: str = "postgresql+asyncpg://rag_user:rag_secure_pass@localhost:5432/rag_medical"

    # Redis / Celery
    REDIS_URL: str = "redis://localhost:6379/0"
    CELERY_BROKER_URL: str = "redis://localhost:6379/1"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/2"

    # MinIO
    MINIO_ENDPOINT: str = "localhost:9000"
    MINIO_ACCESS_KEY: str = "minioadmin"
    MINIO_SECRET_KEY: str = "minio_secure_pass"
    MINIO_BUCKET: str = "documents"
    MINIO_SECURE: bool = False

    # Modelos locales (rutas dentro del contenedor)
    MEDALPACA_MODEL_PATH: str = "/models/medalpaca-7b-q4_k_m.gguf"
    BGE_M3_MODEL_NAME: str = "BAAI/bge-m3"
    RERANKER_MODEL_NAME: str = "BAAI/bge-reranker-v2-m3"
    RERANKER_ENABLED: bool = False
    SURYA_MODEL_DIR: str = "/models/surya"

    # LLM Config
    LLM_N_CTX: int = 4096
    LLM_N_THREADS: int = 8
    LLM_N_GPU_LAYERS: int = 0
    LLM_N_BATCH: int = 512
    LLM_MAX_TOKENS: int = 256
    LLM_ENABLED: bool = True

    # Procesamiento
    MAX_UPLOAD_SIZE_BYTES: int = 50 * 1024 * 1024
    MAX_PAGES_PER_DOC: int = 100
    OCR_DPI: int = 200
    MIN_TEXT_CHARS_PER_PAGE: int = 50
    CHUNK_MAX_TOKENS: int = 512
    CHUNK_OVERLAP_TOKENS: int = 50
    EMBEDDING_BATCH_SIZE: int = 32
    RETRIEVAL_TOP_K: int = 20
    RERANK_TOP_K: int = 10
    CONTEXT_MAX_TOKENS: int = 2500

    # Seguridad
    SECRET_KEY: str = ""
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    AUTH_REQUIRED: bool = True
    DEV_AUTH_BYPASS: bool = False
    CORS_ORIGINS: str = "http://localhost:3000,http://localhost:8000"

    # Auth
    ALLOW_REGISTRATION: bool = False
    DEFAULT_ADMIN_USERNAME: str = "admin"
    DEFAULT_ADMIN_PASSWORD: str = "admin123"

    # Logging
    LOG_LEVEL: str = "INFO"

    @model_validator(mode="after")
    def validate_security(self) -> "Settings":
        if self.AUTH_REQUIRED and (
            len(self.SECRET_KEY) < 32 or "replace-with" in self.SECRET_KEY
        ):
            raise ValueError("SECRET_KEY must be a non-placeholder value with at least 32 characters")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
