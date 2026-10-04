FROM python:3.11-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake libopenblas-dev libomp-dev git curl \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml .

# Install torch CPU first
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install core deps
RUN pip install --no-cache-dir \
       "fastapi>=0.115" "uvicorn[standard]>=0.30" "sqlalchemy[asyncio]>=2.0" \
       "asyncpg>=0.29" "pgvector>=0.3" "pydantic-settings>=2.0" \
       python-multipart "redis>=5.0" "celery>=5.4" "minio>=7.2" \
       "alembic>=1.13" "pymupdf>=1.24" "pillow>=10.0" "tiktoken>=0.7" \
       "PyJWT>=2.8" httpx python-dotenv psycopg "reportlab>=4.0" "openpyxl>=3.1" \
       pytesseract

# Install ML deps with pinned compatible versions
RUN pip install --no-cache-dir \
       "transformers==4.44.2" \
       "sentence-transformers==3.1.1" \
       llama-cpp-python \
    && pip cache purge

# Engine síncrono del worker (src/workers/ingestion_tasks.py usa el dialecto
# postgresql+psycopg2). Capa aparte para no invalidar la cache de torch/ML.
RUN pip install --no-cache-dir "psycopg2-binary>=2.9"

# OCR ML para manuscritos (PLAN-009 Fase B): RapidOCR (ONNX Runtime) con los
# modelos PP-OCRv5 pre-descargados → 100% offline y ligero (sin paddlepaddle).
# Solo lo usa el worker de ingesta (import perezoso); la API no lo carga.
RUN pip install --no-cache-dir "rapidocr>=3.7,<4.0" "onnxruntime>=1.18"
COPY src/services/rapidocr_params.py ./src/services/rapidocr_params.py
COPY scripts/ocr_models_download.py ./scripts/
RUN python scripts/ocr_models_download.py && pip cache purge
FROM python:3.11-slim AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    libopenblas0 libomp5 libglib2.0-0 libsm6 libxext6 libxrender1 \
    libgomp1 libgl1 poppler-utils libpq5 \
    tesseract-ocr tesseract-ocr-spa tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

WORKDIR /app
COPY src/ ./src/
COPY frontend/ ./frontend/
COPY alembic/ ./alembic/
COPY alembic.ini .
COPY scripts/ ./scripts/

RUN useradd -m -u 1000 appuser
USER appuser

EXPOSE 8000
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
