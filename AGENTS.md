# AGENTS.md - Sistema RAG Médico Local

## Objetivo
Sistema RAG 100% local, privado y gratuito para documentos médicos escaneados/manuscritos (PDF) con OCR avanzado, búsqueda semántica híbrida y LLM médico local.

---

## Stack Tecnológico Definitivo

| Capa | Tecnología | Versión/Detalle |
|------|------------|-----------------|
| API Framework | **FastAPI** | 0.115+, async, OpenAPI auto |
| Base de Datos | **PostgreSQL 16** + **pgvector 0.7+** | HNSW/IVFFlat, pg_trgm |
| OCR Principal | **Surya OCR** (Microsoft) | Layout + Text + Tables + Handwriting |
| OCR Fallback | **PaddleOCR** (PP-OCRv4) | Multilingüe, ligero |
| Extracción Estructural | **Marker** (VikParuchuri) | PDF → Markdown + JSON layout |
| Embeddings | **BGE-M3** (BAAI) | 1024-dim, multilingüe, instrucción-aware |
| Reranker | **BGE-Reranker-v2-M3** | Cross-encoder |
| LLM Local | **MedAlpaca-7B-GGUF** (mlabonne) | Q4_K_M ~4.5GB, llama-cpp-python |
| Cola Async | **Celery 5** + **Redis 7** | Workers OCR/Embedding |
| Almacenamiento Objetos | **MinIO** (S3-compatible) | PDFs originales, imágenes |
| Orquestación | **Docker Compose** (dev) / **Kubernetes** (prod) | |
| Observabilidad | **Prometheus + Grafana + OpenTelemetry** | |

---

## Esquema de Base de Datos (PostgreSQL + pgvector)

```sql
-- Extensiones
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Documentos
CREATE TABLE documents (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    filename        VARCHAR(512) NOT NULL,
    original_name   VARCHAR(512) NOT NULL,
    file_hash       CHAR(64) NOT NULL UNIQUE,
    mime_type       VARCHAR(100),
    file_size       BIGINT,
    page_count      INT,
    language        VARCHAR(10) DEFAULT 'es',
    doc_type        VARCHAR(50),
    status          VARCHAR(20) DEFAULT 'pending',
    metadata        JSONB DEFAULT '{}',
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW(),
    processed_at    TIMESTAMPTZ
);

-- Chunks con embeddings
CREATE TABLE document_chunks (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id     UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index     INT NOT NULL,
    content         TEXT NOT NULL,
    content_md      TEXT,
    token_count     INT,
    char_start      INT,
    char_end        INT,
    page_numbers    INT[],
    bbox            JSONB,
    chunk_metadata  JSONB DEFAULT '{}',
    embedding       VECTOR(1024),
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- Índices críticos
CREATE INDEX idx_chunks_doc_id ON document_chunks(document_id);
CREATE INDEX idx_chunks_embedding_hnsw ON document_chunks 
    USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);
CREATE INDEX idx_chunks_content_trgm ON document_chunks 
    USING gin (content gin_trgm_ops);
CREATE INDEX idx_chunks_metadata ON document_chunks USING gin (chunk_metadata);

-- Entidades médicas (NER)
CREATE TABLE medical_entities (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    chunk_id        UUID NOT NULL REFERENCES document_chunks(id) ON DELETE CASCADE,
    entity_text     TEXT NOT NULL,
    entity_type     VARCHAR(50) NOT NULL,
    icd10_code      VARCHAR(20),
    atc_code        VARCHAR(20),
    confidence      FLOAT,
    start_char      INT,
    end_char        INT,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX idx_entities_chunk ON medical_entities(chunk_id);
CREATE INDEX idx_entities_type ON medical_entities(entity_type);
```

---

## Estructura de Directorios

```
rag-medical/
├── AGENTS.md                 # Este archivo
├── docker-compose.yml        # Infraestructura completa
├── Dockerfile                # Imagen API + Workers
├── pyproject.toml            # Dependencias Python
├── alembic/                  # Migraciones BD
│   ├── env.py
│   ├── script.py.mako
│   └── versions/
├── models/                   # Modelos GGUF (gitignored)
│   └── medalpaca-7b-q4_k_m.gguf
├── src/
│   ├── __init__.py
│   ├── main.py               # FastAPI app entrypoint
│   ├── config.py             # Settings (Pydantic Settings)
│   ├── database.py           # SQLAlchemy async engine/session
│   ├── models/               # SQLAlchemy models
│   │   ├── __init__.py
│   │   ├── document.py
│   │   ├── chunk.py
│   │   └── entity.py
│   ├── schemas/              # Pydantic schemas (API)
│   │   ├── __init__.py
│   │   ├── document.py
│   │   ├── query.py
│   │   └── search.py
│   ├── api/                  # Endpoints FastAPI
│   │   ├── __init__.py
│   │   ├── v1/
│   │   │   ├── __init__.py
│   │   │   ├── documents.py
│   │   │   ├── query.py
│   │   │   ├── search.py
│   │   │   └── entities.py
│   │   └── deps.py           # Dependencias (DB, auth, etc.)
│   ├── services/             # Lógica de negocio
│   │   ├── __init__.py
│   │   ├── ingestion.py      # Pipeline ingesta completo
│   │   ├── ocr.py            # Surya + PaddleOCR wrapper
│   │   ├── chunking.py       # Chunking semántico médico
│   │   ├── embeddings.py     # BGE-M3 inference
│   │   ├── retrieval.py      # Hybrid search + RRF + Rerank
│   │   ├── llm.py            # llama-cpp-python (MedAlpaca)
│   │   ├── ner.py            # spaCy médico + entity linking
│   │   └── context.py        # Context building para RAG
│   ├── workers/              # Celery tasks
│   │   ├── __init__.py
│   │   ├── celery_app.py
│   │   ├── ingestion_tasks.py
│   │   └── embedding_tasks.py
│   └── utils/                # Utilidades
│       ├── __init__.py
│       ├── pdf.py
│       ├── hashing.py
│       └── metrics.py
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_ingestion.py
│   ├── test_retrieval.py
│   └── test_api.py
├── scripts/
│   ├── download_models.py    # Descarga modelos HF
│   └── init_db.py            # Inicialización BD
└── monitoring/
    ├── prometheus.yml
    └── grafana-dashboards/
```

---

## Pipeline de Ingesta (Fases)

### 1. Validación y Deduplicación
```python
# services/ingestion.py - etapa 1
async def validate_and_deduplicate(file: UploadFile) -> Document:
    content = await file.read()
    file_hash = sha256(content).hexdigest()
    
    # Verificar duplicado
    existing = await db.query(Document).filter(Document.file_hash == file_hash).first()
    if existing:
        raise DuplicateDocumentError(existing.id)
    
    # Validar PDF
    pdf_info = validate_pdf(content)
    if pdf_info.page_count > MAX_PAGES:
        raise DocumentTooLargeError()
    
    # Guardar en MinIO
    object_key = f"{file_hash[:2]}/{file_hash}.pdf"
    await minio.put_object("documents", object_key, content)
    
    # Crear registro BD
    doc = Document(
        filename=object_key,
        original_name=file.filename,
        file_hash=file_hash,
        file_size=len(content),
        page_count=pdf_info.page_count,
        status="processing"
    )
    db.add(doc)
    await db.commit()
    return doc
```

### 2. Extracción Estructural + OCR (Celery Task)
```python
# workers/ingestion_tasks.py
@celery_app.task(bind=True, max_retries=3)
def process_document(self, document_id: str):
    doc = get_document(document_id)
    
    # 1. Marker: PDF → Markdown + Layout JSON
    markdown, layout = marker_extract(doc.minio_path)
    
    # 2. Extraer imágenes por página para OCR
    page_images = extract_page_images(doc.minio_path)
    
    # 3. Surya OCR por página (layout + texto + tablas)
    ocr_results = []
    for page_num, img in enumerate(page_images, 1):
        surya_result = surya_ocr(img)  # layout, text_lines, tables
        if surya_result.needs_fallback:
            paddle_result = paddle_ocr(img)
            surya_result = merge_ocr(surya_result, paddle_result)
        ocr_results.append(surya_result)
    
    # 4. Fusionar Marker + OCR por coordenadas (bbox)
    merged_content = merge_marker_ocr(markdown, layout, ocr_results)
    
    # 5. Chunking semántico médico
    chunks = semantic_medical_chunking(merged_content, doc.id)
    
    # 6. Guardar chunks (sin embeddings aún)
    save_chunks(chunks)
    
    # 7. Disparar task de embeddings
    generate_embeddings_task.delay(document_id)
    
    # 8. Actualizar status
    update_document_status(document_id, "completed")
```

### 3. Embeddings Batch (Celery Task)
```python
# workers/embedding_tasks.py
@celery_app.task(bind=True)
def generate_embeddings(self, document_id: str):
    chunks = get_chunks_without_embedding(document_id)
    
    # Batch inference BGE-M3
    texts = [c.content for c in chunks]
    embeddings = bge_m3_encode(texts, batch_size=32)
    
    # Bulk update
    for chunk, emb in zip(chunks, embeddings):
        chunk.embedding = emb
    bulk_update_chunks(chunks)
    
    # Refresh indexes (async, non-blocking)
    refresh_vector_indexes.delay()
```

---

## Pipeline de Consulta (RAG Chain)

```python
# services/retrieval.py
async def hybrid_search(query: str, filters: dict, top_k: int = 20) -> list[Chunk]:
    # 1. Embedding de query
    query_emb = bge_m3_encode([query])[0]
    
    # 2. Vector Search (pgvector HNSW) - Top 50
    vector_results = await vector_search(query_emb, limit=50, filters=filters)
    
    # 3. Keyword Search (pg_trgm) - Top 50
    keyword_results = await keyword_search(query, limit=50, filters=filters)
    
    # 4. Reciprocal Rank Fusion
    fused = reciprocal_rank_fusion(vector_results, keyword_results, k=60)
    
    # 5. Cross-Encoder Reranking (BGE-Reranker-v2-M3) - Top 20
    reranked = await rerank(query, fused[:50], top_k=top_k)
    
    # 6. Deduplicación por documento (max 3 chunks/doc)
    return deduplicate_by_document(reranked, max_per_doc=3)
```

```python
# services/llm.py
async def generate_answer(query: str, chunks: list[Chunk]) -> AsyncGenerator[str, None]:
    context = build_context(chunks, max_tokens=2500)  # Deja espacio para respuesta
    messages = build_medalpaca_messages(context, query)
    
    llm = get_medalpaca()
    stream = llm.create_chat_completion(
        messages=messages,
        stream=True,
        temperature=0.1,
        top_p=0.9,
        stop=["</s>", "<<SYS>>"],
        max_tokens=1024
    )
    
    for chunk in stream:
        delta = chunk["choices"][0]["delta"]
        if "content" in delta:
            yield delta["content"]
```

---

## Endpoints API (FastAPI)

```python
# api/v1/documents.py
router = APIRouter(prefix="/api/v1/documents", tags=["documents"])

@router.post("", status_code=202)
async def upload_document(file: UploadFile, background_tasks: BackgroundTasks):
    doc = await validate_and_deduplicate(file)
    background_tasks.add_task(process_document_task, str(doc.id))
    return {"task_id": doc.id, "status": "processing"}

@router.get("")
async def list_documents(
    status: Optional[str] = None,
    doc_type: Optional[str] = None,
    date_from: Optional[date] = None,
    date_to: Optional[date] = None,
    q: Optional[str] = None,
    page: int = 1,
    size: int = 20
):
    return await search_documents(...)

@router.get("/{doc_id}")
async def get_document(doc_id: UUID):
    return await get_document_with_chunks(doc_id)

@router.get("/{doc_id}/pages/{page_num}")
async def get_page_image(doc_id: UUID, page_num: int):
    return StreamingResponse(get_page_with_ocr_overlay(doc_id, page_num))
```

```python
# api/v1/query.py
@router.post("")
async def query_rag(request: QueryRequest):
    # 1. Hybrid retrieval
    chunks = await hybrid_search(request.query, request.filters, top_k=20)
    
    # 2. Streaming response
    return StreamingResponse(
        generate_answer(request.query, chunks),
        media_type="text/plain"
    )

@router.post("/search")
async def search_only(request: SearchRequest):
    chunks = await hybrid_search(request.query, request.filters, request.top_k)
    return [ChunkResponse.from_orm(c) for c in chunks]

@router.post("/extract")
async def structured_extraction(request: ExtractRequest):
    # JSON schema-guided extraction
    return await extract_structured(request.query, request.schema, request.filters)
```

---

## Configuración Crítica (config.py)

```python
# src/config.py
from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    # Database
    DATABASE_URL: str = "postgresql+asyncpg://user:pass@localhost:5432/rag_medical"
    
    # Redis / Celery
    REDIS_URL: str = "redis://localhost:6379/0"
    CELERY_BROKER_URL: str = "redis://localhost:6379/1"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/2"
    
    # MinIO
    MINIO_ENDPOINT: str = "localhost:9000"
    MINIO_ACCESS_KEY: str = "minioadmin"
    MINIO_SECRET_KEY: str = "minioadmin"
    MINIO_BUCKET: str = "documents"
    MINIO_SECURE: bool = False
    
    # Modelos locales (rutas dentro del contenedor)
    MEDALPACA_MODEL_PATH: str = "/models/medalpaca-7b-q4_k_m.gguf"
    BGE_M3_MODEL_NAME: str = "BAAI/bge-m3"
    RERANKER_MODEL_NAME: str = "BAAI/bge-reranker-v2-m3"
    SURYA_MODEL_DIR: str = "/models/surya"
    
    # LLM Config
    LLM_N_CTX: int = 4096
    LLM_N_THREADS: int = 8
    LLM_N_GPU_LAYERS: int = 0  # Cambiar si hay GPU
    LLM_N_BATCH: int = 512
    
    # Procesamiento
    MAX_PAGES_PER_DOC: int = 100
    CHUNK_MAX_TOKENS: int = 512
    CHUNK_OVERLAP_TOKENS: int = 50
    EMBEDDING_BATCH_SIZE: int = 32
    RETRIEVAL_TOP_K: int = 20
    RERANK_TOP_K: int = 10
    CONTEXT_MAX_TOKENS: int = 2500
    
    # Seguridad
    SECRET_KEY: str = "change-in-production"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    class Config:
        env_file = ".env"

@lru_cache
def get_settings() -> Settings:
    return Settings()
```

---

## Docker Compose (Desarrollo)

```yaml
# docker-compose.yml
version: "3.9"

services:
  postgres:
    image: pgvector/pgvector:pg16
    environment:
      POSTGRES_DB: rag_medical
      POSTGRES_USER: rag_user
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./scripts/init-db.sql:/docker-entrypoint-initdb.d/init-db.sql
    ports: ["5432:5432"]
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U rag_user"]
      interval: 5s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    ports: ["6379:6379"]
    volumes: [redis_data:/data]
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 5s

  minio:
    image: minio/minio:latest
    command: server /data --console-address ":9001"
    environment:
      MINIO_ROOT_USER: minioadmin
      MINIO_ROOT_PASSWORD: ${MINIO_PASSWORD}
    ports: ["9000:9000", "9001:9001"]
    volumes: [minio_data:/data]
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9000/minio/health/live"]
      interval: 30s

  api:
    build: .
    command: uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
    ports: ["8000:8000"]
    volumes:
      - ./src:/app/src
      - ./models:/models:ro
    environment:
      - DATABASE_URL=postgresql+asyncpg://rag_user:${POSTGRES_PASSWORD}@postgres:5432/rag_medical
      - REDIS_URL=redis://redis:6379/0
      - CELERY_BROKER_URL=redis://redis:6379/1
      - MINIO_ENDPOINT=minio:9000
      - MINIO_ACCESS_KEY=minioadmin
      - MINIO_SECRET_KEY=${MINIO_PASSWORD}
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
      minio:
        condition: service_healthy

  worker-ingestion:
    build: .
    command: celery -A src.workers.celery_app worker -Q ingestion -c 2 --loglevel=info
    volumes:
      - ./src:/app/src
      - ./models:/models:ro
    environment: *same_as_api
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    depends_on: [postgres, redis, minio]

  worker-embeddings:
    build: .
    command: celery -A src.workers.celery_app worker -Q embeddings -c 1 --loglevel=info
    volumes:
      - ./src:/app/src
      - ./models:/models:ro
    environment: *same_as_api
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    depends_on: [postgres, redis]

  prometheus:
    image: prom/prometheus:latest
    ports: ["9090:9090"]
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus

  grafana:
    image: grafana/grafana:latest
    ports: ["3000:3000"]
    volumes:
      - ./monitoring/grafana-dashboards:/etc/grafana/provisioning/dashboards
      - grafana_data:/var/lib/grafana
    environment:
      GF_SECURITY_ADMIN_PASSWORD: admin

volumes:
  postgres_data:
  redis_data:
  minio_data:
  prometheus_data:
  grafana_data:
```

---

## Descarga de Modelos (scripts/download_models.py)

```python
#!/usr/bin/env python3
"""Descargar modelos necesarios al directorio models/"""
import os
from huggingface_hub import hf_hub_download

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

# MedAlpaca-7B GGUF Q4_K_M
hf_hub_download(
    repo_id="mlabonne/MedAlpaca-7B-GGUF",
    filename="medalpaca-7b-q4_k_m.gguf",
    local_dir=MODELS_DIR,
    local_dir_use_symlinks=False
)

# Surya OCR models (se descargan automáticamente en primera ejecución)
# Pero se pueden pre-cachear:
# from surya.detection import DetectionPredictor
# from surya.recognition import RecognitionPredictor
# DetectionPredictor()
# RecognitionPredictor()

print("Modelos descargados en", MODELS_DIR)
```

---

## Pasos para Empezar (Orden Obligatorio)

1. **Clonar y configurar entorno**
   ```bash
   cd rag-medical
   cp .env.example .env  # Editar passwords
   mkdir -p models
   python scripts/download_models.py
   ```

2. **Levantar infraestructura**
   ```bash
   docker compose up -d postgres redis minio
   docker compose logs -f postgres  # Esperar "database system is ready"
   ```

3. **Inicializar BD y migraciones**
   ```bash
   docker compose run --rm api alembic upgrade head
   python scripts/init_db.py  # Crear bucket MinIO, índices extra
   ```

4. **Levantar API y Workers**
   ```bash
   docker compose up -d api worker-ingestion worker-embeddings
   docker compose logs -f api
   ```

5. **Verificar salud**
   ```bash
   curl http://localhost:8000/api/v1/health
   # Debe responder: {"status":"ok","dependencies":{"db":true,"redis":true,"minio":true}}
   ```

6. **Probar ingesta**
   ```bash
   curl -X POST -F "file=@ejemplo.pdf" http://localhost:8000/api/v1/documents
   # Respuesta: {"task_id": "uuid", "status": "processing"}
   
   # Ver progreso (WebSocket o polling)
   ws://localhost:8000/api/v1/ws/ingest/{task_id}
   ```

7. **Probar consulta**
   ```bash
   curl -X POST http://localhost:8000/api/v1/query \
     -H "Content-Type: application/json" \
     -d '{"query": "¿Qué diagnóstico tiene el paciente?", "filters": {}}'
   ```

---

## Métricas de Éxito (Definition of Done)

| Métrica | Objetivo Mínimo |
|---------|-----------------|
| Ingesta PDF (10 páginas, OCR + embeddings) | < 60 segundos |
| Latencia query RAG (p50) | < 3 segundos |
| Latencia query RAG (p95) | < 8 segundos |
| Throughput ingesta | 100 docs/hora |
| Throughput queries | 50 queries/min concurrentes |
| Recall@10 (dataset médico interno) | > 0.85 |
| Precisión extracción entidades (F1) | > 0.80 |

---

## Notas de Implementación Críticas

1. **Surya OCR** requiere `torch` + `transformers` + `timm`. Instalar con `--extra-index-url https://download.pytorch.org/whl/cu121` si hay GPU.

2. **Marker** necesita `pymupdf` + `pillow` + `huggingface_hub`. Usar `marker-pdf` fork mantenido.

3. **BGE-M3/Reranker**: Usar `FlagEmbedding` library (oficial BAAI) para inference optimizado.

4. **spaCy médico**: `pip install es_core_med7lg` (o modelo custom entrenado con `spacy train`).

5. **pgvector HNSW**: `m=16, ef_construction=64` para inicio. Ajustar a `m=32, ef_construction=128` en producción >1M vectores.

6. **Memoria workers**: Limitar `celery -c 2` para OCR (pesado), `celery -c 1` para embeddings (GPU memory).

7. **Prompt MedAlpaca**: Obligatorio usar template LLaMA-2 con `<<SYS>>`...`<</SYS>>` y `</s>` como stop token.

8. **Anonimización**: Integrar `presidio-analyzer` + `presidio-anonymizer` en pipeline ingesta ANTES de chunking/embeddings.

---

## Próximas Tareas (Post-MVP)

- [ ] Fine-tuning embeddings en corpus médico propio
- [ ] Evaluación automática RAG (RAGAS, TruLens)
- [ ] UI Web (React/Vue) para subida, visor PDF con highlights, chat
- [ ] Multi-tenancy (aislamiento datos por organización)
- [ ] Audit logging inmutable (WORM storage)
- [ ] Backup/Restore automatizado (PG + MinIO)
- [ ] Escalado K8s: HPA por cola Celery, PG read replicas