# RAG Intelligence BSC — Sistema RAG médico local

Sistema **100% local** (sin nube ni APIs pagadas) para documentos médicos en PDF:
subes un PDF —incluso escaneado o manuscrito, gracias al OCR—, lo visualizas y
haces preguntas en lenguaje natural. El sistema busca la evidencia en el documento
y responde **citando** documento, página y fragmento. Incluye un módulo **BSC**
(Balanced Scorecard) con KPIs, semáforos y reportes que mide la calidad del sistema.

## Qué hace

- Carga de un PDF con validación (tipo, tamaño, páginas, duplicados).
- Visor integrado (PDF.js) con **resaltado** de coincidencias sobre el texto.
- Extracción de texto (PyMuPDF) y **OCR en cascada** para escaneados/manuscritos.
- Chunking semántico + embeddings (MiniLM multilingüe, 384-d) + pgvector.
- Búsqueda híbrida (vectorial + léxica) con fusión RRF y re-ranking (cross-encoder).
- Consultas RAG con citas y **abstención** si no hay evidencia (no inventa).
- Chat con memoria, corrección de palabras OCR y feedback (👍/👎).
- Panel automático de datos del paciente detectados por OCR.
- Módulo **BSC**: KPIs, tendencias, alertas, planes de acción y reportes.

> El OCR puede fallar con manuscritos difíciles, sellos o imágenes borrosas.
> Los datos médicos deben confirmarse siempre en el documento original.

## Stack

| Capa | Tecnología |
|---|---|
| Backend | Python 3.11, FastAPI, SQLAlchemy 2 (async) + asyncpg |
| Base de datos | PostgreSQL 16 + **pgvector** (HNSW) + **pg_trgm** |
| Cola | Celery + Redis |
| Embeddings | MiniLM multilingüe **384-d** (sentence-transformers) |
| Re-ranking | `BAAI/bge-reranker-base` (cross-encoder) |
| LLM | **Qwen2.5-1.5B-Instruct GGUF** (llama-cpp-python) |
| OCR | PyMuPDF → Tesseract → RapidOCR (PP-OCRv5, ONNX) en cascada |
| Frontend | SPA `frontend/index.html` + `frontend/bsc.html` (JS vanilla, sin build) |
| Deploy | Docker Compose |

## Servicios (`docker-compose.yml`)

| Servicio | Función | Puerto |
|---|---|---:|
| `api` | FastAPI + interfaz web + LLM | 8000 |
| `worker` | Ingesta: OCR, chunks, embeddings | interno |
| `worker-extract` | Extracción de datos del paciente | interno |
| `postgres` | Base de datos y vectores | 5555 → 5432 |
| `redis` | Cola de Celery | 6379 |
| `minio` | Almacenamiento de originales | 9000 / 9001 |

## Requisitos

- **Docker Desktop** con motor Linux iniciado + **Docker Compose v2**.
- **WSL2** (en Windows).
- **≥ 8 GB de RAM** (Docker Desktop con ~7 GB asignados).
- **Internet** la primera vez (descarga de modelos).
- Python 3.11 solo si vas a correr los tests fuera de Docker.

Comprobar:

~~~powershell
docker info
docker compose version
~~~

## Instalación paso a paso

**1) Clonar**

~~~powershell
git clone https://github.com/Juanguz095/RAG.git
cd RAG
~~~

**2) Crear el `.env` y cambiar las claves**

~~~powershell
Copy-Item .env.example .env        # Linux/Mac: cp .env.example .env
~~~

Editar `.env`: cambia `SECRET_KEY` y **`ADMIN_PASSWORD`** (esta última será la
contraseña del usuario admin).

**3) Descargar el modelo del LLM (~1 GB)**

~~~powershell
pip install huggingface-hub
python scripts/download_models.py
~~~

> Los **embeddings** (MiniLM) y el **re-ranker** (`bge-reranker-base`) se descargan
> solos la primera vez que arranca la API. Los modelos de **OCR** van dentro de la
> imagen Docker.

**4) Levantar todo** (la primera vez compila; tarda unos minutos)

~~~powershell
docker compose up -d --build
~~~

**5) Comprobar**

~~~powershell
docker compose ps
Invoke-RestMethod http://localhost:8000/api/v1/health
~~~

Debe responder `{"status":"healthy","checks":{"api":true,"database":true,"redis":true}}`.
Si la API aún está cargando modelos, espera ~15 s y reintenta.

**6) Entrar** — abrir **http://localhost:8000** y hacer login:

- Usuario: **`admin`**
- Contraseña: la **`ADMIN_PASSWORD`** de tu `.env`.

El primer arranque crea ese admin si la tabla de usuarios está vacía. También
puedes crearlo a mano:

~~~powershell
docker compose run --rm api python scripts/seed_admin.py
~~~

## Uso de la interfaz

1. Abrir http://localhost:8000 y hacer login.
2. Comprobar el estado (backend / base de datos / Redis).
3. Seleccionar un PDF y pulsar **Subir documento**.
4. Esperar a que el estado sea **completed**.
5. Clic en el nombre para abrir el visor; usar el buscador para saltar y resaltar.
6. Revisar **Lo importante del PDF** (datos del paciente).
7. Escribir una pregunta en el chat y pulsar **Consultar**.
8. Eliminar el documento antes de subir otro (se permite uno a la vez).

Tablero BSC (solo admin): http://localhost:8000/bsc.html

## API (resumen)

- `GET  /api/v1/health`
- `POST /api/v1/auth/login`
- `POST /api/v1/documents` (multipart, campo `file`, responde 202)
- `GET  /api/v1/documents`
- `GET  /api/v1/documents/{id}/file`
- `POST /api/v1/query` (RAG con citas; se abstiene sin evidencia)
- `POST /api/v1/query/search`
- Swagger: http://localhost:8000/docs

## Modelos

- **LLM**: `models/qwen2.5-1.5b-instruct-q4_k_m.gguf` — se descarga con el script; **no** se versiona.
- **Embeddings / re-ranker**: caché de HuggingFace en `hf_cache/` — se descarga sola.
- **OCR**: RapidOCR PP-OCRv5, dentro de la imagen Docker.

Los modelos grandes **no** van en Git (GitHub rechaza archivos > 100 MB).

## Pruebas

~~~powershell
python -m pytest -q
python -m compileall -q src
~~~

La suite corre **sin infraestructura externa** (sin Postgres, Redis ni modelos).

## Solución de problemas

### El build tarda muchísimo o falla
El build descarga torch y compila `llama-cpp-python`. Asegúrate de tener el
`.dockerignore` (incluido) para que el contexto no arrastre `models/`, `hf_cache/`
ni `uploads/`.

### No puedo entrar (login)
El admin se crea con `ADMIN_PASSWORD` del `.env`. Si la cambiaste, vuelve a crearlo:

~~~powershell
docker compose run --rm api python scripts/seed_admin.py
~~~

### Docker no responde
Abrir Docker Desktop y esperar a que el motor Linux esté listo; luego `docker info`.

### El documento queda en "processing" / "ocr"
~~~powershell
docker compose logs --tail=200 worker
~~~

### Las consultas tardan mucho la primera vez
El primer `/query` carga los modelos (90-150 s en frío). Haz una consulta de
calentamiento antes de una demo; después, ~1 min por consulta.

## Mantenimiento

~~~powershell
docker compose logs -f api
docker compose ps
docker compose down          # parar (los datos viven en volúmenes)
docker compose down -v       # borrar también volúmenes (BD, Redis, MinIO)
~~~

## Seguridad y limitaciones

- No subir `.env` ni contraseñas reales a Git.
- No exponer PostgreSQL, Redis o MinIO directamente a Internet.
- El sistema es apoyo documental y **no sustituye** a un profesional de salud.

## Estructura

~~~text
src/            backend (main, config, api/v1, services, workers, models, schemas)
frontend/       SPA (index.html, bsc.html)
alembic/        migraciones
scripts/        download_models, seed_admin, init_db, reindex, bench_*
tests/          pytest (host, sin infra)
models/         GGUF del LLM (no versionado)
hf_cache/       caché HuggingFace (no versionado)
uploads/        PDFs subidos (no versionado)
docker-compose.yml
Dockerfile
.dockerignore
.env.example
pyproject.toml
~~~

---

Proyecto académico — RAG Intelligence BSC · Repositorio: https://github.com/Juanguz095/RAG
