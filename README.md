# RAG Médico Local

Sistema local para cargar un único PDF médico, visualizarlo, extraer texto con OCR y consultar su contenido mediante RAG.

## Qué hace

- Carga de un solo PDF a la vez.
- Validación de tipo, tamaño, páginas y duplicados.
- Visor integrado del PDF con desplazamiento por páginas.
- Extracción de texto de PDFs digitales y OCR para escaneados.
- Procesamiento asíncrono con Celery y Redis.
- Fragmentación semántica y embeddings BGE-M3.
- Búsqueda vectorial, por palabras clave e híbrida.
- Consultas RAG con citas.
- Panel automático de datos importantes detectados por OCR.
- Eliminación y reprocesamiento del documento.
- Almacenamiento de originales en MinIO.

El OCR puede fallar con escritura manuscrita, sellos, fotografías borrosas o campos de baja calidad. Los datos médicos deben confirmarse en el documento original.

## Arquitectura

~~~mermaid
flowchart LR
    U[Usuario] --> UI[Interfaz web]
    UI --> API[FastAPI]
    API --> PG[(PostgreSQL + pgvector)]
    API --> R[(Redis)]
    API --> M[(MinIO)]
    R --> W[Celery worker]
    W --> OCR[PyMuPDF + Tesseract]
    W --> E[BGE-M3]
    E --> PG
    API --> S[Búsqueda híbrida]
    S --> PG
    S --> L[LLM local]
~~~

## Servicios

| Servicio | Función | Puerto |
|---|---|---:|
| postgres | Base de datos y vectores | 5432 |
| redis | Cola y resultados de Celery | 6379 |
| minio | Archivos PDF originales | 9000, 9001 |
| api | FastAPI e interfaz web | 8000 |
| worker-ingestion | OCR, chunks, embeddings y entidades | interno |
| worker-embeddings | Tareas de embeddings | interno |
| prometheus | Métricas | 9090 |
| grafana | Panel de métricas | 3000 |

## Estructura

~~~text
RAG/
├── frontend/index.html       # Interfaz web
├── src/
│   ├── main.py               # FastAPI y ruta principal
│   ├── config.py             # Configuración
│   ├── api/v1/               # Endpoints
│   ├── services/             # OCR, ingestión, chunks, embeddings y RAG
│   ├── workers/              # Tareas Celery
│   ├── models/               # Modelos SQLAlchemy
│   └── schemas/              # Esquemas Pydantic
├── alembic/                  # Migraciones
├── scripts/                  # Utilidades
├── models/                   # Modelos locales
├── tests/                    # Pruebas
├── docker-compose.yml
├── docker-compose.test.yml
├── Dockerfile
├── .env.example
└── pyproject.toml
~~~

## Requisitos

- Windows 10/11.
- Docker Desktop con motor Linux iniciado.
- Docker Compose v2.
- WSL2.
- Al menos 8 GB de RAM.
- Python 3.11 para pruebas o desarrollo fuera de Docker.
- Modelos locales en models/ cuando se use el LLM.

Comprobar:

~~~powershell
docker info
docker compose version
wsl --status
~~~

## Configuración

~~~powershell
Copy-Item .env.example .env
notepad .env
~~~

Configurar claves largas y aleatorias:

~~~dotenv
POSTGRES_PASSWORD=una-clave-larga-y-aleatoria
MINIO_PASSWORD=otra-clave-larga-y-aleatoria
SECRET_KEY=una-clave-de-al-menos-32-caracteres
GRAFANA_ADMIN_PASSWORD=otra-clave-larga-y-aleatoria
~~~

Para desarrollo local:

~~~dotenv
AUTH_REQUIRED=false
DEV_AUTH_BYPASS=true
~~~

No usar esa configuración en producción.

## Descargar los modelos

El **código fuente no incluye los modelos** (son grandes; GitHub no admite archivos
de ese tamaño). Se obtienen una sola vez:

~~~powershell
# LLM local (Qwen2.5-1.5B-Instruct, ~1 GB) -> models/
pip install huggingface-hub
python scripts/download_models.py
~~~

- Los **embeddings** (MiniLM) y el **reranker** (`bge-reranker-base`) se descargan
  solos la primera vez que arranca la API (requiere internet una vez; quedan en
  `hf_cache/`).
- Los modelos de **OCR** (RapidOCR PP-OCRv5) van dentro de la imagen Docker.

> Sin el GGUF en `models/`, la interfaz arranca pero el asistente no genera respuestas.

## Arranque

~~~powershell
docker compose up -d --build
~~~

URLs:

- Interfaz: http://localhost:8000/
- Tablero BSC (admin): http://localhost:8000/bsc.html
- Swagger: http://localhost:8000/docs
- MinIO: http://localhost:9001/

Login por defecto: **`admin` / `admin123`**.

## Uso de la interfaz

1. Abrir http://localhost:8000/.
2. Comprobar DB, Redis y MinIO.
3. Seleccionar un PDF y pulsar Subir documento.
4. Esperar al estado completed.
5. Hacer clic en el nombre para abrir el visor.
6. Revisar Lo importante del PDF.
7. Escribir una pregunta y pulsar Consultar.
8. Eliminar el documento antes de subir otro.

Solo se permite un PDF. El visor muestra sus páginas con desplazamiento interno.

## Estados

| Estado | Significado |
|---|---|
| processing | Documento recibido. |
| ocr | Extracción de texto u OCR. |
| chunking | División en fragmentos. |
| embedding | Generación de vectores. |
| completed | Disponible para consultas. |
| failed | Procesamiento con error. |

## Flujo de ingesta

1. Validación del PDF y cálculo de SHA-256.
2. Rechazo de duplicados.
3. Guardado del original en MinIO.
4. Extracción con PyMuPDF.
5. OCR de páginas escaneadas con Tesseract en español e inglés.
6. Anonimización.
7. Chunking semántico.
8. Embeddings BGE-M3.
9. Persistencia de chunks, vectores y entidades.
10. Estado completed.

El worker de ingestión utiliza una sola tarea simultánea para evitar que dos reprocesamientos compitan por memoria.

## API

### Salud

~~~http
GET /api/v1/health
~~~

~~~powershell
Invoke-RestMethod http://localhost:8000/api/v1/health
~~~

Debe responder con status ok y dependencias db, redis y minio verdaderas.

### Documentos

~~~http
POST   /api/v1/documents
GET    /api/v1/documents?size=10
GET    /api/v1/documents/{id}
GET    /api/v1/documents/{id}/file
GET    /api/v1/documents/{id}/pages/{page}
POST   /api/v1/documents/{id}/reprocess
DELETE /api/v1/documents/{id}
~~~

La subida usa multipart/form-data con el campo file y devuelve 202 Accepted.

~~~powershell
$id="ID_DEL_DOCUMENTO"
Invoke-RestMethod -Method Post -Uri "http://localhost:8000/api/v1/documents/$id/reprocess"
~~~

No reprocesar mientras el documento esté processing, ocr, chunking o embedding.

### Consulta RAG

~~~http
POST /api/v1/query
~~~

~~~json
{
  "query": "¿Qué medicamentos aparecen en el documento?",
  "top_k": 5,
  "stream": false
}
~~~

Con stream false devuelve answer, citations, chunks_used y latency_ms. Si no hay evidencia suficiente, no se inventa información.

### Búsqueda y entidades

~~~http
POST /api/v1/query/search
POST /api/v1/search/vector
POST /api/v1/search/keyword
POST /api/v1/search/hybrid
GET  /api/v1/entities
GET  /api/v1/entities/stats
~~~

## Modelos

BGE-M3 se carga durante el primer procesamiento y puede tardar. Produce embeddings de 1024 dimensiones.

Configurar el LLM local:

~~~dotenv
MEDALPACA_MODEL_PATH=/models/medalpaca-7b-q4_k_m.gguf
~~~

Los modelos grandes no deben incluirse en Git. Marker, Surya, PaddleOCR y spaCy son opcionales; la ruta principal usa PyMuPDF y Tesseract.

## Pruebas

~~~powershell
python -m pytest -m "not integration" -q
python -m compileall -q src
python -m ruff check src tests
python -m mypy src
~~~

Pruebas Docker:

~~~powershell
docker compose -f docker-compose.test.yml up --build --abort-on-container-exit --exit-code-from test
~~~

## Solución de problemas

### Docker no responde

Si aparece dockerDesktopLinuxEngine, abrir Docker Desktop y esperar el motor Linux.

~~~powershell
docker info
docker compose ps
~~~

### Failed to fetch

~~~powershell
Invoke-RestMethod http://localhost:8000/api/v1/health
docker compose ps
~~~

Si responde correctamente, recargar con Ctrl+F5.

### El documento queda en ocr

~~~powershell
docker compose logs --tail=200 worker-ingestion
~~~

No pulsar varias veces Reprocesar OCR. Después del OCR puede tardar la fase embedding.

### El PDF no aparece en el visor

~~~powershell
$id="ID_DEL_DOCUMENTO"
curl.exe -I "http://localhost:8000/api/v1/documents/$id/pages/1"
~~~

Debe responder 200 OK y content-type image/png. Recargar con Ctrl+F5.

### El resumen muestra texto extraño

Los formularios escaneados pueden producir ruido OCR. El panel muestra medicamentos, procedimientos y datos institucionales cuando son reconocibles, pero no inventa nombres, DNI, fechas ni diagnósticos. Verificar esos datos en el visor.

### Los cambios no aparecen

~~~powershell
docker compose restart api worker-ingestion
~~~

Si se cambió el comando de un servicio:

~~~powershell
docker compose up -d --force-recreate api worker-ingestion
~~~

## Mantenimiento

~~~powershell
docker compose logs -f api
docker compose logs -f worker-ingestion
docker compose logs -f worker-embeddings
docker compose ps
docker compose down
~~~

Para borrar también todos los datos de los volúmenes:

~~~powershell
docker compose down -v
~~~

Este último comando elimina la base de datos, Redis, MinIO y Grafana almacenados localmente.

## Seguridad y limitaciones

- No guardar contraseñas reales en Git.
- No exponer PostgreSQL, Redis o MinIO directamente a Internet.
- Usar autenticación en producción.
- La anonimización ocurre antes del chunking y los embeddings.
- El OCR no garantiza la lectura de manuscritos o imágenes borrosas.
- El sistema es apoyo documental y no sustituye a un profesional de salud.
