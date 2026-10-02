# HANDOFF — RAG INTELLIGENCE BSC

> Documento de traspaso. Léelo entero antes de tocar el código.
> **Última actualización:** 2026-10-01 · **Estado:** proyecto APROBADO (93.2%), listo para demo/entrega.

---

## 1. Qué es y cómo se aprueba

Sistema **RAG 100% local** para documentos médicos en PDF: subes PDFs (incluso **escaneados o manuscritos**, vía OCR), preguntas en lenguaje natural y el sistema responde **citando la evidencia** (documento, página, fragmento). Incluye un módulo **BSC (Balanced Scorecard)** que mide la calidad del sistema con KPIs, semáforos y reportes.

Todo corre en la máquina local: **sin APIs pagadas, sin nube, sin tokens**. Esto es un **criterio crítico de la rúbrica** — no lo rompas.

- **Repo:** `C:\Users\juang\Downloads\RAG`
- **Fuente de verdad de los requisitos:** `documentos/` (la spec del profesor, la rúbrica y los casos de prueba). Ante cualquier duda, manda ese texto, no la intuición.
- **Umbral de aprobación:** **≥90% global y ningún criterio crítico incumplido.**

---

## 2. Estado actual

| Métrica | Valor |
|---|---|
| **Score global ponderado** | **93.2%** (≥90% ✓) |
| **Críticos de peso 4** | todos 4/4 con evidencia E2E |
| **Único crítico parcial** | RAG-004 (Login) 3/4 — no falla, la spec no exige OAuth/2FA |
| **Casos de prueba oficiales** | **CP-001…CP-010: 10/10 PASS E2E real** |
| **Tests host** | 84 verdes en los módulos núcleo; ~5 requieren infra del contenedor |
| **Velocidad consulta (warm)** | **~59 s** (retrieval 9.3 s + rerank 7.3 s + LLM 41.2 s) |

Evidencia: `docs/veredicto_final.md` (score), `documentos/QA_MATRIZ_CP.csv` (los 10 CP con evidencia).

---

## 3. Stack REAL (varios docs mienten — confía en el código)

| Capa | Tecnología real |
|---|---|
| Backend | Python 3.11, FastAPI, SQLAlchemy 2 (async) + asyncpg |
| BD | PostgreSQL 16 + **pgvector** (HNSW) + **pg_trgm**; migraciones Alembic |
| Cola | Celery + Redis |
| Embeddings | **BGE-M3 (1024-d)** — `EMBED_MODEL_NAME` |
| Re-ranking | **bge-reranker-base** (cross-encoder CPU) |
| LLM | **Qwen2.5-1.5B-Instruct GGUF** vía llama-cpp-python — `LLM_MODEL_PATH` |
| OCR | **PyMuPDF** (texto nativo) → **Tesseract** (`OCR_ENGINE=tesseract`); hay adaptador Paddle sin usar |
| Frontend | SPA de **un solo archivo** `frontend/index.html` + `frontend/bsc.html` (JS vanilla, visor PDF.js, sin build) |
| Deploy | Docker Compose: `api`, `worker`, `worker-extract`, `postgres`, `redis`, `minio` |

**Correcciones a la documentación vieja:**
- `README.md` está **desactualizado** (habla de Surya/Marker, MedAlpaca, Prometheus, K8s… que no existen). No lo uses como referencia.
- `AGENTS.md` está casi al día, pero su sección "Cómo funciona por dentro" aún dice *"MiniLM multilingüe 384-d"* y *"TinyLlama"*: lo real es **BGE-M3 1024-d + Qwen2.5-1.5B**.
- `docs/RESUMEN_HANDOFF.md` era del 2026-09-26 y decía "NO APROBADO, 29-45%": **es historia antigua**; se eliminó por confuso.

---

## 4. Cómo levantarlo, probarlo y presentarlo

### Levantar todo
```bash
cd C:/Users/juang/Downloads/RAG
docker compose up -d          # primera vez o tras cambiar deps/código de imagen: --build
docker ps                     # api, worker, postgres, redis, minio deben estar Up
curl -s localhost:8000/api/v1/health   # {"status":"healthy",...}
```
La **API sirve la interfaz web** en <http://localhost:8000>. Login demo: usuario `admin` (el frontend lo precarga), contraseña en `.env` / `scripts/seed_admin.py`.

### Tests
```bash
python -m pytest              # desde la raíz; corre SIN infraestructura externa
```
- `tests/conftest.py` degrada Postgres/Redis/modelos si no están → la suite corre en el host.
- **Convención TDD:** primero el test en rojo, luego el código.
- **Aviso:** `tests/test_smoke_pipeline.py` **se cuelga en host** (intenta cargar tesseract/modelos reales). No es una regresión; evítalo o córrelo en el contenedor.
- **Aviso:** `tests/test_fase2_rbac_auditoria.py` tarda **mucho** por el *rate-limiter* de login (~30 s por login repetido). Usa timeout amplio.

### Antes de una demo (importante)
El **primer** `/query` paga la carga de modelos (~90-150 s en frío): embeddings ~5-13 s, reranker ~17-27 s, LLM ~30-48 s. **Haz una consulta de calentamiento** antes de presentar; a partir de ahí, ~59 s por consulta.

### Parar
```bash
docker compose stop           # o `docker compose down` (los datos viven en volúmenes)
```

---

## 5. Arquitectura: los dos flujos

### A) Ingesta (subir un PDF)
1. La API guarda el archivo, detecta duplicados por hash y **encola** el trabajo (no bloquea la subida).
2. El worker Celery (`--pool=solo`) extrae texto con PyMuPDF. Páginas vacías/escaneadas → **OCR Tesseract** (con coordenadas por palabra para el resaltado del visor).
3. Corrige rotación (OSD; si falla, compara 0°/180°).
4. Trocea por oraciones (*chunks*), calcula embeddings BGE-M3 1024-d y los indexa en pgvector.
5. Guarda `ocr_s` / `embed_s` / `total_s` en el documento (visibles como "Procesado en X s").

### B) Consulta
1. Expande la pregunta con **sinónimos médicos**.
2. **Búsqueda híbrida:** semántica (pgvector) + léxica (tsquery/pg_trgm), fusionadas con **RRF**.
3. **Re-ranking** con cross-encoder → `rerank_score` (sigmoid) → base del **umbral de evidencia**.
4. Arma el contexto con fuentes numeradas `[n]` y lo pasa al **LLM local**.
5. Respuesta con **citas** (doc + página + chunk + snippet). Si no hay evidencia → **se abstiene** (no inventa).

---

## 6. Mapa del repo

```
src/
  main.py            # app FastAPI + preload de embeddings + health
  config.py          # TODA la configuración por env (Settings). No hardcodees.
  database.py        # modelos SQLAlchemy (Document, Chunk, Keyword, AuditLog, Kpi…)
  api/v1/            # auth, documents, query, chat, keywords, synonyms, audit, bsc, users
  api/deps.py        # seguridad/JWT
  core/permissions.py# matriz RBAC
  services/          # ocr, chunking, embeddings, retrieval, reranker, context, llm,
                     # anonymizer, synonyms, bsc, bsc_seed, audit
  workers/           # celery_app, ingestion_tasks, kpi_tasks (beat 06:00)
frontend/
  index.html         # SPA principal (visor PDF.js, chat, documentos, descargas)
  bsc.html           # tablero ejecutivo BSC
alembic/versions/    # migraciones (…última: chat 007, BGE-M3 1024-d 002)
tests/               # pytest (host, sin infra)
scripts/             # seed_admin, reindex, download_models, bench_*, generate_synthetic_dataset
docs/                # veredicto_final, bench_velocidad, despliegue_free_tier, análisis de brechas
planes/              # PLAN-001…011 (planes de implementación)
documentos/          # spec del profesor + rúbrica + casos de prueba (FUENTE DE VERDAD)
datossinteticos/     # 3 PDFs demo generados
models/              # GGUF del LLM (solo qwen2.5-1.5b, 1.1 GB) — no se commitea
hf_cache/            # caché HuggingFace — no se commitea
uploads/             # originales subidos — no se commitea
```

---

## 7. Decisiones y tuning (respetar)

- **`docker-compose.yml` es la fuente del runtime**, no el README.
- **`PERFIL=rapido`** es el default (recorta contexto y candidatos de rerank para velocidad). `PERFIL=calidad` restaura los máximos.
- **API `mem_limit: 4.5g`** — BGE-M3 (2.2 GB) + reranker (0.6 GB) + Qwen (1 GB) deben caber juntos durante una consulta. Con 3 GB hacía *thrashing* (consultas de 90-110 s). No lo bajes.
- **Worker `--pool=solo`** — evita que los *forks* de Celery carguen BGE-M3 y revienten el cgroup (OOM SIGKILL). No vuelvas a `prefork`.
- **El worker NO precarga el LLM** (`celery_app.py`): la cola `celery` (tareas periódicas) no lo usa. La API sí lo carga para las consultas.
- **Reranker `max_length=128`**: el cross-encoder en CPU era el 2º mayor costo. Subirlo encarece cada consulta.
- **Recordatorio de citar en el primer prompt** (`llm.py`): sin él, el modelo olvida `[n]` y se dispara una **segunda generación** completa (+26 s/consulta).
- **`.wslconfig`** (en `C:\Users\juang\`) tiene `memory=7GB` para Docker Desktop. La máquina tiene 8 GB y no hay GPU.
- **Todo por variables de entorno** (`src/config.py`), nunca hardcodeado.

---

## 8. Deuda y MINAS conocidas (no las empeores)

**Seguridad (deuda documentada, no bloquea la nota):**
- Contraseñas con **SHA-256 sin salt** (pendiente bcrypt/argon2).
- El login del frontend **precarga el usuario** `admin`.
- `AUTH_REQUIRED` existe como interruptor (hoy `true`). No construyas nada nuevo encima de estos huecos sin un plan.

**Trampas de código (¡cuidado!):**
- **`Document.metadata_` es un JSONB pesado** (incluye `word_boxes`). El listado y el detalle hacen `defer(Document.metadata_)`. **NUNCA accedas a `doc.metadata_` en esos endpoints** → lanza `MissingGreenlet` (500). Para los `timings` se selecciona solo `Document.metadata_["timings"]`. (Este bug ya se introdujo y se corrigió una vez.)
- **La dimensión del vector es 1024 (BGE-M3).** Cambiar a MiniLM (384-d) **rompe la columna** de pgvector: requiere migración + `scripts/reindex.py`.
- **Defaults de embeddings:** el default debe ser `BAAI/bge-m3` en `config.py` Y en el compose del worker. Si se desalinean, el worker indexa con una dimensión y la API consulta con otra.
- **tesseract/pytesseract viven en el contenedor.** El `Dockerfile` los declara (líneas 21, 34), pero si el contenedor se recrea desde una imagen vieja puede haberlos perdido → reinstala o haz `docker compose up --build`.
- **La API no hace OCR** (lo delega a Celery). No necesita tesseract.
- **Rate-limiter en el login**: ralentiza los tests de RBAC. No es un fallo.

**Deuda de rúbrica (parcial, no crítica):**
- RAG-004 (Login) 3/4 · RAG-022 (filtros por categoría, parcial) · RAG-055 (benchmark con >1 documento).

---

## 9. Planes: estado

Los planes están en `planes/PLAN-NNN-*.md` (convención y formato en `planes/README.md`). Los commits siguen el estilo `RAG-0xx parcial/completo: resumen`.

| Plan | Tema | Estado |
|---|---|---|
| PLAN-001 | Migración de stack IA/OCR (BGE-M3, reranker, Qwen) | ✅ terminado |
| PLAN-002 | Fase 1: seguridad/core | ✅ |
| PLAN-003 | Velocidad/eficiencia | ✅ |
| PLAN-004 | Fase 2: RBAC + auditoría | ✅ |
| PLAN-005 | BSC + dashboard | ✅ |
| PLAN-006 | Cierre de críticos de rúbrica | ✅ |
| PLAN-007 | Pulido final + evidencia (veredicto 93.2%) | ✅ |
| PLAN-008 | Memoria técnica (documento) | ✅ |
| PLAN-009 | OCR manuscritos + rotación | ⚠️ **Fase A hecha; Fase B (PaddleOCR) PENDIENTE** |
| PLAN-010 | Guía de mejoras de aula (WP1–WP11) | ✅ 10/11 (falta WP3-FaseB) |
| PLAN-011 | Reorganización de la navegación del frontend | ❌ **listo para implementar, sin hacer** |

---

## 10. Pendientes concretos (si continúas el proyecto)

1. **PLAN-011 — navegación del frontend** (el más pequeño y visible): el navbar mezcla *destinos* con *acciones* ("Historial"/"Bitácora"/"Descargar todo" disparan descargas a ciegas), y el Dashboard tiene una "Búsqueda Inteligente" deshabilitada (UI muerta). El plan propone el rediseño completo.
2. **PLAN-009 Fase B — manuscritos con PaddleOCR**: instalar `paddleocr`+`paddlepaddle` solo en el worker y activar `OCR_ENGINE=auto` (cascada que solo paga Paddle en páginas manuscritas). **Riesgo:** 27-156 s/página en CPU y posible OOM en 8 GB. Está documentado por qué no se hizo aún.
3. **RAG-004 (Login) 3/4**: valorar si algo de la spec falta (OAuth/2FA no están exigidos explícitamente).
4. **Migrar contraseñas a bcrypt/argon2** y quitar la precarga de `admin` en el login.

---

## 11. Convenciones de trabajo

1. **Un plan = un entregable verificable.** Antes de codear, lee el plan y los docs que referencie.
2. **Respeta los contratos** (tests, endpoints, formatos). Si algo del plan contradice el código real, **detente y repórtalo** — no improvises arquitectura.
3. **Verifica con `python -m pytest`** y deja la suite en verde (o con exactamente los tests rojos que el plan declare).
4. **Cambia solo lo necesario** — nada de refactors ajenos al plan.
5. **Commits** con el estilo `RAG-0xx parcial/completo: resumen` (mira `git log`).
6. **Regla de fondo:** todo debe funcionar **local y sin servicios pagados**.

---

## 12. Accesos rápidos

```bash
# Estado
docker ps ; curl -s localhost:8000/api/v1/health
git log --oneline -15

# Reiniciar tras editar ./src (uvicorn/celery NO tienen --reload)
docker restart rag-api-1 rag-worker-1

# Ver logs
docker logs -f rag-worker-1        # ingesta/OCR
docker logs -f rag-api-1           # queries

# Re-indexar tras cambiar el modelo de embeddings
docker exec rag-api-1 python -m scripts.reindex

# BD directa
docker exec -it rag-postgres-1 psql -U rag_user -d rag_medical
```

- **URLs:** UI `http://localhost:8000` · BSC `http://localhost:8000/bsc.html` · API docs `http://localhost:8000/docs`
- **Postgres** expuesto en `localhost:5555` · **MinIO** consola en `localhost:9001`
