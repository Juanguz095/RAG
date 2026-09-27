# AGENTS.md — RAG INTELLIGENCE BSC

## Qué es este proyecto

Un sistema RAG (Retrieval-Augmented Generation) **100% local** para documentos médicos en PDF. Permite cargar PDFs — incluso escaneados o manuscritos, gracias al OCR — y hacerles preguntas en lenguaje natural; el sistema busca la evidencia en los documentos y responde citando las fuentes exactas (documento, página, fragmento). Todo corre en la máquina del usuario: sin APIs pagadas, sin nube, sin tokens.

Además del motor RAG, el entregable académico incluye un módulo **BSC (Balanced Scorecard)** que mide la calidad del sistema (cobertura de conocimiento, respuestas fundamentadas, etc.) con KPIs, semáforos automáticos y reportes.

Este es un proyecto académico: se evalúa contra una rúbrica (`documentos/RUBRICA.csv`, criterios RAG-001…RAG-060, de los cuales 18 son críticos) y unos casos de prueba oficiales (`documentos/CASOS_PRUEBAS.csv`, CP-001…CP-010). Para aprobar hay que llegar a ≥90% global sin incumplir ningún crítico.

## Cómo funciona por dentro

El sistema tiene dos flujos principales:

**Cargar y entender documentos.** El usuario sube un PDF por la interfaz web. La API guarda el archivo, detecta duplicados por hash y deja el procesamiento en manos de un worker de Celery (para no bloquear la subida). El worker extrae el texto con PyMuPDF; si una página viene vacía o escaneada, la pasa por Tesseract (OCR) obteniendo incluso las coordenadas de cada palabra, que luego sirven para resaltar texto en el visor. Después trocea el texto en fragmentos (*chunks*) por oraciones, calcula un vector de embeddings de 384 dimensiones con un modelo MiniLM multilingüe, y lo indexa todo en PostgreSQL con pgvector.

**Consultar.** El usuario escribe una pregunta. El sistema expande términos con sinónimos médicos, hace una búsqueda *híbrida* — semántica (por similitud de vectores) + léxica (por palabras clave, con índices de trigramas) — combina ambos resultados con fusión RRF, arma un contexto con las fuentes etiquetadas y se lo pasa a un LLM local (TinyLlama, corriendo sobre llama-cpp). La respuesta llega con las citas: documento, página, chunk y un snippet verificable. Si no hay evidencia suficiente, el sistema debe abstenerse en vez de inventar.

## Tecnologías

- **Backend:** Python, FastAPI, SQLAlchemy 2 (async) + asyncpg.
- **Base de datos:** PostgreSQL con pgvector (búsqueda vectorial, índice HNSW) y pg_trgm (búsqueda por texto). Migraciones con Alembic.
- **Procesamiento en background:** Celery + Redis.
- **IA local:** sentence-transformers (MiniLM, 384-d) para embeddings; llama-cpp-python + TinyLlama GGUF para generación de texto. Los modelos viven en `models/` (no se commitean).
- **OCR:** PyMuPDF + Tesseract.
- **Frontend:** una SPA en un solo archivo, `frontend/index.html` (JavaScript vanilla, visor PDF.js). Sin framework ni paso de build.
- **Despliegue:** Docker + docker-compose (Postgres, Redis, MinIO, API y workers).

## Recorrido del código

Todo el backend vive en `src/`: la app y el arranque en `main.py`, la configuración por variables de entorno en `config.py`, los modelos de datos en `database.py`. La API está dividida en `src/api/v1/` (autenticación, documentos, consultas, sinónimos) con las dependencias de seguridad en `src/api/deps.py`. La lógica de negocio está en `src/services/` — cada servicio hace una cosa: `ocr.py`, `chunking.py`, `embeddings.py`, `retrieval.py`, `synonyms.py`, `context.py`, `llm.py`. Las tareas asíncronas están en `src/workers/`. La interfaz web está en `frontend/`.

Los requisitos oficiales viven en `documentos/` (la spec del profesor, la rúbrica y los casos de prueba) y son la fuente de verdad: ante cualquier duda sobre qué debe hacer el sistema, manda ese texto, no la intuición. El análisis del estado actual y la hoja de ruta están en `docs/`. Los planes de trabajo para los agentes de desarrollo están en `planes/` (ver más abajo).

> **Advertencia:** el `README.md` describe un stack que no es el implementado (habla de Surya, BGE-M3, MedAlpaca, MinIO para originales…). El código real usa PyMuPDF + Tesseract, MiniLM 384-d, TinyLlama y disco local. Si algo no coincide, confía en el código.

## Cómo ejecutarlo y probarlo

- **Con Docker:** `docker-compose up --build` levanta todo (la API queda sirviendo la interfaz web).
- **Tests:** `python -m pytest` desde la raíz. La suite corre **sin infraexterna** (sin Postgres, Redis, MinIO ni modelos): `tests/conftest.py` se encarga de degradar lo que haga falta. Hay una convención TDD: primero se escriben los tests en rojo (como hace `tests/test_fase1_seguridad.py`), y luego el código los pone en verde.
- **Configuración:** todo por variables de entorno, con defaults en `src/config.py`. No hardcodees valores de configuración en el código.

## Cómo se trabaja aquí

El trabajo se organiza en **planes**: documentos en `planes/` que describen, con precisión, qué construir para que un agente de desarrollo pueda implementarlo sin tomar decisiones de arquitectura por su cuenta. Cada plan indica alcance, archivos a tocar, contratos que deben cumplirse (normalmente expresados como tests) y cómo verificar que quedó bien.

Al implementar un plan:

1. Lee el plan completo y los documentos que referencie antes de escribir código.
2. Respeta los contratos del plan (tests, endpoints, formatos). Si algo del plan es imposible o contradice el código real, detente y repórtalo en vez de improvisar.
3. Verifica con `python -m pytest` y deja la suite en verde (o con exactamente los tests rojos que el plan declare como pendientes).
4. Cambia solo lo necesario — nada de refactors ajenos al plan.
5. Los commits siguen el estilo `RAG-0xx parcial/completo: resumen` (mira `git log`).

Un par de reglas que el proyecto arrastra como deuda conocida y no hay que empeorar: las contraseñas se guardan con SHA-256 sin salt (pendiente migrar a bcrypt/argon2), el login del frontend precarga `admin123`, y `AUTH_REQUIRED` apaga la autenticación por defecto. No construyas nada nuevo encima de esos huecos sin que un plan lo cubra.

Y una regla de fondo: **todo debe funcionar local y sin servicios pagados** — es un criterio crítico de la rúbrica.
