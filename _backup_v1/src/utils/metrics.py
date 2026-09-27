"""Service-level Prometheus metrics.

HTTP request metrics (REQUEST_COUNT, REQUEST_LATENCY) are defined directly in
main.py via the middleware.  The metrics below are declared here for future
instrumentation but are NOT currently incremented anywhere in the codebase.
"""

from prometheus_client import Counter, Gauge, Histogram

# Ingestion metrics
DOCS_INGESTED = Counter("docs_ingested_total", "Total documents ingested", ["status"])
DOCS_INGESTED_BYTES = Counter("docs_ingested_bytes_total", "Total bytes ingested")
OCR_DURATION = Histogram("ocr_duration_seconds", "OCR processing duration", ["engine"])
CHUNK_COUNT = Histogram("chunks_per_document", "Chunks generated per document")
EMBEDDING_DURATION = Histogram("embedding_duration_seconds", "Embedding generation duration")

# Retrieval metrics
SEARCH_LATENCY = Histogram("search_latency_seconds", "Search query latency", ["search_type"])
RERANK_LATENCY = Histogram("rerank_latency_seconds", "Reranking latency")
LLM_LATENCY = Histogram("llm_latency_seconds", "LLM inference latency")
CHUNKS_RETRIEVED = Histogram("chunks_retrieved", "Chunks retrieved per query", ["search_type"])

# System metrics
ACTIVE_DOCUMENTS = Gauge("active_documents", "Number of processed documents")
ACTIVE_CHUNKS = Gauge("active_chunks", "Number of chunks with embeddings")
