"""Public service APIs."""

from src.services.anonymization import anonymize_text
from src.services.chunking import semantic_medical_chunking
from src.services.context import build_citations, build_context
from src.services.embeddings import encode_single, encode_texts
from src.services.ingestion import (
    delete_document,
    generate_embeddings_batch,
    get_document_with_chunks,
    get_page_with_ocr_overlay,
    marker_extract,
    process_document_full,
    save_chunks_with_embeddings,
    search_documents,
    validate_and_deduplicate,
)
from src.services.llm import extract_structured, generate_answer_stream
from src.services.ner import (
    extract_medical_entities,
    get_entities_stats,
    list_medical_entities,
)
from src.services.ocr import ocr_document, ocr_page
from src.services.retrieval import (
    hybrid_search,
    keyword_search,
    reciprocal_rank_fusion,
    rerank,
    vector_search,
)

__all__ = [
    "semantic_medical_chunking",
    "anonymize_text",
    "build_context",
    "build_citations",
    "encode_single",
    "encode_texts",
    "delete_document",
    "generate_embeddings_batch",
    "get_document_with_chunks",
    "get_page_with_ocr_overlay",
    "marker_extract",
    "process_document_full",
    "save_chunks_with_embeddings",
    "search_documents",
    "validate_and_deduplicate",
    "extract_structured",
    "generate_answer_stream",
    "extract_medical_entities",
    "get_entities_stats",
    "list_medical_entities",
    "ocr_document",
    "ocr_page",
    "hybrid_search",
    "keyword_search",
    "reciprocal_rank_fusion",
    "rerank",
    "vector_search",
]
