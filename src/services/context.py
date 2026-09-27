from __future__ import annotations

from src.config import get_settings
from src.services.retrieval import RetrievalResult

settings = get_settings()


def _estimate_tokens(text: str) -> int:
    return len(text) // 3


def build_context(results: list[RetrievalResult]) -> str:
    parts = []
    total_tokens = 0
    max_tokens = settings.CONTEXT_MAX_CHARS // 3
    for r in results:
        chunk_text = f"[Documento: {r.document_name} | Paginas: {r.page_numbers}]\n{r.content}"
        chunk_tokens = _estimate_tokens(chunk_text)
        if total_tokens + chunk_tokens > max_tokens:
            remaining = max_tokens - total_tokens
            if remaining > 30:
                chunk_text = chunk_text[:remaining * 3] + "..."
                parts.append(chunk_text)
            break
        parts.append(chunk_text)
        total_tokens += chunk_tokens
    return "\n\n---\n\n".join(parts)
