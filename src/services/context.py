from __future__ import annotations

from src.config import get_settings
from src.services.retrieval import RetrievalResult

settings = get_settings()


def _estimate_tokens(text: str) -> int:
    # Estimación conservadora: español/OCR real rinde ~2.2 chars/token
    # (len/3 subestimaba y provocaba overflow del ctx del LLM).
    return int(len(text) / 2.2)


def build_context(results: list[RetrievalResult]) -> str:
    parts = []
    total_tokens = 0
    max_tokens = settings.CONTEXT_MAX_CHARS // 3
    for i, r in enumerate(results):
        # WP8: inyectar la edad del chunk (si se extrajo durante la ingesta)
        # para que el LLM pueda responder consultas de edad (CP-006).
        edad = (getattr(r, "chunk_metadata", None) or {}).get("edad")
        edad_str = f" | Edad: {edad} anios" if edad is not None else ""
        chunk_text = f"[{i + 1}] Documento: {r.document_name} | Paginas: {r.page_numbers}{edad_str}\n{r.content}"
        chunk_tokens = _estimate_tokens(chunk_text)
        if total_tokens + chunk_tokens > max_tokens:
            remaining = max_tokens - total_tokens
            if remaining > 30:
                chunk_text = chunk_text[:int(remaining * 2.2)] + "..."
                parts.append(chunk_text)
            break
        parts.append(chunk_text)
        total_tokens += chunk_tokens
    return "\n\n---\n\n".join(parts)
