

def build_context(chunks: list[dict], max_tokens: int = 2500) -> str:
    if not chunks:
        return "No se encontró contexto relevante."

    parts = []
    current_tokens = 0.0
    estimated_tokens_per_char = 0.25

    for i, chunk in enumerate(chunks):
        content = chunk.get("content", "")
        page_info = ""
        if chunk.get("page_numbers"):
            pages = chunk["page_numbers"]
            page_info = f" [Página {', '.join(map(str, pages))}]"

        part = f"[Fuente {i + 1}]{page_info}\n{content}"
        part_tokens = len(part) * estimated_tokens_per_char

        if current_tokens + part_tokens > max_tokens:
            remaining = max_tokens - current_tokens
            chars_to_take = int(remaining / estimated_tokens_per_char)
            if chars_to_take > 50:
                part = part[:chars_to_take] + "..."
                parts.append(part)
            break

        parts.append(part)
        current_tokens += part_tokens

    return "\n\n---\n\n".join(parts)


def build_citations(chunks: list[dict]) -> list[dict]:
    citations = []
    for i, chunk in enumerate(chunks):
        citation = {
            "index": i + 1,
            "chunk_id": chunk.get("id", ""),
            "document_id": chunk.get("document_id", ""),
            "page_numbers": chunk.get("page_numbers", []),
            "score": chunk.get("score", 0.0),
            "content_preview": chunk.get("content", "")[:200],
        }
        citations.append(citation)
    return citations
