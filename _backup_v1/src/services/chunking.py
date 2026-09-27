"""Structure-aware chunking for medical documents."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

import tiktoken

from src.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


@dataclass
class Chunk:
    content: str
    chunk_index: int
    document_id: str = ""
    token_count: int = 0
    char_start: int = 0
    char_end: int = 0
    page_numbers: list[int] = field(default_factory=list)
    chunk_metadata: dict = field(default_factory=dict)


def _count_tokens(text: str) -> int:
    try:
        return len(tiktoken.get_encoding("cl100k_base").encode(text))
    except Exception:
        return max(1, len(text) // 4)


def _token_tail(text: str, token_count: int) -> str:
    if token_count <= 0:
        return ""
    try:
        encoder = tiktoken.get_encoding("cl100k_base")
        return encoder.decode(encoder.encode(text)[-token_count:])
    except Exception:
        return " ".join(text.split()[-token_count:])


def _section_for(text: str) -> str | None:
    for line in text.splitlines():
        candidate = line.strip().lstrip("#").strip()
        if candidate and (line.lstrip().startswith("#") or candidate.endswith(":")):
            return candidate[:200]
    return None


def _paragraphs(text: str) -> list[tuple[str, int, int]]:
    """Return non-empty paragraphs while retaining source offsets."""
    result = []
    for match in re.finditer(r"\S(?:.*?\S)?(?=\n\s*\n|\Z)", text, re.DOTALL):
        value = match.group().strip()
        if value:
            start = text.find(value, match.start(), match.end())
            result.append((value, start, start + len(value)))
    return result


def _pages_for_range(start: int, end: int, boundaries: list[tuple[int, int, int]]) -> list[int]:
    return [page for page, page_start, page_end in boundaries if start < page_end and end > page_start]


def semantic_medical_chunking(
    text: str,
    document_id: str = "",
    max_tokens: int | None = None,
    overlap_tokens: int | None = None,
    page_boundaries: list[tuple[int, int, int]] | None = None,
) -> list[Chunk]:
    """Pack paragraphs without splitting headings, lists or tables when possible."""
    max_tokens = max_tokens or settings.CHUNK_MAX_TOKENS
    overlap_tokens = overlap_tokens or settings.CHUNK_OVERLAP_TOKENS
    if max_tokens <= overlap_tokens:
        raise ValueError("max_tokens debe ser mayor que overlap_tokens")
    if not text.strip():
        return []

    paragraphs = _paragraphs(text)
    if not paragraphs:
        paragraphs = [(text.strip(), 0, len(text.strip()))]
    chunks: list[tuple[str, int, int]] = []
    current: list[tuple[str, int, int]] = []
    current_tokens = 0

    def flush() -> None:
        nonlocal current, current_tokens
        if current:
            chunks.append(
                (
                    "\n\n".join(item[0] for item in current),
                    current[0][1],
                    current[-1][2],
                )
            )
            current = []
            current_tokens = 0

    for paragraph, start, end in paragraphs:
        paragraph_tokens = _count_tokens(paragraph)
        if paragraph_tokens > max_tokens:
            flush()
            words = paragraph.split()
            for offset in range(0, len(words), max(1, max_tokens - overlap_tokens)):
                part = " ".join(words[offset : offset + max_tokens])
                part_start = text.find(part[:40], start)
                chunks.append((part, part_start if part_start >= 0 else start, end))
            continue
        if current and current_tokens + paragraph_tokens > max_tokens:
            flush()
        current.append((paragraph, start, end))
        current_tokens += paragraph_tokens
    flush()

    result: list[Chunk] = []
    for index, (content, start, end) in enumerate(chunks):
        if index and overlap_tokens:
            overlap = _token_tail(chunks[index - 1][0], overlap_tokens)
            if overlap:
                content = f"{overlap}\n\n{content}"
        result.append(
            Chunk(
                content=content,
                chunk_index=index,
                document_id=document_id,
                token_count=_count_tokens(content),
                char_start=start,
                char_end=end,
                page_numbers=_pages_for_range(start, end, page_boundaries or []),
                chunk_metadata={
                    "medical_domain": True,
                    "section": _section_for(content),
                    "source_type": "text_or_ocr",
                },
            )
        )
    logger.info("Created %d chunks for document %s", len(result), document_id)
    return result
