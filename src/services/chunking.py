from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from typing import Any

from src.config import get_settings

settings = get_settings()


def _count_tokens(text: str) -> int:
    return max(1, len(text) // 3)


@dataclass
class Chunk:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""
    page_numbers: list[int] = field(default_factory=list)
    chunk_index: int = 0
    token_count: int = 0
    bbox: dict | None = None
    chunk_metadata: dict = field(default_factory=dict)


def _split_sentences(text: str) -> list[str]:
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'(?<=[.!?])\s+(?=[A-ZÁÉÍÓÚ])', '\n', text)
    return [s.strip() for s in text.split('\n') if s.strip()]


def chunk_text(text: str, page_numbers: list[int], chunk_idx_start: int = 0) -> list[Chunk]:
    max_tokens = settings.CHUNK_SIZE
    overlap = settings.CHUNK_OVERLAP
    chunks = []

    sentences = _split_sentences(text)
    if not sentences:
        return []

    current_tokens = 0
    current_sentences: list[str] = []
    current_pages: set[int] = set()
    sent_idx = 0

    for sent in sentences:
        sent_tokens = _count_tokens(sent)

        if current_tokens + sent_tokens > max_tokens and current_sentences:
            content = " ".join(current_sentences)
            chunks.append(Chunk(
                content=content,
                page_numbers=sorted(current_pages) if current_pages else page_numbers[:1],
                chunk_index=chunk_idx_start + len(chunks),
                token_count=current_tokens,
            ))

            overlap_tokens = 0
            overlap_sents = []
            for s in reversed(current_sentences):
                st = _count_tokens(s)
                if overlap_tokens + st > overlap:
                    break
                overlap_sents.insert(0, s)
                overlap_tokens += st
            current_sentences = overlap_sents
            current_tokens = overlap_tokens
            current_pages = set()

        current_sentences.append(sent)
        current_tokens += sent_tokens
        if page_numbers:
            current_pages.add(page_numbers[min(sent_idx, len(page_numbers)-1)])
        sent_idx += 1

    if current_sentences:
        content = " ".join(current_sentences)
        chunks.append(Chunk(
            content=content,
            page_numbers=sorted(current_pages) if current_pages else page_numbers[:1],
            chunk_index=chunk_idx_start + len(chunks),
            token_count=current_tokens,
        ))

    return chunks


def chunk_document(pages: dict[int, str]) -> list[Chunk]:
    all_chunks: list[Chunk] = []
    current_text = ""
    current_pages: list[int] = []
    current_token_count = 0

    sorted_pages = sorted(pages.keys())

    for page_num in sorted_pages:
        text = pages[page_num].strip()
        if not text:
            continue

        page_tokens = _count_tokens(text)

        if current_token_count + page_tokens > settings.CHUNK_SIZE * 2 and current_text:
            page_chunks = chunk_text(current_text, current_pages, len(all_chunks))
            all_chunks.extend(page_chunks)
            current_text = ""
            current_pages = []
            current_token_count = 0

        current_text += ("\n\n" if current_text else "") + text
        current_pages.append(page_num)
        current_token_count += page_tokens

    if current_text:
        page_chunks = chunk_text(current_text, current_pages, len(all_chunks))
        all_chunks.extend(page_chunks)

    for i, chunk in enumerate(all_chunks):
        chunk.chunk_index = i

    return all_chunks
