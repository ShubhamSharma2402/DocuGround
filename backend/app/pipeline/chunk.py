"""Stage 2 — Chunk: split page text into overlapping chunks."""

from __future__ import annotations

import re
from dataclasses import dataclass

import structlog

from app.core.config import get_settings
from app.pipeline.parse import PageContent

log = structlog.get_logger(__name__)


@dataclass
class TextChunk:
    doc_id: str
    user_id: str
    page: int
    chunk_index: int
    text: str


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def _trim_to_overlap(text: str, overlap: int) -> str:
    """Return the last *overlap* characters of *text* aligned to a word boundary."""
    if len(text) <= overlap:
        return text
    candidate = text[-overlap:]
    space_idx = candidate.find(" ")
    if space_idx != -1:
        return candidate[space_idx:].strip()
    return candidate


def chunk_pages(
    pages: list[PageContent],
    doc_id: str,
    user_id: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[TextChunk]:
    """Produce overlapping text chunks from parsed pages."""
    settings = get_settings()
    size = chunk_size or settings.chunk_size
    overlap = chunk_overlap or settings.chunk_overlap

    if overlap >= size:
        raise ValueError(f"chunk_overlap ({overlap}) must be less than chunk_size ({size})")

    chunks: list[TextChunk] = []
    chunk_index = 0

    for page in pages:
        text = page.text.strip()
        if not text:
            continue

        sentences = _split_sentences(text)
        if not sentences:
            continue

        buffer = ""
        i = 0
        while i < len(sentences):
            sentence = sentences[i]

            if len(sentence) > size:
                words = sentence.split()
                for word in words:
                    if len(buffer) + len(word) + 1 > size and buffer:
                        chunks.append(TextChunk(
                            doc_id=doc_id, user_id=user_id, page=page.page_num,
                            chunk_index=chunk_index, text=buffer.strip(),
                        ))
                        chunk_index += 1
                        buffer = _trim_to_overlap(buffer, overlap) + " " + word
                    else:
                        buffer = (buffer + " " + word).strip()
                i += 1
                continue

            candidate = (buffer + " " + sentence).strip() if buffer else sentence
            if len(candidate) <= size:
                buffer = candidate
                i += 1
            else:
                if buffer:
                    chunks.append(TextChunk(
                        doc_id=doc_id, user_id=user_id, page=page.page_num,
                        chunk_index=chunk_index, text=buffer.strip(),
                    ))
                    chunk_index += 1
                    buffer = _trim_to_overlap(buffer, overlap)
                else:
                    buffer = sentence
                    i += 1

        if buffer.strip():
            chunks.append(TextChunk(
                doc_id=doc_id, user_id=user_id, page=page.page_num,
                chunk_index=chunk_index, text=buffer.strip(),
            ))
            chunk_index += 1

    log.info("Chunking complete", doc_id=doc_id, chunk_count=len(chunks))
    return chunks
