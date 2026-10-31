"""Stage 3 — Embed: produce dense vectors for each chunk."""

from __future__ import annotations

import structlog

from app.pipeline.chunk import TextChunk
from app.services.embedder import Embedder

log = structlog.get_logger(__name__)


async def embed_chunks(chunks: list[TextChunk], embedder: Embedder) -> list[list[float]]:
    texts = [c.text for c in chunks]
    log.info("Embedding chunks", count=len(texts))
    vectors = await embedder.aembed_texts(texts)
    log.info("Embedding complete", vectors=len(vectors))
    return vectors
