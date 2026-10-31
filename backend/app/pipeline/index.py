"""Stage 4 — Index: upsert chunks + vectors into Qdrant."""

from __future__ import annotations

import structlog

from app.pipeline.chunk import TextChunk
from app.services.vector_store import ChunkPayload, VectorStore

log = structlog.get_logger(__name__)


async def index_chunks(
    chunks: list[TextChunk],
    vectors: list[list[float]],
    vector_store: VectorStore,
) -> None:
    payloads = [
        ChunkPayload(
            doc_id=c.doc_id, user_id=c.user_id, page=c.page,
            chunk_index=c.chunk_index, text=c.text,
        )
        for c in chunks
    ]
    await vector_store.upsert_chunks(payloads, vectors)
    log.info("Indexed chunks", count=len(chunks),
             doc_id=chunks[0].doc_id if chunks else "none")
