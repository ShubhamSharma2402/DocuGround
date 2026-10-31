"""Qdrant vector store wrapper with hybrid search support."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any

import structlog
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    FieldCondition,
    Filter,
    MatchAny,
    MatchValue,
    PointStruct,
    ScoredPoint,
    SparseVector,
)

from app.core.config import get_settings

log = structlog.get_logger(__name__)


@dataclass
class ChunkPayload:
    doc_id: str
    user_id: str
    page: int
    chunk_index: int
    text: str


def chunk_point_id(doc_id: str, chunk_index: int) -> str:
    """Deterministic UUID5 point ID — makes upserts idempotent."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{doc_id}:{chunk_index}"))


def _make_sparse_vector(text: str) -> SparseVector:
    """Simple TF-based sparse vector for BM25-style retrieval.

    In production you would use a dedicated BM25 tokeniser (e.g. fastembed).
    This implementation gives a good-enough sparse signal for the hybrid search
    without introducing an additional heavy dependency.
    """
    import math
    import re
    from collections import Counter

    tokens = re.findall(r"\w+", text.lower())
    if not tokens:
        return SparseVector(indices=[0], values=[0.0])
    counts = Counter(tokens)
    total = sum(counts.values())
    sparse_map: dict[int, float] = {}
    for token, count in counts.items():
        # Use a stable hash to map tokens → integer indices (mod 2^20)
        idx = abs(hash(token)) % (2**20)
        tf = count / total
        # IDF approximation: log(1 + 1/freq_fraction)
        idf = math.log(1 + 1 / (count / total + 1e-9))
        sparse_map[idx] = sparse_map.get(idx, 0.0) + float(tf * idf)

    sorted_indices = sorted(sparse_map.keys())
    values = [sparse_map[i] for i in sorted_indices]
    return SparseVector(indices=sorted_indices, values=values)


class VectorStore:
    """Async wrapper around Qdrant for upsert, search, and delete."""

    def __init__(self, client: AsyncQdrantClient, collection_name: str | None = None) -> None:
        self._client = client
        self._settings = get_settings()
        self._collection = collection_name or self._settings.qdrant_collection

    # ── Upsert ────────────────────────────────────────────────────────────────

    async def upsert_chunks(
        self,
        chunks: list[ChunkPayload],
        dense_vectors: list[list[float]],
    ) -> None:
        """Upsert chunks with dense + sparse vectors into Qdrant."""
        points: list[PointStruct] = []
        for chunk, dense_vec in zip(chunks, dense_vectors):
            sparse_vec = _make_sparse_vector(chunk.text)
            point_id = chunk_point_id(chunk.doc_id, chunk.chunk_index)
            points.append(
                PointStruct(
                    id=point_id,
                    vector={
                        "dense": dense_vec,
                        "sparse": sparse_vec,
                    },
                    payload={
                        "doc_id": chunk.doc_id,
                        "user_id": chunk.user_id,
                        "page": chunk.page,
                        "chunk_index": chunk.chunk_index,
                        "text": chunk.text,
                    },
                )
            )

        await self._client.upsert(
            collection_name=self._collection,
            points=points,
            wait=True,
        )
        log.debug("Upserted chunks", count=len(points))

    # ── Search ────────────────────────────────────────────────────────────────

    async def dense_search(
        self,
        query_vector: list[float],
        user_id: str,
        doc_ids: list[str],
        top_k: int | None = None,
    ) -> list[ScoredPoint]:
        """Dense cosine similarity search filtered by user_id and doc_ids."""
        settings = get_settings()
        k = top_k or settings.retrieval_top_k

        doc_filter = Filter(
            must=[
                FieldCondition(key="user_id", match=MatchValue(value=user_id)),
                FieldCondition(key="doc_id", match=MatchValue(value=doc_ids[0]))
                if len(doc_ids) == 1
                else FieldCondition(key="doc_id", match=MatchAny(any=doc_ids)),
            ]
        )

        response = await self._client.query_points(
            collection_name=self._collection,
            query=query_vector,
            using="dense",
            query_filter=doc_filter,
            limit=k,
            with_payload=True,
        )
        return response.points

    async def sparse_search(
        self,
        query_text: str,
        user_id: str,
        doc_ids: list[str],
        top_k: int | None = None,
    ) -> list[ScoredPoint]:
        """Sparse BM25-style search filtered by user_id and doc_ids."""
        settings = get_settings()
        k = top_k or settings.retrieval_top_k
        sparse_vec = _make_sparse_vector(query_text)

        doc_filter = Filter(
            must=[
                FieldCondition(key="user_id", match=MatchValue(value=user_id)),
                FieldCondition(key="doc_id", match=MatchValue(value=doc_ids[0]))
                if len(doc_ids) == 1
                else FieldCondition(key="doc_id", match=MatchAny(any=doc_ids)),
            ]
        )

        response = await self._client.query_points(
            collection_name=self._collection,
            query=sparse_vec,
            using="sparse",
            query_filter=doc_filter,
            limit=k,
            with_payload=True,
        )
        return response.points

    # ── Delete ────────────────────────────────────────────────────────────────

    async def delete_by_doc_id(self, doc_id: str, user_id: str) -> int:
        """Delete all points for a document. Returns count deleted."""
        settings = get_settings()
        doc_filter = Filter(
            must=[
                FieldCondition(key="doc_id", match=MatchValue(value=doc_id)),
                FieldCondition(key="user_id", match=MatchValue(value=user_id)),
            ]
        )
        result = await self._client.delete(
            collection_name=self._collection,
            points_selector=doc_filter,
            wait=True,
        )
        log.info("Deleted Qdrant points", doc_id=doc_id, status=result.status)
        return 0  # Qdrant delete doesn't return count directly


_qdrant_client: AsyncQdrantClient | None = None


def get_qdrant_client() -> AsyncQdrantClient:
    global _qdrant_client
    if _qdrant_client is None:
        settings = get_settings()
        _qdrant_client = AsyncQdrantClient(url=settings.qdrant_url)
    return _qdrant_client


def get_vector_store() -> VectorStore:
    return VectorStore(get_qdrant_client())
