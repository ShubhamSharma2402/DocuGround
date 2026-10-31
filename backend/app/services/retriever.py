"""Hybrid retriever: dense + sparse → RRF fusion → rerank → gate."""

from __future__ import annotations

from dataclasses import dataclass

import structlog
from qdrant_client.models import ScoredPoint

from app.core.config import get_settings
from app.services.embedder import Embedder
from app.services.reranker import Reranker
from app.services.vector_store import VectorStore

log = structlog.get_logger(__name__)


@dataclass
class RetrievedChunk:
    point_id: str
    doc_id: str
    user_id: str
    page: int
    chunk_index: int
    text: str
    dense_score: float
    sparse_score: float
    rrf_score: float
    rerank_score: float | None = None


@dataclass
class RetrievalResult:
    chunks: list[RetrievedChunk]
    gate_passed: bool
    best_dense_score: float
    best_rerank_score: float | None
    refusal_reason: str | None  # Set when gate_passed=False


def reciprocal_rank_fusion(
    dense_results: list[ScoredPoint],
    sparse_results: list[ScoredPoint],
    k: int,
) -> list[tuple[str, float, float, float]]:
    """Fuse dense + sparse rankings using RRF.

    Returns list of (point_id, dense_score, sparse_score, rrf_score),
    sorted by rrf_score descending.
    """
    dense_map: dict[str, tuple[int, float]] = {
        str(p.id): (rank + 1, p.score or 0.0)
        for rank, p in enumerate(dense_results)
    }
    sparse_map: dict[str, tuple[int, float]] = {
        str(p.id): (rank + 1, p.score or 0.0)
        for rank, p in enumerate(sparse_results)
    }

    all_ids = set(dense_map) | set(sparse_map)
    scored: list[tuple[str, float, float, float]] = []
    for pid in all_ids:
        d_rank, d_score = dense_map.get(pid, (len(dense_results) + 1, 0.0))
        s_rank, s_score = sparse_map.get(pid, (len(sparse_results) + 1, 0.0))
        rrf = 1.0 / (k + d_rank) + 1.0 / (k + s_rank)
        scored.append((pid, d_score, s_score, rrf))

    scored.sort(key=lambda x: x[3], reverse=True)
    return scored


class HybridRetriever:
    """Orchestrates hybrid retrieval with optional reranking and gate logic."""

    def __init__(
        self,
        vector_store: VectorStore,
        embedder: Embedder,
        reranker: Reranker | None = None,
    ) -> None:
        self._vs = vector_store
        self._embedder = embedder
        self._reranker = reranker

    async def retrieve(
        self,
        query: str,
        user_id: str,
        doc_ids: list[str],
        *,
        top_k: int | None = None,
        rerank_n: int | None = None,
        dense_cutoff: float | None = None,
        reranker_cutoff: float | None = None,
    ) -> RetrievalResult:
        """Full hybrid retrieval pipeline: embed → search → RRF → rerank → gate."""
        settings = get_settings()
        k = top_k or settings.retrieval_top_k
        n = rerank_n or settings.rerank_top_n
        d_cut = dense_cutoff if dense_cutoff is not None else settings.dense_score_cutoff
        r_cut = reranker_cutoff if reranker_cutoff is not None else settings.reranker_score_cutoff

        # Step 1: Embed query
        query_vec = await self._embedder.aembed_query(query)

        # Step 2: Dense + sparse search
        dense_results = await self._vs.dense_search(query_vec, user_id, doc_ids, top_k=k)
        sparse_results = await self._vs.sparse_search(query, user_id, doc_ids, top_k=k)

        log.debug(
            "Search results",
            dense_count=len(dense_results),
            sparse_count=len(sparse_results),
        )

        if not dense_results and not sparse_results:
            return RetrievalResult(
                chunks=[],
                gate_passed=False,
                best_dense_score=0.0,
                best_rerank_score=None,
                refusal_reason="no_results",
            )

        # Step 3: RRF fusion
        fused = reciprocal_rank_fusion(dense_results, sparse_results, k=settings.rrf_k)
        fused = fused[:k]

        # Build a payload lookup from results
        payload_map: dict[str, dict] = {}
        for point in dense_results + sparse_results:
            pid = str(point.id)
            if pid not in payload_map and point.payload:
                payload_map[pid] = point.payload

        # Build dense/sparse score maps
        dense_score_map = {str(p.id): p.score or 0.0 for p in dense_results}
        sparse_score_map = {str(p.id): p.score or 0.0 for p in sparse_results}

        best_dense = max(dense_score_map.values()) if dense_score_map else 0.0

        # Step 4: Gate check on dense score
        # When a cross-encoder reranker is present, it computes joint cross-attention with
        # high precision (scoring relevant chunks > 0.8 even when cosine similarity is ~0.3).
        # We only early-refuse on dense if there is zero / negligible semantic signal (< 0.15).
        # If no reranker is configured, we apply the configured d_cut.
        early_dense_cutoff = 0.15 if self._reranker is not None else d_cut
        if best_dense < early_dense_cutoff and not sparse_results:
            log.info(
                "Refusal gate triggered (dense)",
                best_dense=best_dense,
                cutoff=early_dense_cutoff,
            )
            return RetrievalResult(
                chunks=[],
                gate_passed=False,
                best_dense_score=best_dense,
                best_rerank_score=None,
                refusal_reason=f"dense_score_below_cutoff:{best_dense:.3f}<{early_dense_cutoff}",
            )

        # Step 5: Build chunk objects from fused results
        chunks: list[RetrievedChunk] = []
        for pid, d_score, s_score, rrf_score in fused[:n]:
            payload = payload_map.get(pid)
            if payload is None:
                continue
            chunks.append(
                RetrievedChunk(
                    point_id=pid,
                    doc_id=payload.get("doc_id", ""),
                    user_id=payload.get("user_id", ""),
                    page=payload.get("page", 0),
                    chunk_index=payload.get("chunk_index", 0),
                    text=payload.get("text", ""),
                    dense_score=d_score,
                    sparse_score=s_score,
                    rrf_score=rrf_score,
                )
            )

        # Step 6: Rerank
        best_rerank: float | None = None
        if self._reranker is not None and chunks:
            texts = [c.text for c in chunks]
            scores = await self._reranker.arerank(query, texts)
            for chunk, score in zip(chunks, scores):
                chunk.rerank_score = score
            chunks.sort(key=lambda c: c.rerank_score or 0.0, reverse=True)
            chunks = chunks[:n]
            best_rerank = chunks[0].rerank_score if chunks else None

            # Step 7: Dual-signal refusal gate
            # Pass if cross-encoder confirms relevance (best_rerank >= r_cut)
            # OR if dense semantic similarity is already solid (best_dense >= 0.25,
            # which accommodates short keyword queries where cross-encoders may score slightly lower).
            # True out-of-domain queries (e.g. baking recipes, geography) have best_dense < 0.05 AND rerank < 0.05.
            is_relevant = (best_rerank is not None and best_rerank >= r_cut) or (best_dense >= 0.25)
            if not is_relevant:
                log.info(
                    "Refusal gate triggered (reranker)",
                    best_rerank=best_rerank,
                    best_dense=best_dense,
                    cutoff=r_cut,
                )
                return RetrievalResult(
                    chunks=chunks,
                    gate_passed=False,
                    best_dense_score=best_dense,
                    best_rerank_score=best_rerank,
                    refusal_reason=f"reranker_score_below_cutoff:{best_rerank:.3f}<{r_cut}",
                )

        return RetrievalResult(
            chunks=chunks,
            gate_passed=True,
            best_dense_score=best_dense,
            best_rerank_score=best_rerank,
            refusal_reason=None,
        )
