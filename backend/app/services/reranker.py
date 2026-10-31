"""Reranker interface + BGE cross-encoder implementation."""

from __future__ import annotations

import asyncio
from abc import abstractmethod
from concurrent.futures import ThreadPoolExecutor
from typing import Protocol, runtime_checkable

from app.core.config import get_settings
from app.core.logging import get_logger

log = get_logger(__name__)

_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="reranker")


@runtime_checkable
class Reranker(Protocol):
    """Interface for reranking providers."""

    def rerank(self, query: str, texts: list[str]) -> list[float]:
        """Return relevance scores (higher = more relevant) for each text."""
        ...

    async def arerank(self, query: str, texts: list[str]) -> list[float]:
        ...


class CrossEncoderReranker:
    """BGE cross-encoder reranker (local model)."""

    def __init__(self, model_name: str | None = None) -> None:
        settings = get_settings()
        self._model_name = model_name or settings.reranker_model
        self._model = None

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            settings = get_settings()
            log.info("Loading reranker model", model=self._model_name)
            self._model = CrossEncoder(
                self._model_name,
                cache_folder=str(settings.models_cache_dir),
            )
        return self._model

    def rerank(self, query: str, texts: list[str]) -> list[float]:
        model = self._get_model()
        pairs = [[query, t] for t in texts]
        scores = model.predict(pairs, show_progress_bar=False)
        return scores.tolist()

    async def arerank(self, query: str, texts: list[str]) -> list[float]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(_executor, self.rerank, query, texts)


class FakeReranker:
    """Deterministic fake reranker for tests."""

    def rerank(self, query: str, texts: list[str]) -> list[float]:
        # Return scores based on text length overlap with query words
        query_words = set(query.lower().split())
        scores = []
        for t in texts:
            words = set(t.lower().split())
            overlap = len(query_words & words) / (len(query_words) + 1)
            scores.append(float(overlap))
        return scores

    async def arerank(self, query: str, texts: list[str]) -> list[float]:
        return self.rerank(query, texts)


_default_reranker: CrossEncoderReranker | None = None


def get_reranker() -> CrossEncoderReranker:
    global _default_reranker
    if _default_reranker is None:
        _default_reranker = CrossEncoderReranker()
    return _default_reranker
