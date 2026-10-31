"""Embedder interface + SentenceTransformer implementation.

The Embedder protocol makes it easy to swap implementations in tests or
to switch to a remote embedding API without touching caller code.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from typing import Protocol, runtime_checkable

import numpy as np
from sentence_transformers import SentenceTransformer

from app.core.config import get_settings
from app.core.logging import get_logger

log = get_logger(__name__)

# Shared thread pool for CPU-bound model inference (keeps the event loop free)
_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="embedder")


@runtime_checkable
class Embedder(Protocol):
    """Interface for text embedding providers."""

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Return dense embeddings for *texts* (synchronous, runs in executor)."""
        ...

    async def aembed_texts(self, texts: list[str]) -> list[list[float]]:
        """Async wrapper — runs embed_texts off the event loop."""
        ...

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string (synchronous)."""
        ...

    async def aembed_query(self, text: str) -> list[float]:
        """Async wrapper for embed_query."""
        ...


class SentenceTransformerEmbedder:
    """Dense embedder backed by a local SentenceTransformer model."""

    def __init__(self, model_name: str | None = None) -> None:
        settings = get_settings()
        self._model_name = model_name or settings.embedder_model
        self._batch_size = settings.embed_batch_size
        self._model: SentenceTransformer | None = None

    def _get_model(self) -> SentenceTransformer:
        if self._model is None:
            settings = get_settings()
            log.info("Loading embedding model", model=self._model_name)
            self._model = SentenceTransformer(
                self._model_name,
                cache_folder=str(settings.models_cache_dir),
            )
        return self._model

    # ── Sync API ─────────────────────────────────────────────────────────────

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed *texts* in batches; returns list of float vectors."""
        model = self._get_model()
        vectors: list[list[float]] = []
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            embs = model.encode(batch, normalize_embeddings=True, show_progress_bar=False)
            vectors.extend(embs.tolist())
        return vectors

    def embed_query(self, text: str) -> list[float]:
        model = self._get_model()
        vec = model.encode([text], normalize_embeddings=True, show_progress_bar=False)
        return vec[0].tolist()

    # ── Async API ─────────────────────────────────────────────────────────────

    async def aembed_texts(self, texts: list[str]) -> list[list[float]]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(_executor, self.embed_texts, texts)

    async def aembed_query(self, text: str) -> list[float]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(_executor, self.embed_query, text)


class FakeEmbedder:
    """Deterministic fake embedder for unit/integration tests (no GPU/model needed)."""

    def __init__(self, size: int = 768) -> None:
        self._size = size

    def _vec(self, text: str) -> list[float]:
        rng = np.random.default_rng(abs(hash(text)) % (2**31))
        v = rng.random(self._size).astype(np.float32)
        v /= np.linalg.norm(v) + 1e-9
        return v.tolist()

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)

    async def aembed_texts(self, texts: list[str]) -> list[list[float]]:
        return self.embed_texts(texts)

    async def aembed_query(self, text: str) -> list[float]:
        return self.embed_query(text)


# ── Module-level singleton (lazy) ────────────────────────────────────────────
_default_embedder: SentenceTransformerEmbedder | None = None


def get_embedder() -> SentenceTransformerEmbedder:
    global _default_embedder
    if _default_embedder is None:
        _default_embedder = SentenceTransformerEmbedder()
    return _default_embedder
