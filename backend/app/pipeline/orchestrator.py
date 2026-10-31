"""Pipeline orchestrator: runs the 5-stage pipeline with saga-style rollback."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Coroutine, Any

import structlog
from motor.motor_asyncio import AsyncIOMotorDatabase

from app.pipeline.chunk import chunk_pages
from app.pipeline.embed import embed_chunks
from app.pipeline.finalize import finalize_document
from app.pipeline.index import index_chunks
from app.pipeline.parse import parse_pdf
from app.services.embedder import Embedder
from app.services.vector_store import VectorStore

log = structlog.get_logger(__name__)


@dataclass
class PipelineContext:
    job_id: str
    doc_id: str
    user_id: str
    file_path: Path
    db: AsyncIOMotorDatabase
    vector_store: VectorStore
    embedder: Embedder
    stage_timings: dict[str, float] = field(default_factory=dict)
    _compensations: list[Callable[[], Coroutine[Any, Any, None]]] = field(
        default_factory=list, repr=False
    )

    def register_compensation(self, fn: Callable[[], Coroutine[Any, Any, None]]) -> None:
        self._compensations.append(fn)

    async def rollback(self) -> None:
        log.warning("Pipeline rollback started", doc_id=self.doc_id, job_id=self.job_id)
        for comp in reversed(self._compensations):
            try:
                await comp()
            except Exception as exc:
                log.error("Compensation failed", error=str(exc))


@dataclass
class PipelineResult:
    success: bool
    page_count: int = 0
    chunk_count: int = 0
    error: str | None = None
    stage_timings: dict[str, float] = field(default_factory=dict)


async def run_pipeline(
    ctx: PipelineContext,
    progress_callback: Callable[[str, int], Coroutine[Any, Any, None]] | None = None,
) -> PipelineResult:
    async def _progress(stage: str, pct: int) -> None:
        if progress_callback:
            await progress_callback(stage, pct)

    try:
        t0 = time.monotonic()
        await _progress("parse", 0)
        pages = await asyncio.get_event_loop().run_in_executor(None, parse_pdf, ctx.file_path)
        ctx.stage_timings["parse"] = time.monotonic() - t0
        await _progress("parse", 20)

        t0 = time.monotonic()
        await _progress("chunk", 20)
        chunks = chunk_pages(pages, doc_id=ctx.doc_id, user_id=ctx.user_id)
        ctx.stage_timings["chunk"] = time.monotonic() - t0
        await _progress("chunk", 40)

        t0 = time.monotonic()
        await _progress("embed", 40)
        vectors = await embed_chunks(chunks, ctx.embedder)
        ctx.stage_timings["embed"] = time.monotonic() - t0
        await _progress("embed", 65)

        # Register compensation: delete Qdrant points if indexed
        ctx.register_compensation(
            lambda: ctx.vector_store.delete_by_doc_id(ctx.doc_id, ctx.user_id)
        )

        t0 = time.monotonic()
        await _progress("index", 65)
        await index_chunks(chunks, vectors, ctx.vector_store)
        ctx.stage_timings["index"] = time.monotonic() - t0
        await _progress("index", 85)

        t0 = time.monotonic()
        await _progress("finalize", 85)
        await finalize_document(
            doc_id=ctx.doc_id, db=ctx.db,
            page_count=len(pages), chunk_count=len(chunks),
            temp_file=ctx.file_path,
        )
        ctx.stage_timings["finalize"] = time.monotonic() - t0
        await _progress("done", 100)

        return PipelineResult(
            success=True, page_count=len(pages),
            chunk_count=len(chunks), stage_timings=ctx.stage_timings,
        )

    except Exception as exc:
        log.error("Pipeline failed, rolling back", job_id=ctx.job_id,
                  doc_id=ctx.doc_id, error=str(exc))
        await ctx.rollback()
        return PipelineResult(success=False, error=str(exc), stage_timings=ctx.stage_timings)
