"""Redis Streams consumer loop with XAUTOCLAIM crash recovery, retries, and saga rollback."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import random
from pathlib import Path
from typing import Any

import redis.asyncio as aioredis
from redis.exceptions import ResponseError
import structlog
from motor.motor_asyncio import AsyncIOMotorDatabase

from app.core.config import get_settings
from app.core.metrics import DOCUGROUND_JOBS_TOTAL
from app.db.models import DocumentStatus
from app.db.redis import get_job_record, get_redis_client, update_job_record
from app.pipeline.orchestrator import PipelineContext, run_pipeline
from app.services.embedder import Embedder, get_embedder
from app.services.vector_store import VectorStore, get_vector_store
from app.worker.dlq import push_to_dlq

log = structlog.get_logger(__name__)


async def ensure_consumer_group(
    client: aioredis.Redis,
    stream: str,
    group: str,
) -> None:
    """Create the Redis Stream consumer group idempotently."""
    try:
        await client.xgroup_create(stream, group, id="0", mkstream=True)
        log.info("Created consumer group", stream=stream, group=group)
    except ResponseError as exc:
        if "BUSYGROUP" in str(exc) or "already exists" in str(exc).lower():
            log.debug("Consumer group already exists", stream=stream, group=group)
        else:
            raise


def calculate_backoff(attempt: int, base: float = 2.0, max_delay: float = 30.0) -> float:
    """Exponential backoff with full random jitter."""
    exp = min(max_delay, base * (2 ** attempt))
    jitter = random.uniform(0.1, 1.0)
    return exp + jitter


async def process_single_job(
    message_id: str,
    payload: dict[str, str],
    db: AsyncIOMotorDatabase,
    vector_store: VectorStore,
    embedder: Embedder,
    client: aioredis.Redis | None = None,
) -> bool:
    """Process an ingestion job payload through the 5-stage pipeline."""
    settings = get_settings()
    redis_client = client or get_redis_client()

    job_id = payload.get("job_id", "")
    doc_id = payload.get("doc_id", "")
    user_id = payload.get("user_id", "")
    file_path = Path(payload.get("file_path", ""))

    log.info("Processing job message", message_id=message_id, job_id=job_id, doc_id=doc_id)

    # Fetch current attempt count from Redis Hash
    job_hash = await get_job_record(job_id)
    current_attempts = int(job_hash.get("attempts", "0"))

    async def _on_progress(stage: str, progress: int) -> None:
        await update_job_record(job_id, stage=stage, progress=progress)

    ctx = PipelineContext(
        job_id=job_id,
        doc_id=doc_id,
        user_id=user_id,
        file_path=file_path,
        db=db,
        vector_store=vector_store,
        embedder=embedder,
    )

    result = await run_pipeline(ctx, progress_callback=_on_progress)

    if result.success:
        DOCUGROUND_JOBS_TOTAL.labels(status="done").inc()
        await update_job_record(job_id, stage="done", progress=100, error="")
        await redis_client.xack(settings.ingest_stream, settings.consumer_group, message_id)
        log.info(
            "Job processed and acknowledged",
            job_id=job_id,
            doc_id=doc_id,
            pages=result.page_count,
            chunks=result.chunk_count,
        )
        return True

    # Failure handling
    new_attempts = current_attempts + 1
    error_msg = result.error or "Pipeline execution error"
    await update_job_record(job_id, attempts=new_attempts, error=error_msg)

    if new_attempts < settings.worker_max_retries:
        DOCUGROUND_JOBS_TOTAL.labels(status="retry").inc()
        backoff = calculate_backoff(
            new_attempts,
            base=settings.worker_retry_base_delay,
            max_delay=settings.worker_retry_max_delay,
        )
        log.warning(
            "Job failed, scheduled for retry",
            job_id=job_id,
            attempt=new_attempts,
            max_retries=settings.worker_max_retries,
            backoff_sec=round(backoff, 2),
            error=error_msg,
        )
        # Message remains unacknowledged; will be picked up on next cycle
        return False

    # Max retries exceeded: final failure & dead-letter queue
    DOCUGROUND_JOBS_TOTAL.labels(status="failed").inc()
    log.error(
        "Job exhausted all retries — rolling back and routing to DLQ",
        job_id=job_id,
        doc_id=doc_id,
        attempts=new_attempts,
        error=error_msg,
    )

    # 1. Update MongoDB document status to failed
    now = datetime.now(timezone.utc)
    await db.documents.update_one(
        {"_id": doc_id},
        {
            "$set": {
                "status": DocumentStatus.failed.value,
                "error": error_msg,
                "updated_at": now,
            }
        },
    )

    # 2. Mark Redis job hash failed
    await update_job_record(job_id, stage="failed", progress=0, error=error_msg)

    # 3. Push to DLQ stream
    await push_to_dlq(
        job_id=job_id,
        doc_id=doc_id,
        user_id=user_id,
        error=error_msg,
        attempts=new_attempts,
        file_path=str(file_path),
    )

    # 4. Acknowledge original message to unblock consumer group
    await redis_client.xack(settings.ingest_stream, settings.consumer_group, message_id)

    # 5. Clean up temp file
    if file_path.exists():
        file_path.unlink(missing_ok=True)

    return False


async def reclaim_stuck_messages(
    client: aioredis.Redis,
    stream: str,
    group: str,
    consumer: str,
    min_idle_ms: int,
) -> list[tuple[str, dict[str, str]]]:
    """Call XAUTOCLAIM to reclaim messages idle longer than min_idle_ms."""
    try:
        # xautoclaim returns (next_start_id, messages, deleted_ids)
        result = await client.xautoclaim(
            stream,
            group,
            consumer,
            min_idle_time=min_idle_ms,
            start_id="0-0",
            count=10,
        )
        if result and len(result) >= 2:
            return result[1] or []
    except Exception as exc:
        log.warning("XAUTOCLAIM failed", error=str(exc))
    return []


async def run_consumer_loop(
    db: AsyncIOMotorDatabase,
    vector_store: VectorStore,
    embedder: Embedder,
    stop_event: asyncio.Event | None = None,
    poll_interval_ms: int = 100,
) -> None:
    """Continuous worker consumer loop reading from Redis Streams."""
    settings = get_settings()
    client = get_redis_client()

    await ensure_consumer_group(client, settings.ingest_stream, settings.consumer_group)
    log.info(
        "Worker consumer loop running",
        stream=settings.ingest_stream,
        group=settings.consumer_group,
        consumer=settings.consumer_name,
    )

    last_autoclaim_check = 0.0

    while True:
        if stop_event and stop_event.is_set():
            log.info("Worker consumer loop stopping gracefully")
            break

        now = asyncio.get_event_loop().time()

        # Periodically reclaim stuck messages from crashed workers
        if now - last_autoclaim_check > (settings.autoclaim_min_idle_ms / 1000.0) / 2:
            last_autoclaim_check = now
            reclaimed = await reclaim_stuck_messages(
                client,
                settings.ingest_stream,
                settings.consumer_group,
                settings.consumer_name,
                settings.autoclaim_min_idle_ms,
            )
            for msg_id, payload in reclaimed:
                if payload:
                    await process_single_job(msg_id, payload, db, vector_store, embedder, client)

        # Read new messages with XREADGROUP
        try:
            entries = await client.xreadgroup(
                groupname=settings.consumer_group,
                consumername=settings.consumer_name,
                streams={settings.ingest_stream: ">"},
                count=1,
                block=settings.stream_block_ms,
            )
        except Exception as exc:
            log.error("XREADGROUP error", error=str(exc))
            await asyncio.sleep(1.0)
            continue

        if not entries:
            continue

        for stream_name, messages in entries:
            for message_id, payload in messages:
                if payload:
                    await process_single_job(message_id, payload, db, vector_store, embedder, client)
