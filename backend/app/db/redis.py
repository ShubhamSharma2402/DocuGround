"""Redis client and job queue / hash management."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import redis.asyncio as aioredis
import structlog

from app.core.config import get_settings

log = structlog.get_logger(__name__)

_redis_client: aioredis.Redis | None = None


def get_redis_client() -> aioredis.Redis:
    """Return the global Redis client singleton."""
    global _redis_client
    if _redis_client is None:
        settings = get_settings()
        _redis_client = aioredis.from_url(
            settings.redis_url,
            decode_responses=True,
            health_check_interval=30,
            protocol=2,
        )
    return _redis_client


def set_redis_client(client: aioredis.Redis | None) -> None:
    """Override the global Redis client (e.g. for testing with FakeRedis)."""
    global _redis_client
    _redis_client = client


async def close_redis() -> None:
    """Close the global Redis client connection."""
    global _redis_client
    if _redis_client is not None:
        await _redis_client.close()
        _redis_client = None


def _job_key(job_id: str) -> str:
    return f"job:{job_id}"


async def create_job_record(
    job_id: str,
    doc_id: str,
    user_id: str,
    filename: str,
    file_path: str,
) -> None:
    """Create a Redis Hash job:{job_id} and set TTL."""
    client = get_redis_client()
    settings = get_settings()
    now_iso = datetime.now(timezone.utc).isoformat()

    key = _job_key(job_id)
    mapping = {
        "job_id": job_id,
        "doc_id": doc_id,
        "user_id": user_id,
        "filename": filename,
        "file_path": file_path,
        "stage": "queued",
        "progress": "0",
        "attempts": "0",
        "error": "",
        "created_at": now_iso,
        "updated_at": now_iso,
    }
    await client.hset(key, mapping=mapping)
    await client.expire(key, settings.job_ttl_seconds)
    log.debug("Created Redis job record", job_id=job_id, doc_id=doc_id)


async def get_job_record(job_id: str) -> dict[str, str]:
    """Retrieve all fields of the Redis Hash for job_id."""
    client = get_redis_client()
    data = await client.hgetall(_job_key(job_id))
    return data or {}


async def update_job_record(
    job_id: str,
    stage: str | None = None,
    progress: int | None = None,
    error: str | None = None,
    attempts: int | None = None,
) -> None:
    """Update fields on the Redis Hash job:{job_id}."""
    client = get_redis_client()
    key = _job_key(job_id)
    mapping: dict[str, str] = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    if stage is not None:
        mapping["stage"] = stage
    if progress is not None:
        mapping["progress"] = str(progress)
    if error is not None:
        mapping["error"] = error
    if attempts is not None:
        mapping["attempts"] = str(attempts)

    await client.hset(key, mapping=mapping)


async def enqueue_ingest_job(
    job_id: str,
    doc_id: str,
    user_id: str,
    file_path: str,
) -> str:
    """XADD the ingestion job to the jobs:ingest stream."""
    client = get_redis_client()
    settings = get_settings()
    payload = {
        "job_id": job_id,
        "doc_id": doc_id,
        "user_id": user_id,
        "file_path": file_path,
    }
    msg_id = await client.xadd(settings.ingest_stream, payload)
    log.info(
        "Job enqueued to stream",
        stream=settings.ingest_stream,
        job_id=job_id,
        doc_id=doc_id,
        msg_id=msg_id,
    )
    return msg_id
