"""Dead Letter Queue (DLQ) dispatch for poisoned / permanently failed jobs."""

from __future__ import annotations

from datetime import datetime, timezone

import structlog

from app.core.config import get_settings
from app.db.redis import get_redis_client

log = structlog.get_logger(__name__)


async def push_to_dlq(
    job_id: str,
    doc_id: str,
    user_id: str,
    error: str,
    attempts: int,
    file_path: str = "",
) -> str:
    """Push permanently failed job to the dead-letter stream (jobs:dead)."""
    settings = get_settings()
    client = get_redis_client()
    now_iso = datetime.now(timezone.utc).isoformat()

    payload = {
        "job_id": job_id,
        "doc_id": doc_id,
        "user_id": user_id,
        "error": error,
        "attempts": str(attempts),
        "file_path": file_path,
        "failed_at": now_iso,
    }

    msg_id = await client.xadd(settings.dlq_stream, payload)
    log.error(
        "Job permanently failed — pushed to DLQ",
        dlq_stream=settings.dlq_stream,
        job_id=job_id,
        doc_id=doc_id,
        user_id=user_id,
        attempts=attempts,
        error=error,
        dlq_msg_id=msg_id,
    )
    return msg_id
