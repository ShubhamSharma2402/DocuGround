"""Janitor task: finds stuck processing documents and cleans orphan records."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import structlog
from motor.motor_asyncio import AsyncIOMotorDatabase

from app.core.config import get_settings
from app.db.models import DocumentStatus
from app.db.redis import get_job_record
from app.services.vector_store import VectorStore

log = structlog.get_logger(__name__)


async def cleanup_stuck_documents(
    db: AsyncIOMotorDatabase,
    vector_store: VectorStore,
    threshold_seconds: int | None = None,
) -> int:
    """Find documents stuck in processing beyond threshold with no active job, clean up and mark failed."""
    settings = get_settings()
    cutoff_sec = threshold_seconds or settings.stuck_job_threshold_seconds
    cutoff_time = datetime.now(timezone.utc) - timedelta(seconds=cutoff_sec)

    cursor = db.documents.find({
        "status": DocumentStatus.processing.value,
        "updated_at": {"$lt": cutoff_time},
    })
    stuck_docs = await cursor.to_list(length=100)

    cleaned_count = 0
    for doc in stuck_docs:
        doc_id = str(doc["_id"])
        user_id = doc.get("user_id", "")

        log.warning(
            "Janitor detected stuck document",
            doc_id=doc_id,
            user_id=user_id,
            updated_at=doc.get("updated_at"),
        )

        # 1. Clean up any partial Qdrant vector points
        try:
            await vector_store.delete_by_doc_id(doc_id, user_id)
        except Exception as exc:
            log.warning("Janitor: Qdrant deletion failed or unreachable", error=str(exc))

        # 2. Mark document failed in MongoDB
        now = datetime.now(timezone.utc)
        await db.documents.update_one(
            {"_id": doc_id},
            {
                "$set": {
                    "status": DocumentStatus.failed.value,
                    "error": "Job timed out or worker crashed during processing",
                    "updated_at": now,
                }
            },
        )
        cleaned_count += 1

    if cleaned_count > 0:
        log.info("Janitor cleanup completed", cleaned_count=cleaned_count)

    return cleaned_count


async def run_janitor_loop(
    db: AsyncIOMotorDatabase,
    vector_store: VectorStore,
    interval_seconds: int | None = None,
    stop_event: asyncio.Event | None = None,
) -> None:
    """Run the periodic janitor loop until stop_event is signaled."""
    settings = get_settings()
    interval = interval_seconds or settings.janitor_interval_seconds
    log.info("Starting janitor loop", interval_seconds=interval)

    while True:
        if stop_event and stop_event.is_set():
            log.info("Janitor loop stopping")
            break

        try:
            await cleanup_stuck_documents(db, vector_store)
        except Exception as exc:
            log.error("Janitor execution failed", error=str(exc))

        try:
            if stop_event:
                await asyncio.wait_for(stop_event.wait(), timeout=interval)
                break
            else:
                await asyncio.sleep(interval)
        except asyncio.TimeoutError:
            continue
