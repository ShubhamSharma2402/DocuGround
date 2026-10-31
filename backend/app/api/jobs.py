"""Job status inspection and SSE live event stream endpoints."""

from __future__ import annotations

import asyncio
import json

import structlog
from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from motor.motor_asyncio import AsyncIOMotorDatabase

from app.api.deps import get_current_user
from app.core.errors import ForbiddenError, NotFoundError
from app.db.client import get_db
from app.db.models import JobStatus, UserDB
from app.db.redis import get_job_record

log = structlog.get_logger(__name__)

router = APIRouter(prefix="/jobs", tags=["jobs"])


@router.get("/{job_id}", response_model=JobStatus)
async def get_job_status(
    job_id: str,
    current_user: UserDB = Depends(get_current_user),
    db: AsyncIOMotorDatabase = Depends(get_db),
) -> JobStatus:
    """Get the current stage, progress, and status of an ingestion job."""
    data = await get_job_record(job_id)

    if not data:
        # Fallback check MongoDB documents if Redis hash expired
        doc = await db.documents.find_one({"_id": job_id})
        if not doc:
            raise NotFoundError(f"Job {job_id} not found")
        if doc["user_id"] != str(current_user.id):
            raise ForbiddenError("You do not have access to this job")

        return JobStatus(
            job_id=job_id,
            doc_id=str(doc["_id"]),
            user_id=doc["user_id"],
            stage="done" if doc["status"] == "ready" else doc["status"],
            progress=100 if doc["status"] == "ready" else 0,
            attempts=1,
            error=doc.get("error") or "",
            created_at=doc["created_at"].isoformat(),
            updated_at=doc["updated_at"].isoformat(),
        )

    if data.get("user_id") != str(current_user.id):
        raise ForbiddenError("You do not have access to this job")

    return JobStatus(
        job_id=data.get("job_id", job_id),
        doc_id=data.get("doc_id", ""),
        user_id=data.get("user_id", ""),
        stage=data.get("stage", "queued"),
        progress=int(data.get("progress", 0)),
        attempts=int(data.get("attempts", 0)),
        error=data.get("error", ""),
        created_at=data.get("created_at", ""),
        updated_at=data.get("updated_at", ""),
    )


@router.get("/{job_id}/events")
async def stream_job_events(
    job_id: str,
    request: Request,
    current_user: UserDB = Depends(get_current_user),
) -> StreamingResponse:
    """Stream live stage and progress updates via Server-Sent Events (SSE)."""
    # Verify ownership before opening the stream
    data = await get_job_record(job_id)
    if not data:
        raise NotFoundError(f"Job {job_id} not found")
    if data.get("user_id") != str(current_user.id):
        raise ForbiddenError("You do not have access to this job")

    async def event_generator():
        last_stage = None
        last_progress = None

        while True:
            if await request.is_disconnected():
                log.debug("SSE client disconnected", job_id=job_id)
                break

            current_data = await get_job_record(job_id)
            if not current_data:
                break

            stage = current_data.get("stage", "queued")
            progress = int(current_data.get("progress", 0))
            error = current_data.get("error", "")

            # Emit whenever stage or progress changes
            if stage != last_stage or progress != last_progress:
                payload = json.dumps({
                    "job_id": job_id,
                    "stage": stage,
                    "progress": progress,
                    "error": error,
                })
                yield f"data: {payload}\n\n"
                last_stage = stage
                last_progress = progress

            if stage in ("done", "failed"):
                break

            await asyncio.sleep(0.5)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
