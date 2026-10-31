"""Stage 5 — Finalize: mark document ready and clean up temp file."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import structlog
from motor.motor_asyncio import AsyncIOMotorDatabase

log = structlog.get_logger(__name__)


async def finalize_document(
    doc_id: str,
    db: AsyncIOMotorDatabase,
    page_count: int,
    chunk_count: int,
    temp_file: Path | None = None,
) -> None:
    now = datetime.now(timezone.utc)
    await db.documents.update_one(
        {"_id": doc_id},
        {"$set": {
            "status": "ready",
            "page_count": page_count,
            "chunk_count": chunk_count,
            "updated_at": now,
            "error": None,
        }},
    )
    # Only unlink if it's explicitly a temporary file (starting with tmp_)
    if temp_file and temp_file.exists() and temp_file.name.startswith("tmp_"):
        temp_file.unlink(missing_ok=True)
        log.debug("Temp file deleted", path=str(temp_file))

    log.info("Document finalized", doc_id=doc_id, pages=page_count, chunks=chunk_count)
