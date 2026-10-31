"""Integration tests for background worker, consumer group, retries, DLQ, and janitor."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from motor.motor_asyncio import AsyncIOMotorDatabase

from app.core.config import get_settings
from app.db.client import get_db
from app.db.models import DocumentStatus
from app.db.redis import get_job_record, get_redis_client
from app.services.embedder import FakeEmbedder
from app.services.vector_store import VectorStore
from app.worker.consumer import (
    ensure_consumer_group,
    process_single_job,
    reclaim_stuck_messages,
)
from app.worker.janitor import cleanup_stuck_documents


class MockVectorStore(VectorStore):
    """In-memory mock for vector store during worker tests."""

    def __init__(self) -> None:
        self.upserted_points = []
        self.deleted_docs = []

    async def upsert_chunks(self, chunks, dense_vectors) -> None:
        self.upserted_points.extend(chunks)

    async def delete_by_doc_id(self, doc_id: str, user_id: str) -> int:
        self.deleted_docs.append(doc_id)
        return len(self.deleted_docs)


@pytest.mark.asyncio
async def test_worker_happy_path(
    client: httpx.AsyncClient,
    registered_user: dict,
    make_pdf,
):
    settings = get_settings()
    redis = get_redis_client()
    db = get_db()
    mock_vs = MockVectorStore()
    fake_embedder = FakeEmbedder()

    await ensure_consumer_group(redis, settings.ingest_stream, settings.consumer_group)

    # 1. Upload valid PDF
    pdf_bytes = make_pdf("Worker happy path document content", pages=2)
    files = {"file": ("worker_happy.pdf", pdf_bytes, "application/pdf")}
    upload_resp = await client.post("/documents", headers=registered_user["headers"], files=files)
    assert upload_resp.status_code == 202
    job_id = upload_resp.json()["job_id"]
    doc_id = upload_resp.json()["document_id"]

    # 2. Read message from Redis stream via consumer group
    entries = await redis.xreadgroup(
        groupname=settings.consumer_group,
        consumername=settings.consumer_name,
        streams={settings.ingest_stream: ">"},
        count=1,
    )
    assert len(entries) >= 1
    stream_name, messages = entries[0]
    assert len(messages) >= 1
    msg_id, payload = messages[0]
    assert payload["job_id"] == job_id

    # 3. Process job
    success = await process_single_job(
        message_id=msg_id,
        payload=payload,
        db=db,
        vector_store=mock_vs,
        embedder=fake_embedder,
        client=redis,
    )
    assert success is True

    # 4. Assert MongoDB document status is ready with chunk and page counts
    doc = await db.documents.find_one({"_id": doc_id})
    assert doc is not None
    assert doc["status"] == DocumentStatus.ready.value
    assert doc["page_count"] == 2
    assert doc["chunk_count"] >= 1
    assert doc["error"] is None

    # 5. Assert Redis Hash is updated to done
    job_hash = await get_job_record(job_id)
    assert job_hash["stage"] == "done"
    assert job_hash["progress"] == "100"

    # 6. Assert message was acknowledged (pending entries in group is 0)
    pending = await redis.xpending(settings.ingest_stream, settings.consumer_group)
    assert pending["pending"] == 0

    # 7. Assert vector store received chunks
    assert len(mock_vs.upserted_points) >= 1


@pytest.mark.asyncio
async def test_worker_retries_and_dlq_on_forced_failure(
    client: httpx.AsyncClient,
    registered_user: dict,
    make_pdf,
    monkeypatch: pytest.MonkeyPatch,
):
    settings = get_settings()
    redis = get_redis_client()
    db = get_db()
    mock_vs = MockVectorStore()
    fake_embedder = FakeEmbedder()

    # Max retries = 3
    monkeypatch.setattr(settings, "worker_max_retries", 3)
    monkeypatch.setattr(settings, "worker_retry_base_delay", 0.01)
    monkeypatch.setattr(settings, "worker_retry_max_delay", 0.05)

    await ensure_consumer_group(redis, settings.ingest_stream, settings.consumer_group)

    pdf_bytes = make_pdf("Forced failure document", pages=1)
    upload_resp = await client.post(
        "/documents",
        headers=registered_user["headers"],
        files={"file": ("fail_doc.pdf", pdf_bytes, "application/pdf")},
    )
    job_id = upload_resp.json()["job_id"]
    doc_id = upload_resp.json()["document_id"]

    # Read message from stream
    entries = await redis.xreadgroup(
        groupname=settings.consumer_group,
        consumername=settings.consumer_name,
        streams={settings.ingest_stream: ">"},
        count=1,
    )
    msg_id, payload = entries[0][1][0]

    # Corrupt the file_path to trigger failure
    corrupt_payload = dict(payload)
    corrupt_payload["file_path"] = "non_existent_file.pdf"

    # Attempt 1: fails, scheduled for retry
    success_1 = await process_single_job(
        msg_id, corrupt_payload, db, mock_vs, fake_embedder, redis
    )
    assert success_1 is False
    hash_1 = await get_job_record(job_id)
    assert hash_1["attempts"] == "1"

    # Attempt 2: fails, scheduled for retry
    success_2 = await process_single_job(
        msg_id, corrupt_payload, db, mock_vs, fake_embedder, redis
    )
    assert success_2 is False
    hash_2 = await get_job_record(job_id)
    assert hash_2["attempts"] == "2"

    # Attempt 3: exhausted (attempts >= 3) -> routes to DLQ and rollbacks
    success_3 = await process_single_job(
        msg_id, corrupt_payload, db, mock_vs, fake_embedder, redis
    )
    assert success_3 is False

    # Verify MongoDB document marked failed
    doc = await db.documents.find_one({"_id": doc_id})
    assert doc["status"] == DocumentStatus.failed.value
    assert "non_existent_file" in doc["error"]

    # Verify Redis Hash marked failed
    job_hash = await get_job_record(job_id)
    assert job_hash["stage"] == "failed"

    # Verify pushed to DLQ stream (jobs:dead)
    dlq_messages = await redis.xrange(settings.dlq_stream)
    assert len(dlq_messages) >= 1
    dlq_payload = dlq_messages[-1][1]
    assert dlq_payload["job_id"] == job_id
    assert dlq_payload["doc_id"] == doc_id

    # Verify original message acknowledged from ingest stream
    pending = await redis.xpending(settings.ingest_stream, settings.consumer_group)
    assert pending["pending"] == 0


@pytest.mark.asyncio
async def test_worker_crash_recovery_autoclaim(
    client: httpx.AsyncClient,
    registered_user: dict,
    make_pdf,
):
    settings = get_settings()
    redis = get_redis_client()
    db = get_db()
    mock_vs = MockVectorStore()
    fake_embedder = FakeEmbedder()

    await ensure_consumer_group(redis, settings.ingest_stream, settings.consumer_group)

    pdf_bytes = make_pdf("Crash recovery test", pages=1)
    upload_resp = await client.post(
        "/documents",
        headers=registered_user["headers"],
        files={"file": ("crash_recovery.pdf", pdf_bytes, "application/pdf")},
    )
    job_id = upload_resp.json()["job_id"]
    doc_id = upload_resp.json()["document_id"]

    # Worker 1 reads message into PEL (pending entries list) and "crashes"
    entries = await redis.xreadgroup(
        groupname=settings.consumer_group,
        consumername="worker-crashed",
        streams={settings.ingest_stream: ">"},
        count=1,
    )
    assert len(entries) >= 1

    # Worker 2 runs reclaim_stuck_messages with min_idle_ms=0 to reclaim pending message
    reclaimed = await reclaim_stuck_messages(
        redis,
        stream=settings.ingest_stream,
        group=settings.consumer_group,
        consumer="worker-recovered",
        min_idle_ms=0,
    )
    assert len(reclaimed) >= 1
    rec_msg_id, rec_payload = reclaimed[0]
    assert rec_payload["job_id"] == job_id

    # Process reclaimed message to completion
    success = await process_single_job(
        rec_msg_id, rec_payload, db, mock_vs, fake_embedder, redis
    )
    assert success is True

    # Document ready with no duplicates
    doc = await db.documents.find_one({"_id": doc_id})
    assert doc["status"] == DocumentStatus.ready.value


@pytest.mark.asyncio
async def test_janitor_cleanup(make_pdf):
    db = get_db()
    mock_vs = MockVectorStore()

    old_time = datetime.now(timezone.utc) - timedelta(minutes=20)
    stuck_doc_id = "stuck_doc_123"

    await db.documents.insert_one({
        "_id": stuck_doc_id,
        "user_id": "user_xyz",
        "filename": "abandoned.pdf",
        "size_bytes": 1024,
        "status": DocumentStatus.processing.value,
        "created_at": old_time,
        "updated_at": old_time,
    })

    # Run janitor with 10 minute threshold
    cleaned = await cleanup_stuck_documents(db, mock_vs, threshold_seconds=600)
    assert cleaned >= 1

    # Check document marked failed
    doc = await db.documents.find_one({"_id": stuck_doc_id})
    assert doc["status"] == DocumentStatus.failed.value
    assert "timed out" in doc["error"].lower()

    # Verify vector cleanup called
    assert stuck_doc_id in mock_vs.deleted_docs
