"""Integration tests for document upload, listing, and deletion."""

from __future__ import annotations

import httpx
import pytest

from app.core.config import get_settings
from app.db.redis import get_job_record, get_redis_client


@pytest.mark.asyncio
async def test_upload_valid_pdf_returns_202(
    client: httpx.AsyncClient,
    registered_user: dict,
    make_pdf,
):
    pdf_bytes = make_pdf("DocuGround integration test", pages=2)
    files = {"file": ("integration_test.pdf", pdf_bytes, "application/pdf")}

    resp = await client.post(
        "/documents",
        headers=registered_user["headers"],
        files=files,
    )
    assert resp.status_code == 202
    data = resp.json()
    assert "job_id" in data
    assert "document_id" in data

    job_id = data["job_id"]
    doc_id = data["document_id"]

    # Verify Redis Hash was created
    job_hash = await get_job_record(job_id)
    assert job_hash["stage"] == "queued"
    assert job_hash["doc_id"] == doc_id
    assert job_hash["user_id"] == registered_user["user_id"]
    assert job_hash["progress"] == "0"

    # Verify Redis stream has the job
    redis = get_redis_client()
    settings = get_settings()
    messages = await redis.xrange(settings.ingest_stream)
    assert len(messages) >= 1
    stream_payload = messages[-1][1]
    assert stream_payload["job_id"] == job_id
    assert stream_payload["doc_id"] == doc_id


@pytest.mark.asyncio
async def test_upload_invalid_magic_bytes_rejected(
    client: httpx.AsyncClient,
    registered_user: dict,
):
    fake_content = b"This is plain text pretending to be a pdf."
    files = {"file": ("fake.pdf", fake_content, "application/pdf")}

    resp = await client.post(
        "/documents",
        headers=registered_user["headers"],
        files=files,
    )
    assert resp.status_code == 422
    assert resp.json()["error"] == "invalid_pdf"


@pytest.mark.asyncio
async def test_upload_exceeds_max_size_rejected_413(
    client: httpx.AsyncClient,
    registered_user: dict,
    monkeypatch: pytest.MonkeyPatch,
):
    settings = get_settings()
    # Temporarily set max_upload_mb to 0 (trigger cutoff with small payload)
    monkeypatch.setattr(settings, "max_upload_mb", 0)

    pdf_bytes = b"%PDF-1.4 " + (b"0" * 1000)
    files = {"file": ("too_large.pdf", pdf_bytes, "application/pdf")}

    resp = await client.post(
        "/documents",
        headers=registered_user["headers"],
        files=files,
    )
    assert resp.status_code == 413
    assert resp.json()["error"] == "file_too_large"


@pytest.mark.asyncio
async def test_upload_encrypted_pdf_rejected(
    client: httpx.AsyncClient,
    registered_user: dict,
    make_pdf,
):
    pdf_bytes = make_pdf("Encrypted test", pages=1, encrypted=True)
    files = {"file": ("encrypted.pdf", pdf_bytes, "application/pdf")}

    resp = await client.post(
        "/documents",
        headers=registered_user["headers"],
        files=files,
    )
    assert resp.status_code == 422
    assert "encrypted" in resp.json()["message"].lower() or resp.json()["error"] == "invalid_pdf"


@pytest.mark.asyncio
async def test_list_and_get_documents(
    client: httpx.AsyncClient,
    registered_user: dict,
    make_pdf,
):
    pdf_bytes = make_pdf("Doc 1", pages=1)
    upload_resp = await client.post(
        "/documents",
        headers=registered_user["headers"],
        files={"file": ("doc1.pdf", pdf_bytes, "application/pdf")},
    )
    doc_id = upload_resp.json()["document_id"]

    # List
    list_resp = await client.get("/documents", headers=registered_user["headers"])
    assert list_resp.status_code == 200
    docs = list_resp.json()
    assert len(docs) >= 1
    assert any(d["id"] == doc_id for d in docs)

    # Get single
    get_resp = await client.get(f"/documents/{doc_id}", headers=registered_user["headers"])
    assert get_resp.status_code == 200
    assert get_resp.json()["id"] == doc_id


@pytest.mark.asyncio
async def test_cross_user_isolation_document_access(
    client: httpx.AsyncClient,
    registered_user: dict,
    second_user: dict,
    make_pdf,
):
    # User 1 uploads
    pdf_bytes = make_pdf("User 1 private doc", pages=1)
    upload_resp = await client.post(
        "/documents",
        headers=registered_user["headers"],
        files={"file": ("private.pdf", pdf_bytes, "application/pdf")},
    )
    doc_id = upload_resp.json()["document_id"]

    # User 2 tries to GET
    get_resp = await client.get(f"/documents/{doc_id}", headers=second_user["headers"])
    assert get_resp.status_code == 403

    # User 2 tries to DELETE
    del_resp = await client.delete(f"/documents/{doc_id}", headers=second_user["headers"])
    assert del_resp.status_code == 403


@pytest.mark.asyncio
async def test_delete_document_success(
    client: httpx.AsyncClient,
    registered_user: dict,
    make_pdf,
):
    pdf_bytes = make_pdf("To be deleted", pages=1)
    upload_resp = await client.post(
        "/documents",
        headers=registered_user["headers"],
        files={"file": ("delete_me.pdf", pdf_bytes, "application/pdf")},
    )
    doc_id = upload_resp.json()["document_id"]

    del_resp = await client.delete(f"/documents/{doc_id}", headers=registered_user["headers"])
    assert del_resp.status_code == 200

    # Ensure it no longer exists
    get_resp = await client.get(f"/documents/{doc_id}", headers=registered_user["headers"])
    assert get_resp.status_code == 404
