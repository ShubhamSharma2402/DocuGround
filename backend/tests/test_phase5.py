"""Unit and integration tests for Phase 5: Rate Limiting, Metrics, Security Headers, and Verification."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest
import httpx
from bson import ObjectId

from app.core.config import get_settings
from app.db.models import DocumentStatus, VerificationVerdict
from app.services.llm import FakeLLMProvider, REFUSAL_MESSAGE
from app.services.retriever import RetrievalResult, RetrievedChunk


@pytest.mark.asyncio
async def test_security_headers_and_correlation_id(client: httpx.AsyncClient):
    """Verify security response headers and X-Request-ID propagation."""
    # 1. Custom request ID passed by client
    custom_id = "test-custom-req-id-12345"
    resp = await client.get("/health", headers={"X-Request-ID": custom_id})
    assert resp.status_code == 200
    assert resp.headers.get("X-Request-ID") == custom_id
    assert resp.headers.get("X-Content-Type-Options") == "nosniff"
    assert resp.headers.get("X-Frame-Options") == "DENY"
    assert resp.headers.get("X-XSS-Protection") == "1; mode=block"
    assert resp.headers.get("Referrer-Policy") == "strict-origin-when-cross-origin"

    # 2. Auto-generated request ID if none provided
    resp_auto = await client.get("/health")
    assert resp_auto.status_code == 200
    assert "X-Request-ID" in resp_auto.headers
    assert len(resp_auto.headers["X-Request-ID"]) > 10


@pytest.mark.asyncio
async def test_metrics_endpoint(client: httpx.AsyncClient):
    """Verify /metrics returns Prometheus text format containing DocuGround counters."""
    resp = await client.get("/metrics")
    assert resp.status_code == 200
    assert "text/plain" in resp.headers.get("content-type", "")
    content = resp.text
    assert "docuground_uploads_total" in content
    assert "docuground_jobs_total" in content
    assert "docuground_gate_refusals_total" in content
    assert "docuground_chat_requests_total" in content
    assert "docuground_query_latency_seconds" in content


@pytest.mark.asyncio
async def test_rate_limiting_upload(client: httpx.AsyncClient, registered_user: dict[str, str], make_pdf):
    """Verify that exceeding upload rate limit triggers 429 with Retry-After header."""
    settings = get_settings()
    # Temporarily lower the upload rate limit to 2 for this test
    original_limit = settings.rate_limit_upload_per_minute
    settings.rate_limit_upload_per_minute = 2
    try:
        pdf_bytes = make_pdf("Rate limit test", pages=1)
        files = {"file": ("test.pdf", pdf_bytes, "application/pdf")}

        # 1st upload -> 202
        r1 = await client.post("/documents", headers=registered_user["headers"], files={"file": ("test.pdf", pdf_bytes, "application/pdf")})
        assert r1.status_code == 202

        # 2nd upload -> 202
        r2 = await client.post("/documents", headers=registered_user["headers"], files={"file": ("test.pdf", pdf_bytes, "application/pdf")})
        assert r2.status_code == 202

        # 3rd upload -> 429 RateLimitError
        r3 = await client.post("/documents", headers=registered_user["headers"], files={"file": ("test.pdf", pdf_bytes, "application/pdf")})
        assert r3.status_code == 429
        assert "Retry-After" in r3.headers
        data = r3.json()
        assert data["error"] == "rate_limit_exceeded"
        assert "Rate limit exceeded" in data["message"]

    finally:
        settings.rate_limit_upload_per_minute = original_limit


@pytest.mark.asyncio
async def test_verification_pass_in_chat(
    client: httpx.AsyncClient,
    registered_user: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
):
    """Test chat verification pass: emits verification event with verdict and handles unsupported rejection."""
    settings = get_settings()
    monkeypatch.setattr(settings, "verification_enabled", True)
    monkeypatch.setattr(settings, "verification_reject_unsupported", True)

    from app.db.client import get_db
    from app.main import app
    db = app.dependency_overrides[get_db]()

    # Create ready document
    doc_id = str(ObjectId())
    await db.documents.insert_one({
        "_id": doc_id,
        "user_id": registered_user["user_id"],
        "filename": "verified_doc.pdf",
        "size_bytes": 1024,
        "status": DocumentStatus.ready.value,
        "page_count": 1,
        "chunk_count": 1,
    })

    # Mock retriever to pass gate with 1 chunk
    mock_chunk = RetrievedChunk(
        point_id="p-1",
        doc_id=doc_id,
        user_id=registered_user["user_id"],
        page=1,
        chunk_index=0,
        text="The revenue of Acme Corp in 2023 was $10M.",
        dense_score=0.92,
        sparse_score=0.88,
        rrf_score=0.033,
        rerank_score=0.85,
    )
    mock_retrieval = RetrievalResult(
        chunks=[mock_chunk],
        gate_passed=True,
        best_dense_score=0.92,
        best_rerank_score=0.85,
        refusal_reason=None,
    )

    with patch("app.api.chat.HybridRetriever.retrieve", new_callable=AsyncMock) as mock_retrieve, \
         patch("app.api.chat.get_llm_provider", return_value=FakeLLMProvider("Acme Corp had $10M revenue [1].")), \
         patch("app.api.chat.verify_answer", new_callable=AsyncMock) as mock_verify:

        mock_retrieve.return_value = mock_retrieval
        mock_verify.return_value = VerificationVerdict.supported

        # Send chat request
        resp = await client.post(
            "/chat",
            headers=registered_user["headers"],
            json={
                "document_ids": [doc_id],
                "question": "What was the revenue of Acme Corp in 2023?",
            },
        )
        assert resp.status_code == 200
        events = resp.text.split("\n\n")

        # Verify event: verification was emitted
        verif_event = [e for e in events if "event: verification" in e]
        assert len(verif_event) == 1
        verif_data = json.loads(verif_event[0].split("data: ")[1])
        assert verif_data["verdict"] == "supported"
        assert verif_data["rejected"] is False

        # Verify message stored in DB has verdict supported
        stored_msg = await db.messages.find_one({"role": "assistant"})
        assert stored_msg is not None
        assert stored_msg["verdict"] == "supported"
        assert "Acme Corp had $10M revenue" in stored_msg["content"]


@pytest.mark.asyncio
async def test_verification_reject_unsupported(
    client: httpx.AsyncClient,
    registered_user: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
):
    """Test that when verification verdict is 'unsupported' and reject_unsupported is enabled, the stored answer is replaced with refusal."""
    settings = get_settings()
    monkeypatch.setattr(settings, "verification_enabled", True)
    monkeypatch.setattr(settings, "verification_reject_unsupported", True)

    from app.db.client import get_db
    from app.main import app
    db = app.dependency_overrides[get_db]()

    doc_id = str(ObjectId())
    await db.documents.insert_one({
        "_id": doc_id,
        "user_id": registered_user["user_id"],
        "filename": "hallucinated_doc.pdf",
        "size_bytes": 1024,
        "status": DocumentStatus.ready.value,
        "page_count": 1,
        "chunk_count": 1,
    })

    mock_chunk = RetrievedChunk(
        point_id="p-2",
        doc_id=doc_id,
        user_id=registered_user["user_id"],
        page=1,
        chunk_index=0,
        text="The company has 50 employees.",
        dense_score=0.85,
        sparse_score=0.75,
        rrf_score=0.030,
        rerank_score=0.80,
    )
    mock_retrieval = RetrievalResult(
        chunks=[mock_chunk],
        gate_passed=True,
        best_dense_score=0.85,
        best_rerank_score=0.80,
        refusal_reason=None,
    )

    with patch("app.api.chat.HybridRetriever.retrieve", new_callable=AsyncMock) as mock_retrieve, \
         patch("app.api.chat.get_llm_provider", return_value=FakeLLMProvider("The CEO is John Doe.")), \
         patch("app.api.chat.verify_answer", new_callable=AsyncMock) as mock_verify:

        mock_retrieve.return_value = mock_retrieval
        mock_verify.return_value = VerificationVerdict.unsupported

        resp = await client.post(
            "/chat",
            headers=registered_user["headers"],
            json={
                "document_ids": [doc_id],
                "question": "Who is the CEO?",
            },
        )
        assert resp.status_code == 200
        events = resp.text.split("\n\n")

        verif_event = [e for e in events if "event: verification" in e]
        assert len(verif_event) == 1
        verif_data = json.loads(verif_event[0].split("data: ")[1])
        assert verif_data["verdict"] == "unsupported"
        assert verif_data["rejected"] is True

        stored_msg = await db.messages.find_one({"role": "assistant"})
        assert stored_msg is not None
        assert stored_msg["verdict"] == "unsupported"
        # Since verification rejected it, persisted content is the refusal message
        assert stored_msg["content"] == REFUSAL_MESSAGE
