"""Integration tests for the Chat endpoint: streaming, retrieval, gate refusal, rewrite, and isolation."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from bson import ObjectId

from app.core.config import get_settings
from app.db.client import get_db
from app.db.models import DocumentStatus, MessageRole, utc_now
from app.services.llm import FakeLLMProvider, REFUSAL_MESSAGE
from app.services.retriever import RetrievedChunk, RetrievalResult


@pytest.fixture
def mock_retrieval_pass():
    """Returns a RetrievalResult where gate has passed with good scores."""
    chunks = [
        RetrievedChunk(
            point_id="chunk-1",
            doc_id="doc-123",
            user_id="user-123",
            page=1,
            chunk_index=0,
            text="DocuGround is a production-grade RAG system built on FastAPI and Qdrant [1].",
            dense_score=0.88,
            sparse_score=0.75,
            rrf_score=0.032,
            rerank_score=0.82,
        ),
        RetrievedChunk(
            point_id="chunk-2",
            doc_id="doc-123",
            user_id="user-123",
            page=2,
            chunk_index=1,
            text="It enforces strict refusal gates when queries are out of scope [2].",
            dense_score=0.81,
            sparse_score=0.69,
            rrf_score=0.030,
            rerank_score=0.74,
        ),
    ]
    return RetrievalResult(
        chunks=chunks,
        gate_passed=True,
        best_dense_score=0.88,
        best_rerank_score=0.82,
        refusal_reason=None,
    )


@pytest.fixture
def mock_retrieval_refusal():
    """Returns a RetrievalResult where gate failed (score below threshold)."""
    return RetrievalResult(
        chunks=[],
        gate_passed=False,
        best_dense_score=0.35,
        best_rerank_score=None,
        refusal_reason="dense_score_below_cutoff:0.35<0.7",
    )


@pytest.mark.asyncio
async def test_chat_happy_path_streaming(
    client: httpx.AsyncClient,
    registered_user: dict,
    mock_retrieval_pass,
):
    db = get_db()
    user_id = registered_user["user_id"]
    doc_id = str(ObjectId())

    # Create ready document
    await db.documents.insert_one({
        "_id": doc_id,
        "user_id": user_id,
        "filename": "guide.pdf",
        "size_bytes": 1024,
        "status": DocumentStatus.ready.value,
        "page_count": 2,
        "chunk_count": 2,
        "created_at": utc_now(),
        "updated_at": utc_now(),
    })

    # Align mock chunks with the actual doc_id
    for chunk in mock_retrieval_pass.chunks:
        chunk.doc_id = doc_id

    fake_llm = FakeLLMProvider("DocuGround is built on FastAPI and Qdrant [1].")

    with patch("app.api.chat.HybridRetriever.retrieve", new=AsyncMock(return_value=mock_retrieval_pass)), \
         patch("app.api.chat.get_llm_provider", return_value=fake_llm):

        payload = {
            "document_ids": [doc_id],
            "question": "What is DocuGround built on?",
        }

        events = []
        async with client.stream("POST", "/chat", headers=registered_user["headers"], json=payload) as resp:
            assert resp.status_code == 200
            assert "text/event-stream" in resp.headers["content-type"]

            current_event = None
            async for line in resp.aiter_lines():
                if line.startswith("event: "):
                    current_event = line[7:].strip()
                elif line.startswith("data: "):
                    data = json.loads(line[6:])
                    events.append((current_event, data))

    # Assert SSE event ordering: citations first, then tokens, then done
    event_types = [e[0] for e in events]
    assert event_types[0] == "citations"
    assert "token" in event_types
    assert event_types[-1] == "done"

    # Verify citations payload structure
    citations_data = events[0][1]
    assert len(citations_data) == 2
    assert citations_data[0]["page"] == 1
    assert citations_data[0]["doc_name"] == "guide.pdf"

    # Verify token payload
    token_texts = [e[1]["token"] for e in events if e[0] == "token"]
    full_text = "".join(token_texts)
    assert "DocuGround" in full_text

    # Verify messages saved to MongoDB
    conv_id = events[-1][1]["conversation_id"]
    db_messages = await db.messages.find({"conversation_id": conv_id}).sort("created_at", 1).to_list(length=10)
    assert len(db_messages) == 2
    assert db_messages[0]["role"] == MessageRole.user.value
    assert db_messages[0]["content"] == "What is DocuGround built on?"
    assert db_messages[1]["role"] == MessageRole.assistant.value
    assert db_messages[1]["gate_passed"] is True
    assert len(db_messages[1]["citations"]) == 2


@pytest.mark.asyncio
async def test_chat_refusal_gate_skips_llm(
    client: httpx.AsyncClient,
    registered_user: dict,
    mock_retrieval_refusal,
):
    db = get_db()
    user_id = registered_user["user_id"]
    doc_id = str(ObjectId())

    await db.documents.insert_one({
        "_id": doc_id,
        "user_id": user_id,
        "filename": "guide.pdf",
        "size_bytes": 1024,
        "status": DocumentStatus.ready.value,
        "created_at": utc_now(),
        "updated_at": utc_now(),
    })

    fake_llm = FakeLLMProvider("Should not be called")

    with patch("app.api.chat.HybridRetriever.retrieve", new=AsyncMock(return_value=mock_retrieval_refusal)), \
         patch("app.api.chat.get_llm_provider", return_value=fake_llm):

        payload = {
            "document_ids": [doc_id],
            "question": "What is the capital of Mars?",
        }

        events = []
        async with client.stream("POST", "/chat", headers=registered_user["headers"], json=payload) as resp:
            assert resp.status_code == 200
            current_event = None
            async for line in resp.aiter_lines():
                if line.startswith("event: "):
                    current_event = line[7:].strip()
                elif line.startswith("data: "):
                    data = json.loads(line[6:])
                    events.append((current_event, data))

    # CRITICAL ACCEPTANCE TEST: Assert LLM generation was NEVER invoked!
    assert fake_llm.call_count == 0

    # Stream returned exact refusal sentence
    token_events = [e[1]["token"] for e in events if e[0] == "token"]
    assert REFUSAL_MESSAGE in "".join(token_events)

    done_event = [e[1] for e in events if e[0] == "done"][0]
    assert done_event["gate_passed"] is False

    # Verify refusal recorded in database
    conv_id = done_event["conversation_id"]
    assistant_msg = await db.messages.find_one({
        "conversation_id": conv_id,
        "role": MessageRole.assistant.value,
    })
    assert assistant_msg["content"] == REFUSAL_MESSAGE
    assert assistant_msg["gate_passed"] is False


@pytest.mark.asyncio
async def test_cross_user_chat_isolation(
    client: httpx.AsyncClient,
    registered_user: dict,
    second_user: dict,
):
    db = get_db()
    user1_id = registered_user["user_id"]
    doc_id = str(ObjectId())

    # User 1 owns document
    await db.documents.insert_one({
        "_id": doc_id,
        "user_id": user1_id,
        "filename": "private_user1.pdf",
        "size_bytes": 1024,
        "status": DocumentStatus.ready.value,
        "created_at": utc_now(),
        "updated_at": utc_now(),
    })

    # User 2 tries to chat on User 1's document
    payload = {
        "document_ids": [doc_id],
        "question": "Tell me user 1 secrets",
    }
    resp = await client.post("/chat", headers=second_user["headers"], json=payload)
    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_document_not_ready_rejected(
    client: httpx.AsyncClient,
    registered_user: dict,
):
    db = get_db()
    user_id = registered_user["user_id"]
    doc_id = str(ObjectId())

    # Document still in processing
    await db.documents.insert_one({
        "_id": doc_id,
        "user_id": user_id,
        "filename": "in_progress.pdf",
        "size_bytes": 1024,
        "status": DocumentStatus.processing.value,
        "created_at": utc_now(),
        "updated_at": utc_now(),
    })

    payload = {"document_ids": [doc_id], "question": "Are you done?"}
    resp = await client.post("/chat", headers=registered_user["headers"], json=payload)
    assert resp.status_code == 422
    assert "not ready" in resp.json()["message"].lower()


@pytest.mark.asyncio
async def test_conversation_crud_and_messages(
    client: httpx.AsyncClient,
    registered_user: dict,
    second_user: dict,
):
    db = get_db()
    user_id = registered_user["user_id"]
    conv_id = str(ObjectId())

    await db.conversations.insert_one({
        "_id": conv_id,
        "user_id": user_id,
        "document_ids": ["doc-1"],
        "title": "CRUD Test Conversation",
        "created_at": utc_now(),
    })

    await db.messages.insert_one({
        "_id": str(ObjectId()),
        "conversation_id": conv_id,
        "role": MessageRole.user.value,
        "content": "Hello",
        "citations": [],
        "retrieval_scores": [],
        "gate_passed": None,
        "verdict": None,
        "latency_ms": None,
        "created_at": utc_now(),
    })

    # 1. List conversations
    resp = await client.get("/conversations", headers=registered_user["headers"])
    assert resp.status_code == 200
    conv_list = resp.json()
    assert len(conv_list) >= 1
    assert any(c["id"] == conv_id for c in conv_list)

    # 2. Get conversation detail
    detail_resp = await client.get(f"/conversations/{conv_id}", headers=registered_user["headers"])
    assert detail_resp.status_code == 200
    detail = detail_resp.json()
    assert detail["id"] == conv_id
    assert len(detail["messages"]) == 1

    # 3. User 2 cannot access User 1 conversation
    unauth_resp = await client.get(f"/conversations/{conv_id}", headers=second_user["headers"])
    assert unauth_resp.status_code == 403

    # 4. Delete conversation
    del_resp = await client.delete(f"/conversations/{conv_id}", headers=registered_user["headers"])
    assert del_resp.status_code == 200

    # Ensure conversation and messages removed
    assert await db.conversations.find_one({"_id": conv_id}) is None
    assert await db.messages.count_documents({"conversation_id": conv_id}) == 0
