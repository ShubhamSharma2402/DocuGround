"""Chat endpoint: query rewriting, hybrid retrieval, rerank, refusal gate, streaming SSE, and persistence."""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import structlog
from bson import ObjectId
from fastapi import APIRouter, Depends, Request, status
from fastapi.responses import StreamingResponse
from motor.motor_asyncio import AsyncIOMotorDatabase

from app.api.deps import get_current_user
from app.core.config import get_settings
from app.core.errors import ForbiddenError, NotFoundError, ValidationError
from app.core.metrics import (
    DOCUGROUND_CHAT_REQUESTS_TOTAL,
    DOCUGROUND_GATE_REFUSALS_TOTAL,
    DOCUGROUND_QUERY_LATENCY_SECONDS,
)
from app.core.rate_limit import RateLimiter
from app.db.client import get_db
from app.db.models import (
    ChatRequest,
    Citation,
    ConversationDB,
    ConversationDetail,
    ConversationRead,
    DocumentStatus,
    MessageDB,
    MessageRead,
    MessageRole,
    UserDB,
    VerificationVerdict,
    utc_now,
)
from app.services.embedder import get_embedder
from app.services.llm import REFUSAL_MESSAGE, get_llm_provider
from app.services.query_rewriter import rewrite_query
from app.services.reranker import get_reranker
from app.services.retriever import HybridRetriever
from app.services.vector_store import get_vector_store
from app.services.verifier import verify_answer

log = structlog.get_logger(__name__)

router = APIRouter(tags=["chat"])

SYSTEM_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "system.txt"


def _load_system_prompt_template() -> str:
    if SYSTEM_PROMPT_PATH.exists():
        return SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
    return (
        "You are a document assistant. Answer strictly based on the provided context.\n"
        "Context:\n{context}\n"
    )


@router.post("/chat")
async def chat_endpoint(
    payload: ChatRequest,
    request: Request,
    current_user: UserDB = Depends(get_current_user),
    db: AsyncIOMotorDatabase = Depends(get_db),
    _rate_limit: None = Depends(RateLimiter("chat", "rate_limit_chat_per_minute")),
) -> StreamingResponse:
    """Live streaming chat with documents: rewrite -> retrieve -> rerank -> gate -> stream SSE."""
    t0 = time.monotonic()
    DOCUGROUND_CHAT_REQUESTS_TOTAL.inc()
    user_id = str(current_user.id)
    settings = get_settings()

    # 1. Verify ownership and ready status of every document_id
    doc_names_map: dict[str, str] = {}
    for doc_id in payload.document_ids:
        doc = await db.documents.find_one({"_id": doc_id})
        if not doc:
            raise NotFoundError(f"Document {doc_id} not found")
        if doc["user_id"] != user_id:
            raise ForbiddenError(f"You do not own document {doc_id}")
        if doc["status"] != DocumentStatus.ready.value:
            raise ValidationError(
                f"Document '{doc['filename']}' is not ready yet (status: {doc['status']})"
            )
        doc_names_map[doc_id] = doc["filename"]

    # 2. Conversation handling
    conv_id = payload.conversation_id
    if conv_id:
        conv = await db.conversations.find_one({"_id": conv_id})
        if not conv:
            raise NotFoundError(f"Conversation {conv_id} not found")
        if conv["user_id"] != user_id:
            raise ForbiddenError("You do not have access to this conversation")
    else:
        conv_id = str(ObjectId())
        title = payload.question[:60] + ("..." if len(payload.question) > 60 else "")
        await db.conversations.insert_one({
            "_id": conv_id,
            "user_id": user_id,
            "document_ids": payload.document_ids,
            "title": title,
            "created_at": utc_now(),
        })

    # 3. Fetch conversation history for query rewriting
    cursor = db.messages.find({"conversation_id": conv_id}).sort("created_at", 1)
    history_docs = await cursor.to_list(length=10)
    history = [
        {"role": m["role"], "content": m["content"]}
        for m in history_docs
        if m.get("content")
    ]

    llm = get_llm_provider()

    # 4. Query rewriting (skipped automatically if history is empty)
    t_rw0 = time.monotonic()
    effective_query = await rewrite_query(payload.question, history, llm)
    DOCUGROUND_QUERY_LATENCY_SECONDS.labels(step="rewrite").observe(time.monotonic() - t_rw0)

    # 5. Hybrid retrieval + cross-encoder rerank + refusal gate
    vector_store = get_vector_store()
    embedder = get_embedder()
    reranker = get_reranker()
    retriever = HybridRetriever(vector_store, embedder, reranker)

    t_ret0 = time.monotonic()
    retrieval_result = await retriever.retrieve(
        query=effective_query,
        user_id=user_id,
        doc_ids=payload.document_ids,
    )
    DOCUGROUND_QUERY_LATENCY_SECONDS.labels(step="retrieve").observe(time.monotonic() - t_ret0)

    # Persist user question message
    user_msg_id = str(ObjectId())
    await db.messages.insert_one({
        "_id": user_msg_id,
        "conversation_id": conv_id,
        "role": MessageRole.user.value,
        "content": payload.question,
        "citations": [],
        "retrieval_scores": [],
        "gate_passed": None,
        "verdict": None,
        "latency_ms": None,
        "created_at": utc_now(),
    })

    assistant_msg_id = str(ObjectId())

    # 6. SSE Event Streaming Generator
    async def sse_event_generator():
        # Case A: Refusal Gate Triggered (DO NOT call LLM for generation!)
        if not retrieval_result.gate_passed:
            DOCUGROUND_GATE_REFUSALS_TOTAL.inc()
            log.info(
                "Chat query refused by threshold gate",
                reason=retrieval_result.refusal_reason,
                best_dense=retrieval_result.best_dense_score,
                best_rerank=retrieval_result.best_rerank_score,
                conv_id=conv_id,
            )

            # Emit refusal as token and finish
            refusal_payload = json.dumps({"token": REFUSAL_MESSAGE})
            yield f"event: token\ndata: {refusal_payload}\n\n"

            latency_ms = round((time.monotonic() - t0) * 1000.0, 2)
            done_payload = json.dumps({
                "conversation_id": conv_id,
                "message_id": assistant_msg_id,
                "gate_passed": False,
                "refusal_reason": retrieval_result.refusal_reason,
                "verdict": None,
                "latency_ms": latency_ms,
            })
            yield f"event: done\ndata: {done_payload}\n\n"

            # Persist assistant refusal message
            await db.messages.insert_one({
                "_id": assistant_msg_id,
                "conversation_id": conv_id,
                "role": MessageRole.assistant.value,
                "content": REFUSAL_MESSAGE,
                "citations": [],
                "retrieval_scores": [retrieval_result.best_dense_score],
                "gate_passed": False,
                "verdict": None,
                "latency_ms": latency_ms,
                "created_at": utc_now(),
            })
            return

        # Case B: Gate Passed -> Build citations and generate answer
        citations: list[Citation] = [
            Citation(
                chunk_id=c.point_id,
                doc_id=c.doc_id,
                doc_name=doc_names_map.get(c.doc_id, "Document"),
                page=c.page,
                snippet=c.text[:240].strip(),
                score=round(c.rerank_score if c.rerank_score is not None else c.dense_score, 4),
            )
            for c in retrieval_result.chunks
        ]

        # 1. Emit citations event first
        citations_data = [cit.model_dump() for cit in citations]
        yield f"event: citations\ndata: {json.dumps(citations_data)}\n\n"

        # 2. Build strict prompt
        context_blocks = [
            f"[{i+1}] (Source: {c.doc_name}, Page {c.page})\n{c.snippet}"
            for i, c in enumerate(citations)
        ]
        system_template = _load_system_prompt_template()
        system_prompt = system_template.replace("{context}", "\n\n".join(context_blocks))

        collected_tokens: list[str] = []

        # 3. Stream LLM tokens
        t_gen0 = time.monotonic()
        try:
            async for token in llm.stream_complete(
                system=system_prompt,
                messages=[{"role": "user", "content": effective_query}],
                temperature=0.1,
            ):
                if await request.is_disconnected():
                    log.info("Client disconnected mid-stream", conv_id=conv_id)
                    break

                collected_tokens.append(token)
                token_payload = json.dumps({"token": token})
                yield f"event: token\ndata: {token_payload}\n\n"

        except Exception as exc:
            log.error("LLM stream error", error=str(exc))
            err_payload = json.dumps({"error": "Generation error"})
            yield f"event: error\ndata: {err_payload}\n\n"

        DOCUGROUND_QUERY_LATENCY_SECONDS.labels(step="generate").observe(time.monotonic() - t_gen0)

        full_answer = "".join(collected_tokens).strip() or REFUSAL_MESSAGE

        # 4. Optional verification pass
        verdict = None
        if settings.verification_enabled and full_answer != REFUSAL_MESSAGE and citations:
            t_v0 = time.monotonic()
            try:
                verdict = await verify_answer(
                    answer=full_answer,
                    context_chunks=[c.snippet for c in citations],
                    llm=llm,
                )
            except Exception as exc:
                log.warning("Verification pass failed", error=str(exc))
                verdict = VerificationVerdict.partially

            DOCUGROUND_QUERY_LATENCY_SECONDS.labels(step="verify").observe(time.monotonic() - t_v0)

            is_rejected = (
                settings.verification_reject_unsupported
                and verdict == VerificationVerdict.unsupported
            )
            verification_payload = json.dumps({
                "verdict": verdict.value,
                "rejected": is_rejected,
            })
            yield f"event: verification\ndata: {verification_payload}\n\n"

            if is_rejected:
                log.info(
                    "Answer rejected by verification pass (unsupported claims)",
                    conv_id=conv_id,
                )
                full_answer = REFUSAL_MESSAGE

        latency_ms = round((time.monotonic() - t0) * 1000.0, 2)

        # 5. Emit done event
        done_payload = json.dumps({
            "conversation_id": conv_id,
            "message_id": assistant_msg_id,
            "gate_passed": True,
            "verdict": verdict.value if verdict else None,
            "latency_ms": latency_ms,
        })
        yield f"event: done\ndata: {done_payload}\n\n"

        # 6. Persist assistant response
        await db.messages.insert_one({
            "_id": assistant_msg_id,
            "conversation_id": conv_id,
            "role": MessageRole.assistant.value,
            "content": full_answer,
            "citations": [c.model_dump() for c in citations],
            "retrieval_scores": [
                c.rerank_score or c.dense_score for c in retrieval_result.chunks
            ],
            "gate_passed": True,
            "verdict": verdict.value if verdict else None,
            "latency_ms": latency_ms,
            "created_at": utc_now(),
        })

    return StreamingResponse(
        sse_event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/conversations", response_model=list[ConversationRead])
async def list_conversations(
    current_user: UserDB = Depends(get_current_user),
    db: AsyncIOMotorDatabase = Depends(get_db),
) -> list[ConversationRead]:
    """List all conversations for the authenticated user."""
    user_id = str(current_user.id)
    cursor = db.conversations.find({"user_id": user_id}).sort("created_at", -1)
    convs = await cursor.to_list(length=100)

    return [
        ConversationRead(
            id=str(c["_id"]),
            user_id=c["user_id"],
            document_ids=c.get("document_ids", []),
            title=c.get("title", "Conversation"),
            created_at=c["created_at"],
        )
        for c in convs
    ]


@router.get("/conversations/{conversation_id}", response_model=ConversationDetail)
async def get_conversation(
    conversation_id: str,
    current_user: UserDB = Depends(get_current_user),
    db: AsyncIOMotorDatabase = Depends(get_db),
) -> ConversationDetail:
    """Get a conversation by ID with all its messages."""
    user_id = str(current_user.id)
    conv = await db.conversations.find_one({"_id": conversation_id})
    if not conv:
        raise NotFoundError("Conversation not found")
    if conv["user_id"] != user_id:
        raise ForbiddenError("You do not have access to this conversation")

    cursor = db.messages.find({"conversation_id": conversation_id}).sort("created_at", 1)
    msg_docs = await cursor.to_list(length=200)

    messages = [
        MessageRead(
            id=str(m["_id"]),
            conversation_id=m["conversation_id"],
            role=MessageRole(m["role"]),
            content=m["content"],
            citations=[Citation(**c) for c in m.get("citations", [])],
            retrieval_scores=m.get("retrieval_scores", []),
            gate_passed=m.get("gate_passed"),
            verdict=m.get("verdict"),
            latency_ms=m.get("latency_ms"),
            created_at=m["created_at"],
        )
        for m in msg_docs
    ]

    return ConversationDetail(
        id=str(conv["_id"]),
        user_id=conv["user_id"],
        document_ids=conv.get("document_ids", []),
        title=conv.get("title", "Conversation"),
        created_at=conv["created_at"],
        messages=messages,
    )


@router.delete("/conversations/{conversation_id}")
async def delete_conversation(
    conversation_id: str,
    current_user: UserDB = Depends(get_current_user),
    db: AsyncIOMotorDatabase = Depends(get_db),
) -> dict[str, str]:
    """Delete a conversation and all its messages."""
    user_id = str(current_user.id)
    conv = await db.conversations.find_one({"_id": conversation_id})
    if not conv:
        raise NotFoundError("Conversation not found")
    if conv["user_id"] != user_id:
        raise ForbiddenError("You do not have access to this conversation")

    await db.messages.delete_many({"conversation_id": conversation_id})
    await db.conversations.delete_one({"_id": conversation_id})

    return {"message": "Conversation deleted successfully"}
