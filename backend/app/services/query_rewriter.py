"""Query rewriting: turn a follow-up question + history into a standalone question."""

from __future__ import annotations

import structlog

from app.services.llm import LLMProvider

log = structlog.get_logger(__name__)

_REWRITE_SYSTEM = """You are a query reformulation assistant.
Given a conversation history and a follow-up question, rewrite the follow-up question
into a self-contained, standalone question that can be understood without the history.
Output ONLY the rewritten question — no explanation, no preamble."""


async def rewrite_query(
    question: str,
    history: list[dict[str, str]],
    llm: LLMProvider,
) -> str:
    """Return a standalone question. If history is empty, returns question unchanged (no LLM call)."""
    if not history:
        log.debug("Query rewrite skipped: no history")
        return question

    history_text = "\n".join(
        f"{msg['role'].upper()}: {msg['content']}" for msg in history[-6:]
    )
    prompt = f"Conversation history:\n{history_text}\n\nFollow-up question: {question}"

    rewritten = await llm.complete(
        system=_REWRITE_SYSTEM,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    rewritten = rewritten.strip()
    log.info("Query rewritten", original=question, rewritten=rewritten)
    return rewritten
