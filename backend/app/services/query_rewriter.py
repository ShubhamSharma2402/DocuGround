"""Query rewriting: turn a follow-up question + history into a standalone question."""

from __future__ import annotations

import structlog

from app.services.llm import LLMProvider

log = structlog.get_logger(__name__)

_REWRITE_SYSTEM = """You are a query reformulation assistant for a document search system.
Your job is to resolve ambiguous references, pronouns (it, its, they, their, this, that), or elliptical follow-ups using the conversation history.

GUIDELINES:
1. If the follow-up question uses pronouns (e.g. "how much does it weigh?", "what is its capacity?"), replace the pronoun with the specific entity discussed.
2. If the follow-up question introduces a distinct feature, category, or general question (e.g. "what are orbital benefits?", "who is the manufacturer?", "what warranty is included?"), DO NOT inappropriately narrow or restrict it to a specific previously mentioned product unless the user explicitly refers to it.
3. Preserve the core search intent without adding unnecessary constraints.
4. Output ONLY the standalone rewritten question — no explanation, no preamble."""


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
