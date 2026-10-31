"""Unit tests for query rewriter."""

from __future__ import annotations

import pytest

from app.services.llm import FakeLLMProvider
from app.services.query_rewriter import rewrite_query


class TestQueryRewriter:
    @pytest.mark.asyncio
    async def test_no_history_no_llm_call(self):
        llm = FakeLLMProvider("Rewritten?")
        result = await rewrite_query("What is revenue?", history=[], llm=llm)
        assert result == "What is revenue?"
        assert llm.call_count == 0

    @pytest.mark.asyncio
    async def test_with_history_calls_llm(self):
        llm = FakeLLMProvider("What was the total revenue in 2023?")
        history = [
            {"role": "user", "content": "Tell me about 2023 financials."},
            {"role": "assistant", "content": "Revenue was $10M."},
        ]
        result = await rewrite_query("And the total?", history=history, llm=llm)
        assert llm.call_count == 1

    @pytest.mark.asyncio
    async def test_empty_question_no_call(self):
        llm = FakeLLMProvider("")
        result = await rewrite_query("", history=[], llm=llm)
        assert result == ""
        assert llm.call_count == 0
