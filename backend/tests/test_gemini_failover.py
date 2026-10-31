"""Tests for Google Gemini Provider and automated dual-key failover / rotation."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from app.core.config import get_settings
from app.services.llm import GeminiProvider


@pytest.mark.asyncio
async def test_gemini_provider_dual_key_failover(monkeypatch: pytest.MonkeyPatch):
    """Verify that when primary key encounters 429 quota exhaustion, it fails over to the backup key seamlessly."""
    settings = get_settings()
    monkeypatch.setattr(settings, "gemini_api_key", "primary-key-111")
    monkeypatch.setattr(settings, "gemini_api_key_backup", "backup-key-222")
    monkeypatch.setattr(settings, "gemini_model", "gemini-2.5-flash")

    provider = GeminiProvider()
    assert len(provider._keys) == 2
    assert provider._keys[0] == "primary-key-111"
    assert provider._keys[1] == "backup-key-222"

    call_records = []

    async def mock_post(url: str, json: dict | None = None, **kwargs):
        call_records.append(url)
        # First call (with primary key) fails with 429 quota exhausted
        if "primary-key-111" in url:
            mock_resp = MagicMock(spec=httpx.Response)
            mock_resp.status_code = 429
            mock_resp.text = '{"error": {"message": "Resource has been exhausted (e.g. check quota)."}}'
            return mock_resp
        # Second call (with backup key) succeeds
        elif "backup-key-222" in url:
            mock_resp = MagicMock(spec=httpx.Response)
            mock_resp.status_code = 200
            mock_resp.json.return_value = {
                "candidates": [
                    {
                        "content": {
                            "parts": [{"text": "Answer from backup key"}]
                        }
                    }
                ]
            }
            return mock_resp
        return MagicMock(status_code=500)

    with patch.object(provider._http, "post", side_effect=mock_post):
        result = await provider.complete(
            system="System instruction",
            messages=[{"role": "user", "content": "Hello"}],
        )

        assert result == "Answer from backup key"
        # Assert that both keys were called sequentially
        assert len(call_records) == 2
        assert "primary-key-111" in call_records[0]
        assert "backup-key-222" in call_records[1]


@pytest.mark.asyncio
async def test_gemini_provider_streaming_failover(monkeypatch: pytest.MonkeyPatch):
    """Verify that streaming fails over to backup key on primary 429."""
    settings = get_settings()
    monkeypatch.setattr(settings, "gemini_api_key", "primary-key-111")
    monkeypatch.setattr(settings, "gemini_api_key_backup", "backup-key-222")

    provider = GeminiProvider()
    stream_calls = []

    class MockStreamContext:
        def __init__(self, status_code: int, lines: list[str]):
            self.status_code = status_code
            self.lines = lines

        async def __aenter__(self):
            resp = MagicMock()
            resp.status_code = self.status_code
            resp.aread = AsyncMock(return_value=b"Quota exhausted")
            async def _lines():
                for line in self.lines:
                    yield line
            resp.aiter_lines = _lines
            return resp

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

    def mock_stream(method: str, url: str, json: dict):
        stream_calls.append(url)
        if "primary-key-111" in url:
            return MockStreamContext(429, [])
        else:
            return MockStreamContext(200, [
                'data: {"candidates": [{"content": {"parts": [{"text": "Streaming "}]}}]}',
                'data: {"candidates": [{"content": {"parts": [{"text": "success!"}]}}]}',
            ])

    with patch.object(provider._http, "stream", side_effect=mock_stream):
        tokens = []
        async for token in provider.stream_complete(
            system="System prompt",
            messages=[{"role": "user", "content": "Question"}],
        ):
            tokens.append(token)

        assert "".join(tokens) == "Streaming success!"
        assert len(stream_calls) == 2
        assert "primary-key-111" in stream_calls[0]
        assert "backup-key-222" in stream_calls[1]
