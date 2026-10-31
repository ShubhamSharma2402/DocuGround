"""LLM provider interface with Gemini and Ollama implementations."""

from __future__ import annotations

import abc
from typing import AsyncIterator

from app.core.config import get_settings
from app.core.logging import get_logger

log = get_logger(__name__)

REFUSAL_MESSAGE = "I couldn't find this in the document(s)."

# Gemini API base — v1beta with header-based x-goog-api-key auth
GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"


class LLMProvider(abc.ABC):
    @abc.abstractmethod
    async def stream_complete(
        self, system: str, messages: list[dict[str, str]], temperature: float = 0.1
    ) -> AsyncIterator[str]: ...

    @abc.abstractmethod
    async def complete(
        self, system: str, messages: list[dict[str, str]], temperature: float = 0.1
    ) -> str: ...


class GeminiProvider(LLMProvider):
    """Google Gemini provider with automated dual-key failover and rotation."""

    def __init__(self) -> None:
        import httpx

        settings = get_settings()
        self._keys = [
            k.strip()
            for k in [settings.gemini_api_key, settings.gemini_api_key_backup]
            if k and k.strip()
        ]
        self._current_key_idx = 0
        self._model = settings.gemini_model
        self._http = httpx.AsyncClient(timeout=120.0)

    def _get_key(self) -> str:
        if not self._keys:
            return ""
        return self._keys[self._current_key_idx]

    def _rotate_key(self, reason: str) -> bool:
        if len(self._keys) <= 1:
            log.warning("No secondary Gemini API key configured to fail over to", reason=reason)
            return False
        old_idx = self._current_key_idx
        self._current_key_idx = (self._current_key_idx + 1) % len(self._keys)
        log.warning(
            "Gemini API key error, automatically failing over to backup key",
            previous_key_index=old_idx + 1,
            new_key_index=self._current_key_idx + 1,
            reason=reason,
        )
        return True

    def _build_payload(self, system: str, messages: list[dict[str, str]], temperature: float) -> dict:
        contents = []
        for m in messages:
            role = "model" if m.get("role") == "assistant" else "user"
            contents.append({
                "role": role,
                "parts": [{"text": m.get("content", "")}],
            })
        if not contents:
            contents = [{"role": "user", "parts": [{"text": "Hello"}]}]

        return {
            "system_instruction": {
                "parts": [{"text": system}],
            },
            "contents": contents,
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": 2048,
            },
        }

    async def stream_complete(
        self, system: str, messages: list[dict[str, str]], temperature: float = 0.1
    ) -> AsyncIterator[str]:
        import json
        import httpx

        max_attempts = max(1, len(self._keys))
        for attempt in range(max_attempts):
            api_key = self._get_key()
            url = f"{GEMINI_API_BASE}/{self._model}:streamGenerateContent?alt=sse"
            headers = {
                "Content-Type": "application/json",
                "x-goog-api-key": api_key,
            }
            payload = self._build_payload(system, messages, temperature)

            try:
                async with self._http.stream("POST", url, headers=headers, json=payload) as resp:
                    if resp.status_code in (429, 403, 503) or resp.status_code >= 400:
                        err_bytes = await resp.aread()
                        err_msg = f"HTTP {resp.status_code}: {err_bytes.decode('utf-8', errors='replace')[:200]}"
                        if attempt < max_attempts - 1 and self._rotate_key(err_msg):
                            continue
                        resp.raise_for_status()

                    async for line in resp.aiter_lines():
                        if line.startswith("data: "):
                            raw_json = line[6:].strip()
                            if not raw_json:
                                continue
                            try:
                                chunk_data = json.loads(raw_json)
                                candidates = chunk_data.get("candidates", [])
                                if candidates:
                                    parts = candidates[0].get("content", {}).get("parts", [])
                                    for p in parts:
                                        # Skip internal "thought" parts from thinking models (e.g. gemini-2.5+)
                                        if p.get("thought"):
                                            continue
                                        if text := p.get("text"):
                                            yield text
                            except Exception:
                                pass
                return
            except (httpx.HTTPStatusError, httpx.RequestError) as exc:
                if attempt < max_attempts - 1 and self._rotate_key(str(exc)):
                    continue
                log.error("Gemini streaming failed across all available keys", error=str(exc))
                raise

    async def complete(
        self, system: str, messages: list[dict[str, str]], temperature: float = 0.1
    ) -> str:
        import httpx

        max_attempts = max(1, len(self._keys))
        for attempt in range(max_attempts):
            api_key = self._get_key()
            url = f"{GEMINI_API_BASE}/{self._model}:generateContent"
            headers = {
                "Content-Type": "application/json",
                "x-goog-api-key": api_key,
            }
            payload = self._build_payload(system, messages, temperature)

            try:
                resp = await self._http.post(url, headers=headers, json=payload)
                if resp.status_code in (429, 403, 503) or resp.status_code >= 400:
                    err_msg = f"HTTP {resp.status_code}: {resp.text[:200]}"
                    if attempt < max_attempts - 1 and self._rotate_key(err_msg):
                        continue
                    resp.raise_for_status()

                data = resp.json()
                candidates = data.get("candidates", [])
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    # Filter out internal "thought" parts from thinking models (e.g. gemini-2.5+)
                    return "".join(p.get("text", "") for p in parts if not p.get("thought"))
                return ""
            except (httpx.HTTPStatusError, httpx.RequestError) as exc:
                if attempt < max_attempts - 1 and self._rotate_key(str(exc)):
                    continue
                log.error("Gemini completion failed across all available keys", error=str(exc))
                raise


class OllamaProvider(LLMProvider):
    def __init__(self) -> None:
        import httpx
        settings = get_settings()
        self._base_url = settings.ollama_base_url
        self._model = settings.ollama_model
        self._http = httpx.AsyncClient(base_url=self._base_url, timeout=120.0)

    async def stream_complete(self, system, messages, temperature=0.1):
        import json
        payload = {
            "model": self._model, "stream": True,
            "options": {"temperature": temperature},
            "messages": [{"role": "system", "content": system}] + messages,
        }
        async with self._http.stream("POST", "/api/chat", json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if line:
                    data = json.loads(line)
                    if token := data.get("message", {}).get("content", ""):
                        yield token
                    if data.get("done"):
                        break

    async def complete(self, system, messages, temperature=0.1):
        payload = {
            "model": self._model, "stream": False,
            "options": {"temperature": temperature},
            "messages": [{"role": "system", "content": system}] + messages,
        }
        resp = await self._http.post("/api/chat", json=payload)
        resp.raise_for_status()
        return resp.json()["message"]["content"]


class FakeLLMProvider(LLMProvider):
    """Deterministic fake LLM for offline tests."""
    def __init__(self, response: str = "This is a test answer.") -> None:
        self._response = response
        self.call_count = 0

    async def stream_complete(self, system, messages, temperature=0.1):
        self.call_count += 1
        for word in self._response.split():
            yield word + " "

    async def complete(self, system, messages, temperature=0.1):
        self.call_count += 1
        return self._response


_provider: LLMProvider | None = None


def get_llm_provider() -> LLMProvider:
    global _provider
    if _provider is None:
        settings = get_settings()
        if settings.llm_provider == "gemini":
            _provider = GeminiProvider()
        else:
            _provider = OllamaProvider()
    return _provider
