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
        # Active 3.x models recommended directly by Google API
        self._model_candidates = [
            self._model,
            "gemini-3.5-flash-lite",
            "gemini-3.8-flash",
        ]
        # Deduplicate while preserving order
        self._model_candidates = list(dict.fromkeys(self._model_candidates))
        # Responsive timeouts: connect 8s, read 30s (so 503/hung connections fail fast)
        self._http = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=8.0, read=30.0, write=10.0, pool=10.0)
        )

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
        import time
        import httpx

        t_start = time.monotonic()
        first_token_received = False
        payload = self._build_payload(system, messages, temperature)

        for model in self._model_candidates:
            max_attempts = max(1, len(self._keys))
            for attempt in range(max_attempts):
                api_key = self._get_key()
                url = f"{GEMINI_API_BASE}/{model}:streamGenerateContent?alt=sse"
                headers = {
                    "Content-Type": "application/json",
                    "x-goog-api-key": api_key,
                }

                log.info(
                    "Calling Gemini stream API",
                    model=model,
                    attempt=attempt + 1,
                    max_attempts=max_attempts,
                )

                try:
                    async with self._http.stream("POST", url, headers=headers, json=payload) as resp:
                        if resp.status_code in (429, 403, 503) or resp.status_code >= 400:
                            err_bytes = await resp.aread()
                            err_msg = f"HTTP {resp.status_code}: {err_bytes.decode('utf-8', errors='replace')[:300]}"
                            log.warning(
                                "Gemini API returned error",
                                status=resp.status_code,
                                error=err_msg,
                                model=model,
                                attempt=attempt + 1,
                            )
                            # If 503 (High Demand) or 404 (Not Found), try next model candidate immediately
                            if resp.status_code in (503, 404):
                                break
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
                                            if p.get("thought"):
                                                continue
                                            if text := p.get("text"):
                                                if not first_token_received:
                                                    first_token_received = True
                                                    log.info(
                                                        "Gemini first token streamed",
                                                        ttft_ms=round((time.monotonic() - t_start) * 1000, 1),
                                                        model=model,
                                                    )
                                                yield text
                                except Exception:
                                    pass

                    if first_token_received:
                        log.info(
                            "Gemini streaming completed successfully",
                            duration_ms=round((time.monotonic() - t_start) * 1000, 1),
                            model=model,
                        )
                        return
                except (httpx.HTTPStatusError, httpx.RequestError) as exc:
                    log.warning(
                        "Gemini request exception",
                        error=str(exc),
                        model=model,
                        attempt=attempt + 1,
                    )
                    # If this model had a read timeout or 503/connect error, fall through to next model
                    if attempt < max_attempts - 1 and self._rotate_key(str(exc)):
                        continue
                    break

        log.error("Gemini streaming failed across all available models and keys")
        raise RuntimeError("Gemini streaming unavailable across all models and keys")

    async def complete(
        self, system: str, messages: list[dict[str, str]], temperature: float = 0.1
    ) -> str:
        import httpx

        payload = self._build_payload(system, messages, temperature)

        for model in self._model_candidates:
            max_attempts = max(1, len(self._keys))
            for attempt in range(max_attempts):
                api_key = self._get_key()
                url = f"{GEMINI_API_BASE}/{model}:generateContent"
                headers = {
                    "Content-Type": "application/json",
                    "x-goog-api-key": api_key,
                }

                try:
                    resp = await self._http.post(url, headers=headers, json=payload)
                    if resp.status_code in (429, 403, 503) or resp.status_code >= 400:
                        err_msg = f"HTTP {resp.status_code}: {resp.text[:200]}"
                        if resp.status_code in (503, 404):
                            break
                        if attempt < max_attempts - 1 and self._rotate_key(err_msg):
                            continue
                        resp.raise_for_status()

                    data = resp.json()
                    candidates = data.get("candidates", [])
                    if candidates:
                        parts = candidates[0].get("content", {}).get("parts", [])
                        return "".join(p.get("text", "") for p in parts if not p.get("thought"))
                    return ""
                except (httpx.HTTPStatusError, httpx.RequestError) as exc:
                    if attempt < max_attempts - 1 and self._rotate_key(str(exc)):
                        continue
                    break

        log.error("Gemini completion failed across all available models and keys")
        raise RuntimeError("Gemini completion unavailable across all models and keys")


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
