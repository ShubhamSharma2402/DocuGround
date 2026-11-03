"""Application configuration loaded from environment variables / .env file."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(".env", "backend/.env", "../.env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Application ──────────────────────────────────────────────
    app_env: Literal["development", "staging", "production"] = "development"
    secret_key: str = "change-me-in-production-min-32-chars"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 60

    # ── Upload / Processing ──────────────────────────────────────
    max_upload_mb: int = 20
    max_pages: int = 400
    min_text_chars_per_page: int = 50
    chunk_size: int = 800
    chunk_overlap: int = 100
    embed_batch_size: int = 32
    retrieval_top_k: int = 30
    rerank_top_n: int = 10
    max_chunks_per_page: int = 3
    dense_score_cutoff: float = 0.4
    reranker_score_cutoff: float = 0.4
    rrf_k: int = 60

    # ── Redis ────────────────────────────────────────────────────
    redis_url: str = "redis://localhost:6379/0"
    ingest_stream: str = "jobs:ingest"
    dlq_stream: str = "jobs:dead"
    consumer_group: str = "docuground-workers"
    consumer_name: str = "worker-1"
    stream_block_ms: int = 2000
    autoclaim_min_idle_ms: int = 60_000
    job_ttl_seconds: int = 86_400
    worker_max_retries: int = 3
    worker_retry_base_delay: float = 2.0
    worker_retry_max_delay: float = 30.0

    # ── MongoDB ──────────────────────────────────────────────────
    mongodb_url: str = "mongodb://localhost:27017"
    mongodb_db: str = "docuground"

    # ── Qdrant ───────────────────────────────────────────────────
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "docuground_chunks"
    qdrant_vector_size: int = 384

    # ── Models ───────────────────────────────────────────────────
    embedder_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    llm_provider: Literal["gemini", "ollama"] = "gemini"
    gemini_api_key: str = ""
    gemini_api_key_backup: str = ""
    gemini_model: str = "gemini-3.5-flash-lite"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2"

    # ── Verification ─────────────────────────────────────────────
    verification_enabled: bool = False
    verification_reject_unsupported: bool = False

    # ── CORS ─────────────────────────────────────────────────────
    cors_origins: str = "http://localhost:5173,http://localhost:3000"

    # ── Rate Limiting ────────────────────────────────────────────
    rate_limit_upload_per_minute: int = 10
    rate_limit_chat_per_minute: int = 30

    # ── Storage ──────────────────────────────────────────────────
    upload_dir: Path = Path("./uploads")
    models_cache_dir: Path = Path("./model_cache")

    # ── Janitor ──────────────────────────────────────────────────
    janitor_interval_seconds: int = 300
    stuck_job_threshold_seconds: int = 600

    # ── Derived ──────────────────────────────────────────────────
    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @field_validator("upload_dir", "models_cache_dir", mode="before")
    @classmethod
    def _make_path(cls, v: object) -> Path:
        return Path(str(v))

    @model_validator(mode="after")
    def _ensure_dirs(self) -> "Settings":
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.models_cache_dir.mkdir(parents=True, exist_ok=True)
        return self


def get_settings() -> Settings:
    """Return a cached Settings instance."""
    return _settings


_settings = Settings()
