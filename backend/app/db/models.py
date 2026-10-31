"""Pydantic v2 models that serve as both API schemas and MongoDB document shapes.

MongoDB does not enforce a schema; these models are the single source of truth.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Annotated, Any

from bson import ObjectId
from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator


# ── ObjectId helper ──────────────────────────────────────────────────────────

class PyObjectId(str):
    """Custom type that accepts both str and bson.ObjectId."""

    @classmethod
    def __get_validators__(cls):  # noqa: D105 (pydantic v1 compat kept for safety)
        yield cls.validate

    @classmethod
    def validate(cls, v: Any) -> str:
        if isinstance(v, ObjectId):
            return str(v)
        if isinstance(v, str) and ObjectId.is_valid(v):
            return v
        raise ValueError(f"Invalid ObjectId: {v!r}")

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type: Any, handler: Any) -> Any:
        from pydantic_core import core_schema

        return core_schema.no_info_plain_validator_function(
            cls.validate,
            serialization=core_schema.to_string_ser_schema(),
        )


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


# ── Enums ────────────────────────────────────────────────────────────────────

class DocumentStatus(str, Enum):
    processing = "processing"
    ready = "ready"
    failed = "failed"


class MessageRole(str, Enum):
    user = "user"
    assistant = "assistant"


class VerificationVerdict(str, Enum):
    supported = "supported"
    partially = "partially"
    unsupported = "unsupported"


# ── Base ─────────────────────────────────────────────────────────────────────

class MongoBase(BaseModel):
    """Base model for documents stored in MongoDB."""

    model_config = ConfigDict(
        populate_by_name=True,
        arbitrary_types_allowed=True,
    )

    id: PyObjectId | None = Field(default=None, alias="_id")

    @field_serializer("id")
    def serialize_id(self, v: PyObjectId | None) -> str | None:
        return str(v) if v else None


# ── User ─────────────────────────────────────────────────────────────────────

class UserDB(MongoBase):
    """MongoDB `users` collection document."""

    email: str
    password_hash: str
    created_at: datetime = Field(default_factory=utc_now)


class UserCreate(BaseModel):
    email: str
    password: str

    @field_validator("email")
    @classmethod
    def email_lower(cls, v: str) -> str:
        return v.strip().lower()

    @field_validator("password")
    @classmethod
    def password_length(cls, v: str) -> str:
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters")
        return v


class UserRead(BaseModel):
    id: str
    email: str
    created_at: datetime


class UserLogin(BaseModel):
    email: str
    password: str

    @field_validator("email")
    @classmethod
    def email_lower(cls, v: str) -> str:
        return v.strip().lower()


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserRead


class UploadResponse(BaseModel):
    job_id: str
    document_id: str


class JobStatus(BaseModel):
    job_id: str
    doc_id: str
    user_id: str
    stage: str
    progress: int
    attempts: int = 0
    error: str = ""
    created_at: str
    updated_at: str


# ── Document ─────────────────────────────────────────────────────────────────

class DocumentDB(MongoBase):
    """MongoDB `documents` collection document."""

    user_id: str
    filename: str
    size_bytes: int
    status: DocumentStatus = DocumentStatus.processing
    page_count: int | None = None
    chunk_count: int | None = None
    error: str | None = None
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class DocumentRead(BaseModel):
    id: str
    user_id: str
    filename: str
    size_bytes: int
    status: DocumentStatus
    page_count: int | None
    chunk_count: int | None
    error: str | None
    created_at: datetime
    updated_at: datetime


# ── Conversation ──────────────────────────────────────────────────────────────

class ConversationDB(MongoBase):
    """MongoDB `conversations` collection document."""

    user_id: str
    document_ids: list[str] = Field(default_factory=list)
    title: str = "New Conversation"
    created_at: datetime = Field(default_factory=utc_now)


class ConversationRead(BaseModel):
    id: str
    user_id: str
    document_ids: list[str]
    title: str
    created_at: datetime


# ── Citation ─────────────────────────────────────────────────────────────────

class Citation(BaseModel):
    chunk_id: str
    doc_id: str
    doc_name: str
    page: int
    snippet: str
    score: float


# ── Message ──────────────────────────────────────────────────────────────────

class MessageDB(MongoBase):
    """MongoDB `messages` collection document."""

    conversation_id: str
    role: MessageRole
    content: str
    citations: list[Citation] = Field(default_factory=list)
    retrieval_scores: list[float] = Field(default_factory=list)
    gate_passed: bool | None = None
    verdict: VerificationVerdict | None = None
    latency_ms: float | None = None
    created_at: datetime = Field(default_factory=utc_now)


class MessageRead(BaseModel):
    id: str
    conversation_id: str
    role: MessageRole
    content: str
    citations: list[Citation]
    retrieval_scores: list[float]
    gate_passed: bool | None
    verdict: VerificationVerdict | None
    latency_ms: float | None
    created_at: datetime


class ChatRequest(BaseModel):
    conversation_id: str | None = None
    document_ids: list[str] = Field(min_length=1)
    question: str = Field(min_length=1)


class ConversationDetail(BaseModel):
    id: str
    user_id: str
    document_ids: list[str]
    title: str
    created_at: datetime
    messages: list[MessageRead] = Field(default_factory=list)
