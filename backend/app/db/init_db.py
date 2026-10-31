"""Database initialisation: create indexes idempotently on startup.

Also creates the Qdrant collection and payload indexes if they do not exist.
"""

from __future__ import annotations

import structlog
from motor.motor_asyncio import AsyncIOMotorDatabase
from pymongo import ASCENDING, DESCENDING, IndexModel
from qdrant_client import AsyncQdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import (
    Distance,
    PayloadSchemaType,
    SparseVectorParams,
    VectorParams,
)

from app.core.config import get_settings

log = structlog.get_logger(__name__)


async def init_mongo(db: AsyncIOMotorDatabase) -> None:
    """Create all MongoDB indexes idempotently."""
    settings = get_settings()

    # users: unique email
    await db.users.create_indexes(
        [IndexModel([("email", ASCENDING)], unique=True, name="users_email_unique")]
    )

    # documents: lookup by user, janitor query
    await db.documents.create_indexes(
        [
            IndexModel([("user_id", ASCENDING)], name="docs_user_id"),
            IndexModel(
                [("status", ASCENDING), ("updated_at", ASCENDING)],
                name="docs_status_updated",
            ),
            # Optional TTL on failed documents (30 days)
            IndexModel(
                [("updated_at", ASCENDING)],
                name="docs_failed_ttl",
                expireAfterSeconds=30 * 24 * 3600,
                partialFilterExpression={"status": "failed"},
            ),
        ]
    )

    # messages: conversation history retrieval
    await db.messages.create_indexes(
        [
            IndexModel(
                [("conversation_id", ASCENDING), ("created_at", ASCENDING)],
                name="msgs_conv_time",
            )
        ]
    )

    # conversations: per-user listing
    await db.conversations.create_indexes(
        [IndexModel([("user_id", ASCENDING)], name="convs_user_id")]
    )

    log.info("MongoDB indexes ensured")


async def init_qdrant(client: AsyncQdrantClient) -> None:
    """Create Qdrant collection and payload indexes if missing."""
    settings = get_settings()
    collection = settings.qdrant_collection

    try:
        await client.get_collection(collection)
        log.info("Qdrant collection already exists", collection=collection)
    except (UnexpectedResponse, Exception):
        log.info("Creating Qdrant collection", collection=collection)
        await client.create_collection(
            collection_name=collection,
            vectors_config={
                "dense": VectorParams(
                    size=settings.qdrant_vector_size,
                    distance=Distance.COSINE,
                ),
            },
            sparse_vectors_config={
                "sparse": SparseVectorParams(),
            },
        )

    # Payload indexes for filtering
    for field in ("user_id", "doc_id"):
        try:
            await client.create_payload_index(
                collection_name=collection,
                field_name=field,
                field_schema=PayloadSchemaType.KEYWORD,
            )
        except Exception:
            pass  # Already exists

    log.info("Qdrant collection and indexes ensured", collection=collection)
