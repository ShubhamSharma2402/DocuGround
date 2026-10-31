"""Shared pytest fixtures for API and database tests."""

from __future__ import annotations

import io
from typing import AsyncIterator

import fakeredis.aioredis
import fitz
import httpx
import pytest
import pytest_asyncio
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

from app.core.config import get_settings
from app.db.client import close_mongo, get_db
from app.db.init_db import init_mongo
from app.db.redis import set_redis_client
from app.main import app


@pytest_asyncio.fixture(autouse=True)
async def setup_test_environment(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[None]:
    """Isolate tests using an in-memory Redis and a dedicated test MongoDB database."""
    settings = get_settings()

    # Use in-memory FakeRedis
    fake_redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
    set_redis_client(fake_redis)

    # Use test database in Mongo
    test_db_name = "docuground_test_db"
    settings.__dict__["mongodb_db"] = test_db_name

    await close_mongo()
    client = AsyncIOMotorClient(settings.mongodb_url)
    test_db = client[test_db_name]

    # Clean collections before test
    await test_db.users.delete_many({})
    await test_db.documents.delete_many({})
    await test_db.conversations.delete_many({})
    await test_db.messages.delete_many({})

    await init_mongo(test_db)

    # FastAPI dependency override
    app.dependency_overrides[get_db] = lambda: test_db

    yield

    # Clean up after test
    await test_db.users.delete_many({})
    await test_db.documents.delete_many({})
    await test_db.conversations.delete_many({})
    await test_db.messages.delete_many({})
    app.dependency_overrides.clear()
    client.close()
    await fake_redis.aclose()
    set_redis_client(None)


@pytest_asyncio.fixture
async def client() -> AsyncIterator[httpx.AsyncClient]:
    """Async HTTP test client."""
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as ac:
        yield ac


@pytest.fixture
def make_pdf():
    """Helper to create valid PDF bytes in memory."""
    def _create(text: str = "DocuGround test content", pages: int = 1, encrypted: bool = False) -> bytes:
        doc = fitz.open()
        for i in range(pages):
            page = doc.new_page()
            page.insert_text((50, 72), f"{text} - page {i+1}")
        
        if encrypted:
            pdf_bytes = doc.tobytes(
                encryption=fitz.PDF_ENCRYPT_AES_256,
                user_pw="secret",
                owner_pw="secret",
            )
        else:
            pdf_bytes = doc.tobytes()
        doc.close()
        return pdf_bytes

    return _create


@pytest_asyncio.fixture
async def registered_user(client: httpx.AsyncClient) -> dict[str, str]:
    """Register a test user and return their auth headers and info."""
    resp = await client.post(
        "/auth/register",
        json={"email": "alice@example.com", "password": "password123"},
    )
    assert resp.status_code == 201
    data = resp.json()
    token = data["access_token"]
    user_id = data["user"]["id"]
    return {
        "user_id": user_id,
        "email": "alice@example.com",
        "token": token,
        "headers": {"Authorization": f"Bearer {token}"},
    }


@pytest_asyncio.fixture
async def second_user(client: httpx.AsyncClient) -> dict[str, str]:
    """Register a second user for cross-user isolation tests."""
    resp = await client.post(
        "/auth/register",
        json={"email": "bob@example.com", "password": "password456"},
    )
    assert resp.status_code == 201
    data = resp.json()
    token = data["access_token"]
    user_id = data["user"]["id"]
    return {
        "user_id": user_id,
        "email": "bob@example.com",
        "token": token,
        "headers": {"Authorization": f"Bearer {token}"},
    }
