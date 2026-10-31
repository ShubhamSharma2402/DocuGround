"""Integration tests for Auth endpoints."""

from __future__ import annotations

import httpx
import pytest


@pytest.mark.asyncio
async def test_register_success(client: httpx.AsyncClient):
    resp = await client.post(
        "/auth/register",
        json={"email": "newuser@example.com", "password": "secure_password_123"},
    )
    assert resp.status_code == 201
    data = resp.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"
    assert data["user"]["email"] == "newuser@example.com"
    assert "id" in data["user"]


@pytest.mark.asyncio
async def test_register_duplicate_email(client: httpx.AsyncClient, registered_user: dict):
    resp = await client.post(
        "/auth/register",
        json={"email": registered_user["email"], "password": "another_password"},
    )
    assert resp.status_code == 409
    data = resp.json()
    assert data["error"] == "conflict"


@pytest.mark.asyncio
async def test_register_short_password(client: httpx.AsyncClient):
    resp = await client.post(
        "/auth/register",
        json={"email": "user@example.com", "password": "123"},
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_login_success(client: httpx.AsyncClient, registered_user: dict):
    resp = await client.post(
        "/auth/login",
        json={"email": registered_user["email"], "password": "password123"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "access_token" in data
    assert data["user"]["email"] == registered_user["email"]


@pytest.mark.asyncio
async def test_login_wrong_password(client: httpx.AsyncClient, registered_user: dict):
    resp = await client.post(
        "/auth/login",
        json={"email": registered_user["email"], "password": "incorrect_password"},
    )
    assert resp.status_code == 401
    assert resp.json()["error"] == "unauthorized"


@pytest.mark.asyncio
async def test_login_nonexistent_user(client: httpx.AsyncClient):
    resp = await client.post(
        "/auth/login",
        json={"email": "doesnotexist@example.com", "password": "some_password"},
    )
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_get_me_success(client: httpx.AsyncClient, registered_user: dict):
    resp = await client.get("/auth/me", headers=registered_user["headers"])
    assert resp.status_code == 200
    data = resp.json()
    assert data["id"] == registered_user["user_id"]
    assert data["email"] == registered_user["email"]


@pytest.mark.asyncio
async def test_get_me_unauthorized(client: httpx.AsyncClient):
    resp = await client.get("/auth/me")
    assert resp.status_code == 401

    resp = await client.get("/auth/me", headers={"Authorization": "Bearer badtoken"})
    assert resp.status_code == 401
