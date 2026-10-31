"""Integration tests for job status, SSE streaming, and health endpoints."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from app.db.redis import update_job_record


@pytest.mark.asyncio
async def test_get_job_status_success(
    client: httpx.AsyncClient,
    registered_user: dict,
    make_pdf,
):
    pdf_bytes = make_pdf("Job test doc")
    upload_resp = await client.post(
        "/documents",
        headers=registered_user["headers"],
        files={"file": ("job_test.pdf", pdf_bytes, "application/pdf")},
    )
    job_id = upload_resp.json()["job_id"]

    resp = await client.get(f"/jobs/{job_id}", headers=registered_user["headers"])
    assert resp.status_code == 200
    data = resp.json()
    assert data["job_id"] == job_id
    assert data["stage"] == "queued"
    assert data["progress"] == 0
    assert data["attempts"] == 0


@pytest.mark.asyncio
async def test_cross_user_isolation_job_status(
    client: httpx.AsyncClient,
    registered_user: dict,
    second_user: dict,
    make_pdf,
):
    pdf_bytes = make_pdf("Private job test")
    upload_resp = await client.post(
        "/documents",
        headers=registered_user["headers"],
        files={"file": ("user1_job.pdf", pdf_bytes, "application/pdf")},
    )
    job_id = upload_resp.json()["job_id"]

    # User 2 cannot inspect User 1's job
    resp = await client.get(f"/jobs/{job_id}", headers=second_user["headers"])
    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_job_not_found(client: httpx.AsyncClient, registered_user: dict):
    resp = await client.get("/jobs/00000000-0000-0000-0000-000000000000", headers=registered_user["headers"])
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_stream_job_events_sse(
    client: httpx.AsyncClient,
    registered_user: dict,
    make_pdf,
):
    pdf_bytes = make_pdf("SSE job test")
    upload_resp = await client.post(
        "/documents",
        headers=registered_user["headers"],
        files={"file": ("sse_test.pdf", pdf_bytes, "application/pdf")},
    )
    job_id = upload_resp.json()["job_id"]

    # Concurrently simulate worker advancing stage to "parse" (progress 20) then "done" (progress 100)
    async def simulate_worker():
        await asyncio.sleep(0.1)
        await update_job_record(job_id, stage="parse", progress=20)
        await asyncio.sleep(0.1)
        await update_job_record(job_id, stage="done", progress=100)

    sim_task = asyncio.create_task(simulate_worker())

    received_events = []
    async with client.stream("GET", f"/jobs/{job_id}/events", headers=registered_user["headers"]) as resp:
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]
        async for line in resp.aiter_lines():
            if line.startswith("data: "):
                payload = json.loads(line[6:])
                received_events.append(payload)
                if payload.get("stage") == "done":
                    break

    await sim_task
    assert len(received_events) >= 1
    stages = [e["stage"] for e in received_events]
    assert "done" in stages


@pytest.mark.asyncio
async def test_health_and_readiness(client: httpx.AsyncClient):
    # Liveness
    resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}

    # Readiness
    resp = await client.get("/ready")
    assert resp.status_code in (200, 503)
    assert "services" in resp.json()
