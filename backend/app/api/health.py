"""Health and readiness check endpoints."""

from __future__ import annotations

import structlog
from fastapi import APIRouter, Response, status

from app.db.client import get_db
from app.db.redis import get_redis_client
from app.services.vector_store import get_qdrant_client

log = structlog.get_logger(__name__)

router = APIRouter(tags=["health"])


@router.get("/health")
async def health_check() -> dict[str, str]:
    """Liveness probe: verifies process is responsive."""
    return {"status": "ok"}


@router.get("/ready")
async def readiness_check(response: Response) -> dict[str, object]:
    """Readiness probe: verifies connectivity to MongoDB, Redis, and Qdrant."""
    services: dict[str, str] = {}
    is_ready = True

    # 1. MongoDB check
    try:
        db = get_db()
        await db.command("ping")
        services["mongodb"] = "ok"
    except Exception as exc:
        log.warning("Readiness probe: MongoDB unreachable", error=str(exc))
        services["mongodb"] = "unreachable"
        is_ready = False

    # 2. Redis check
    try:
        redis = get_redis_client()
        await redis.ping()
        services["redis"] = "ok"
    except Exception as exc:
        log.warning("Readiness probe: Redis unreachable", error=str(exc))
        services["redis"] = "unreachable"
        is_ready = False

    # 3. Qdrant check
    try:
        qdrant = get_qdrant_client()
        await qdrant.get_collections()
        services["qdrant"] = "ok"
    except Exception as exc:
        log.warning("Readiness probe: Qdrant unreachable", error=str(exc))
        services["qdrant"] = "unreachable"
        is_ready = False

    if not is_ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return {
        "status": "ready" if is_ready else "degraded",
        "services": services,
    }


@router.get("/metrics")
async def metrics_endpoint() -> Response:
    """Prometheus metrics scrape endpoint."""
    from app.core.metrics import export_metrics

    body, media_type = export_metrics()
    return Response(content=body, media_type=media_type)
