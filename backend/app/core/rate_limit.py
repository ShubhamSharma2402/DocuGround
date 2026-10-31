"""Redis sliding window rate limiting."""

from __future__ import annotations

import time
import uuid

import structlog
from fastapi import Depends, Request

from app.api.deps import get_current_user
from app.core.config import get_settings
from app.core.errors import RateLimitError
from app.db.models import UserDB
from app.db.redis import get_redis_client

log = structlog.get_logger(__name__)


async def check_rate_limit(
    key: str,
    limit: int,
    window_seconds: int = 60,
) -> tuple[bool, int]:
    """Sliding-window counter backed by Redis sorted sets.

    Returns (allowed, retry_after_seconds).
    """
    if limit <= 0:
        return True, 0

    try:
        client = get_redis_client()
        now = time.time()
        clear_before = now - window_seconds
        member = f"{now}:{uuid.uuid4().hex[:6]}"

        pipe = client.pipeline()
        pipe.zremrangebyscore(key, 0, clear_before)
        pipe.zadd(key, {member: now})
        pipe.zcard(key)
        pipe.expire(key, window_seconds + 5)
        results = await pipe.execute()

        current_count = results[2]
        if current_count > limit:
            retry_after = int(window_seconds)
            return False, retry_after

        return True, 0
    except Exception as exc:
        log.warning("Rate limiter check failed or redis unavailable, allowing request", error=str(exc))
        return True, 0


class RateLimiter:
    """FastAPI dependency for endpoint rate limiting."""

    def __init__(self, action: str, limit_attr: str) -> None:
        self.action = action
        self.limit_attr = limit_attr

    async def __call__(
        self,
        request: Request,
        current_user: UserDB = Depends(get_current_user),
    ) -> None:
        settings = get_settings()
        limit = getattr(settings, self.limit_attr, 30)

        identifier = str(current_user.id) if current_user else (request.client.host if request.client else "unknown")
        rate_key = f"ratelimit:{self.action}:{identifier}"

        allowed, retry_after = await check_rate_limit(rate_key, limit=limit, window_seconds=60)
        if not allowed:
            log.warning("Rate limit exceeded", action=self.action, identifier=identifier, limit=limit)
            raise RateLimitError(
                message=f"Rate limit exceeded for {self.action}. Please try again later.",
                retry_after=retry_after,
                detail=f"Allowed {limit} requests per minute.",
            )
