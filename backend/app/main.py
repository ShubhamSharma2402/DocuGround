"""FastAPI application entrypoint for DocuGround."""

from __future__ import annotations

from contextlib import asynccontextmanager
import uuid
from typing import AsyncIterator

import structlog
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

from app.api import auth, chat, documents, health, jobs
from app.core.config import get_settings
from app.core.errors import AppError, app_error_handler, generic_error_handler
from app.core.logging import configure_logging
from app.db.client import close_mongo, get_db
from app.db.init_db import init_mongo, init_qdrant
from app.db.redis import close_redis
from app.services.vector_store import get_qdrant_client

log = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Startup and shutdown lifecycle management."""
    configure_logging()
    settings = get_settings()
    log.info("Starting DocuGround API", env=settings.app_env)

    # Initialize databases idempotently (skip if unavailable in unit test mode)
    try:
        db = get_db()
        await init_mongo(db)
    except Exception as exc:
        log.warning("MongoDB initialization skipped or failed", error=str(exc))

    try:
        qdrant = get_qdrant_client()
        await init_qdrant(qdrant)
    except Exception as exc:
        log.warning("Qdrant initialization skipped or failed", error=str(exc))

    yield

    log.info("Shutting down DocuGround API")
    await close_mongo()
    await close_redis()


class RequestCorrelationMiddleware(BaseHTTPMiddleware):
    """Assign or propagate X-Request-ID and bind to structlog context."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(request_id=request_id)

        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add defensive security headers to all HTTP responses."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        return response


def create_app() -> FastAPI:
    """Build and configure the FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title="DocuGround API",
        description="Production RAG application for chatting with PDFs",
        version="0.1.0",
        lifespan=lifespan,
    )

    # Security & Correlation Middlewares
    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(RequestCorrelationMiddleware)

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Error Handlers
    app.add_exception_handler(AppError, app_error_handler)
    app.add_exception_handler(Exception, generic_error_handler)

    # Routers
    app.include_router(health.router)
    app.include_router(auth.router)
    app.include_router(documents.router)
    app.include_router(jobs.router)
    app.include_router(chat.router)

    return app


app = create_app()
