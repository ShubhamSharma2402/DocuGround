"""Application-wide error classes and FastAPI exception handlers."""

from __future__ import annotations

from fastapi import Request
from fastapi.responses import JSONResponse


class AppError(Exception):
    """Base application error."""

    status_code: int = 500
    error_code: str = "internal_error"

    def __init__(self, message: str, *, detail: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail


class NotFoundError(AppError):
    status_code = 404
    error_code = "not_found"


class ValidationError(AppError):
    status_code = 422
    error_code = "validation_error"


class AuthError(AppError):
    status_code = 401
    error_code = "unauthorized"


class ForbiddenError(AppError):
    status_code = 403
    error_code = "forbidden"


class FileTooLargeError(AppError):
    status_code = 413
    error_code = "file_too_large"


class InvalidPDFError(AppError):
    status_code = 422
    error_code = "invalid_pdf"


class ConflictError(AppError):
    status_code = 409
    error_code = "conflict"


class RateLimitError(AppError):
    status_code = 429
    error_code = "rate_limit_exceeded"

    def __init__(self, message: str = "Too many requests. Please try again later.", retry_after: int = 60, detail: str | None = None) -> None:
        super().__init__(message, detail=detail)
        self.retry_after = retry_after


async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
    """Convert AppError subclasses to structured JSON responses (no stack traces)."""
    body: dict[str, object] = {
        "error": exc.error_code,
        "message": exc.message,
    }
    if exc.detail:
        body["detail"] = exc.detail

    headers = {}
    if isinstance(exc, RateLimitError):
        headers["Retry-After"] = str(exc.retry_after)

    return JSONResponse(status_code=exc.status_code, content=body, headers=headers)


async def generic_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all handler that never leaks stack traces."""
    return JSONResponse(
        status_code=500,
        content={"error": "internal_error", "message": "An unexpected error occurred."},
    )
