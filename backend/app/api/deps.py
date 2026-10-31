"""FastAPI dependency injection utilities."""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Header, Request
from motor.motor_asyncio import AsyncIOMotorDatabase

from app.core.errors import AuthError
from app.core.security import decode_access_token
from app.db.client import get_db
from app.db.models import UserDB


async def get_current_user(
    authorization: Annotated[str | None, Header()] = None,
    db: AsyncIOMotorDatabase = Depends(get_db),
) -> UserDB:
    """Validate Bearer token and return the authenticated UserDB."""
    if not authorization:
        raise AuthError("Missing Authorization header")

    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise AuthError("Invalid authorization format. Must be 'Bearer <token>'")

    token = parts[1]
    user_id = decode_access_token(token)

    user_dict = await db.users.find_one({"_id": user_id})
    if not user_dict:
        raise AuthError("User not found or deactivated")

    return UserDB(**user_dict)
