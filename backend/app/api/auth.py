"""Authentication endpoints: register, login, current user."""

from __future__ import annotations

from bson import ObjectId
from fastapi import APIRouter, Depends, status
from motor.motor_asyncio import AsyncIOMotorDatabase

from app.api.deps import get_current_user
from app.core.errors import AuthError, ConflictError
from app.core.security import create_access_token, hash_password, verify_password
from app.db.client import get_db
from app.db.models import TokenResponse, UserCreate, UserDB, UserLogin, UserRead, utc_now

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
async def register(
    payload: UserCreate,
    db: AsyncIOMotorDatabase = Depends(get_db),
) -> TokenResponse:
    """Register a new user account."""
    existing = await db.users.find_one({"email": payload.email})
    if existing:
        raise ConflictError("An account with this email already exists")

    user_id = str(ObjectId())
    pwd_hash = hash_password(payload.password)
    now = utc_now()

    doc = {
        "_id": user_id,
        "email": payload.email,
        "password_hash": pwd_hash,
        "created_at": now,
    }
    await db.users.insert_one(doc)

    user_read = UserRead(id=user_id, email=payload.email, created_at=now)
    token = create_access_token(user_id)
    return TokenResponse(access_token=token, token_type="bearer", user=user_read)


@router.post("/login", response_model=TokenResponse)
async def login(
    payload: UserLogin,
    db: AsyncIOMotorDatabase = Depends(get_db),
) -> TokenResponse:
    """Authenticate with email and password to receive a JWT access token."""
    user_dict = await db.users.find_one({"email": payload.email})
    if not user_dict or not verify_password(payload.password, user_dict.get("password_hash", "")):
        raise AuthError("Invalid email or password")

    user_id = str(user_dict["_id"])
    token = create_access_token(user_id)
    user_read = UserRead(
        id=user_id,
        email=user_dict["email"],
        created_at=user_dict["created_at"],
    )
    return TokenResponse(access_token=token, token_type="bearer", user=user_read)


@router.get("/me", response_model=UserRead)
async def get_me(
    current_user: UserDB = Depends(get_current_user),
) -> UserRead:
    """Get the current authenticated user's profile."""
    return UserRead(
        id=str(current_user.id),
        email=current_user.email,
        created_at=current_user.created_at,
    )
