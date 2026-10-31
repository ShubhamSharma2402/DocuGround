"""Unit tests for JWT and password hashing."""

from __future__ import annotations

import pytest

from app.core.errors import AuthError
from app.core.security import (
    create_access_token, decode_access_token,
    hash_password, verify_password,
)


class TestPasswordHashing:
    def test_hash_not_plaintext(self):
        assert hash_password("mysecret") != "mysecret"

    def test_correct_password_verifies(self):
        h = hash_password("correct-horse-battery")
        assert verify_password("correct-horse-battery", h) is True

    def test_wrong_password_fails(self):
        h = hash_password("correct-horse-battery")
        assert verify_password("wrong", h) is False

    def test_different_hashes_same_password(self):
        assert hash_password("same") != hash_password("same")


class TestJWT:
    def test_roundtrip(self):
        token = create_access_token("user123")
        assert decode_access_token(token) == "user123"

    def test_invalid_token_raises(self):
        with pytest.raises(AuthError):
            decode_access_token("not.a.valid.jwt")

    def test_tampered_token_raises(self):
        token = create_access_token("user123")
        with pytest.raises(AuthError):
            decode_access_token(token[:-5] + "XXXXX")

    def test_expired_token_raises(self):
        from datetime import datetime, timedelta, timezone
        from jose import jwt
        from app.core.config import get_settings
        settings = get_settings()
        payload = {
            "sub": "user123",
            "exp": datetime.now(timezone.utc) - timedelta(seconds=1),
            "iat": datetime.now(timezone.utc) - timedelta(minutes=2),
        }
        token = jwt.encode(payload, settings.secret_key, algorithm=settings.algorithm)
        with pytest.raises(AuthError):
            decode_access_token(token)
