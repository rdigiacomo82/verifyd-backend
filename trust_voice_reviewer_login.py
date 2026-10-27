"""Standard-user Apple App Review login, WITHOUT one-time email codes.

This route is inert until a secret PBKDF2 hash and explicit enable flag are
configured in Render. No passwords or session keys belong in source code.
The only permitted identity is appreview@vfvid.com with @AppReview handle.
"""
from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import threading
import time

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

router = APIRouter(prefix="/trust-voice", tags=["Apple App Review"])
_REVIEW_EMAIL = "appreview@vfvid.com"
_REVIEW_HANDLE = "appreview"
_LOCK = threading.Lock()
_FAILURES = {}  # Request source: (failure_count, window_start), single-worker throttle.
_MAX_FAILURES = 8
_WINDOW_SECONDS = 15 * 60


class ReviewerLoginRequest(BaseModel):
    email: str = Field(min_length=3, max_length=254)
    password: str = Field(min_length=1, max_length=256)


def _review_enabled() -> bool:
    return os.environ.get("VERIFYD_APP_REVIEW_LOGIN_ENABLED", "").strip().lower() in {"1", "true", "yes", "on"}


def _password_spec():
    """Fail closed when the operator has not installed a valid password hash."""
    spec = os.environ.get("VERIFYD_APP_REVIEW_PASSWORD_HASH", "").strip()
    try:
        algorithm, count, salt_hex, hash_hex = spec.split("$")
        iterations = int(count)
        salt = bytes.fromhex(salt_hex)
        stored = bytes.fromhex(hash_hex)
        if algorithm != "pbkdf2_sha256" or not (400000 <= iterations <= 1200000):
            raise ValueError("Unsupported hash configuration")
        if len(salt) < 16 or len(stored) != 32:
            raise ValueError("Invalid hash configuration")
        return iterations, salt, stored
    except (ValueError, TypeError):
        return None


def _source_ip(request: Request) -> str:
    # Do not trust a client-supplied X-Forwarded-For header.
    return str(request.client.host if request.client else "unknown")


def _check_throttle(source: str):
    now = time.monotonic()
    with _LOCK:
        if len(_FAILURES) > 1024:
            _FAILURES.clear()  # Limit in-memory state, never persist passwords.
        attempts, start = _FAILURES.get(source, (0, now))
        if now - start >= _WINDOW_SECONDS:
            _FAILURES.pop(source, None)
            return
        if attempts >= _MAX_FAILURES:
            raise HTTPException(
                status_code=429,
                detail={"error": "try_again_later", "message": "Too many login attempts. Try again later."},
                headers={"Retry-After": "900"},
            )


def _failure(source: str):
    now = time.monotonic()
    with _LOCK:
        attempts, start = _FAILURES.get(source, (0, now))
        if now - start >= _WINDOW_SECONDS:
            attempts, start = 0, now
        _FAILURES[source] = (attempts + 1, start)


def _success(source: str):
    with _LOCK:
        _FAILURES.pop(source, None)


@router.post("/session/reviewer-login")
def reviewer_login(payload: ReviewerLoginRequest, request: Request):
    if not _review_enabled():
        raise HTTPException(status_code=404, detail="not_found")
    spec = _password_spec()
    if spec is None:
        raise HTTPException(status_code=503, detail={"error": "review_login_not_configured"})

    # Import at request time to reuse existing session/security logic, with no
    # secondary token format or email verification bypass for normal accounts.
    from trust_voice_signaling import (
        _require_enabled,
        _identity_by_email,
        _create_persistent_session,
        _sign,
        _public_identity,
    )
    _require_enabled()
    source = _source_ip(request)
    _check_throttle(source)

    email = payload.email.strip().lower()
    iterations, salt, stored = spec
    derived = hashlib.pbkdf2_hmac(
        "sha256", payload.password.encode("utf-8"), salt, iterations, dklen=32
    )
    password_ok = hmac.compare_digest(derived, stored)
    # Evaluate the exact identity AFTER the fixed-cost hash check. Do not
    # disclose whether any particular non-review account exists.
    if email != _REVIEW_EMAIL or not password_ok:
        _failure(source)
        raise HTTPException(status_code=401, detail={"error": "invalid_credentials"})

    identity = _identity_by_email(_REVIEW_EMAIL)
    if not identity or str(identity.get("handle_lower") or "").lower() != _REVIEW_HANDLE or not identity.get("email_verified"):
        raise HTTPException(status_code=403, detail={"error": "review_account_unavailable"})

    session_id = _create_persistent_session(identity["id"])
    now = int(time.time())
    token = _sign({
        "purpose": "trust_voice_session",
        "session_id": session_id,
        "identity_id": identity["id"],
        "handle_lower": _REVIEW_HANDLE,
        "iat": now,
        "nonce": secrets.token_hex(8),
    })
    _success(source)
    return {
        "ok": True,
        "token": token,
        "persistent": True,
        "expires_in_seconds": None,
        "identity": _public_identity(identity),
    }
