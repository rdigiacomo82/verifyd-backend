# ============================================================
# VeriFYD Trust Voice — Identity MVP
# VERIFYD_TRUST_VOICE_IDENTITY_MVP_V1
#
# Isolated beta router for:
#   - email-code ownership proof
#   - unique @handle claiming
#   - optional display emoji
#   - public exact-handle lookup
#
# IMPORTANT:
#   This module is disabled unless VERIFYD_TRUST_VOICE_ENABLED=1.
#   It does not modify existing VeriFYD tables or routes.
# ============================================================

from __future__ import annotations

import logging
import os
import re
import uuid
import unicodedata
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from database import create_otp, get_db, get_email_typo_suggestion, is_valid_email, verify_otp
from emailer import send_otp_email

log = logging.getLogger("verifyd.trust_voice.identity")

router = APIRouter(prefix="/trust-voice", tags=["Trust Voice"])

FEATURE_VERSION = "0.1.0"
_HANDLE_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]{2,23}$")

# Protect names that could imply VeriFYD authority or confuse users.
_RESERVED_HANDLES = {
    "admin", "administrator", "api", "billing", "docs", "help", "moderator",
    "official", "root", "security", "staff", "support", "system", "trust",
    "trustmail", "trustmessage", "trustvoice", "vault", "verifyd", "verifydadmin",
    "verified", "verification", "verify", "vfvid", "vfyd",
    "verifydofficial", "verifydsupport",
}

# Decorative emoji must never resemble the authoritative VeriFYD verification badge.
_BANNED_EMOJI_TOKENS = {
    "✅", "☑", "☑️", "✔", "✔️", "✓", "🛡", "🛡️",
}

_STARTER_EMOJIS = [
    "🏌️", "🌻", "⚾", "🐶", "🎧", "🎨", "🚀", "📚", "🎮", "⚽",
    "🏀", "🏈", "🎸", "☕", "🌎", "⭐", "💼", "🛠️", "📷", "🚲",
    "🏃", "🎯", "🧩", "🌴", "🐱", "🐾", "💡", "🧠", "🌊", "⛳",
]


class RequestIdentityCode(BaseModel):
    email: str = Field(min_length=3, max_length=254)


class ClaimIdentityRequest(BaseModel):
    email: str = Field(min_length=3, max_length=254)
    code: str = Field(min_length=6, max_length=12)
    handle: str = Field(min_length=3, max_length=25)
    display_name: str = Field(min_length=1, max_length=80)
    display_emoji: str = Field(default="", max_length=24)


def _feature_enabled() -> bool:
    value = (os.environ.get("VERIFYD_TRUST_VOICE_ENABLED", "") or "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def _require_enabled() -> None:
    if not _feature_enabled():
        # Return 404 rather than advertising an unfinished beta surface.
        raise HTTPException(status_code=404, detail="not_found")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_email(value: str) -> str:
    return (value or "").strip().lower()


def _normalize_handle(value: str) -> str:
    raw = (value or "").strip()
    if raw.startswith("@"):
        raw = raw[1:]
    return raw.strip()


def _validate_handle(value: str) -> tuple[str, str]:
    handle = _normalize_handle(value)
    if not _HANDLE_RE.fullmatch(handle):
        raise HTTPException(
            status_code=400,
            detail={
                "error": "invalid_handle",
                "message": "Handle must be 3-24 characters, start with a letter, and contain only letters, numbers, or underscores.",
            },
        )
    handle_lower = handle.lower()
    if handle_lower in _RESERVED_HANDLES:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "reserved_handle",
                "message": "That handle is reserved by VeriFYD.",
            },
        )
    return handle, handle_lower


def _validate_display_name(value: str) -> str:
    cleaned = " ".join((value or "").strip().split())
    if not cleaned:
        raise HTTPException(status_code=400, detail={"error": "display_name_required"})
    if len(cleaned) > 80:
        raise HTTPException(status_code=400, detail={"error": "display_name_too_long"})
    return cleaned


def _validate_display_emoji(value: str) -> str:
    emoji = (value or "").strip()
    if not emoji:
        return ""
    if len(emoji) > 12:
        raise HTTPException(
            status_code=400,
            detail={"error": "emoji_too_long", "message": "Choose one short display emoji."},
        )
    if any(token in emoji for token in _BANNED_EMOJI_TOKENS):
        raise HTTPException(
            status_code=400,
            detail={
                "error": "emoji_reserved",
                "message": "Verification-style symbols are reserved for VeriFYD trust badges.",
            },
        )
    # Keep the field decorative. Permit Unicode symbols/marks used by emoji,
    # plus ZWJ / variation-selector characters needed for emoji sequences.
    for ch in emoji:
        if ch in {"\u200d", "\ufe0f"}:
            continue
        category = unicodedata.category(ch)
        if not (category.startswith("S") or category.startswith("M")):
            raise HTTPException(
                status_code=400,
                detail={"error": "invalid_emoji", "message": "Display emoji must contain emoji/symbol characters only."},
            )
    return emoji


def _ensure_identity_schema() -> None:
    """Create only the isolated Trust Voice identity table. Safe and idempotent."""
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS verifyd_identities (
                id                    TEXT PRIMARY KEY,
                email_lower           TEXT UNIQUE NOT NULL,
                handle                TEXT NOT NULL,
                handle_lower          TEXT UNIQUE NOT NULL,
                display_name          TEXT NOT NULL,
                display_emoji         TEXT NOT NULL DEFAULT '',
                email_verified        BOOLEAN NOT NULL DEFAULT TRUE,
                identity_verified     BOOLEAN NOT NULL DEFAULT FALSE,
                organization_verified BOOLEAN NOT NULL DEFAULT FALSE,
                verification_level    TEXT NOT NULL DEFAULT 'email_verified',
                can_receive_calls     BOOLEAN NOT NULL DEFAULT FALSE,
                call_privacy          TEXT NOT NULL DEFAULT 'verified_users',
                status                TEXT NOT NULL DEFAULT 'active',
                created_at            TEXT NOT NULL,
                updated_at            TEXT NOT NULL
            )
            """
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_verifyd_identities_handle_lower ON verifyd_identities(handle_lower)"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_verifyd_identities_status ON verifyd_identities(status)"
        )


def _public_identity(row: dict) -> dict:
    handle = (row.get("handle") or "").strip()
    emoji = (row.get("display_emoji") or "").strip()
    return {
        "identity_id": row.get("id"),
        "handle": f"@{handle}",
        "display_name": row.get("display_name") or "",
        "display_emoji": emoji,
        "display_label": f"{emoji + ' ' if emoji else ''}@{handle}",
        "verification": {
            "email_verified": bool(row.get("email_verified")),
            "identity_verified": bool(row.get("identity_verified")),
            "organization_verified": bool(row.get("organization_verified")),
            "level": row.get("verification_level") or "email_verified",
        },
        "trust_voice": {
            "can_receive_calls": bool(row.get("can_receive_calls")),
            "call_privacy": row.get("call_privacy") or "verified_users",
        },
        "status": row.get("status") or "active",
        "created_at": row.get("created_at") or "",
    }


def _recent_otp_request(email_lower: str, cooldown_seconds: int = 60) -> bool:
    """Prevent accidental/resend abuse without changing the existing OTP table."""
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT created_at FROM email_otp WHERE email_lower = %s ORDER BY id DESC LIMIT 1",
            (email_lower,),
        )
        row = cur.fetchone()
    if not row or not row.get("created_at"):
        return False
    try:
        created = datetime.fromisoformat(row["created_at"])
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        age = (datetime.now(timezone.utc) - created).total_seconds()
        return age < cooldown_seconds
    except Exception:
        return False


@router.get("/health")
def trust_voice_health():
    _require_enabled()
    return {
        "status": "ok",
        "feature": "trust_voice_identity",
        "version": FEATURE_VERSION,
        "voice_transport": "not_enabled",
    }


@router.get("/emoji-options")
def trust_voice_emoji_options():
    _require_enabled()
    return {
        "starter_emojis": _STARTER_EMOJIS,
        "note": "Emoji are decorative only. VeriFYD verification badges are separate and cannot be selected as profile emoji.",
    }


@router.post("/identity/request-code")
def request_identity_code(payload: RequestIdentityCode):
    _require_enabled()
    email = _normalize_email(payload.email)
    if not is_valid_email(email):
        raise HTTPException(status_code=400, detail={"error": "invalid_email"})
    suggestion = get_email_typo_suggestion(email)
    if suggestion:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "possible_email_typo",
                "message": f"Did you mean {suggestion}?",
                "suggested_email": suggestion,
            },
        )

    if _recent_otp_request(email):
        raise HTTPException(
            status_code=429,
            detail={"error": "code_recently_sent", "message": "Please wait before requesting another code."},
        )

    code = create_otp(email)
    if not send_otp_email(email, code):
        raise HTTPException(
            status_code=503,
            detail={"error": "email_send_failed", "message": "Verification email could not be sent."},
        )

    log.info("Trust Voice identity verification code sent to %s", email)
    return {
        "ok": True,
        "email": email,
        "expires_minutes": 10,
    }


@router.get("/handle-availability/{handle}")
def handle_availability(handle: str):
    _require_enabled()
    display_handle, handle_lower = _validate_handle(handle)
    _ensure_identity_schema()

    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT 1 FROM verifyd_identities WHERE handle_lower = %s LIMIT 1",
            (handle_lower,),
        )
        exists = bool(cur.fetchone())

    return {
        "handle": f"@{display_handle}",
        "available": not exists,
    }


@router.get("/handle/{handle}")
def lookup_handle(handle: str):
    _require_enabled()
    _, handle_lower = _validate_handle(handle)
    _ensure_identity_schema()

    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT *
            FROM verifyd_identities
            WHERE handle_lower = %s
              AND status = 'active'
            LIMIT 1
            """,
            (handle_lower,),
        )
        row = cur.fetchone()

    if not row:
        raise HTTPException(status_code=404, detail={"error": "handle_not_found"})
    return {"identity": _public_identity(dict(row))}


@router.post("/identity/claim")
def claim_identity(payload: ClaimIdentityRequest):
    _require_enabled()

    email = _normalize_email(payload.email)
    if not is_valid_email(email):
        raise HTTPException(status_code=400, detail={"error": "invalid_email"})

    handle, handle_lower = _validate_handle(payload.handle)
    display_name = _validate_display_name(payload.display_name)
    display_emoji = _validate_display_emoji(payload.display_emoji)

    # Ownership proof is intentionally required for each initial identity claim.
    # This is stronger than relying on a browser-local "verified email" flag.
    ok, message = verify_otp(email, payload.code)
    if not ok:
        raise HTTPException(
            status_code=400,
            detail={"error": "invalid_verification_code", "message": message},
        )

    _ensure_identity_schema()

    with get_db() as conn:
        cur = conn.cursor()

        cur.execute(
            "SELECT * FROM verifyd_identities WHERE email_lower = %s LIMIT 1",
            (email,),
        )
        existing_email = cur.fetchone()
        if existing_email:
            existing = dict(existing_email)
            if (existing.get("handle_lower") or "") == handle_lower:
                return {
                    "created": False,
                    "message": "identity_already_exists",
                    "identity": _public_identity(existing),
                }
            raise HTTPException(
                status_code=409,
                detail={
                    "error": "email_already_has_identity",
                    "message": "This verified email already owns a VeriFYD Identity.",
                },
            )

        cur.execute(
            "SELECT 1 FROM verifyd_identities WHERE handle_lower = %s LIMIT 1",
            (handle_lower,),
        )
        if cur.fetchone():
            raise HTTPException(
                status_code=409,
                detail={"error": "handle_taken", "message": "That handle is already in use."},
            )

        identity_id = "vfyd_usr_" + uuid.uuid4().hex
        now = _now_iso()
        cur.execute(
            """
            INSERT INTO verifyd_identities (
                id, email_lower, handle, handle_lower,
                display_name, display_emoji,
                email_verified, identity_verified, organization_verified,
                verification_level, can_receive_calls, call_privacy,
                status, created_at, updated_at
            )
            VALUES (
                %s, %s, %s, %s,
                %s, %s,
                TRUE, FALSE, FALSE,
                'email_verified', FALSE, 'verified_users',
                'active', %s, %s
            )
            RETURNING *
            """,
            (
                identity_id,
                email,
                handle,
                handle_lower,
                display_name,
                display_emoji,
                now,
                now,
            ),
        )
        row = cur.fetchone()

    identity = _public_identity(dict(row))
    log.info("Trust Voice identity claimed: %s by %s", identity["handle"], email)
    return {
        "created": True,
        "identity": identity,
        "next_phase": "voice_calling_not_enabled_yet",
    }
