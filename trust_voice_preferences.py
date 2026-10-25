# ============================================================
# VeriFYD Circle — Multilingual Account Preferences
# VERIFYD_TRUST_VOICE_PREFERENCES_V1
#
# Stores account-level UI/speech locale without modifying identity/session
# tables. Device-local cache remains the fast startup fallback; this table is
# authoritative after an authenticated session is available.
# ============================================================

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel

from database import get_db
from trust_voice_messages import _identity_from_bearer

router = APIRouter(prefix="/trust-voice", tags=["Trust Voice Preferences"])

FEATURE_VERSION = "0.1.0"
DEFAULT_LOCALE = "en-US"
SUPPORTED_LOCALES = {
    "en-US",
    "es-MX",
    "zh-CN",
    "zh-TW",
    "fr-FR",
    "pt-BR",
    "ar",
    "hi-IN",
    "ja-JP",
}


class PreferenceUpdate(BaseModel):
    preferred_locale: str


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_locale(value: str) -> str:
    locale = str(value or "").strip()
    if locale not in SUPPORTED_LOCALES:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "unsupported_locale",
                "message": "Choose a supported VeriFYD Circle language.",
                "supported_locales": sorted(SUPPORTED_LOCALES),
            },
        )
    return locale


def ensure_preferences_schema() -> None:
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS verifyd_voice_preferences (
                identity_id       TEXT PRIMARY KEY,
                preferred_locale  TEXT NOT NULL DEFAULT 'en-US',
                created_at        TEXT NOT NULL,
                updated_at        TEXT NOT NULL
            )
            """
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_verifyd_voice_preferences_locale
            ON verifyd_voice_preferences (preferred_locale)
            """
        )


def _read_preferences(identity_id: str) -> dict:
    ensure_preferences_schema()
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT identity_id, preferred_locale, created_at, updated_at
            FROM verifyd_voice_preferences
            WHERE identity_id=%s
            LIMIT 1
            """,
            (identity_id,),
        )
        row = cur.fetchone()

    if not row:
        return {
            "identity_id": identity_id,
            "preferred_locale": DEFAULT_LOCALE,
            "source": "default",
            "created_at": None,
            "updated_at": None,
        }

    item = dict(row)
    locale = item.get("preferred_locale") or DEFAULT_LOCALE
    if locale not in SUPPORTED_LOCALES:
        locale = DEFAULT_LOCALE
    return {
        "identity_id": identity_id,
        "preferred_locale": locale,
        "source": "account",
        "created_at": item.get("created_at"),
        "updated_at": item.get("updated_at"),
    }


def preferred_locale_for_identity(identity_id: str) -> str:
    """Return the account locale for server-side notifications/copy."""
    try:
        return str(_read_preferences(str(identity_id or ""))["preferred_locale"])
    except Exception:
        return DEFAULT_LOCALE


@router.get("/preferences/health")
def preferences_health():
    return {
        "status": "ok",
        "feature": "trust_voice_preferences",
        "version": FEATURE_VERSION,
        "default_locale": DEFAULT_LOCALE,
        "supported_locales": sorted(SUPPORTED_LOCALES),
        "storage": "postgresql_v1",
    }


@router.get("/preferences")
def get_preferences(
    authorization: Optional[str] = Header(default=None),
):
    identity = _identity_from_bearer(authorization)
    return {"preferences": _read_preferences(str(identity["id"]))}


@router.put("/preferences")
def update_preferences(
    payload: PreferenceUpdate,
    authorization: Optional[str] = Header(default=None),
):
    identity = _identity_from_bearer(authorization)
    identity_id = str(identity["id"])
    locale = _normalize_locale(payload.preferred_locale)
    ensure_preferences_schema()
    now = _now_iso()

    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO verifyd_voice_preferences (
                identity_id, preferred_locale, created_at, updated_at
            )
            VALUES (%s,%s,%s,%s)
            ON CONFLICT (identity_id)
            DO UPDATE SET
                preferred_locale=EXCLUDED.preferred_locale,
                updated_at=EXCLUDED.updated_at
            """,
            (identity_id, locale, now, now),
        )

    return {
        "ok": True,
        "preferences": _read_preferences(identity_id),
    }
