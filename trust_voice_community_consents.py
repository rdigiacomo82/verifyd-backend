"""VeriFYD Circle: versioned, authenticated Community Board acknowledgements.

Separate acceptance of Community rules from acknowledgement of public-posting
moderation/privacy notice. Stored by immutable identity_id, not device storage.
Never treat a previous version's acceptance as consent to a new rules version.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Literal, Optional

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel

from database import get_db
from trust_voice_messages import _identity_from_bearer

router = APIRouter(prefix="/trust-voice/community/acknowledgements", tags=["Community Board Acknowledgements"])
RULES_VERSION = "community-rules-2026-11-v1"
PUBLIC_NOTICE_VERSION = "public-posting-2026-11-v1"
CURRENT_VERSIONS = {"rules": RULES_VERSION, "public_notice": PUBLIC_NOTICE_VERSION}
_TRUE = {"1", "true", "yes", "on"}


def _require_board():
    if os.getenv("VERIFYD_TRUST_VOICE_COMMUNITY_ENABLED", "").strip().lower() not in _TRUE:
        raise HTTPException(503, detail={"error": "feature_not_enabled"})


def _actor_id(authorization: Optional[str]):
    _require_board()
    actor = _identity_from_bearer(authorization)
    if not actor.get("id") or actor.get("status") != "active":
        raise HTTPException(401, detail={"error": "session_required"})
    return str(actor["id"])


def _schema():
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("""CREATE TABLE IF NOT EXISTS verifyd_voice_community_acknowledgements (
          identity_id TEXT NOT NULL REFERENCES verifyd_identities(id) ON DELETE CASCADE,
          notice_key TEXT NOT NULL CHECK (notice_key IN ('rules','public_notice')),
          version TEXT NOT NULL,
          acknowledged_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
          PRIMARY KEY(identity_id,notice_key,version)
        )""")


def _accepted_for(cur, identity_id: str, notice_key: str) -> bool:
    version = CURRENT_VERSIONS[notice_key]
    cur.execute("""SELECT 1 FROM verifyd_voice_community_acknowledgements
        WHERE identity_id=%s AND notice_key=%s AND version=%s LIMIT 1""",
                (identity_id, notice_key, version))
    return bool(cur.fetchone())


def verify_post_acknowledgements(identity_id: str, *, is_public: bool):
    """Enforce only when the deployment's migration-safe rollout flag is enabled."""
    if os.getenv("VERIFYD_CIRCLE_BOARD_ACK_ENFORCEMENT_ENABLED", "").strip().lower() not in _TRUE:
        return
    _schema()
    with get_db() as conn:
        cur = conn.cursor()
        if not _accepted_for(cur, identity_id, "rules"):
            raise HTTPException(403, detail={"error": "community_rules_acceptance_required"})
        if is_public and not _accepted_for(cur, identity_id, "public_notice"):
            raise HTTPException(403, detail={"error": "public_posting_acknowledgement_required"})


@router.get("")
def get_acknowledgements(authorization: Optional[str] = Header(default=None)):
    identity_id = _actor_id(authorization)
    _schema()
    with get_db() as conn:
        cur = conn.cursor()
        acknowledged = {key: _accepted_for(cur, identity_id, key) for key in CURRENT_VERSIONS}
    return {"versions": dict(CURRENT_VERSIONS), "acknowledged": acknowledged,
            "persistent": "server_per_identity_and_version"}


class Acknowledge(BaseModel):
    notice_key: Literal["rules", "public_notice"]
    version: str


@router.post("")
def acknowledge(payload: Acknowledge, authorization: Optional[str] = Header(default=None)):
    identity_id = _actor_id(authorization)
    version = CURRENT_VERSIONS[payload.notice_key]
    if payload.version != version:
        raise HTTPException(409, detail={"error": "policy_version_changed",
                                          "message": "Please review the current notice."})
    _schema()
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("""INSERT INTO verifyd_voice_community_acknowledgements
          (identity_id,notice_key,version)
          VALUES (%s,%s,%s) ON CONFLICT(identity_id,notice_key,version) DO NOTHING""",
            (identity_id, payload.notice_key, version))
        cur.execute("""SELECT acknowledged_at FROM verifyd_voice_community_acknowledgements
          WHERE identity_id=%s AND notice_key=%s AND version=%s""",
            (identity_id, payload.notice_key, version))
        row = cur.fetchone()
    recorded = row.get("acknowledged_at") if row else None
    if isinstance(recorded, datetime):
        recorded = recorded.astimezone(timezone.utc).isoformat()
    return {"ok": True, "notice_key": payload.notice_key,
            "version": version, "acknowledged": True, "acknowledged_at": recorded}
