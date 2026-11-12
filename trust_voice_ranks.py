"""VeriFYD Circle Achievements & Trust Cards pilot v0.1.

Disabled by default (VERIFYD_CIRCLE_RANKING_ENABLED=1). Dedicated tables;
no mutation of identity or clinical session records. This is an ACHIEVEMENT
rank, not proof of identity/organization verification or safety.

Games remain recreational: they DO NOT contribute to Trust Rank. Only
eligible server-validated learning or other reviewed non-game achievements
may affect achievement ranking. There is no public points-award endpoint.
"""
from __future__ import annotations

import os
import threading
import uuid
from typing import Literal, Optional

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel

from database import get_db
from trust_voice_messages import _identity_from_bearer

router = APIRouter(prefix="/trust-voice/ranks", tags=["Circle Achievements"])
_schema_lock = threading.Lock()
_schema_ready = False

# Printed card numbering follows the user-provided 01/05 ... 05/05 artwork.
TIERS = [
    {"level": 0, "name": "Newcomer", "color": "#94A3B8", "card": None, "min_points": 0, "min_unique": 0, "min_games": 0},
    {"level": 1, "name": "Nova", "color": "#38BDF8", "card": "NOVA", "min_points": 150, "min_unique": 3, "min_games": 0},
    {"level": 2, "name": "Pixel", "color": "#FB923C", "card": "PIXEL", "min_points": 500, "min_unique": 8, "min_games": 0},
    {"level": 3, "name": "Echo", "color": "#C084FC", "card": "ECHO", "min_points": 1100, "min_unique": 16, "min_games": 0},
    {"level": 4, "name": "Spark", "color": "#A3E635", "card": "SPARK", "min_points": 2500, "min_unique": 30, "min_games": 0},
    {"level": 5, "name": "Luna", "color": "#F0ABFC", "card": "LUNA", "min_points": 4500, "min_unique": 50, "min_games": 0},
]

# Controlled, server-only challenge IDs. No public client-award endpoint.
# Every additional challenge must be explicitly reviewed before release.
CHALLENGES = {}  # Game points disabled; games unlock separately.
AVATAR_UNLOCKS = {
    level: ["default", *["nova", "pixel", "echo", "spark", "luna"][:level]]
    for level in range(6)
}


def enabled():
    return os.getenv("VERIFYD_CIRCLE_RANKING_ENABLED", "0").strip().lower() in ("1", "true", "on", "yes")


def _require_enabled():
    if not enabled():
        raise HTTPException(404, detail={"error": "circle_ranks_unavailable"})


def _owner(authorization: Optional[str]):
    _require_enabled()
    identity = _identity_from_bearer(authorization)
    if identity.get("status") != "active":
        raise HTTPException(401, detail={"error": "session_required"})
    return identity


def _schema():
    global _schema_ready
    if _schema_ready:
        return
    with _schema_lock:
        if _schema_ready:
            return
        with get_db() as conn:
            cur = conn.cursor()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS verifyd_circle_rank_profiles (
                    identity_id TEXT PRIMARY KEY REFERENCES verifyd_identities(id) ON DELETE CASCADE,
                    show_public_number BOOLEAN NOT NULL DEFAULT FALSE,
                    selected_avatar TEXT NOT NULL DEFAULT 'default',
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS verifyd_circle_rank_awards (
                    award_id TEXT PRIMARY KEY,
                    identity_id TEXT NOT NULL REFERENCES verifyd_identities(id) ON DELETE CASCADE,
                    challenge_id TEXT NOT NULL,
                    kind TEXT NOT NULL CHECK(kind IN ('game', 'challenge')),
                    points INTEGER NOT NULL CHECK(points BETWEEN 1 AND 1000),
                    evidence_ref TEXT NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    UNIQUE(identity_id, challenge_id),
                    UNIQUE(challenge_id, evidence_ref)
                )
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_circle_rank_awards_identity
                ON verifyd_circle_rank_awards(identity_id)
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS verifyd_circle_rank_card_requests (
                    request_id TEXT PRIMARY KEY,
                    identity_id TEXT NOT NULL REFERENCES verifyd_identities(id) ON DELETE CASCADE,
                    level INTEGER NOT NULL CHECK(level BETWEEN 1 AND 5),
                    status TEXT NOT NULL DEFAULT 'requested'
                        CHECK(status IN ('requested', 'review', 'approved', 'mailed', 'cancelled')),
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    UNIQUE(identity_id, level)
                )
            """)
        _schema_ready = True


def tier_for(points, unique, games):
    winner = TIERS[0]
    for tier in TIERS[1:]:
        if points >= tier["min_points"] and unique >= tier["min_unique"]:
            winner = tier
    return winner


def _snapshot(identity_id: str, owner: bool = False):
    _schema()
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("""SELECT show_public_number, selected_avatar
                       FROM verifyd_circle_rank_profiles WHERE identity_id=%s""", (identity_id,))
        profile = cur.fetchone() or {}
        cur.execute("""SELECT COALESCE(SUM(points) FILTER (WHERE kind='challenge'),0) AS points,
                           COUNT(*) FILTER (WHERE kind='challenge') AS unique_count,
                           COUNT(*) FILTER (WHERE kind='game') AS game_count
                       FROM verifyd_circle_rank_awards WHERE identity_id=%s""", (identity_id,))
        totals = cur.fetchone() or {}
        cur.execute("""SELECT challenge_id, points, created_at FROM verifyd_circle_rank_awards
                       WHERE identity_id=%s ORDER BY created_at DESC LIMIT 100""", (identity_id,))
        awards = [dict(row) for row in cur.fetchall()] if owner else []
        if owner:
            cur.execute("""SELECT level, status, created_at FROM verifyd_circle_rank_card_requests
                           WHERE identity_id=%s ORDER BY level""", (identity_id,))
            requests = [dict(row) for row in cur.fetchall()]
        else:
            requests = []
    points = int(totals.get("points") or 0)
    unique = int(totals.get("unique_count") or 0)
    games = int(totals.get("game_count") or 0)
    tier = tier_for(points, unique, games)
    show = bool(profile.get("show_public_number", False))
    selected = profile.get("selected_avatar") or "default"
    if selected not in AVATAR_UNLOCKS[tier["level"]]:
        selected = "default"
    data = {
        "identity_id": identity_id,
        "tier": tier,
        "show_public_number": show,
        "public_number": points if show else None,
        "score_label": "Activity points — not an identity verification",
        "selected_avatar": selected,
    }
    if owner:
        data.update({
            "points": points,
            "unique_achievements": unique,
            "unique_games": games,
            "next_tier": TIERS[tier["level"] + 1] if tier["level"] < 5 else None,
            "unlocked_avatars": AVATAR_UNLOCKS[tier["level"]],
            "awards": awards,
            "card_requests": requests,
            "eligible_cards": [t["card"] for t in TIERS[1:tier["level"] + 1]],
        })
    return data


class RankSettings(BaseModel):
    show_public_number: Optional[bool] = None
    selected_avatar: Optional[Literal["default", "nova", "pixel", "echo", "spark", "luna"]] = None


@router.get("/catalog")
def catalog():
    _require_enabled()
    return {
        "version": "0.1.0", "tiers": TIERS,
        "challenges": [{**c, "id": key, "status": "verification_integration_pending"}
                       for key, c in CHALLENGES.items()],
        "warning": "Learning achievements, not gameplay or identity verification, drive Trust Rank.",
    }


@router.get("/me")
def my_rank(authorization: Optional[str] = Header(default=None)):
    identity = _owner(authorization)
    return _snapshot(identity["id"], owner=True)


@router.patch("/me/settings")
def settings(payload: RankSettings, authorization: Optional[str] = Header(default=None)):
    identity = _owner(authorization)
    if payload.show_public_number is None and payload.selected_avatar is None:
        raise HTTPException(400, detail={"error": "no_changes"})
    before = _snapshot(identity["id"], owner=True)
    selected = payload.selected_avatar if payload.selected_avatar is not None else before["selected_avatar"]
    if selected not in before["unlocked_avatars"]:
        raise HTTPException(403, detail={"error": "avatar_locked"})
    show = payload.show_public_number if payload.show_public_number is not None else before["show_public_number"]
    with get_db() as conn:
        conn.cursor().execute("""
            INSERT INTO verifyd_circle_rank_profiles (identity_id, show_public_number, selected_avatar)
            VALUES (%s,%s,%s)
            ON CONFLICT (identity_id) DO UPDATE SET
                show_public_number=EXCLUDED.show_public_number,
                selected_avatar=EXCLUDED.selected_avatar,
                updated_at=NOW()
        """, (identity["id"], show, selected))
    return _snapshot(identity["id"], owner=True)


@router.get("/public/{identity_id}")
def public_rank(identity_id: str):
    _require_enabled()
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("SELECT 1 FROM verifyd_identities WHERE id=%s AND status=%s", (identity_id, "active"))
        if not cur.fetchone():
            raise HTTPException(404, detail={"error": "identity_not_found"})
    return _snapshot(identity_id, owner=False)


@router.post("/me/cards/{level}/request", status_code=202)
def request_card(level: int, authorization: Optional[str] = Header(default=None)):
    identity = _owner(authorization)
    if level < 1 or level > 5:
        raise HTTPException(400, detail={"error": "invalid_level"})
    snapshot = _snapshot(identity["id"], owner=True)
    if snapshot["tier"]["level"] < level:
        raise HTTPException(403, detail={"error": "card_locked"})
    # No address input accepted here; manual review is required for fulfillment.
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO verifyd_circle_rank_card_requests(request_id, identity_id, level)
            VALUES (%s,%s,%s)
            ON CONFLICT(identity_id, level) DO NOTHING
        """, (str(uuid.uuid4()), identity["id"], level))
        cur.execute("""SELECT status FROM verifyd_circle_rank_card_requests
                       WHERE identity_id=%s AND level=%s""", (identity["id"], level))
        result = cur.fetchone() or {}
    return {"level": level, "status": result.get("status", "requested"),
            "fulfillment": "pending_manual_review", "mailing_address_submitted": False}


def award_verified_challenge(identity_id: str, challenge_id: str, evidence_ref: str):
    """ONLY call after server validation. Never expose as a raw request handler.

    DB uniqueness prevents replay of a challenge or server-issued evidence.
    """
    _require_enabled()
    definition = CHALLENGES.get(challenge_id)
    if not definition or not evidence_ref or not identity_id:
        raise ValueError("Unknown challenge or missing verified evidence")
    _schema()
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO verifyd_circle_rank_awards
            (award_id, identity_id, challenge_id, kind, points, evidence_ref)
            VALUES (%s,%s,%s,%s,%s,%s)
            ON CONFLICT DO NOTHING
            RETURNING award_id
        """, (str(uuid.uuid4()), identity_id, challenge_id, definition["kind"], definition["points"], evidence_ref))
        awarded = bool(cur.fetchone())
    return {"awarded": awarded, "challenge_id": challenge_id,
            "points_added": definition["points"] if awarded else 0,
            "rank": _snapshot(identity_id, owner=True)}
