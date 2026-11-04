"""
VeriFYD Circle -- consent-controlled location sharing (Phase A).

Opt-in, latest-position-only API for existing Trust Circle identities.
Not a background-location implementation: native iOS/Android clients require
separate permissions, lifecycle integration, and privacy review.

Feature is disabled by default: VERIFYD_TRUST_VOICE_LOCATION_ENABLED=1
is required before any location-sharing endpoint is available.
"""
from __future__ import annotations

import hashlib
import math
import os
import secrets
import threading
from datetime import datetime, timedelta, timezone
from typing import Literal, Optional

from fastapi import APIRouter, Header, HTTPException, Response
from pydantic import BaseModel, Field

from database import get_db
from trust_voice_blocking import identities_blocked
from trust_voice_permissions import trust_circle_contains
from trust_voice_signaling import _identity_by_handle, _identity_by_id, _verify_token

router = APIRouter(prefix="/trust-voice/location", tags=["Trust Voice Location"])
_SCHEMA_READY = False
_SCHEMA_LOCK = threading.Lock()
MAX_LOCATION_AGE_SECONDS = 24 * 3600
STALE_AFTER_SECONDS = 15 * 60


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: Optional[datetime]) -> Optional[str]:
    return value.isoformat() if value is not None else None


def _enabled() -> bool:
    return os.getenv("VERIFYD_TRUST_VOICE_LOCATION_ENABLED", "0").strip().lower() in (
        "1", "true", "yes", "on"
    )


def _identity(authorization: Optional[str]) -> dict:
    if not _enabled():
        raise HTTPException(404, detail={"error": "feature_unavailable"})
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(401, detail={"error": "session_required"})
    token = authorization.split(" ", 1)[1].strip()
    if not token:
        raise HTTPException(401, detail={"error": "session_required"})
    try:
        payload = _verify_token(token)
    except HTTPException:
        # Preserve 503 from the persistent-session database; never sign the user
        # out because Render or Postgres temporarily cannot validate a token.
        raise
    except Exception:
        raise HTTPException(401, detail={"error": "session_required"})
    user = _identity_by_id(str(payload.get("identity_id") or ""))
    if not user or str(user.get("status") or "active") != "active":
        raise HTTPException(401, detail={"error": "identity_not_found"})
    return user


def _schema() -> None:
    """Create isolated tables once per process. No existing data tables are altered."""
    global _SCHEMA_READY
    if _SCHEMA_READY:
        return
    with _SCHEMA_LOCK:
        if _SCHEMA_READY:
            return
        with get_db() as conn:
            cur = conn.cursor()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS verifyd_circle_location_shares (
                    owner_identity_id TEXT PRIMARY KEY,
                    enabled BOOLEAN NOT NULL DEFAULT FALSE,
                    share_scope TEXT NOT NULL DEFAULT 'selected',
                    precision_mode TEXT NOT NULL DEFAULT 'approximate',
                    expires_at TIMESTAMPTZ NULL,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS verifyd_circle_location_grants (
                    owner_identity_id TEXT NOT NULL,
                    viewer_identity_id TEXT NOT NULL,
                    PRIMARY KEY (owner_identity_id, viewer_identity_id)
                )
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_circle_location_grants_viewer
                ON verifyd_circle_location_grants(viewer_identity_id)
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS verifyd_circle_latest_locations (
                    owner_identity_id TEXT PRIMARY KEY,
                    latitude DOUBLE PRECISION NOT NULL,
                    longitude DOUBLE PRECISION NOT NULL,
                    accuracy_m DOUBLE PRECISION NULL,
                    recorded_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS verifyd_circle_location_native_tokens (
                    owner_identity_id TEXT PRIMARY KEY,
                    token_hash TEXT NOT NULL UNIQUE,
                    session_id TEXT NOT NULL,
                    expires_at TIMESTAMPTZ NOT NULL,
                    issued_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
            """)
        _SCHEMA_READY = True


class SharingUpdate(BaseModel):
    enabled: bool = False
    scope: Literal["selected", "all"] = "selected"
    precision: Literal["precise", "approximate"] = "approximate"
    selected_handles: list[str] = Field(default_factory=list, max_length=100)
    duration_minutes: Optional[Literal[15, 60, 480, 1440]] = None


class PositionUpdate(BaseModel):
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    accuracy_m: Optional[float] = Field(default=None, ge=0, le=100000)


def _no_cache(response: Response) -> None:
    response.headers["Cache-Control"] = "private, no-store"


def _purge_expired(cur) -> None:
    """Never retain historical positions; discard old or inactive positions."""
    cur.execute("""
        DELETE FROM verifyd_circle_latest_locations p
        WHERE p.recorded_at < NOW() - INTERVAL '24 hours'
          OR NOT EXISTS (
            SELECT 1 FROM verifyd_circle_location_shares s
            WHERE s.owner_identity_id=p.owner_identity_id
              AND s.enabled=TRUE
              AND (s.expires_at IS NULL OR s.expires_at > NOW())
          )
    """)


@router.get("/sharing")
def get_sharing(response: Response, authorization: Optional[str] = Header(default=None)):
    owner = _identity(authorization)
    _no_cache(response)
    _schema()
    owner_id = str(owner["id"])
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT enabled, share_scope, precision_mode, expires_at
            FROM verifyd_circle_location_shares WHERE owner_identity_id=%s
        """, (owner_id,))
        row = cur.fetchone()
        cur.execute("""
            SELECT viewer_identity_id FROM verifyd_circle_location_grants
            WHERE owner_identity_id=%s
        """, (owner_id,))
        grants = [str(item["viewer_identity_id"]) for item in (cur.fetchall() or [])]
    if not row:
        return {
            "enabled": False, "scope": "selected", "precision": "approximate",
            "selected_identity_ids": [], "expires_at": None,
        }
    active = bool(row["enabled"] and (
        row["expires_at"] is None or row["expires_at"] > _now()
    ))
    return {
        "enabled": active, "scope": row["share_scope"],
        "precision": row["precision_mode"], "selected_identity_ids": grants,
        "expires_at": _iso(row["expires_at"]),
    }


@router.put("/sharing")
def set_sharing(
    body: SharingUpdate, response: Response, authorization: Optional[str] = Header(default=None)
):
    owner = _identity(authorization)
    _no_cache(response)
    _schema()
    owner_id = str(owner["id"])
    grants: list[str] = []
    if body.enabled and body.scope == "selected":
        if not body.selected_handles:
            raise HTTPException(422, detail={"error": "select_one_or_more_contacts"})
        for raw in body.selected_handles:
            handle = str(raw or "").strip().lstrip("@").lower()
            member = _identity_by_handle(handle) if handle else None
            if not member:
                raise HTTPException(422, detail={"error": "invalid_circle_member"})
            member_id = str(member["id"])
            if (
                member_id == owner_id
                or not trust_circle_contains(owner_id, member_id)
                or identities_blocked(owner_id, member_id)
            ):
                raise HTTPException(403, detail={"error": "invalid_circle_member"})
            grants.append(member_id)
        grants = list(dict.fromkeys(grants))
    expiry = (
        _now() + timedelta(minutes=body.duration_minutes)
        if body.enabled and body.duration_minutes is not None else None
    )
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO verifyd_circle_location_shares
                (owner_identity_id,enabled,share_scope,precision_mode,expires_at,updated_at)
            VALUES (%s,%s,%s,%s,%s,NOW())
            ON CONFLICT(owner_identity_id) DO UPDATE SET
                enabled=EXCLUDED.enabled,
                share_scope=EXCLUDED.share_scope,
                precision_mode=EXCLUDED.precision_mode,
                expires_at=EXCLUDED.expires_at,
                updated_at=NOW()
        """, (owner_id, body.enabled, body.scope, body.precision, expiry))
        cur.execute(
            "DELETE FROM verifyd_circle_location_grants WHERE owner_identity_id=%s",
            (owner_id,),
        )
        # Clear prior location and invalidate native writers whenever consent
        # or recipients change; a previous position must not leak to new viewers.
        cur.execute("DELETE FROM verifyd_circle_location_native_tokens WHERE owner_identity_id=%s", (owner_id,))
        cur.execute("DELETE FROM verifyd_circle_latest_locations WHERE owner_identity_id=%s", (owner_id,))
        if body.enabled:
            for member_id in grants:
                cur.execute("""
                    INSERT INTO verifyd_circle_location_grants
                        (owner_identity_id,viewer_identity_id)
                    VALUES (%s,%s)
                """, (owner_id, member_id))
        else:
            cur.execute(
                "DELETE FROM verifyd_circle_latest_locations WHERE owner_identity_id=%s",
                (owner_id,),
            )
    return {
        "ok": True, "enabled": body.enabled,
        "scope": body.scope, "expires_at": _iso(expiry),
    }


@router.delete("/sharing")
def stop_sharing(response: Response, authorization: Optional[str] = Header(default=None)):
    owner = _identity(authorization)
    _no_cache(response)
    _schema()
    owner_id = str(owner["id"])
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("""
            UPDATE verifyd_circle_location_shares
            SET enabled=FALSE,updated_at=NOW()
            WHERE owner_identity_id=%s
        """, (owner_id,))
        cur.execute(
            "DELETE FROM verifyd_circle_location_grants WHERE owner_identity_id=%s",
            (owner_id,),
        )
        cur.execute(
            "DELETE FROM verifyd_circle_latest_locations WHERE owner_identity_id=%s",
            (owner_id,),
        )
        cur.execute("DELETE FROM verifyd_circle_location_native_tokens WHERE owner_identity_id=%s", (owner_id,))
    return {"ok": True, "enabled": False}


@router.post("/position")
def update_position(
    body: PositionUpdate, response: Response, authorization: Optional[str] = Header(default=None)
):
    owner = _identity(authorization)
    _no_cache(response)
    _schema()
    if (
        not math.isfinite(body.latitude) or not math.isfinite(body.longitude)
        or (body.accuracy_m is not None and not math.isfinite(body.accuracy_m))
    ):
        raise HTTPException(422, detail={"error": "invalid_coordinates"})
    owner_id = str(owner["id"])
    with get_db() as conn:
        cur = conn.cursor()
        # Locks same row as Stop Sharing updates: a delayed position cannot
        # silently re-enable visibility after explicit revocation.
        cur.execute("""
            SELECT enabled,precision_mode,expires_at
            FROM verifyd_circle_location_shares
            WHERE owner_identity_id=%s FOR UPDATE
        """, (owner_id,))
        share = cur.fetchone()
        if not share or not share["enabled"] or (
            share["expires_at"] is not None and share["expires_at"] <= _now()
        ):
            raise HTTPException(403, detail={"error": "sharing_disabled"})
        cur.execute("""
            SELECT recorded_at FROM verifyd_circle_latest_locations
            WHERE owner_identity_id=%s
        """, (owner_id,))
        previous = cur.fetchone()
        if previous and (_now() - previous["recorded_at"]).total_seconds() < 10:
            return {"ok": True, "throttled": True}
        approximate = share["precision_mode"] == "approximate"
        lat = round(body.latitude, 2) if approximate else body.latitude
        lon = round(body.longitude, 2) if approximate else body.longitude
        accuracy = None if approximate else body.accuracy_m
        cur.execute("""
            INSERT INTO verifyd_circle_latest_locations
                (owner_identity_id,latitude,longitude,accuracy_m,recorded_at)
            VALUES (%s,%s,%s,%s,NOW())
            ON CONFLICT(owner_identity_id) DO UPDATE SET
                latitude=EXCLUDED.latitude,longitude=EXCLUDED.longitude,
                accuracy_m=EXCLUDED.accuracy_m,recorded_at=NOW()
        """, (owner_id, lat, lon, accuracy))
    return {"ok": True, "throttled": False}


@router.get("/visible")
def visible_locations(response: Response, authorization: Optional[str] = Header(default=None)):
    viewer = _identity(authorization)
    _no_cache(response)
    _schema()
    viewer_id = str(viewer["id"])
    with get_db() as conn:
        cur = conn.cursor()
        _purge_expired(cur)
        cur.execute("""
            SELECT s.owner_identity_id,s.precision_mode,
                   p.latitude,p.longitude,p.accuracy_m,p.recorded_at
            FROM verifyd_circle_location_shares s
            LEFT JOIN verifyd_circle_latest_locations p
              ON p.owner_identity_id=s.owner_identity_id
            WHERE s.enabled=TRUE
              AND (s.expires_at IS NULL OR s.expires_at>NOW())
              AND s.owner_identity_id<>%s
              AND (
                s.share_scope='all' OR EXISTS (
                    SELECT 1 FROM verifyd_circle_location_grants g
                    WHERE g.owner_identity_id=s.owner_identity_id
                      AND g.viewer_identity_id=%s
                )
              )
              AND EXISTS (
                SELECT 1 FROM verifyd_trust_circle_members m
                WHERE m.owner_identity_id=s.owner_identity_id
                  AND m.member_identity_id=%s
              )
            ORDER BY s.updated_at DESC
            LIMIT 100
        """, (viewer_id, viewer_id, viewer_id))
        rows = list(cur.fetchall() or [])
    data = []
    now = _now()
    for row in rows:
        owner_id = str(row["owner_identity_id"])
        if identities_blocked(owner_id, viewer_id):
            continue
        owner = _identity_by_id(owner_id)
        if not owner or str(owner.get("status") or "active") != "active":
            continue
        recorded = row["recorded_at"]
        age = max(0, int((now - recorded).total_seconds())) if recorded else None
        current = age is not None and age <= MAX_LOCATION_AGE_SECONDS
        data.append({
            "identity_id": owner_id,
            "handle": owner.get("handle") or "",
            "display_name": owner.get("display_name") or "",
            "precision": row["precision_mode"],
            "latitude": row["latitude"] if current else None,
            "longitude": row["longitude"] if current else None,
            "accuracy_m": row["accuracy_m"] if current else None,
            "updated_at": _iso(recorded) if current else None,
            "age_seconds": age if current else None,
            "stale": age is None or age > STALE_AFTER_SECONDS,
        })
    return {"locations": data}


# The native bridge never stores the user's primary session token in the
# background-location provider. Instead it receives a narrow upload-only
# token bound to the revocable session that issued it.
class NativePositionUpdate(BaseModel):
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    accuracy: Optional[float] = Field(default=None, ge=0, le=100000)
    time: Optional[float] = Field(default=None, ge=0)


@router.post("/native-token")
def issue_native_upload_token(
    response: Response,
    authorization: Optional[str] = Header(default=None),
):
    owner = _identity(authorization)
    _no_cache(response)
    _schema()
    auth = authorization.split(" ", 1)[1].strip()
    payload = _verify_token(auth)
    session_id = str(payload.get("session_id") or "")
    if not session_id:
        raise HTTPException(403, detail={"error": "persistent_session_required"})
    owner_id = str(owner["id"])
    now = _now()
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT enabled,expires_at FROM verifyd_circle_location_shares
            WHERE owner_identity_id=%s FOR UPDATE
        """, (owner_id,))
        share = cur.fetchone()
        if not share or not share["enabled"] or (
            share["expires_at"] and share["expires_at"] <= now
        ):
            raise HTTPException(403, detail={"error": "sharing_disabled"})
        expiry = now + timedelta(days=30)
        if share["expires_at"]:
            expiry = min(expiry, share["expires_at"])
        raw = "loc_" + secrets.token_urlsafe(32)
        digest = hashlib.sha256(raw.encode("ascii")).hexdigest()
        cur.execute("""
            INSERT INTO verifyd_circle_location_native_tokens
                (owner_identity_id,token_hash,session_id,expires_at,issued_at)
            VALUES (%s,%s,%s,%s,NOW())
            ON CONFLICT(owner_identity_id) DO UPDATE SET
                token_hash=EXCLUDED.token_hash,
                session_id=EXCLUDED.session_id,
                expires_at=EXCLUDED.expires_at,issued_at=NOW()
        """, (owner_id, digest, session_id, expiry))
    return {"upload_token": raw, "expires_at": _iso(expiry)}


@router.delete("/native-token")
def revoke_native_upload_token(
    response: Response,
    authorization: Optional[str] = Header(default=None),
):
    owner = _identity(authorization)
    _no_cache(response)
    _schema()
    with get_db() as conn:
        conn.cursor().execute(
            "DELETE FROM verifyd_circle_location_native_tokens WHERE owner_identity_id=%s",
            (str(owner["id"]),),
        )
    return {"ok": True}


@router.post("/native-position")
def upload_native_position(
    body: NativePositionUpdate,
    response: Response,
    authorization: Optional[str] = Header(default=None),
):
    if not _enabled():
        raise HTTPException(404, detail={"error": "feature_unavailable"})
    _no_cache(response)
    supplied = str(authorization or "").strip()
    token = supplied[7:].strip() if supplied.lower().startswith("bearer ") else ""
    if not token.startswith("loc_") or len(token) > 120:
        raise HTTPException(401, detail={"error": "invalid_location_upload_token"})
    # Capgo's documented Location.time field can be null. Null timestamps
    # indicate that native reported the fix without provider time; use server
    # receipt time, but validate an actual timestamp when it is available.
    if not all(math.isfinite(v) for v in (
        body.latitude, body.longitude,
        *([body.time] if body.time is not None else []),
        *([body.accuracy] if body.accuracy is not None else []),
    )):
        raise HTTPException(422, detail={"error": "invalid_native_position"})
    now = _now()
    if body.time is not None:
        try:
            produced = datetime.fromtimestamp(body.time / 1000, tz=timezone.utc)
        except (ValueError, OverflowError, OSError):
            raise HTTPException(422, detail={"error": "invalid_position_time"})
        if not -60 <= (now - produced).total_seconds() <= 300:
            raise HTTPException(422, detail={"error": "stale_position"})
    _schema()
    digest = hashlib.sha256(token.encode("ascii")).hexdigest()
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT owner_identity_id FROM verifyd_circle_location_native_tokens
            WHERE token_hash=%s AND expires_at>NOW()
        """, (digest,))
        token_row = cur.fetchone()
        if not token_row:
            raise HTTPException(401, detail={"error": "invalid_location_upload_token"})
        owner_id = str(token_row["owner_identity_id"])
        # Serialize uploads with Stop Sharing and changes to recipients.
        cur.execute("""
            SELECT enabled,precision_mode,expires_at
            FROM verifyd_circle_location_shares
            WHERE owner_identity_id=%s FOR UPDATE
        """, (owner_id,))
        share = cur.fetchone()
        if not share or not share["enabled"] or (
            share["expires_at"] and share["expires_at"] <= now
        ):
            raise HTTPException(403, detail={"error": "sharing_disabled"})
        # Recheck credential AFTER share lock, and check account/session revoke.
        cur.execute("""
            SELECT 1 FROM verifyd_circle_location_native_tokens t
            JOIN verifyd_voice_sessions sess ON sess.session_id=t.session_id
            JOIN verifyd_identities ident ON ident.id=t.owner_identity_id
            WHERE t.owner_identity_id=%s AND t.token_hash=%s
              AND t.expires_at>NOW() AND sess.revoked_at IS NULL
              AND sess.identity_id=t.owner_identity_id
              AND ident.status='active'
            LIMIT 1
        """, (owner_id, digest))
        if not cur.fetchone():
            raise HTTPException(401, detail={"error": "invalid_location_upload_token"})
        cur.execute("""
            SELECT recorded_at FROM verifyd_circle_latest_locations
            WHERE owner_identity_id=%s
        """, (owner_id,))
        prior = cur.fetchone()
        if prior and (now - prior["recorded_at"]).total_seconds() < 10:
            return {"ok": True, "throttled": True}
        approximate = share["precision_mode"] == "approximate"
        lat = round(body.latitude, 2) if approximate else body.latitude
        lon = round(body.longitude, 2) if approximate else body.longitude
        accuracy = None if approximate else body.accuracy
        cur.execute("""
            INSERT INTO verifyd_circle_latest_locations
                (owner_identity_id,latitude,longitude,accuracy_m,recorded_at)
            VALUES (%s,%s,%s,%s,NOW())
            ON CONFLICT(owner_identity_id) DO UPDATE SET
                latitude=EXCLUDED.latitude,longitude=EXCLUDED.longitude,
                accuracy_m=EXCLUDED.accuracy_m,recorded_at=NOW()
        """, (owner_id, lat, lon, accuracy))
    return {"ok": True, "throttled": False}
