# ============================================================
# VeriFYD Trust Voice — Guest Recording Shares
# VERIFYD_TRUST_VOICE_GROUP_VIDEO_GUEST_SHARE_V1
#
# Additive capability-link access for a completed Trust Voice
# group-video recording. Guest links are view-only, time-limited,
# revocable, and stored as SHA-256 token hashes only.
# ============================================================

from __future__ import annotations

import hashlib
import logging
import secrets
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import APIRouter, Header, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from database import get_db
from trust_voice_groups import _require_group_member, ensure_group_schema
from trust_voice_messages import _identity_from_bearer
from trust_voice_group_video_recording import (
    _duration_seconds,
    _r2_bucket,
    _r2_client,
    _recording_by_id,
    ensure_recording_schema,
)

log = logging.getLogger("verifyd.trust_voice.group_video_guest_share")

router = APIRouter(
    prefix="/trust-voice",
    tags=["Trust Voice Group Video Guest Share"],
)

FEATURE_VERSION = "0.1.0"
_schema_ready = False


class GuestShareCreatePayload(BaseModel):
    expires_in_hours: int = Field(default=168, ge=1, le=720)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _now_iso() -> str:
    return _now().isoformat()


def _token_hash(token: str) -> str:
    return hashlib.sha256(str(token or "").encode("utf-8")).hexdigest()


def ensure_guest_share_schema() -> None:
    global _schema_ready
    if _schema_ready:
        return

    ensure_group_schema()
    ensure_recording_schema()

    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS
            verifyd_voice_group_video_guest_shares (
                share_id                 TEXT PRIMARY KEY,
                token_hash               TEXT UNIQUE NOT NULL,
                recording_id             TEXT NOT NULL,
                session_id               TEXT NOT NULL,
                conversation_id          TEXT NOT NULL,
                created_by_identity_id    TEXT NOT NULL,
                created_at               TEXT NOT NULL,
                expires_at               TEXT NOT NULL,
                revoked_at               TEXT,
                view_count               INTEGER NOT NULL DEFAULT 0,
                last_viewed_at           TEXT
            )
            """
        )
        cur.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS
            uq_verifyd_voice_guest_share_token
            ON verifyd_voice_group_video_guest_shares (token_hash)
            """
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS
            idx_verifyd_voice_guest_share_recording
            ON verifyd_voice_group_video_guest_shares (
                recording_id,
                created_at DESC
            )
            """
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS
            idx_verifyd_voice_guest_share_expiry
            ON verifyd_voice_group_video_guest_shares (expires_at)
            WHERE revoked_at IS NULL
            """
        )

    _schema_ready = True


def _share_by_id(share_id: str) -> Optional[dict]:
    ensure_guest_share_schema()
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT *
            FROM verifyd_voice_group_video_guest_shares
            WHERE share_id = %s
            LIMIT 1
            """,
            (share_id,),
        )
        row = cur.fetchone()
    return dict(row) if row else None


def _active_share_by_token(token: str) -> Optional[dict]:
    ensure_guest_share_schema()
    token = str(token or "").strip()
    if not token:
        return None

    now = _now_iso()
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT
                gs.*,
                r.status AS recording_status,
                r.storage_key,
                r.file_size_bytes,
                r.started_at AS recording_started_at,
                r.ended_at AS recording_ended_at,
                s.started_at AS meeting_started_at,
                s.ended_at AS meeting_ended_at,
                g.name AS group_name
            FROM verifyd_voice_group_video_guest_shares gs
            INNER JOIN verifyd_voice_group_video_recordings r
                ON r.recording_id = gs.recording_id
            INNER JOIN verifyd_voice_conversation_members vm
                ON vm.conversation_id = gs.conversation_id
               AND vm.identity_id = gs.created_by_identity_id
            LEFT JOIN verifyd_voice_group_video_sessions s
                ON s.session_id = gs.session_id
            LEFT JOIN verifyd_voice_groups g
                ON g.conversation_id = gs.conversation_id
            WHERE gs.token_hash = %s
              AND gs.revoked_at IS NULL
              AND gs.expires_at::timestamptz > %s::timestamptz
              AND r.status = 'complete'
              AND r.storage_key IS NOT NULL
              AND r.storage_key <> ''
            LIMIT 1
            """,
            (_token_hash(token), now),
        )
        row = cur.fetchone()

    return dict(row) if row else None


def _require_active_guest_share(token: str) -> dict:
    share = _active_share_by_token(token)
    if not share:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "guest_recording_share_unavailable",
                "message": "This shared recording is no longer available.",
            },
        )
    return share


def _mark_guest_view(share_id: str) -> None:
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            UPDATE verifyd_voice_group_video_guest_shares
            SET
                view_count = COALESCE(view_count, 0) + 1,
                last_viewed_at = %s
            WHERE share_id = %s
              AND revoked_at IS NULL
            """,
            (_now_iso(), share_id),
        )


def _public_guest_metadata(share: dict) -> dict:
    return {
        "available": True,
        "view_only": True,
        "download_allowed": False,
        "group_name": share.get("group_name") or "Trust Voice Meeting",
        "meeting_started_at": share.get("meeting_started_at"),
        "meeting_ended_at": share.get("meeting_ended_at"),
        "recording_started_at": share.get("recording_started_at"),
        "recording_ended_at": share.get("recording_ended_at"),
        "duration_seconds": _duration_seconds(
            share.get("recording_started_at"),
            share.get("recording_ended_at"),
        ),
        "file_size_bytes": (
            int(share.get("file_size_bytes") or 0)
            if share.get("file_size_bytes") is not None
            else None
        ),
        "expires_at": share.get("expires_at"),
    }


def _stream_guest_recording(
    share: dict,
    *,
    range_header: Optional[str] = None,
):
    storage_key = str(share.get("storage_key") or "")
    if not storage_key:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "guest_recording_share_unavailable",
                "message": "This shared recording is no longer available.",
            },
        )

    args = {"Bucket": _r2_bucket(), "Key": storage_key}
    if range_header and str(range_header).lower().startswith("bytes="):
        args["Range"] = str(range_header)

    try:
        obj = _r2_client().get_object(**args)
    except Exception as exc:
        log.exception("Unable to open guest-shared group video recording")
        raise HTTPException(
            status_code=404,
            detail={
                "error": "guest_recording_share_unavailable",
                "message": "This shared recording is no longer available.",
            },
        ) from exc

    body = obj["Body"]

    def iterator():
        try:
            while True:
                chunk = body.read(1024 * 1024)
                if not chunk:
                    break
                yield chunk
        finally:
            try:
                body.close()
            except Exception:
                pass

    headers = {
        "Accept-Ranges": "bytes",
        "Cache-Control": "private, no-store, max-age=0",
        "Content-Disposition": 'inline; filename="VeriFYD-Shared-Recording.mp4"',
        "Referrer-Policy": "no-referrer",
        "X-Robots-Tag": "noindex, nofollow, noarchive",
    }
    if obj.get("ContentLength") is not None:
        headers["Content-Length"] = str(obj["ContentLength"])
    if obj.get("ContentRange"):
        headers["Content-Range"] = str(obj["ContentRange"])

    return StreamingResponse(
        iterator(),
        status_code=206 if obj.get("ContentRange") else 200,
        media_type="video/mp4",
        headers=headers,
    )


@router.get("/group-video/guest-share/health")
def guest_share_health():
    ensure_guest_share_schema()
    return {
        "status": "ok",
        "feature": "trust_voice_group_video_guest_share",
        "version": FEATURE_VERSION,
        "access": "capability_token_v1",
        "token_storage": "sha256_only_v1",
        "default_expiry_hours": 168,
        "maximum_expiry_hours": 720,
        "guest_scope": "recording_view_only_v1",
        "download": "disabled_v1",
        "revocation": "creator_or_group_owner_v1",
        "creator_membership": "required_while_share_active_v1",
    }


@router.post("/group-video/recordings/{recording_id}/guest-share")
def create_guest_share(
    recording_id: str,
    payload: GuestShareCreatePayload,
    authorization: Optional[str] = Header(default=None),
):
    identity = _identity_from_bearer(authorization)
    identity_id = str(identity["id"])

    recording = _recording_by_id(recording_id)
    if not recording:
        raise HTTPException(
            status_code=404,
            detail={"error": "group_video_recording_not_found"},
        )

    conversation_id = str(recording.get("conversation_id") or "")
    _require_group_member(conversation_id, identity_id)

    if recording.get("status") != "complete" or not recording.get("storage_key"):
        raise HTTPException(
            status_code=409,
            detail={
                "error": "guest_share_recording_not_ready",
                "message": "This recording is not ready to share yet.",
            },
        )

    raw_token = secrets.token_urlsafe(32)
    share_id = "tvgvgs_" + uuid.uuid4().hex
    created_at = _now()
    expires_at = created_at + timedelta(hours=int(payload.expires_in_hours))

    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO verifyd_voice_group_video_guest_shares (
                share_id,
                token_hash,
                recording_id,
                session_id,
                conversation_id,
                created_by_identity_id,
                created_at,
                expires_at,
                revoked_at,
                view_count,
                last_viewed_at
            )
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,NULL,0,NULL)
            """,
            (
                share_id,
                _token_hash(raw_token),
                recording_id,
                str(recording.get("session_id") or ""),
                conversation_id,
                identity_id,
                created_at.isoformat(),
                expires_at.isoformat(),
            ),
        )

    return {
        "ok": True,
        "share": {
            "share_id": share_id,
            "guest_token": raw_token,
            "created_at": created_at.isoformat(),
            "expires_at": expires_at.isoformat(),
            "expires_in_hours": int(payload.expires_in_hours),
            "view_only": True,
            "download_allowed": False,
        },
    }


@router.get("/group-video/recordings/{recording_id}/guest-shares")
def list_guest_shares(
    recording_id: str,
    authorization: Optional[str] = Header(default=None),
):
    identity = _identity_from_bearer(authorization)
    identity_id = str(identity["id"])

    recording = _recording_by_id(recording_id)
    if not recording:
        raise HTTPException(
            status_code=404,
            detail={"error": "group_video_recording_not_found"},
        )

    conversation_id = str(recording.get("conversation_id") or "")
    _require_group_member(conversation_id, identity_id)
    ensure_guest_share_schema()

    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT
                share_id,
                created_by_identity_id,
                created_at,
                expires_at,
                revoked_at,
                view_count,
                last_viewed_at
            FROM verifyd_voice_group_video_guest_shares
            WHERE recording_id = %s
            ORDER BY created_at DESC
            """,
            (recording_id,),
        )
        rows = [dict(row) for row in (cur.fetchall() or [])]

    now = _now()
    shares = []
    for row in rows:
        expires_at = None
        try:
            expires_at = datetime.fromisoformat(
                str(row.get("expires_at") or "").replace("Z", "+00:00")
            )
            if expires_at.tzinfo is None:
                expires_at = expires_at.replace(tzinfo=timezone.utc)
            expires_at = expires_at.astimezone(timezone.utc)
        except Exception:
            expires_at = None

        shares.append(
            {
                "share_id": row.get("share_id") or "",
                "created_at": row.get("created_at"),
                "expires_at": row.get("expires_at"),
                "revoked_at": row.get("revoked_at"),
                "view_count": int(row.get("view_count") or 0),
                "last_viewed_at": row.get("last_viewed_at"),
                "created_by_me": (
                    str(row.get("created_by_identity_id") or "") == identity_id
                ),
                "active": bool(
                    not row.get("revoked_at")
                    and expires_at is not None
                    and expires_at > now
                ),
            }
        )

    return {"recording_id": recording_id, "shares": shares}


@router.delete("/group-video/guest-shares/{share_id}")
def revoke_guest_share(
    share_id: str,
    authorization: Optional[str] = Header(default=None),
):
    identity = _identity_from_bearer(authorization)
    identity_id = str(identity["id"])

    share = _share_by_id(share_id)
    if not share:
        raise HTTPException(
            status_code=404,
            detail={"error": "guest_recording_share_not_found"},
        )

    group = _require_group_member(
        str(share.get("conversation_id") or ""),
        identity_id,
    )

    owner_id = str(group.get("owner_identity_id") or "")
    creator_id = str(share.get("created_by_identity_id") or "")

    if identity_id not in {creator_id, owner_id}:
        raise HTTPException(
            status_code=403,
            detail={
                "error": "guest_recording_share_revoke_not_allowed",
                "message": "Only the link creator or group owner can revoke this guest link.",
            },
        )

    if share.get("revoked_at"):
        return {"ok": True, "revoked": False, "reason": "already_revoked"}

    revoked_at = _now_iso()
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            UPDATE verifyd_voice_group_video_guest_shares
            SET revoked_at = %s
            WHERE share_id = %s
              AND revoked_at IS NULL
            """,
            (revoked_at, share_id),
        )

    return {
        "ok": True,
        "revoked": True,
        "share_id": share_id,
        "revoked_at": revoked_at,
    }


@router.get("/public/group-video/guest/{guest_token}")
def public_guest_recording_metadata(guest_token: str):
    share = _require_active_guest_share(guest_token)
    _mark_guest_view(str(share.get("share_id") or ""))
    return {
        "status": "ok",
        "recording": _public_guest_metadata(share),
    }


@router.get("/public/group-video/guest/{guest_token}/media")
def public_guest_recording_media(
    guest_token: str,
    request: Request,
):
    share = _require_active_guest_share(guest_token)
    return _stream_guest_recording(
        share,
        range_header=request.headers.get("range"),
    )
