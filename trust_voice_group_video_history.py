# ============================================================
# VeriFYD Trust Voice — Group Video Reliability + History
# VERIFYD_TRUST_VOICE_GROUP_VIDEO_HISTORY_V1
# Additive companion module. Existing group-video behavior is unchanged.
# ============================================================

from __future__ import annotations

import hashlib
import os
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Header, HTTPException, Query, Request
from livekit import api

from database import get_db
from trust_voice_groups import _group_member_rows, ensure_group_schema
from trust_voice_messages import _identity_from_bearer, _public_contact
from trust_voice_signaling import _identity_by_id

router = APIRouter(
    prefix="/trust-voice",
    tags=["Trust Voice Group Video History"],
)

FEATURE_VERSION = "0.1.0"
_schema_ready = False


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _now_iso() -> str:
    return _now().isoformat()


def _parse_iso(value: Optional[str]) -> Optional[datetime]:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def _duration_seconds(started_at: Optional[str], ended_at: Optional[str]) -> Optional[int]:
    start = _parse_iso(started_at)
    if not start:
        return None
    end = _parse_iso(ended_at) or _now()
    return max(0, int((end - start).total_seconds()))


def _livekit_key() -> str:
    return (os.environ.get("LIVEKIT_API_KEY", "") or "").strip()


def _livekit_secret() -> str:
    return (os.environ.get("LIVEKIT_API_SECRET", "") or "").strip()


def _participant_identity(identity_id: str) -> str:
    return "tvp_" + hashlib.sha256(
        str(identity_id or "").encode("utf-8")
    ).hexdigest()[:32]


def ensure_schema() -> None:
    global _schema_ready
    if _schema_ready:
        return

    ensure_group_schema()

    with get_db() as conn:
        cur = conn.cursor()

        cur.execute(
            """
            ALTER TABLE verifyd_voice_group_video_sessions
            ADD COLUMN IF NOT EXISTS first_connected_at TEXT
            """
        )
        cur.execute(
            """
            ALTER TABLE verifyd_voice_group_video_sessions
            ADD COLUMN IF NOT EXISTS last_activity_at TEXT
            """
        )
        cur.execute(
            """
            ALTER TABLE verifyd_voice_group_video_sessions
            ADD COLUMN IF NOT EXISTS livekit_room_sid TEXT
            """
        )
        cur.execute(
            """
            ALTER TABLE verifyd_voice_group_video_sessions
            ADD COLUMN IF NOT EXISTS peak_participant_count INTEGER NOT NULL DEFAULT 0
            """
        )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS verifyd_voice_group_video_participants (
                session_id             TEXT NOT NULL,
                identity_id            TEXT NOT NULL,
                participant_identity   TEXT NOT NULL,
                first_joined_at        TEXT,
                last_joined_at         TEXT,
                left_at                TEXT,
                join_count             INTEGER NOT NULL DEFAULT 0,
                total_seconds          DOUBLE PRECISION NOT NULL DEFAULT 0,
                currently_connected    BOOLEAN NOT NULL DEFAULT FALSE,
                PRIMARY KEY (session_id, identity_id)
            )
            """
        )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS verifyd_voice_group_video_webhook_events (
                event_id       TEXT PRIMARY KEY,
                event_type     TEXT NOT NULL,
                received_at    TEXT NOT NULL
            )
            """
        )

    _schema_ready = True


def _session_by_room(room_name: str) -> Optional[dict]:
    ensure_schema()
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT *
            FROM verifyd_voice_group_video_sessions
            WHERE room_name = %s
            ORDER BY started_at DESC
            LIMIT 1
            """,
            (room_name,),
        )
        row = cur.fetchone()
    return dict(row) if row else None


def _identity_id_from_participant(
    conversation_id: str,
    participant_identity: str,
) -> Optional[str]:
    for row in _group_member_rows(conversation_id):
        identity_id = str(
            row.get("identity_id") or row.get("id") or ""
        )
        if identity_id and _participant_identity(identity_id) == participant_identity:
            return identity_id
    return None


def _event_time_iso(event) -> str:
    created_at = int(getattr(event, "created_at", 0) or 0)
    if created_at:
        return datetime.fromtimestamp(
            created_at,
            tz=timezone.utc,
        ).isoformat()
    return _now_iso()


def _webhook_seen(event_id: str) -> bool:
    if not event_id:
        return False
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT 1
            FROM verifyd_voice_group_video_webhook_events
            WHERE event_id = %s
            """,
            (event_id,),
        )
        return bool(cur.fetchone())


def _remember_webhook(event_id: str, event_type: str) -> None:
    if not event_id:
        return
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO verifyd_voice_group_video_webhook_events (
                event_id,
                event_type,
                received_at
            )
            VALUES (%s, %s, %s)
            ON CONFLICT (event_id) DO NOTHING
            """,
            (event_id, event_type, _now_iso()),
        )


def _participant_joined(
    session: dict,
    participant_identity: str,
    event_time: str,
) -> None:
    conversation_id = str(session.get("conversation_id") or "")
    session_id = str(session.get("session_id") or "")
    identity_id = _identity_id_from_participant(
        conversation_id,
        participant_identity,
    )
    if not identity_id:
        return

    with get_db() as conn:
        cur = conn.cursor()

        cur.execute(
            """
            INSERT INTO verifyd_voice_group_video_participants (
                session_id,
                identity_id,
                participant_identity,
                first_joined_at,
                last_joined_at,
                left_at,
                join_count,
                total_seconds,
                currently_connected
            )
            VALUES (%s, %s, %s, %s, %s, NULL, 1, 0, TRUE)
            ON CONFLICT (session_id, identity_id)
            DO UPDATE SET
                participant_identity = EXCLUDED.participant_identity,
                first_joined_at = COALESCE(
                    verifyd_voice_group_video_participants.first_joined_at,
                    EXCLUDED.first_joined_at
                ),
                last_joined_at = EXCLUDED.last_joined_at,
                left_at = NULL,
                join_count = verifyd_voice_group_video_participants.join_count + 1,
                currently_connected = TRUE
            """,
            (
                session_id,
                identity_id,
                participant_identity,
                event_time,
                event_time,
            ),
        )

        cur.execute(
            """
            SELECT COUNT(*) AS active_count
            FROM verifyd_voice_group_video_participants
            WHERE session_id = %s
              AND currently_connected = TRUE
            """,
            (session_id,),
        )
        active_count = int(
            (cur.fetchone() or {}).get("active_count") or 0
        )

        cur.execute(
            """
            UPDATE verifyd_voice_group_video_sessions
            SET
                first_connected_at = COALESCE(first_connected_at, %s),
                last_activity_at = %s,
                peak_participant_count = GREATEST(
                    COALESCE(peak_participant_count, 0),
                    %s
                )
            WHERE session_id = %s
            """,
            (event_time, event_time, active_count, session_id),
        )


def _participant_left(
    session: dict,
    participant_identity: str,
    event_time: str,
) -> None:
    conversation_id = str(session.get("conversation_id") or "")
    session_id = str(session.get("session_id") or "")
    identity_id = _identity_id_from_participant(
        conversation_id,
        participant_identity,
    )
    if not identity_id:
        return

    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT last_joined_at, currently_connected
            FROM verifyd_voice_group_video_participants
            WHERE session_id = %s
              AND identity_id = %s
            """,
            (session_id, identity_id),
        )
        row = cur.fetchone()

        extra_seconds = 0.0
        if row and row.get("currently_connected"):
            joined = _parse_iso(row.get("last_joined_at"))
            left = _parse_iso(event_time)
            if joined and left:
                extra_seconds = max(
                    0.0,
                    (left - joined).total_seconds(),
                )

        cur.execute(
            """
            UPDATE verifyd_voice_group_video_participants
            SET
                left_at = %s,
                total_seconds = COALESCE(total_seconds, 0) + %s,
                currently_connected = FALSE
            WHERE session_id = %s
              AND identity_id = %s
            """,
            (event_time, extra_seconds, session_id, identity_id),
        )

        cur.execute(
            """
            UPDATE verifyd_voice_group_video_sessions
            SET last_activity_at = %s
            WHERE session_id = %s
            """,
            (event_time, session_id),
        )


def _room_finished(session: dict, event_time: str) -> None:
    session_id = str(session.get("session_id") or "")

    with get_db() as conn:
        cur = conn.cursor()

        cur.execute(
            """
            SELECT identity_id, last_joined_at
            FROM verifyd_voice_group_video_participants
            WHERE session_id = %s
              AND currently_connected = TRUE
            """,
            (session_id,),
        )
        rows = [dict(row) for row in (cur.fetchall() or [])]

        for row in rows:
            joined = _parse_iso(row.get("last_joined_at"))
            ended = _parse_iso(event_time)
            extra = 0.0
            if joined and ended:
                extra = max(
                    0.0,
                    (ended - joined).total_seconds(),
                )

            cur.execute(
                """
                UPDATE verifyd_voice_group_video_participants
                SET
                    left_at = %s,
                    total_seconds = COALESCE(total_seconds, 0) + %s,
                    currently_connected = FALSE
                WHERE session_id = %s
                  AND identity_id = %s
                """,
                (
                    event_time,
                    extra,
                    session_id,
                    row.get("identity_id"),
                ),
            )

        cur.execute(
            """
            UPDATE verifyd_voice_group_video_sessions
            SET
                status = CASE
                    WHEN status = 'active' THEN 'ended'
                    ELSE status
                END,
                ended_at = COALESCE(ended_at, %s),
                ended_reason = COALESCE(ended_reason, 'room_finished'),
                last_activity_at = %s
            WHERE session_id = %s
            """,
            (event_time, event_time, session_id),
        )


@router.get("/group-video/history/health")
def history_health():
    ensure_schema()
    return {
        "status": "ok",
        "feature": "trust_voice_group_video_history",
        "version": FEATURE_VERSION,
        "webhooks": "signed_livekit_v1",
        "history": "postgresql_v1",
        "participant_history": "postgresql_v1",
        "working_group_video_structure": "unchanged",
    }


@router.post("/group-video/livekit-webhook")
async def livekit_webhook(
    request: Request,
    authorization: Optional[str] = Header(default=None),
):
    ensure_schema()

    body = (await request.body()).decode("utf-8")
    auth_token = str(authorization or "").strip()

    if auth_token.lower().startswith("bearer "):
        auth_token = auth_token.split(" ", 1)[1].strip()

    if not body or not auth_token:
        raise HTTPException(
            status_code=401,
            detail={"error": "invalid_livekit_webhook"},
        )

    try:
        receiver = api.WebhookReceiver(
            api.TokenVerifier(
                _livekit_key(),
                _livekit_secret(),
            )
        )
        event = receiver.receive(body, auth_token)
    except Exception:
        raise HTTPException(
            status_code=401,
            detail={"error": "invalid_livekit_webhook"},
        )

    event_id = str(getattr(event, "id", "") or "")
    event_type = str(getattr(event, "event", "") or "")

    if event_id and _webhook_seen(event_id):
        return {
            "ok": True,
            "duplicate": True,
            "event": event_type,
        }

    room = getattr(event, "room", None)
    room_name = str(getattr(room, "name", "") or "") if room else ""
    session = _session_by_room(room_name) if room_name else None
    event_time = _event_time_iso(event)

    if event_type == "room_started" and session:
        room_sid = str(getattr(room, "sid", "") or "")
        with get_db() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                UPDATE verifyd_voice_group_video_sessions
                SET
                    first_connected_at = COALESCE(first_connected_at, %s),
                    last_activity_at = %s,
                    livekit_room_sid = COALESCE(NULLIF(%s, ''), livekit_room_sid)
                WHERE session_id = %s
                """,
                (
                    event_time,
                    event_time,
                    room_sid,
                    session.get("session_id"),
                ),
            )

    elif event_type == "participant_joined" and session:
        participant = getattr(event, "participant", None)
        participant_identity = str(
            getattr(participant, "identity", "") or ""
        )
        if participant_identity:
            _participant_joined(
                session,
                participant_identity,
                event_time,
            )

    elif event_type in {
        "participant_left",
        "participant_connection_aborted",
    } and session:
        participant = getattr(event, "participant", None)
        participant_identity = str(
            getattr(participant, "identity", "") or ""
        )
        if participant_identity:
            _participant_left(
                session,
                participant_identity,
                event_time,
            )

    elif event_type == "room_finished" and session:
        _room_finished(session, event_time)

    _remember_webhook(event_id, event_type)

    return {
        "ok": True,
        "event": event_type,
        "matched_session": bool(session),
    }


@router.get("/group-video/active")
def active_group_video_sessions(
    authorization: Optional[str] = Header(default=None),
):
    identity = _identity_from_bearer(authorization)
    ensure_schema()

    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT
                s.*,
                g.name AS group_name
            FROM verifyd_voice_group_video_sessions s
            INNER JOIN verifyd_voice_conversation_members vm
                ON vm.conversation_id = s.conversation_id
            LEFT JOIN verifyd_voice_groups g
                ON g.conversation_id = s.conversation_id
            WHERE vm.identity_id = %s
              AND s.status = 'active'
            ORDER BY s.started_at DESC
            """,
            (identity["id"],),
        )
        rows = [dict(row) for row in (cur.fetchall() or [])]

    sessions = []
    for row in rows:
        starter = _identity_by_id(
            str(row.get("started_by_identity_id") or "")
        )
        sessions.append(
            {
                "session_id": row.get("session_id") or "",
                "group_id": row.get("conversation_id") or "",
                "group_name": row.get("group_name") or "Group Video",
                "started_by": _public_contact(starter) if starter else None,
                "started_at": row.get("started_at") or "",
                "participant_peak": int(
                    row.get("peak_participant_count") or 0
                ),
            }
        )

    return {
        "count": len(sessions),
        "sessions": sessions,
    }


@router.get("/group-video/history")
def group_video_history(
    authorization: Optional[str] = Header(default=None),
    limit: int = Query(default=25, ge=1, le=100),
):
    identity = _identity_from_bearer(authorization)
    ensure_schema()

    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT
                s.*,
                g.name AS group_name,
                (
                    SELECT COUNT(*)
                    FROM verifyd_voice_group_video_participants p
                    WHERE p.session_id = s.session_id
                ) AS participant_count
            FROM verifyd_voice_group_video_sessions s
            INNER JOIN verifyd_voice_conversation_members vm
                ON vm.conversation_id = s.conversation_id
            LEFT JOIN verifyd_voice_groups g
                ON g.conversation_id = s.conversation_id
            WHERE vm.identity_id = %s
            ORDER BY s.started_at DESC
            LIMIT %s
            """,
            (identity["id"], limit),
        )
        rows = [dict(row) for row in (cur.fetchall() or [])]

    history = []
    for row in rows:
        starter = _identity_by_id(
            str(row.get("started_by_identity_id") or "")
        )
        history.append(
            {
                "session_id": row.get("session_id") or "",
                "group_id": row.get("conversation_id") or "",
                "group_name": row.get("group_name") or "Group Video",
                "status": row.get("status") or "",
                "started_by": _public_contact(starter) if starter else None,
                "started_at": row.get("started_at") or "",
                "ended_at": row.get("ended_at"),
                "ended_reason": row.get("ended_reason"),
                "duration_seconds": _duration_seconds(
                    row.get("started_at"),
                    row.get("ended_at"),
                ),
                "participant_count": int(
                    row.get("participant_count") or 0
                ),
                "participant_peak": int(
                    row.get("peak_participant_count") or 0
                ),
            }
        )

    return {
        "count": len(history),
        "history": history,
    }


@router.get("/group-video/history/{session_id}")
def group_video_history_detail(
    session_id: str,
    authorization: Optional[str] = Header(default=None),
):
    identity = _identity_from_bearer(authorization)
    ensure_schema()

    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT
                s.*,
                g.name AS group_name
            FROM verifyd_voice_group_video_sessions s
            INNER JOIN verifyd_voice_conversation_members vm
                ON vm.conversation_id = s.conversation_id
            LEFT JOIN verifyd_voice_groups g
                ON g.conversation_id = s.conversation_id
            WHERE s.session_id = %s
              AND vm.identity_id = %s
            LIMIT 1
            """,
            (session_id, identity["id"]),
        )
        session_row = cur.fetchone()

        if not session_row:
            raise HTTPException(
                status_code=404,
                detail={"error": "group_video_history_not_found"},
            )

        session = dict(session_row)

        cur.execute(
            """
            SELECT *
            FROM verifyd_voice_group_video_participants
            WHERE session_id = %s
            ORDER BY first_joined_at ASC NULLS LAST
            """,
            (session_id,),
        )
        participant_rows = [
            dict(row) for row in (cur.fetchall() or [])
        ]

    participants = []
    for row in participant_rows:
        person = _identity_by_id(
            str(row.get("identity_id") or "")
        )
        participants.append(
            {
                "identity": _public_contact(person) if person else None,
                "first_joined_at": row.get("first_joined_at"),
                "left_at": row.get("left_at"),
                "join_count": int(row.get("join_count") or 0),
                "total_seconds": int(
                    float(row.get("total_seconds") or 0)
                ),
            }
        )

    starter = _identity_by_id(
        str(session.get("started_by_identity_id") or "")
    )

    return {
        "session": {
            "session_id": session.get("session_id") or "",
            "group_id": session.get("conversation_id") or "",
            "group_name": session.get("group_name") or "Group Video",
            "status": session.get("status") or "",
            "started_by": _public_contact(starter) if starter else None,
            "started_at": session.get("started_at") or "",
            "ended_at": session.get("ended_at"),
            "ended_reason": session.get("ended_reason"),
            "duration_seconds": _duration_seconds(
                session.get("started_at"),
                session.get("ended_at"),
            ),
            "participant_peak": int(
                session.get("peak_participant_count") or 0
            ),
        },
        "participants": participants,
    }
