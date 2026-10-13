# ============================================================
# VeriFYD Trust Voice — Scheduled Group Video Meetings
# VERIFYD_TRUST_VOICE_GROUP_VIDEO_SCHEDULE_V1
#
# Additive scheduling metadata for the existing Trust Voice
# Group Video system.
#
# Important:
#   - Does NOT pre-create LiveKit rooms or participant tokens.
#   - Does NOT modify working Group Video / recording behavior.
#   - PostgreSQL is authoritative.
#   - WebSocket notifications are best-effort only.
#   - 15-minute in-app reminders are derived from scheduled_for
#     by the client and recovered from the upcoming endpoint.
# ============================================================

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Header, HTTPException, Query
from pydantic import BaseModel, Field

from database import get_db
from trust_voice_groups import (
    _group_member_ids,
    _require_group_member,
    ensure_group_schema,
)
from trust_voice_messages import _identity_from_bearer, _public_contact
from trust_voice_signaling import _identity_by_id, _send_to_identity

log = logging.getLogger("verifyd.trust_voice.group_video_schedule")

router = APIRouter(
    prefix="/trust-voice",
    tags=["Trust Voice Group Video Schedule"],
)

FEATURE_VERSION = "0.1.0"
_schema_ready = False

MAX_TITLE_LENGTH = 120
DEFAULT_REMINDER_MINUTES = 15
MAX_FUTURE_DAYS = 365
EXPIRE_AFTER_MINUTES = 120


class ScheduleCreatePayload(BaseModel):
    scheduled_for: str = Field(min_length=8, max_length=80)
    title: Optional[str] = Field(default=None, max_length=MAX_TITLE_LENGTH)


class ScheduleUpdatePayload(BaseModel):
    scheduled_for: Optional[str] = Field(default=None, min_length=8, max_length=80)
    title: Optional[str] = Field(default=None, max_length=MAX_TITLE_LENGTH)


class ScheduleStartedPayload(BaseModel):
    session_id: str = Field(min_length=8, max_length=160)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _now_iso() -> str:
    return _now().isoformat()


def _parse_datetime(value: str) -> datetime:
    raw = str(value or "").strip()
    if not raw:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "scheduled_for_required",
                "message": "Choose a meeting date and time.",
            },
        )
    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "scheduled_for_invalid",
                "message": "Enter a valid meeting date and time.",
            },
        ) from exc

    if dt.tzinfo is None:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "scheduled_for_timezone_required",
                "message": "Meeting time must include a timezone.",
            },
        )

    dt = dt.astimezone(timezone.utc)
    now = _now()

    # Allow a small amount of client/server clock skew.
    if dt < now - timedelta(minutes=5):
        raise HTTPException(
            status_code=400,
            detail={
                "error": "scheduled_for_in_past",
                "message": "Choose a future meeting time.",
            },
        )

    if dt > now + timedelta(days=MAX_FUTURE_DAYS):
        raise HTTPException(
            status_code=400,
            detail={
                "error": "scheduled_for_too_far",
                "message": "Meetings can be scheduled up to one year in advance.",
            },
        )

    return dt


def _clean_title(value: Optional[str], group_name: str) -> str:
    title = " ".join(str(value or "").split()).strip()
    if not title:
        base = " ".join(str(group_name or "Group").split()).strip() or "Group"
        title = f"{base} Video Meeting"
    if len(title) > MAX_TITLE_LENGTH:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "meeting_title_too_long",
                "message": f"Meeting title must be {MAX_TITLE_LENGTH} characters or fewer.",
            },
        )
    return title


def ensure_schedule_schema() -> None:
    global _schema_ready
    if _schema_ready:
        return

    ensure_group_schema()

    with get_db() as conn:
        cur = conn.cursor()

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS
            verifyd_voice_group_video_schedules (
                schedule_id                TEXT PRIMARY KEY,
                conversation_id            TEXT NOT NULL,
                title                      TEXT NOT NULL,
                scheduled_for              TEXT NOT NULL,
                scheduled_by_identity_id   TEXT NOT NULL,
                status                     TEXT NOT NULL DEFAULT 'scheduled',
                created_at                 TEXT NOT NULL,
                updated_at                 TEXT NOT NULL,
                cancelled_at               TEXT,
                cancelled_by_identity_id   TEXT,
                started_at                 TEXT,
                started_by_identity_id     TEXT,
                session_id                 TEXT
            )
            """
        )

        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS
            idx_verifyd_voice_group_video_schedules_group
            ON verifyd_voice_group_video_schedules (
                conversation_id,
                status,
                scheduled_for
            )
            """
        )

        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS
            idx_verifyd_voice_group_video_schedules_upcoming
            ON verifyd_voice_group_video_schedules (
                status,
                scheduled_for
            )
            """
        )

        cur.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS
            uq_verifyd_voice_group_video_schedule_session
            ON verifyd_voice_group_video_schedules (session_id)
            WHERE session_id IS NOT NULL
            """
        )

    _schema_ready = True


def _expire_old_schedules() -> None:
    ensure_schedule_schema()
    cutoff = (_now() - timedelta(minutes=EXPIRE_AFTER_MINUTES)).isoformat()
    now = _now_iso()

    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            UPDATE verifyd_voice_group_video_schedules
            SET
                status = 'expired',
                updated_at = %s
            WHERE status = 'scheduled'
              AND scheduled_for::timestamptz < %s::timestamptz
            """,
            (now, cutoff),
        )


def _schedule_row(schedule_id: str) -> Optional[dict]:
    ensure_schedule_schema()

    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT
                s.*,
                g.name AS group_name,
                g.owner_identity_id
            FROM verifyd_voice_group_video_schedules s
            LEFT JOIN verifyd_voice_groups g
                ON g.conversation_id = s.conversation_id
            WHERE s.schedule_id = %s
            LIMIT 1
            """,
            (schedule_id,),
        )
        row = cur.fetchone()

    return dict(row) if row else None


def _public_schedule(
    row: dict,
    viewer_identity_id: Optional[str] = None,
) -> dict:
    scheduler_id = str(row.get("scheduled_by_identity_id") or "")
    scheduler = _identity_by_id(scheduler_id) if scheduler_id else None

    scheduled_for_raw = str(row.get("scheduled_for") or "")
    try:
        scheduled_for_dt = datetime.fromisoformat(
            scheduled_for_raw.replace("Z", "+00:00")
        )
        if scheduled_for_dt.tzinfo is None:
            scheduled_for_dt = scheduled_for_dt.replace(tzinfo=timezone.utc)
        scheduled_for_dt = scheduled_for_dt.astimezone(timezone.utc)
    except Exception:
        scheduled_for_dt = None

    now = _now()
    starts_in_seconds = (
        int((scheduled_for_dt - now).total_seconds())
        if scheduled_for_dt
        else None
    )

    reminder_due = bool(
        row.get("status") == "scheduled"
        and starts_in_seconds is not None
        and 0 <= starts_in_seconds <= DEFAULT_REMINDER_MINUTES * 60
    )

    viewer_id = str(viewer_identity_id or "")
    owner_id = str(row.get("owner_identity_id") or "")

    return {
        "schedule_id": row.get("schedule_id") or "",
        "group_id": row.get("conversation_id") or "",
        "group_name": row.get("group_name") or "Group",
        "title": row.get("title") or "Group Video Meeting",
        "scheduled_for": scheduled_for_raw,
        "scheduled_by": _public_contact(scheduler) if scheduler else None,
        "status": row.get("status") or "scheduled",
        "created_at": row.get("created_at") or "",
        "updated_at": row.get("updated_at") or "",
        "cancelled_at": row.get("cancelled_at"),
        "started_at": row.get("started_at"),
        "session_id": row.get("session_id"),
        "starts_in_seconds": starts_in_seconds,
        "reminder_due": reminder_due,
        "reminder_minutes": DEFAULT_REMINDER_MINUTES,
        "can_edit": bool(
            viewer_id
            and viewer_id in {scheduler_id, owner_id}
        ),
    }


def _require_schedule_member(
    schedule_id: str,
    identity_id: str,
) -> dict:
    row = _schedule_row(schedule_id)
    if not row:
        raise HTTPException(
            status_code=404,
            detail={"error": "scheduled_meeting_not_found"},
        )

    _require_group_member(
        str(row.get("conversation_id") or ""),
        identity_id,
    )
    return row


def _require_schedule_editor(
    schedule_id: str,
    identity_id: str,
) -> dict:
    row = _require_schedule_member(
        schedule_id,
        identity_id,
    )

    scheduler_id = str(
        row.get("scheduled_by_identity_id") or ""
    )
    owner_id = str(
        row.get("owner_identity_id") or ""
    )

    if str(identity_id) not in {scheduler_id, owner_id}:
        raise HTTPException(
            status_code=403,
            detail={
                "error": "scheduled_meeting_edit_not_allowed",
                "message": "Only the person who scheduled the meeting or the group owner can make this change.",
            },
        )

    return row


async def _notify_group(
    conversation_id: str,
    payload: dict,
    exclude_identity_id: Optional[str] = None,
) -> None:
    for recipient_id in _group_member_ids(conversation_id):
        if (
            exclude_identity_id
            and recipient_id == exclude_identity_id
        ):
            continue
        await _send_to_identity(
            recipient_id,
            payload,
        )


@router.get("/group-video/schedule/health")
def schedule_health():
    ensure_schedule_schema()
    return {
        "status": "ok",
        "feature": "trust_voice_group_video_schedule",
        "version": FEATURE_VERSION,
        "store": "postgresql_v1",
        "realtime": "best_effort_websocket_v1",
        "livekit_room_creation": "deferred_until_actual_start_v1",
        "meeting_link": "opaque_schedule_id_v1",
        "reminder": "client_15_minute_plus_visibility_recovery_v1",
        "reminder_minutes": DEFAULT_REMINDER_MINUTES,
        "maximum_future_days": MAX_FUTURE_DAYS,
        "expire_after_minutes": EXPIRE_AFTER_MINUTES,
    }


@router.post("/groups/{conversation_id}/video/schedules")
def create_schedule(
    conversation_id: str,
    payload: ScheduleCreatePayload,
    background_tasks: BackgroundTasks,
    authorization: Optional[str] = Header(default=None),
):
    identity = _identity_from_bearer(authorization)
    identity_id = str(identity["id"])

    group = _require_group_member(
        conversation_id,
        identity_id,
    )

    scheduled_for = _parse_datetime(
        payload.scheduled_for
    )
    title = _clean_title(
        payload.title,
        str(group.get("name") or "Group"),
    )

    schedule_id = "tvgvsm_" + uuid.uuid4().hex
    now = _now_iso()

    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO verifyd_voice_group_video_schedules (
                schedule_id,
                conversation_id,
                title,
                scheduled_for,
                scheduled_by_identity_id,
                status,
                created_at,
                updated_at,
                cancelled_at,
                cancelled_by_identity_id,
                started_at,
                started_by_identity_id,
                session_id
            )
            VALUES (
                %s,%s,%s,%s,%s,
                'scheduled',
                %s,%s,
                NULL,NULL,NULL,NULL,NULL
            )
            """,
            (
                schedule_id,
                conversation_id,
                title,
                scheduled_for.isoformat(),
                identity_id,
                now,
                now,
            ),
        )

    row = _schedule_row(schedule_id)
    if not row:
        raise HTTPException(
            status_code=500,
            detail={"error": "scheduled_meeting_create_failed"},
        )

    public = _public_schedule(
        row,
        identity_id,
    )

    background_tasks.add_task(
        _notify_group,
        conversation_id,
        {
            "type": "group_video_schedule_created",
            "group_id": conversation_id,
            "schedule": public,
        },
        identity_id,
    )

    return {
        "ok": True,
        "schedule": public,
    }


@router.get("/group-video/schedules/upcoming")
def list_upcoming_schedules(
    authorization: Optional[str] = Header(default=None),
    limit: int = Query(default=25, ge=1, le=100),
):
    identity = _identity_from_bearer(authorization)
    identity_id = str(identity["id"])

    _expire_old_schedules()

    lower_bound = (
        _now() - timedelta(minutes=EXPIRE_AFTER_MINUTES)
    ).isoformat()

    ensure_schedule_schema()
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT
                s.*,
                g.name AS group_name,
                g.owner_identity_id
            FROM verifyd_voice_group_video_schedules s
            INNER JOIN verifyd_voice_conversation_members vm
                ON vm.conversation_id = s.conversation_id
               AND vm.identity_id = %s
            LEFT JOIN verifyd_voice_groups g
                ON g.conversation_id = s.conversation_id
            WHERE s.status = 'scheduled'
              AND s.scheduled_for::timestamptz >= %s::timestamptz
            ORDER BY s.scheduled_for::timestamptz ASC
            LIMIT %s
            """,
            (
                identity_id,
                lower_bound,
                limit,
            ),
        )
        rows = [
            dict(row)
            for row in (cur.fetchall() or [])
        ]

    schedules = [
        _public_schedule(row, identity_id)
        for row in rows
    ]

    return {
        "count": len(schedules),
        "schedules": schedules,
        "reminder_minutes": DEFAULT_REMINDER_MINUTES,
    }


@router.get("/group-video/schedules/{schedule_id}")
def get_schedule(
    schedule_id: str,
    authorization: Optional[str] = Header(default=None),
):
    identity = _identity_from_bearer(authorization)
    identity_id = str(identity["id"])

    _expire_old_schedules()

    row = _require_schedule_member(
        schedule_id,
        identity_id,
    )

    return {
        "schedule": _public_schedule(
            row,
            identity_id,
        ),
    }


@router.patch("/group-video/schedules/{schedule_id}")
def update_schedule(
    schedule_id: str,
    payload: ScheduleUpdatePayload,
    background_tasks: BackgroundTasks,
    authorization: Optional[str] = Header(default=None),
):
    identity = _identity_from_bearer(authorization)
    identity_id = str(identity["id"])

    row = _require_schedule_editor(
        schedule_id,
        identity_id,
    )

    if row.get("status") != "scheduled":
        raise HTTPException(
            status_code=409,
            detail={
                "error": "scheduled_meeting_not_editable",
                "message": "This meeting can no longer be edited.",
            },
        )

    scheduled_for_value = payload.scheduled_for
    title_value = payload.title

    if scheduled_for_value is None and title_value is None:
        raise HTTPException(
            status_code=400,
            detail={"error": "scheduled_meeting_no_changes"},
        )

    conversation_id = str(
        row.get("conversation_id") or ""
    )
    group = _require_group_member(
        conversation_id,
        identity_id,
    )

    scheduled_for = (
        _parse_datetime(scheduled_for_value)
        if scheduled_for_value is not None
        else datetime.fromisoformat(
            str(row.get("scheduled_for") or "").replace("Z", "+00:00")
        ).astimezone(timezone.utc)
    )

    title = (
        _clean_title(
            title_value,
            str(group.get("name") or "Group"),
        )
        if title_value is not None
        else str(row.get("title") or "Group Video Meeting")
    )

    updated_at = _now_iso()

    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            UPDATE verifyd_voice_group_video_schedules
            SET
                title = %s,
                scheduled_for = %s,
                updated_at = %s
            WHERE schedule_id = %s
              AND status = 'scheduled'
            """,
            (
                title,
                scheduled_for.isoformat(),
                updated_at,
                schedule_id,
            ),
        )

    refreshed = _schedule_row(schedule_id)
    if not refreshed:
        raise HTTPException(
            status_code=404,
            detail={"error": "scheduled_meeting_not_found"},
        )

    public = _public_schedule(
        refreshed,
        identity_id,
    )

    background_tasks.add_task(
        _notify_group,
        conversation_id,
        {
            "type": "group_video_schedule_updated",
            "group_id": conversation_id,
            "schedule": public,
        },
        identity_id,
    )

    return {
        "ok": True,
        "schedule": public,
    }


@router.post("/group-video/schedules/{schedule_id}/cancel")
def cancel_schedule(
    schedule_id: str,
    background_tasks: BackgroundTasks,
    authorization: Optional[str] = Header(default=None),
):
    identity = _identity_from_bearer(authorization)
    identity_id = str(identity["id"])

    row = _require_schedule_editor(
        schedule_id,
        identity_id,
    )

    if row.get("status") == "cancelled":
        return {
            "ok": True,
            "cancelled": False,
            "reason": "already_cancelled",
            "schedule": _public_schedule(
                row,
                identity_id,
            ),
        }

    if row.get("status") != "scheduled":
        raise HTTPException(
            status_code=409,
            detail={
                "error": "scheduled_meeting_not_cancelable",
                "message": "This meeting can no longer be cancelled.",
            },
        )

    now = _now_iso()
    conversation_id = str(
        row.get("conversation_id") or ""
    )

    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            UPDATE verifyd_voice_group_video_schedules
            SET
                status = 'cancelled',
                cancelled_at = %s,
                cancelled_by_identity_id = %s,
                updated_at = %s
            WHERE schedule_id = %s
              AND status = 'scheduled'
            """,
            (
                now,
                identity_id,
                now,
                schedule_id,
            ),
        )

    refreshed = _schedule_row(schedule_id)
    public = _public_schedule(
        refreshed or row,
        identity_id,
    )

    background_tasks.add_task(
        _notify_group,
        conversation_id,
        {
            "type": "group_video_schedule_cancelled",
            "group_id": conversation_id,
            "schedule": public,
        },
        identity_id,
    )

    return {
        "ok": True,
        "cancelled": True,
        "schedule": public,
    }


@router.post("/group-video/schedules/{schedule_id}/started")
def mark_schedule_started(
    schedule_id: str,
    payload: ScheduleStartedPayload,
    background_tasks: BackgroundTasks,
    authorization: Optional[str] = Header(default=None),
):
    identity = _identity_from_bearer(authorization)
    identity_id = str(identity["id"])

    row = _require_schedule_member(
        schedule_id,
        identity_id,
    )

    if row.get("status") == "started":
        return {
            "ok": True,
            "started": False,
            "reason": "already_started",
            "schedule": _public_schedule(
                row,
                identity_id,
            ),
        }

    if row.get("status") != "scheduled":
        raise HTTPException(
            status_code=409,
            detail={
                "error": "scheduled_meeting_not_startable",
                "message": "This scheduled meeting is no longer available to start.",
            },
        )

    conversation_id = str(
        row.get("conversation_id") or ""
    )

    ensure_schedule_schema()
    with get_db() as conn:
        cur = conn.cursor()

        cur.execute(
            """
            SELECT
                session_id,
                conversation_id,
                status
            FROM verifyd_voice_group_video_sessions
            WHERE session_id = %s
            LIMIT 1
            """,
            (payload.session_id,),
        )
        session = cur.fetchone()

        if not session:
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "group_video_session_not_found",
                    "message": "The group video session could not be linked to this scheduled meeting.",
                },
            )

        if str(session.get("conversation_id") or "") != conversation_id:
            raise HTTPException(
                status_code=409,
                detail={
                    "error": "scheduled_meeting_session_group_mismatch",
                },
            )

        now = _now_iso()

        cur.execute(
            """
            UPDATE verifyd_voice_group_video_schedules
            SET
                status = 'started',
                started_at = %s,
                started_by_identity_id = %s,
                session_id = %s,
                updated_at = %s
            WHERE schedule_id = %s
              AND status = 'scheduled'
            """,
            (
                now,
                identity_id,
                payload.session_id,
                now,
                schedule_id,
            ),
        )

    refreshed = _schedule_row(schedule_id)
    if not refreshed:
        raise HTTPException(
            status_code=404,
            detail={"error": "scheduled_meeting_not_found"},
        )

    public = _public_schedule(
        refreshed,
        identity_id,
    )

    background_tasks.add_task(
        _notify_group,
        conversation_id,
        {
            "type": "group_video_schedule_started",
            "group_id": conversation_id,
            "schedule": public,
        },
        identity_id,
    )

    return {
        "ok": True,
        "started": True,
        "schedule": public,
    }
