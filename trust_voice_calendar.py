# ============================================================
# VeriFYD Trust Voice — Calendar Integration
# VERIFYD_TRUST_VOICE_CALENDAR_V1
# ============================================================

from __future__ import annotations

import os
import re
from datetime import datetime, timedelta, timezone
from typing import Optional
from urllib.parse import urlencode

from fastapi import APIRouter, Header, HTTPException
from fastapi.responses import Response

from trust_voice_group_video_schedule import _schedule_row, ensure_schedule_schema
from trust_voice_groups import _require_group_member
from trust_voice_messages import _identity_from_bearer

router = APIRouter(prefix="/trust-voice", tags=["Trust Voice Calendar"])

FEATURE_VERSION = "0.1.0"
DEFAULT_DURATION_MINUTES = 30


def _duration_minutes() -> int:
    try:
        value = int(
            (
                os.environ.get(
                    "VERIFYD_TRUST_VOICE_CALENDAR_DURATION_MINUTES",
                    str(DEFAULT_DURATION_MINUTES),
                )
                or ""
            ).strip()
        )
    except Exception:
        value = DEFAULT_DURATION_MINUTES
    return max(15, min(value, 480))


def _public_app_base() -> str:
    return (
        os.environ.get("VERIFYD_PUBLIC_APP_URL", "https://vfvid.com")
        or "https://vfvid.com"
    ).strip().rstrip("/")


def _parse_iso(value: Optional[str]) -> datetime:
    raw = str(value or "").strip()
    if not raw:
        raise HTTPException(
            status_code=500,
            detail={"error": "scheduled_meeting_time_missing"},
        )
    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail={"error": "scheduled_meeting_time_invalid"},
        ) from exc
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _utc_compact(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _safe_sequence(value: Optional[str]) -> int:
    try:
        return max(0, int(_parse_iso(value).timestamp()))
    except Exception:
        return 0


def _safe_filename(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value or "").strip()).strip("-.")
    return (text or "trust-voice-meeting")[:80] + ".ics"


def _ics_escape(value: str) -> str:
    return (
        str(value or "")
        .replace("\\", "\\\\")
        .replace("\r\n", "\\n")
        .replace("\n", "\\n")
        .replace("\r", "\\n")
        .replace(";", "\\;")
        .replace(",", "\\,")
    )


def _authorized_event(schedule_id: str, authorization: Optional[str]) -> dict:
    identity = _identity_from_bearer(authorization)
    row = _schedule_row(schedule_id)

    if not row:
        raise HTTPException(
            status_code=404,
            detail={"error": "scheduled_meeting_not_found"},
        )

    _require_group_member(
        str(row.get("conversation_id") or ""),
        str(identity.get("id") or ""),
    )

    start = _parse_iso(row.get("scheduled_for"))
    end = start + timedelta(minutes=_duration_minutes())
    group_name = str(row.get("group_name") or "Trust Voice Group").strip() or "Trust Voice Group"
    title = str(row.get("title") or "Trust Voice Video Meeting").strip() or "Trust Voice Video Meeting"
    app_url = f"{_public_app_base()}/voice?scheduled={schedule_id}"
    status = str(row.get("status") or "scheduled").strip().lower()

    return {
        "schedule_id": schedule_id,
        "uid": f"{schedule_id}@vfvid.com",
        "title": title,
        "group_name": group_name,
        "start": start,
        "end": end,
        "duration_minutes": _duration_minutes(),
        "app_url": app_url,
        "status": status,
        "calendar_status": "CANCELLED" if status == "cancelled" else "CONFIRMED",
        "description": f"Trust Voice group: {group_name}\nOpen meeting: {app_url}",
        "location": "VeriFYD Trust Voice",
        "sequence": _safe_sequence(row.get("updated_at")),
    }


def _google_url(event: dict) -> str:
    return "https://calendar.google.com/calendar/render?" + urlencode({
        "action": "TEMPLATE",
        "text": event["title"],
        "dates": f"{_utc_compact(event['start'])}/{_utc_compact(event['end'])}",
        "details": event["description"],
        "location": event["location"],
    })


def _outlook_url(event: dict) -> str:
    return "https://outlook.office.com/calendar/0/deeplink/compose?" + urlencode({
        "path": "/calendar/action/compose",
        "rru": "addevent",
        "subject": event["title"],
        "startdt": event["start"].isoformat(),
        "enddt": event["end"].isoformat(),
        "body": event["description"],
        "location": event["location"],
    })


def _ics_text(event: dict) -> str:
    lines = [
        "BEGIN:VCALENDAR",
        "VERSION:2.0",
        "PRODID:-//Data by Design LLC//VeriFYD Trust Voice//EN",
        "CALSCALE:GREGORIAN",
        "METHOD:PUBLISH",
        "BEGIN:VEVENT",
        f"UID:{_ics_escape(event['uid'])}",
        f"DTSTAMP:{_utc_compact(datetime.now(timezone.utc))}",
        f"DTSTART:{_utc_compact(event['start'])}",
        f"DTEND:{_utc_compact(event['end'])}",
        f"SEQUENCE:{int(event['sequence'])}",
        f"STATUS:{event['calendar_status']}",
        f"SUMMARY:{_ics_escape(event['title'])}",
        f"DESCRIPTION:{_ics_escape(event['description'])}",
        f"LOCATION:{_ics_escape(event['location'])}",
        f"URL:{_ics_escape(event['app_url'])}",
        "END:VEVENT",
        "END:VCALENDAR",
        "",
    ]
    return "\r\n".join(lines)


@router.get("/calendar/health")
def calendar_health():
    ensure_schedule_schema()
    return {
        "status": "ok",
        "feature": "trust_voice_calendar",
        "version": FEATURE_VERSION,
        "source": "scheduled_meetings_v1",
        "authorization": "group_member_required_v1",
        "providers": [
            "google_calendar_link_v1",
            "outlook_calendar_link_v1",
            "ics_download_v1",
        ],
        "default_duration_minutes": _duration_minutes(),
        "event_uid": "stable_schedule_id_v1",
        "calendar_content": "privacy_limited_v1",
        "external_oauth": "not_required_v1",
    }


@router.get("/group-video/schedules/{schedule_id}/calendar")
def calendar_options(
    schedule_id: str,
    authorization: Optional[str] = Header(default=None),
):
    event = _authorized_event(schedule_id, authorization)
    return {
        "event": {
            "schedule_id": event["schedule_id"],
            "uid": event["uid"],
            "title": event["title"],
            "group_name": event["group_name"],
            "start": event["start"].isoformat(),
            "end": event["end"].isoformat(),
            "duration_minutes": event["duration_minutes"],
            "status": event["status"],
            "app_url": event["app_url"],
            "location": event["location"],
        },
        "providers": {
            "google": _google_url(event),
            "outlook": _outlook_url(event),
            "ics_path": f"/trust-voice/group-video/schedules/{schedule_id}/calendar.ics",
        },
    }


@router.get("/group-video/schedules/{schedule_id}/calendar.ics")
def calendar_ics(
    schedule_id: str,
    authorization: Optional[str] = Header(default=None),
):
    event = _authorized_event(schedule_id, authorization)
    return Response(
        content=_ics_text(event),
        media_type="text/calendar; charset=utf-8",
        headers={
            "Content-Disposition": f'attachment; filename="{_safe_filename(event["title"])}"',
            "Cache-Control": "private, no-store, max-age=0",
            "Pragma": "no-cache",
            "X-Content-Type-Options": "nosniff",
        },
    )


# VERIFYD_CIRCLE_HUB_V1: register additive Circle meetings and Community Board routes.
# main.py already registers trust_voice_calendar_router; leave all existing routes intact.
from trust_voice_circle_hub import router as circle_hub_router
router.include_router(circle_hub_router)

# VERIFYD_CIRCLE_MODERATION_V1: additive authenticated moderator routes.
from trust_voice_community_moderation import router as community_moderation_router
router.include_router(community_moderation_router)
