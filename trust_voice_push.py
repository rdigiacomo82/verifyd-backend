# ============================================================
# VeriFYD Trust Voice — Web Push Notifications
# VERIFYD_TRUST_VOICE_PUSH_V1
# ============================================================

from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from fastapi import APIRouter, Header, HTTPException, Request
from pydantic import BaseModel, Field

from database import get_db
from trust_voice_signaling import (
    _identity_by_id,
    _token_from_authorization,
    _verify_token,
)
from trust_voice_preferences import preferred_locale_for_identity
from trust_voice_push_localization import localize_notification

try:
    from pywebpush import WebPushException, webpush
except Exception:
    WebPushException = Exception
    webpush = None

log = logging.getLogger("verifyd.trust_voice.push")

router = APIRouter(
    prefix="/trust-voice",
    tags=["Trust Voice Push Notifications"],
)

FEATURE_VERSION = "0.1.0"
_schema_ready = False
_scheduler_started = False
_scheduler_lock = threading.Lock()

DEFAULT_REMINDER_MINUTES = 15
DEFAULT_DISPATCH_SECONDS = 30
RESULT_LOOKBACK_HOURS = 24
MAX_ENDPOINT_LENGTH = 4096
MAX_KEY_LENGTH = 1024


class PushKeysPayload(BaseModel):
    p256dh: str = Field(min_length=20, max_length=MAX_KEY_LENGTH)
    auth: str = Field(min_length=8, max_length=MAX_KEY_LENGTH)


class PushSubscriptionPayload(BaseModel):
    endpoint: str = Field(min_length=12, max_length=MAX_ENDPOINT_LENGTH)
    keys: PushKeysPayload


class PushUnsubscribePayload(BaseModel):
    endpoint: str = Field(min_length=12, max_length=MAX_ENDPOINT_LENGTH)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _now_iso() -> str:
    return _now().isoformat()


def _env_bool(name: str, default: bool = False) -> bool:
    raw = (os.environ.get(name, "") or "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int((os.environ.get(name, str(default)) or "").strip())
    except Exception:
        value = default
    return max(minimum, min(value, maximum))


def _enabled() -> bool:
    return _env_bool("VERIFYD_WEB_PUSH_ENABLED", False)


def _vapid_private_key() -> str:
    return (
        os.environ.get("VERIFYD_WEB_PUSH_VAPID_PRIVATE_KEY", "")
        or ""
    ).strip()


def _vapid_public_key() -> str:
    return (
        os.environ.get("VERIFYD_WEB_PUSH_VAPID_PUBLIC_KEY", "")
        or ""
    ).strip()


def _vapid_subject() -> str:
    return (
        os.environ.get("VERIFYD_WEB_PUSH_SUBJECT", "")
        or ""
    ).strip()


def _configured() -> bool:
    return bool(
        webpush is not None
        and _vapid_private_key()
        and _vapid_public_key()
        and _vapid_subject()
    )


def _dispatch_seconds() -> int:
    return _env_int(
        "VERIFYD_WEB_PUSH_DISPATCH_SECONDS",
        DEFAULT_DISPATCH_SECONDS,
        15,
        300,
    )


def _validate_endpoint(value: str) -> str:
    endpoint = str(value or "").strip()
    if (
        not endpoint
        or len(endpoint) > MAX_ENDPOINT_LENGTH
        or not endpoint.lower().startswith("https://")
    ):
        raise HTTPException(
            status_code=400,
            detail={
                "error": "invalid_push_endpoint",
                "message": "The browser push subscription endpoint is invalid.",
            },
        )
    return endpoint


def _auth_context(
    authorization: Optional[str],
) -> tuple[dict, dict]:
    token = _token_from_authorization(authorization)
    payload = _verify_token(token)

    identity_id = str(payload.get("identity_id") or "").strip()
    identity = _identity_by_id(identity_id)

    if not identity:
        raise HTTPException(
            status_code=401,
            detail={"error": "identity_not_found"},
        )

    return identity, payload


def ensure_push_schema() -> None:
    global _schema_ready
    if _schema_ready:
        return

    with get_db() as conn:
        cur = conn.cursor()

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS verifyd_voice_push_subscriptions (
                subscription_id   TEXT PRIMARY KEY,
                identity_id       TEXT NOT NULL,
                session_id        TEXT,
                endpoint          TEXT UNIQUE NOT NULL,
                p256dh            TEXT NOT NULL,
                auth_secret       TEXT NOT NULL,
                user_agent        TEXT,
                created_at        TEXT NOT NULL,
                updated_at        TEXT NOT NULL,
                last_success_at   TEXT,
                last_failure_at   TEXT,
                failure_count     INTEGER NOT NULL DEFAULT 0,
                disabled_at       TEXT,
                disabled_reason   TEXT
            )
            """
        )

        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_verifyd_voice_push_identity
            ON verifyd_voice_push_subscriptions (
                identity_id,
                disabled_at
            )
            """
        )

        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_verifyd_voice_push_session
            ON verifyd_voice_push_subscriptions (
                session_id,
                disabled_at
            )
            """
        )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS verifyd_voice_push_deliveries (
                delivery_key       TEXT PRIMARY KEY,
                identity_id        TEXT NOT NULL,
                event_type         TEXT NOT NULL,
                created_at         TEXT NOT NULL,
                sent_at            TEXT,
                status             TEXT NOT NULL DEFAULT 'claimed',
                success_count      INTEGER NOT NULL DEFAULT 0,
                failure_count      INTEGER NOT NULL DEFAULT 0,
                error_text         TEXT
            )
            """
        )

        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_verifyd_voice_push_deliveries_identity
            ON verifyd_voice_push_deliveries (
                identity_id,
                created_at DESC
            )
            """
        )

    _schema_ready = True


def _active_subscriptions(identity_id: str) -> list[dict]:
    ensure_push_schema()

    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT ps.*
            FROM verifyd_voice_push_subscriptions ps
            WHERE ps.identity_id = %s
              AND ps.disabled_at IS NULL
              AND (
                  ps.session_id IS NULL
                  OR EXISTS (
                      SELECT 1
                      FROM verifyd_voice_sessions vs
                      WHERE vs.session_id = ps.session_id
                        AND vs.identity_id = ps.identity_id
                        AND vs.revoked_at IS NULL
                  )
              )
            ORDER BY ps.updated_at DESC
            """,
            (identity_id,),
        )
        rows = [
            dict(row)
            for row in (cur.fetchall() or [])
        ]

    return rows


def _mark_subscription_success(subscription_id: str) -> None:
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            UPDATE verifyd_voice_push_subscriptions
            SET
                last_success_at = %s,
                failure_count = 0,
                last_failure_at = NULL
            WHERE subscription_id = %s
            """,
            (
                _now_iso(),
                subscription_id,
            ),
        )


def _mark_subscription_failure(
    subscription_id: str,
    *,
    status_code: Optional[int],
    reason: str,
) -> None:
    now = _now_iso()
    stale = status_code in {404, 410}

    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            UPDATE verifyd_voice_push_subscriptions
            SET
                failure_count = COALESCE(failure_count, 0) + 1,
                last_failure_at = %s,
                disabled_at = CASE
                    WHEN %s THEN COALESCE(disabled_at, %s)
                    ELSE disabled_at
                END,
                disabled_reason = CASE
                    WHEN %s THEN %s
                    ELSE disabled_reason
                END
            WHERE subscription_id = %s
            """,
            (
                now,
                stale,
                now,
                stale,
                str(reason or "")[:120],
                subscription_id,
            ),
        )


def notify_identity_push(
    identity_id: str,
    *,
    notification_type: str,
    title: str,
    body: str,
    url: str = "/voice",
    tag: str = "",
    data: Optional[dict[str, Any]] = None,
    ttl: int = 600,
) -> dict:
    """Best-effort push delivery; never raises into product paths."""
    if not _enabled() or not _configured():
        return {
            "configured": _configured(),
            "attempted": 0,
            "sent": 0,
            "failed": 0,
        }

    subscriptions = _active_subscriptions(str(identity_id or ""))
    if not subscriptions:
        return {
            "configured": True,
            "attempted": 0,
            "sent": 0,
            "failed": 0,
        }

    locale = preferred_locale_for_identity(str(identity_id or ""))
    title, body = localize_notification(
        locale,
        notification_type,
        title,
        body,
    )

    payload = {
        "version": "trust_voice_push_v1",
        "type": str(notification_type or "notification")[:80],
        "title": str(title or "Trust Voice")[:120],
        "body": str(body or "")[:240],
        "url": str(url or "/voice")[:1000],
        "tag": str(tag or notification_type or "trust-voice")[:160],
        "created_at": _now_iso(),
        "data": data if isinstance(data, dict) else {},
    }

    serialized = json.dumps(
        payload,
        separators=(",", ":"),
        ensure_ascii=False,
    )

    sent = 0
    failed = 0

    for subscription in subscriptions:
        subscription_id = str(
            subscription.get("subscription_id") or ""
        )

        info = {
            "endpoint": subscription.get("endpoint") or "",
            "keys": {
                "p256dh": subscription.get("p256dh") or "",
                "auth": subscription.get("auth_secret") or "",
            },
        }

        try:
            webpush(
                subscription_info=info,
                data=serialized,
                vapid_private_key=_vapid_private_key(),
                vapid_claims={"sub": _vapid_subject()},
                ttl=max(0, min(int(ttl or 0), 86400)),
                timeout=5,
            )
            sent += 1
            _mark_subscription_success(subscription_id)

        except WebPushException as exc:
            failed += 1
            status_code = getattr(exc, "status_code", None)
            _mark_subscription_failure(
                subscription_id,
                status_code=status_code,
                reason=(
                    "expired_or_gone"
                    if status_code in {404, 410}
                    else f"webpush_{status_code or 'error'}"
                ),
            )
            log.warning(
                "Trust Voice push delivery failed subscription_id=%s status=%s",
                subscription_id,
                status_code or "unknown",
            )

        except Exception:
            failed += 1
            _mark_subscription_failure(
                subscription_id,
                status_code=None,
                reason="unexpected_push_error",
            )
            log.warning(
                "Trust Voice push delivery failed subscription_id=%s",
                subscription_id,
                exc_info=True,
            )

    return {
        "configured": True,
        "attempted": len(subscriptions),
        "sent": sent,
        "failed": failed,
    }


def _display_name(public_identity: Optional[dict]) -> str:
    if not isinstance(public_identity, dict):
        return "Someone"
    return (
        str(public_identity.get("display_name") or "").strip()
        or str(public_identity.get("display_label") or "").strip()
        or str(public_identity.get("handle") or "").strip()
        or "Someone"
    )


def _group_name(conversation_id: str) -> str:
    if not conversation_id:
        return "your group"

    try:
        with get_db() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT name
                FROM verifyd_voice_groups
                WHERE conversation_id = %s
                LIMIT 1
                """,
                (conversation_id,),
            )
            row = cur.fetchone() or {}
        return str(row.get("name") or "your group")
    except Exception:
        return "your group"


def dispatch_push_for_event(
    identity_id: str,
    payload: dict,
) -> dict:
    """Map selected existing realtime events into privacy-safe Web Push."""
    if not isinstance(payload, dict):
        return {"ignored": True}

    event_type = str(payload.get("type") or "").strip()

    if event_type == "circle_community_event":
        # Recipient is selected by the Community Board backend from the
        # author's Trust Circle only; never broadcast public posts globally.
        post_id = str(payload.get("post_id") or "")
        return notify_identity_push(
            identity_id,
            notification_type="circle_community_event",
            title="New Trust Circle event",
            body="Someone in your Trust Circle shared a community event.",
            url="/voice",
            tag=f"trust-voice-community-{post_id}" if post_id else "trust-voice-community",
            data={"post_id": post_id},
            ttl=3600,
        )

    if event_type == "message_created":
        sender_name = _display_name(payload.get("sender"))
        conversation_id = str(
            payload.get("conversation_id") or ""
        )
        return notify_identity_push(
            identity_id,
            notification_type="message_created",
            title="New Trust Voice message",
            body=f"{sender_name} sent you a message.",
            url=(
                f"/voice?conversation={conversation_id}"
                if conversation_id
                else "/voice"
            ),
            tag=(
                f"trust-voice-message-{conversation_id}"
                if conversation_id
                else "trust-voice-message"
            ),
            data={"conversation_id": conversation_id},
            ttl=3600,
        )

    if event_type == "group_video_started":
        group_id = str(payload.get("group_id") or "")
        session = (
            payload.get("session")
            if isinstance(payload.get("session"), dict)
            else {}
        )
        session_id = str(session.get("session_id") or "")
        starter_name = _display_name(session.get("started_by"))
        group_name = _group_name(group_id)

        return notify_identity_push(
            identity_id,
            notification_type="group_video_started",
            title="Video meeting started",
            body=f"{starter_name} started a video meeting in {group_name}.",
            url=(
                f"/voice?meeting={session_id}"
                if session_id
                else "/voice"
            ),
            tag=(
                f"trust-voice-video-{session_id}"
                if session_id
                else f"trust-voice-video-{group_id}"
            ),
            data={
                "group_id": group_id,
                "session_id": session_id,
            },
            ttl=300,
        )

    if event_type in {
        "group_video_schedule_created",
        "group_video_schedule_updated",
        "group_video_schedule_cancelled",
    }:
        schedule = (
            payload.get("schedule")
            if isinstance(payload.get("schedule"), dict)
            else {}
        )
        schedule_id = str(schedule.get("schedule_id") or "")
        title = str(
            schedule.get("title") or "Trust Voice meeting"
        ).strip()
        group_name = str(
            schedule.get("group_name") or "your group"
        ).strip()

        title_map = {
            "group_video_schedule_created": "New meeting scheduled",
            "group_video_schedule_updated": "Meeting updated",
            "group_video_schedule_cancelled": "Meeting cancelled",
        }
        verb_map = {
            "group_video_schedule_created": "was scheduled",
            "group_video_schedule_updated": "was updated",
            "group_video_schedule_cancelled": "was cancelled",
        }

        return notify_identity_push(
            identity_id,
            notification_type=event_type,
            title=title_map[event_type],
            body=f"{title} {verb_map[event_type]} in {group_name}.",
            url=(
                f"/voice?scheduled={schedule_id}"
                if schedule_id
                and event_type != "group_video_schedule_cancelled"
                else "/voice"
            ),
            tag=(
                f"trust-voice-schedule-{schedule_id}"
                if schedule_id
                else "trust-voice-schedule"
            ),
            data={
                "schedule_id": schedule_id,
                "group_id": str(schedule.get("group_id") or ""),
            },
            ttl=3600,
        )

    return {"ignored": True, "type": event_type}


def _claim_delivery(
    delivery_key: str,
    identity_id: str,
    event_type: str,
) -> bool:
    ensure_push_schema()

    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO verifyd_voice_push_deliveries (
                delivery_key,
                identity_id,
                event_type,
                created_at,
                sent_at,
                status,
                success_count,
                failure_count,
                error_text
            )
            VALUES (%s,%s,%s,%s,NULL,'claimed',0,0,NULL)
            ON CONFLICT (delivery_key) DO NOTHING
            RETURNING delivery_key
            """,
            (
                delivery_key,
                identity_id,
                event_type,
                _now_iso(),
            ),
        )
        row = cur.fetchone()

    return bool(row)


def _finish_delivery(
    delivery_key: str,
    result: dict,
) -> None:
    sent = int(result.get("sent") or 0)
    failed = int(result.get("failed") or 0)

    if sent <= 0:
        with get_db() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                DELETE FROM verifyd_voice_push_deliveries
                WHERE delivery_key = %s
                  AND status = 'claimed'
                """,
                (delivery_key,),
            )
        return

    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            UPDATE verifyd_voice_push_deliveries
            SET
                sent_at = %s,
                status = 'sent',
                success_count = %s,
                failure_count = %s,
                error_text = NULL
            WHERE delivery_key = %s
            """,
            (
                _now_iso(),
                sent,
                failed,
                delivery_key,
            ),
        )


def _dispatch_due_schedule_reminders() -> int:
    if not _enabled() or not _configured():
        return 0

    ensure_push_schema()

    now = _now()
    window_end = now + timedelta(minutes=DEFAULT_REMINDER_MINUTES)
    recovery_start = now - timedelta(minutes=5)

    try:
        with get_db() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT
                    s.schedule_id,
                    s.conversation_id,
                    s.title,
                    s.scheduled_for,
                    g.name AS group_name,
                    vm.identity_id
                FROM verifyd_voice_group_video_schedules s
                INNER JOIN verifyd_voice_conversation_members vm
                    ON vm.conversation_id = s.conversation_id
                LEFT JOIN verifyd_voice_groups g
                    ON g.conversation_id = s.conversation_id
                WHERE s.status = 'scheduled'
                  AND s.scheduled_for::timestamptz >= %s::timestamptz
                  AND s.scheduled_for::timestamptz <= %s::timestamptz
                ORDER BY s.scheduled_for::timestamptz ASC
                """,
                (
                    recovery_start.isoformat(),
                    window_end.isoformat(),
                ),
            )
            rows = [
                dict(row)
                for row in (cur.fetchall() or [])
            ]
    except Exception:
        log.debug(
            "Trust Voice push reminder scan skipped",
            exc_info=True,
        )
        return 0

    delivered = 0

    for row in rows:
        identity_id = str(row.get("identity_id") or "")
        schedule_id = str(row.get("schedule_id") or "")
        scheduled_for = str(row.get("scheduled_for") or "")

        if not identity_id or not schedule_id:
            continue

        if not _active_subscriptions(identity_id):
            continue

        delivery_key = (
            f"schedule_reminder:{schedule_id}:"
            f"{identity_id}:{scheduled_for}"
        )

        if not _claim_delivery(
            delivery_key,
            identity_id,
            "scheduled_meeting_reminder",
        ):
            continue

        title = str(
            row.get("title") or "Trust Voice meeting"
        ).strip()
        group_name = str(
            row.get("group_name") or "your group"
        ).strip()

        result = notify_identity_push(
            identity_id,
            notification_type="scheduled_meeting_reminder",
            title="Meeting starts in 15 minutes",
            body=f"{title} · {group_name}",
            url=f"/voice?scheduled={schedule_id}",
            tag=f"trust-voice-reminder-{schedule_id}",
            data={
                "schedule_id": schedule_id,
                "group_id": str(row.get("conversation_id") or ""),
            },
            ttl=900,
        )

        _finish_delivery(
            delivery_key,
            result,
        )

        if int(result.get("sent") or 0) > 0:
            delivered += 1

    return delivered


def _dispatch_ready_meeting_results() -> int:
    if not _enabled() or not _configured():
        return 0

    ensure_push_schema()
    cutoff = (_now() - timedelta(hours=RESULT_LOOKBACK_HOURS)).isoformat()

    try:
        with get_db() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT
                    s.session_id,
                    s.conversation_id,
                    s.ended_at,
                    g.name AS group_name,
                    vm.identity_id,
                    (
                        SELECT COUNT(*)
                        FROM verifyd_voice_group_video_recordings r
                        WHERE r.session_id = s.session_id
                          AND r.status = 'complete'
                    ) AS complete_recording_count
                FROM verifyd_voice_group_video_sessions s
                INNER JOIN verifyd_voice_conversation_members vm
                    ON vm.conversation_id = s.conversation_id
                LEFT JOIN verifyd_voice_groups g
                    ON g.conversation_id = s.conversation_id
                WHERE s.status <> 'active'
                  AND s.ended_at IS NOT NULL
                  AND s.ended_at::timestamptz >= %s::timestamptz
                  AND NOT EXISTS (
                      SELECT 1
                      FROM verifyd_voice_group_video_recordings open_r
                      WHERE open_r.session_id = s.session_id
                        AND open_r.status IN (
                            'consent_pending',
                            'ready',
                            'starting',
                            'active',
                            'stopping'
                        )
                  )
                ORDER BY s.ended_at::timestamptz DESC
                """,
                (cutoff,),
            )
            rows = [
                dict(row)
                for row in (cur.fetchall() or [])
            ]
    except Exception:
        log.debug(
            "Trust Voice push results scan skipped",
            exc_info=True,
        )
        return 0

    delivered = 0

    for row in rows:
        identity_id = str(row.get("identity_id") or "")
        session_id = str(row.get("session_id") or "")

        if not identity_id or not session_id:
            continue

        if not _active_subscriptions(identity_id):
            continue

        delivery_key = (
            f"meeting_results:{session_id}:{identity_id}"
        )

        if not _claim_delivery(
            delivery_key,
            identity_id,
            "meeting_results_ready",
        ):
            continue

        group_name = str(
            row.get("group_name") or "Trust Voice meeting"
        ).strip()
        has_recording = int(
            row.get("complete_recording_count") or 0
        ) > 0

        result = notify_identity_push(
            identity_id,
            notification_type="meeting_results_ready",
            title="Meeting results ready",
            body=(
                f"{group_name} summary and recording are ready."
                if has_recording
                else f"{group_name} meeting summary is ready."
            ),
            url=f"/voice?meeting={session_id}&results=1",
            tag=f"trust-voice-results-{session_id}",
            data={
                "session_id": session_id,
                "group_id": str(row.get("conversation_id") or ""),
            },
            ttl=86400,
        )

        _finish_delivery(
            delivery_key,
            result,
        )

        if int(result.get("sent") or 0) > 0:
            delivered += 1

    return delivered


def run_push_dispatch_once() -> dict:
    return {
        "scheduled_reminders": _dispatch_due_schedule_reminders(),
        "meeting_results": _dispatch_ready_meeting_results(),
    }


def _scheduler_loop() -> None:
    time.sleep(10)

    while True:
        try:
            run_push_dispatch_once()
        except Exception:
            log.warning(
                "Trust Voice push scheduler iteration failed",
                exc_info=True,
            )
        time.sleep(_dispatch_seconds())


def start_push_scheduler() -> None:
    global _scheduler_started

    if not _enabled() or not _configured():
        log.info(
            "Trust Voice Web Push scheduler not started enabled=%s configured=%s",
            _enabled(),
            _configured(),
        )
        return

    ensure_push_schema()

    with _scheduler_lock:
        if _scheduler_started:
            return

        thread = threading.Thread(
            target=_scheduler_loop,
            daemon=True,
            name="trust-voice-web-push",
        )
        thread.start()
        _scheduler_started = True

    log.info(
        "Trust Voice Web Push scheduler started interval=%ss",
        _dispatch_seconds(),
    )


@router.get("/push/health")
def push_health():
    ensure_push_schema()
    return {
        "status": "ok",
        "feature": "trust_voice_push",
        "version": FEATURE_VERSION,
        "enabled": _enabled(),
        "configured": _configured(),
        "library": (
            "pywebpush"
            if webpush is not None
            else "unavailable"
        ),
        "subscription_store": "postgresql_v1",
        "session_binding": "persistent_session_v1",
        "stale_subscription_cleanup": "http_404_410_v1",
        "message_privacy": "no_message_body_v1",
        "realtime": "existing_websocket_preserved_v1",
        "scheduled_reminders": "server_durable_15_min_v1",
        "meeting_results": "server_durable_terminal_state_v1",
        "dispatch_seconds": _dispatch_seconds(),
    }


@router.get("/push/config")
def push_config(
    authorization: Optional[str] = Header(default=None),
):
    _auth_context(authorization)
    ensure_push_schema()

    if not _enabled() or not _configured():
        raise HTTPException(
            status_code=503,
            detail={
                "error": "push_not_configured",
                "message": "Trust Voice notifications are not configured yet.",
            },
        )

    return {
        "enabled": True,
        "configured": True,
        "public_key": _vapid_public_key(),
    }


@router.get("/push/status")
def push_status(
    authorization: Optional[str] = Header(default=None),
):
    identity, _payload = _auth_context(authorization)
    subscriptions = _active_subscriptions(
        str(identity["id"])
    )

    return {
        "enabled": _enabled(),
        "configured": _configured(),
        "subscription_count": len(subscriptions),
        "subscribed": bool(subscriptions),
    }


@router.post("/push/subscriptions")
def register_push_subscription(
    payload: PushSubscriptionPayload,
    request: Request,
    authorization: Optional[str] = Header(default=None),
):
    if not _enabled() or not _configured():
        raise HTTPException(
            status_code=503,
            detail={"error": "push_not_configured"},
        )

    identity, token_payload = _auth_context(
        authorization
    )
    ensure_push_schema()

    endpoint = _validate_endpoint(
        payload.endpoint
    )
    p256dh = str(payload.keys.p256dh or "").strip()
    auth_secret = str(payload.keys.auth or "").strip()

    if not p256dh or not auth_secret:
        raise HTTPException(
            status_code=400,
            detail={"error": "push_keys_required"},
        )

    session_id = str(
        token_payload.get("session_id") or ""
    ).strip() or None
    now = _now_iso()
    subscription_id = "tvpush_" + uuid.uuid4().hex
    user_agent = str(
        request.headers.get("user-agent") or ""
    )[:500]

    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO verifyd_voice_push_subscriptions (
                subscription_id,
                identity_id,
                session_id,
                endpoint,
                p256dh,
                auth_secret,
                user_agent,
                created_at,
                updated_at,
                last_success_at,
                last_failure_at,
                failure_count,
                disabled_at,
                disabled_reason
            )
            VALUES (
                %s,%s,%s,%s,%s,%s,%s,
                %s,%s,NULL,NULL,0,NULL,NULL
            )
            ON CONFLICT (endpoint)
            DO UPDATE SET
                identity_id = EXCLUDED.identity_id,
                session_id = EXCLUDED.session_id,
                p256dh = EXCLUDED.p256dh,
                auth_secret = EXCLUDED.auth_secret,
                user_agent = EXCLUDED.user_agent,
                updated_at = EXCLUDED.updated_at,
                failure_count = 0,
                disabled_at = NULL,
                disabled_reason = NULL
            RETURNING subscription_id
            """,
            (
                subscription_id,
                str(identity["id"]),
                session_id,
                endpoint,
                p256dh,
                auth_secret,
                user_agent,
                now,
                now,
            ),
        )
        row = cur.fetchone() or {}

    return {
        "ok": True,
        "subscription_id": row.get("subscription_id") or subscription_id,
        "subscribed": True,
    }


@router.delete("/push/subscriptions")
def unregister_push_subscription(
    payload: PushUnsubscribePayload,
    authorization: Optional[str] = Header(default=None),
):
    identity, _token_payload = _auth_context(
        authorization
    )
    ensure_push_schema()

    endpoint = _validate_endpoint(
        payload.endpoint
    )
    now = _now_iso()

    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            UPDATE verifyd_voice_push_subscriptions
            SET
                disabled_at = COALESCE(disabled_at, %s),
                disabled_reason = 'client_unsubscribed',
                updated_at = %s
            WHERE endpoint = %s
              AND identity_id = %s
              AND disabled_at IS NULL
            """,
            (
                now,
                now,
                endpoint,
                str(identity["id"]),
            ),
        )
        changed = bool(cur.rowcount)

    return {
        "ok": True,
        "unsubscribed": changed,
    }


@router.post("/push/test")
def test_push(
    authorization: Optional[str] = Header(default=None),
):
    identity, _token_payload = _auth_context(
        authorization
    )

    result = notify_identity_push(
        str(identity["id"]),
        notification_type="push_test",
        title="Trust Voice notifications",
        body="Push notifications are working on this device.",
        url="/voice",
        tag="trust-voice-push-test",
        ttl=120,
    )

    return {
        "ok": int(result.get("sent") or 0) > 0,
        "delivery": result,
    }
