# ============================================================
# VeriFYD Trust Voice — Signaling + WebRTC Negotiation
# VERIFYD_TRUST_VOICE_SIGNALING_V1
# VERIFYD_TRUST_VOICE_WEBRTC_V1
# VERIFYD_TRUST_VOICE_MESSAGE_NOTIFY_V1
#
# Adds authenticated beta signaling only:
#   - email OTP session login for existing VeriFYD Identities
#   - persistent, server-revocable HMAC-signed session token
#   - legacy short-lived token validation during migration
#   - online/offline presence
#   - incoming call invite
#   - answer / decline / end signaling
#   - authenticated WebRTC offer / answer / ICE relay
#
# IMPORTANT:
#   - Disabled unless VERIFYD_TRUST_VOICE_SIGNALING_ENABLED=1
#   - Requires VERIFYD_TRUST_VOICE_SESSION_SECRET
#   - Creates/uses verifyd_voice_sessions for persistent session revocation
#   - Does NOT transport microphone/audio through the backend
#   - WebRTC media remains browser-to-browser where network conditions permit
#   - Presence/call state is in-memory beta state only
# ============================================================

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import logging
import os
import psycopg2
import psycopg2.pool
import secrets
import time
import uuid
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional, Set

from fastapi import (
    APIRouter,
    Header,
    HTTPException,
    Query,
    WebSocket,
    WebSocketDisconnect,
)
from pydantic import BaseModel, Field
from fastapi.concurrency import run_in_threadpool

from database import create_otp, get_db, is_valid_email, verify_otp
from emailer import send_otp_email
from trust_voice_permissions import call_permission_decision

log = logging.getLogger("verifyd.trust_voice.signaling")

router = APIRouter(prefix="/trust-voice", tags=["Trust Voice Signaling"])

FEATURE_VERSION = "0.7.1"

# Trust Voice sessions are persistent and server-revocable.
# New sessions do not expire based on elapsed time.
INVITE_TTL_SECONDS = 30
CALL_COOLDOWN_SECONDS = 3
ALLOWED_CALL_TYPES = {"audio", "video"}
MAX_WEBRTC_SDP_BYTES = 128 * 1024
MAX_WEBRTC_ICE_BYTES = 32 * 1024

_connections: Dict[str, Set[WebSocket]] = defaultdict(set)
_connection_meta: Dict[int, Dict[str, Any]] = {}
_active_calls: Dict[str, Dict[str, Any]] = {}
_last_invite_at: Dict[str, float] = {}
_state_lock = asyncio.Lock()
_persistent_session_schema_ready = False
_call_history_schema_ready = False
_call_expiry_tasks: Dict[str, asyncio.Task] = {}


class SessionCodeRequest(BaseModel):
    email: str = Field(min_length=3, max_length=254)


class SessionVerifyRequest(BaseModel):
    email: str = Field(min_length=3, max_length=254)
    code: str = Field(min_length=6, max_length=12)


def _enabled() -> bool:
    value = (
        os.environ.get(
            "VERIFYD_TRUST_VOICE_SIGNALING_ENABLED",
            "",
        )
        or ""
    ).strip().lower()

    return value in {
        "1",
        "true",
        "yes",
        "on",
    }


def _webrtc_enabled() -> bool:
    value = (
        os.environ.get(
            "VERIFYD_TRUST_VOICE_WEBRTC_ENABLED",
            "",
        )
        or ""
    ).strip().lower()

    return value in {
        "1",
        "true",
        "yes",
        "on",
    }


def _json_size_bytes(value: Any) -> int:
    try:
        return len(
            json.dumps(
                value,
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode("utf-8")
        )
    except Exception:
        return MAX_WEBRTC_SDP_BYTES + 1


def _require_enabled() -> None:
    if not _enabled():
        raise HTTPException(
            status_code=404,
            detail="not_found",
        )


def _secret() -> bytes:
    value = os.environ.get(
        "VERIFYD_TRUST_VOICE_SESSION_SECRET",
        "",
    )

    if len(value) < 32:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "signaling_not_configured",
                "message": (
                    "Trust Voice signaling session secret "
                    "is not configured."
                ),
            },
        )

    return value.encode("utf-8")


def _normalize_email(value: str) -> str:
    return (value or "").strip().lower()


def _normalize_handle(value: str) -> str:
    raw = (value or "").strip()

    if raw.startswith("@"):
        raw = raw[1:]

    return raw.lower()


def _b64url_encode(data: bytes) -> str:
    return (
        base64.urlsafe_b64encode(data)
        .decode("ascii")
        .rstrip("=")
    )


def _b64url_decode(value: str) -> bytes:
    padded = value + "=" * (-len(value) % 4)

    return base64.urlsafe_b64decode(
        padded.encode("ascii")
    )


def _sign(
    payload: Dict[str, Any],
) -> str:
    body = json.dumps(
        payload,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")

    encoded = _b64url_encode(body)

    signature = hmac.new(
        _secret(),
        encoded.encode("ascii"),
        hashlib.sha256,
    ).digest()

    return (
        f"{encoded}."
        f"{_b64url_encode(signature)}"
    )


def _now_iso() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


def ensure_call_history_schema() -> None:
    """
    Create the durable Trust Voice call-history table.

    Safe/idempotent. The table stores call outcomes separately from the
    in-memory WebSocket call state so missed calls survive refreshes,
    reconnects, and server restarts.
    """
    global _call_history_schema_ready

    if _call_history_schema_ready:
        return

    now = _now_iso()
    stale_cutoff = (
        datetime.now(timezone.utc)
        - timedelta(seconds=INVITE_TTL_SECONDS)
    ).isoformat()

    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS verifyd_voice_calls (
                call_id                TEXT PRIMARY KEY,
                caller_identity_id     TEXT NOT NULL,
                callee_identity_id     TEXT NOT NULL,
                call_type              TEXT NOT NULL DEFAULT 'audio',
                status                 TEXT NOT NULL,
                started_at             TEXT NOT NULL,
                answered_at            TEXT,
                ended_at               TEXT,
                missed_at              TEXT,
                missed_seen_at         TEXT,
                ended_reason           TEXT
            )
            """
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_verifyd_voice_calls_callee_missed
            ON verifyd_voice_calls (
                callee_identity_id,
                status,
                missed_at DESC
            )
            """
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_verifyd_voice_calls_caller_started
            ON verifyd_voice_calls (
                caller_identity_id,
                started_at DESC
            )
            """
        )

        # Recovery safety: if the service restarted while a call was ringing,
        # convert any stale "ringing" row into a durable missed call.
        cur.execute(
            """
            UPDATE verifyd_voice_calls
            SET
                status = 'missed',
                ended_at = COALESCE(ended_at, %s),
                missed_at = COALESCE(missed_at, %s),
                ended_reason = COALESCE(
                    ended_reason,
                    'server_recovery_timeout'
                )
            WHERE status = 'ringing'
              AND started_at::timestamptz <= %s::timestamptz
            """,
            (
                now,
                now,
                stale_cutoff,
            ),
        )

    _call_history_schema_ready = True


def _create_call_history(
    call_id: str,
    caller_identity_id: str,
    callee_identity_id: str,
    call_type: str = "audio",
) -> None:
    ensure_call_history_schema()

    now = _now_iso()

    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO verifyd_voice_calls (
                call_id,
                caller_identity_id,
                callee_identity_id,
                call_type,
                status,
                started_at,
                answered_at,
                ended_at,
                missed_at,
                missed_seen_at,
                ended_reason
            )
            VALUES (
                %s, %s, %s, %s, 'ringing', %s,
                NULL, NULL, NULL, NULL, NULL
            )
            ON CONFLICT (call_id) DO NOTHING
            """,
            (
                call_id,
                caller_identity_id,
                callee_identity_id,
                call_type,
                now,
            ),
        )


def _record_call_answered(call_id: str) -> None:
    ensure_call_history_schema()

    now = _now_iso()

    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            UPDATE verifyd_voice_calls
            SET
                status = 'answered',
                answered_at = COALESCE(answered_at, %s),
                ended_reason = NULL
            WHERE call_id = %s
              AND status = 'ringing'
            """,
            (
                now,
                call_id,
            ),
        )


def _record_call_terminal(
    call_id: str,
    status: str,
    reason: str,
) -> str:
    """
    Persist a terminal call state and return its terminal timestamp.

    Internal callers use only trusted status values:
    missed, declined, ended, canceled, failed.
    """
    ensure_call_history_schema()

    now = _now_iso()
    missed_at = now if status == "missed" else None

    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            UPDATE verifyd_voice_calls
            SET
                status = %s,
                ended_at = COALESCE(ended_at, %s),
                missed_at = CASE
                    WHEN %s IS NOT NULL
                    THEN COALESCE(missed_at, %s)
                    ELSE missed_at
                END,
                ended_reason = %s
            WHERE call_id = %s
              AND status NOT IN (
                  'missed',
                  'declined',
                  'ended',
                  'canceled',
                  'failed'
              )
            """,
            (
                status,
                now,
                missed_at,
                missed_at,
                reason,
                call_id,
            ),
        )

    return now


def _cancel_call_expiry(call_id: str) -> None:
    task = _call_expiry_tasks.pop(call_id, None)

    if task and not task.done():
        task.cancel()


async def _expire_unanswered_call(call_id: str) -> None:
    try:
        # Push delivery and database work must not extend the ringing deadline.
        call = _active_calls.get(call_id, {})
        remaining = max(0, float(call.get("created_at") or 0) + INVITE_TTL_SECONDS - time.time())
        await asyncio.sleep(remaining)
    except asyncio.CancelledError:
        return

    async with _state_lock:
        call = dict(_active_calls.get(call_id, {}))

        if not call or call.get("state") != "ringing":
            return

        _active_calls.pop(call_id, None)

    caller_id = str(call.get("caller_id") or "")
    callee_id = str(call.get("callee_id") or "")
    missed_at = _record_call_terminal(
        call_id,
        "missed",
        "no_answer",
    )

    caller_identity = (
        _identity_by_id(caller_id)
        if caller_id
        else None
    )

    await _send_to_identity(
        caller_id,
        {
            "type": "call_no_answer",
            "call_id": call_id,
            "reason": "no_answer",
        },
    )

    await _send_to_identity(
        callee_id,
        {
            "type": "call_missed",
            "call_id": call_id,
            "caller": (
                _public_identity(caller_identity)
                if caller_identity
                else None
            ),
            "missed_at": missed_at,
        },
    )

    try:
        from trust_voice_push import notify_identity_push

        caller_public = (
            _public_identity(caller_identity)
            if caller_identity
            else None
        )
        caller_name = (
            str((caller_public or {}).get("display_name") or "").strip()
            or str((caller_public or {}).get("display_label") or "").strip()
            or str((caller_public or {}).get("handle") or "").strip()
            or "Someone"
        )
        call_type = str(call.get("call_type") or "audio").strip().lower()

        await asyncio.to_thread(
            notify_identity_push,
            callee_id,
            notification_type="call_missed",
            title="Missed Trust Voice call",
            body=f"Missed {call_type} call from {caller_name}.",
            url="/voice?missed=1",
            tag=f"trust-voice-missed-{call_id}",
            data={
                "call_id": call_id,
                "call_type": call_type,
            },
            ttl=3600,
        )
    except Exception:
        log.warning(
            "Trust Voice missed-call Web Push dispatch failed",
            exc_info=True,
        )


def _schedule_call_expiry(call_id: str) -> None:
    _cancel_call_expiry(call_id)

    task = asyncio.create_task(
        _expire_unanswered_call(call_id)
    )
    _call_expiry_tasks[call_id] = task

    def _cleanup(done_task: asyncio.Task) -> None:
        current = _call_expiry_tasks.get(call_id)
        if current is done_task:
            _call_expiry_tasks.pop(call_id, None)

    task.add_done_callback(_cleanup)


def ensure_persistent_session_schema() -> None:
    global _persistent_session_schema_ready

    if _persistent_session_schema_ready:
        return

    with get_db() as conn:
        cur = conn.cursor()

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS verifyd_voice_sessions (
                session_id      TEXT PRIMARY KEY,
                identity_id     TEXT NOT NULL,
                created_at      TEXT NOT NULL,
                last_seen_at    TEXT NOT NULL,
                revoked_at      TEXT,
                revoked_reason  TEXT
            )
            """
        )

        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS
            idx_verifyd_voice_sessions_identity
            ON verifyd_voice_sessions (
                identity_id,
                revoked_at
            )
            """
        )

    _persistent_session_schema_ready = True


def _create_persistent_session(
    identity_id: str,
) -> str:
    ensure_persistent_session_schema()

    session_id = (
        "tvsess_"
        + secrets.token_hex(24)
    )

    now = _now_iso()

    with get_db() as conn:
        cur = conn.cursor()

        cur.execute(
            """
            INSERT INTO verifyd_voice_sessions (
                session_id,
                identity_id,
                created_at,
                last_seen_at,
                revoked_at,
                revoked_reason
            )
            VALUES (
                %s,
                %s,
                %s,
                %s,
                NULL,
                NULL
            )
            """,
            (
                session_id,
                identity_id,
                now,
                now,
            ),
        )

    return session_id


def _persistent_session_row(
    session_id: str,
) -> Optional[dict]:
    ensure_persistent_session_schema()

    with get_db() as conn:
        cur = conn.cursor()

        cur.execute(
            """
            SELECT *
            FROM verifyd_voice_sessions
            WHERE session_id = %s
            LIMIT 1
            """,
            (
                session_id,
            ),
        )

        row = cur.fetchone()

    return (
        dict(row)
        if row
        else None
    )


def _token_from_authorization(
    authorization: Optional[str],
) -> str:
    value = (
        authorization
        or ""
    ).strip()

    if not value.lower().startswith(
        "bearer "
    ):
        raise HTTPException(
            status_code=401,
            detail={
                "error": "invalid_session",
            },
        )

    token = value[7:].strip()

    if not token:
        raise HTTPException(
            status_code=401,
            detail={
                "error": "invalid_session",
            },
        )

    return token


def _verify_token(
    token: str,
) -> Dict[str, Any]:
    try:
        encoded, sig = token.split(
            ".",
            1,
        )

        expected = hmac.new(
            _secret(),
            encoded.encode("ascii"),
            hashlib.sha256,
        ).digest()

        supplied = _b64url_decode(
            sig
        )

        if not hmac.compare_digest(
            expected,
            supplied,
        ):
            raise ValueError(
                "bad signature"
            )

        payload = json.loads(
            _b64url_decode(
                encoded
            ).decode("utf-8")
        )

        if (
            payload.get("purpose")
            != "trust_voice_session"
        ):
            raise ValueError(
                "wrong purpose"
            )

        identity_id = str(
            payload.get(
                "identity_id"
            )
            or ""
        ).strip()

        if not identity_id:
            raise ValueError(
                "missing identity"
            )

        session_id = str(
            payload.get(
                "session_id"
            )
            or ""
        ).strip()

        # --------------------------------------------------
        # NEW persistent / revocable session format
        # --------------------------------------------------
        if session_id:
            try:
                row = _persistent_session_row(
                    session_id
                )
            except (psycopg2.Error, psycopg2.pool.PoolError):
                # A brief database outage or exhausted pool must NEVER be
                # interpreted as a revoked/invalid user session. The caller
                # should retain the existing token and retry after recovery.
                raise HTTPException(
                    status_code=503,
                    detail={
                        "error": "session_check_unavailable",
                        "message": "VeriFYD is temporarily busy. Please retry without signing out.",
                    },
                ) from None

            if not row:
                raise ValueError(
                    "session not found"
                )

            if (
                str(
                    row.get(
                        "identity_id"
                    )
                    or ""
                )
                != identity_id
            ):
                raise ValueError(
                    "session identity mismatch"
                )

            if row.get(
                "revoked_at"
            ):
                raise ValueError(
                    "session revoked"
                )

            return payload

        # --------------------------------------------------
        # LEGACY MIGRATION SUPPORT
        #
        # Existing short-lived sessions may still work until
        # their original exp. New sessions are never issued
        # using this format.
        # --------------------------------------------------
        exp = int(
            payload.get(
                "exp",
                0,
            )
            or 0
        )

        if (
            exp <= 0
            or exp < int(
                time.time()
            )
        ):
            raise ValueError(
                "expired"
            )

        return payload

    except HTTPException:
        raise

    except Exception:
        raise HTTPException(
            status_code=401,
            detail={
                "error": "invalid_session",
            },
        )


def _identity_by_email(
    email_lower: str,
) -> Optional[dict]:
    with get_db() as conn:
        cur = conn.cursor()

        cur.execute(
            """
            SELECT *
            FROM verifyd_identities
            WHERE email_lower = %s
              AND status = 'active'
            LIMIT 1
            """,
            (
                email_lower,
            ),
        )

        row = cur.fetchone()

    return (
        dict(row)
        if row
        else None
    )


def _identity_by_id(
    identity_id: str,
) -> Optional[dict]:
    with get_db() as conn:
        cur = conn.cursor()

        cur.execute(
            """
            SELECT *
            FROM verifyd_identities
            WHERE id = %s
              AND status = 'active'
            LIMIT 1
            """,
            (
                identity_id,
            ),
        )

        row = cur.fetchone()

    return (
        dict(row)
        if row
        else None
    )


def _identity_by_handle(
    handle_lower: str,
) -> Optional[dict]:
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
            (
                handle_lower,
            ),
        )

        row = cur.fetchone()

    return (
        dict(row)
        if row
        else None
    )


def _public_identity(
    row: dict,
) -> dict:
    handle = (
        row.get(
            "handle"
        )
        or ""
    ).strip()

    emoji = (
        row.get(
            "display_emoji"
        )
        or ""
    ).strip()

    return {
        "identity_id": (
            row.get(
                "id"
            )
        ),
        "handle": (
            f"@{handle}"
        ),
        "display_name": (
            row.get(
                "display_name"
            )
            or ""
        ),
        "display_emoji": emoji,
        "display_label": (
            f"{emoji + ' ' if emoji else ''}"
            f"@{handle}"
        ),
        "verification": {
            "email_verified": bool(
                row.get(
                    "email_verified"
                )
            ),
            "identity_verified": bool(
                row.get(
                    "identity_verified"
                )
            ),
            "organization_verified": bool(
                row.get(
                    "organization_verified"
                )
            ),
            "level": (
                row.get(
                    "verification_level"
                )
                or "email_verified"
            ),
        },
    }


async def _send(
    ws: WebSocket,
    payload: dict,
) -> bool:
    try:
        await ws.send_json(payload)
        return True
    except Exception:
        return False


async def _send_to_identity(
    identity_id: str,
    payload: dict,
) -> int:
    async with _state_lock:
        sockets = list(
            _connections.get(
                identity_id,
                set(),
            )
        )

    sent = 0

    for ws in sockets:
        try:
            await ws.send_json(
                payload
            )

            sent += 1

        except Exception:
            pass

    return sent


async def notify_message_created(
    recipient_identity_id: str,
    conversation_id: str,
    message_id: str,
    sender: dict,
    created_at: str,
    message_type: str = "text",
) -> int:
    # PostgreSQL/HTTP remains authoritative. WebSocket and Web Push are both
    # best-effort supplemental delivery paths and must never fail the stored
    # message.
    safe_message_type = (
        message_type
        if message_type
        in {
            "text",
            "attachment",
        }
        else "text"
    )

    event_payload = {
        "type": "message_created",
        "conversation_id": conversation_id,
        "message_id": message_id,
        "message_type": safe_message_type,
        "sender": sender,
        "created_at": created_at,
    }

    sent = await _send_to_identity(
        recipient_identity_id,
        event_payload,
    )

    try:
        from trust_voice_push import dispatch_push_for_event

        await asyncio.to_thread(
            dispatch_push_for_event,
            recipient_identity_id,
            event_payload,
        )
    except Exception:
        log.warning(
            "Trust Voice message Web Push dispatch failed",
            exc_info=True,
        )

    return sent



def _incoming_call_push_pilot_enabled(target: dict) -> bool:
    """Do not change call behavior unless BOTH gates explicitly allow this target."""
    flag = os.environ.get("VERIFYD_INCOMING_CALL_PUSH_PILOT_ENABLED", "0").strip().lower()
    if flag not in {"1", "true", "yes", "on"}:
        return False
    allowed_handles = {
        v.strip().lower().lstrip("@")
        for v in os.environ.get("VERIFYD_INCOMING_CALL_PUSH_PILOT_HANDLES", "").split(",")
        if v.strip()
    }
    handle = str(target.get("handle_lower") or target.get("handle") or "").strip().lower().lstrip("@")
    return bool(handle and handle in allowed_handles)


def _incoming_call_push_subscription_ready(identity_id: str) -> bool:
    """Read only current session-bound, nonrevoked subscriptions; fail closed."""
    try:
        from trust_voice_push import (
            _active_subscriptions,
            _enabled as push_enabled,
            _configured as push_configured,
        )
        return bool(
            identity_id and push_enabled() and push_configured()
            and _active_subscriptions(identity_id)
        )
    except Exception:
        log.warning("Incoming-call pilot push subscription lookup failed", exc_info=True)
        return False


async def _send_incoming_call_pilot_push(
    callee_id: str, call_id: str, call_type: str
) -> bool:
    """Best-effort Web Push, NOT a native phone-call or a delivery acknowledgment."""
    try:
        from trust_voice_push import notify_identity_push
        result = await asyncio.to_thread(
            notify_identity_push,
            callee_id,
            notification_type="call_incoming",
            title="Incoming VeriFYD Circle call",
            body="Tap to open VeriFYD Circle and answer.",
            url=f"/voice?incoming={call_id}",
            tag=f"verifyd-incoming-{call_id}",
            data={"call_id": call_id, "call_type": call_type},
            ttl=INVITE_TTL_SECONDS,
        )
        return bool(int((result or {}).get("sent") or 0) > 0)
    except Exception:
        log.warning("Incoming-call pilot Web Push failed", exc_info=True)
        return False


async def _replay_pilot_ringing_calls(identity_id: str, websocket: WebSocket) -> set[str]:
    """Deliver live invitations and return the call IDs sent on this socket."""
    replayed_call_ids = set()
    async with _state_lock:
        pending = [
            dict(call)
            for call in _active_calls.values()
            if call.get("state") == "ringing"
            and (call.get("pilot_incoming_push") or call.get("native_incoming_push") or call.get("android_incoming_push"))
            and str(call.get("callee_id") or "") == identity_id
        ]
    for call in pending:
        call_age = time.time() - float(call.get("created_at") or 0)
        if not (0 <= call_age < INVITE_TTL_SECONDS):
            continue
        try:
            caller = _identity_by_id(str(call.get("caller_id") or ""))
            callee = _identity_by_id(identity_id)
            if not caller or not callee:
                continue
            allowed, _reason = call_permission_decision(caller=caller, callee=callee)
            if not allowed:
                continue
            from trust_voice_blocking import identities_blocked
            if identities_blocked(str(caller["id"]), identity_id):
                continue
            sent = await _send(websocket, {
                "type": "incoming_call",
                "call_id": str(call["call_id"]),
                "caller": _public_identity(caller),
                "expires_in_seconds": max(1, int(INVITE_TTL_SECONDS - call_age)),
                "call_type": str(call.get("call_type") or "audio"),
                "replayed": True,
            })
            log.info("call_trace event=replay call_id=%s result=%s",
                     call["call_id"], "sent" if sent else "send_failed")
            if sent:
                replayed_call_ids.add(str(call["call_id"]))
        except Exception:
            log.warning("Could not replay staged incoming-call invitation", exc_info=True)
    return replayed_call_ids


async def _presence(
    identity_id: str,
) -> bool:
    async with _state_lock:
        return bool(
            _connections.get(
                identity_id
            )
        )


async def _register(
    identity_id: str,
    ws: WebSocket,
    token_payload: dict,
) -> None:
    async with _state_lock:
        _connections[
            identity_id
        ].add(
            ws
        )

        _connection_meta[
            id(ws)
        ] = {
            "identity_id": (
                identity_id
            ),
            "connected_at": (
                time.time()
            ),
            "session_id": str(
                token_payload.get(
                    "session_id"
                )
                or ""
            ),
            "token_exp": int(
                token_payload.get(
                    "exp",
                    0,
                )
                or 0
            ),
            "persistent": bool(
                token_payload.get(
                    "session_id"
                )
            ),
        }


async def _unregister(identity_id: str, ws: WebSocket) -> None:
    async with _state_lock:
        if identity_id in _connections:
            _connections[identity_id].discard(ws)
            if not _connections[identity_id]:
                _connections.pop(identity_id, None)

        _connection_meta.pop(id(ws), None)

        # Only tear down calls if this identity has no remaining live
        # Trust Voice socket on another tab/device.
        identity_still_online = bool(_connections.get(identity_id))
        affected = []

        if not identity_still_online:
            for call_id, call in list(_active_calls.items()):
                if identity_id in {
                    call.get("caller_id"),
                    call.get("callee_id"),
                }:
                    # A locked pilot recipient may briefly reconnect, then
                    # lose its suspended browser socket before answering.
                    # Preserve the invitation until its EXISTING 30-second
                    # timeout, rather than turning it into an immediate miss.
                    # A caller disconnect still cancels the invite normally.
                    if (
                        identity_id == call.get("callee_id")
                        and call.get("state") == "ringing"
                        and (call.get("pilot_incoming_push") or call.get("native_incoming_push") or call.get("android_incoming_push"))
                    ):
                        continue
                    if identity_id == call.get("callee_id") and call.get("android_media_socket") is not None:
                        continue
                    affected.append((call_id, dict(call)))
                    _active_calls.pop(call_id, None)

    for call_id, call in affected:
        _cancel_call_expiry(call_id)

        caller_id = str(call.get("caller_id") or "")
        callee_id = str(call.get("callee_id") or "")
        call_state = str(call.get("state") or "")

        peer_id = (
            callee_id
            if identity_id == caller_id
            else caller_id
        )

        if call_state == "ringing":
            if identity_id == callee_id:
                # Recipient disappeared while ringing: make it a missed call.
                missed_at = _record_call_terminal(
                    call_id,
                    "missed",
                    "callee_disconnected",
                )

                await _send_to_identity(
                    caller_id,
                    {
                        "type": "call_no_answer",
                        "call_id": call_id,
                        "reason": "callee_disconnected",
                    },
                )

                caller_identity = _identity_by_id(caller_id)

                await _send_to_identity(
                    callee_id,
                    {
                        "type": "call_missed",
                        "call_id": call_id,
                        "caller": (
                            _public_identity(caller_identity)
                            if caller_identity
                            else None
                        ),
                        "missed_at": missed_at,
                    },
                )
            else:
                # Caller left before the recipient answered.
                _record_call_terminal(
                    call_id,
                    "canceled",
                    "caller_disconnected",
                )

                await _send_to_identity(
                    peer_id,
                    {
                        "type": "call_ended",
                        "call_id": call_id,
                        "reason": "caller_disconnected",
                    },
                )

            continue

        _record_call_terminal(
            call_id,
            "ended",
            "peer_disconnected",
        )

        event = {"type": "call_ended", "call_id": call_id, "reason": "peer_disconnected"}
        await _send_to_identity(peer_id, event)
        if call.get("android_media_socket") is not None:
            await _send(call["android_media_socket"], event)


def _recent_otp_request(
    email_lower: str,
    cooldown_seconds: int = 60,
) -> bool:
    with get_db() as conn:
        cur = conn.cursor()

        cur.execute(
            """
            SELECT created_at
            FROM email_otp
            WHERE email_lower = %s
            ORDER BY id DESC
            LIMIT 1
            """,
            (
                email_lower,
            ),
        )

        row = cur.fetchone()

    if (
        not row
        or not row.get(
            "created_at"
        )
    ):
        return False

    try:
        created = (
            datetime.fromisoformat(
                row[
                    "created_at"
                ]
            )
        )

        if created.tzinfo is None:
            created = (
                created.replace(
                    tzinfo=timezone.utc
                )
            )

        return (
            datetime.now(
                timezone.utc
            )
            - created
        ).total_seconds() < (
            cooldown_seconds
        )

    except Exception:
        return False


@router.get(
    "/signaling/health"
)
def signaling_health():
    _require_enabled()
    _secret()
    ensure_persistent_session_schema()
    ensure_call_history_schema()

    return {
        "status": "ok",
        "feature": (
            "trust_voice_signaling"
        ),
        "version": (
            FEATURE_VERSION
        ),
        "audio_transport": (
            "webrtc_peer_to_peer_beta"
            if _webrtc_enabled()
            else "not_enabled"
        ),
        "webrtc_signaling": (
            "enabled"
            if _webrtc_enabled()
            else "disabled"
        ),
        "session_mode": (
            "persistent_revocable"
        ),
        "session_expiry": (
            "none"
        ),
        "session_store": (
            "postgresql"
        ),
        "legacy_session_support": (
            "temporary"
        ),
        "ring_timeout_seconds": INVITE_TTL_SECONDS,
        "call_history": "postgresql_v1",
        "missed_calls": "durable_v1",
        "supported_call_types": ["audio", "video"],
        "video_calling": (
            "webrtc_one_to_one_v1"
            if _webrtc_enabled()
            else "not_enabled"
        ),
        "call_control_signaling": (
            "media_state_recording_consent_v1"
        ),
        "recording_transport": (
            "browser_local_only"
        ),
        "recording_consent": (
            "peer_required_v1"
        ),
        "presence_store": (
            "memory_beta"
        ),
    }


@router.post(
    "/session/request-code"
)
def request_session_code(
    payload: SessionCodeRequest,
):
    _require_enabled()
    _secret()

    email = _normalize_email(
        payload.email
    )

    if not is_valid_email(
        email
    ):
        raise HTTPException(
            status_code=400,
            detail={
                "error": (
                    "invalid_email"
                ),
            },
        )

    identity = _identity_by_email(
        email
    )

    if not identity:
        raise HTTPException(
            status_code=404,
            detail={
                "error": (
                    "identity_not_found"
                ),
                "message": (
                    "Create a VeriFYD Identity "
                    "before starting a Trust "
                    "Voice session."
                ),
            },
        )

    # Reviewers use an explicitly configured password sign-in. No email code
    # is sent to the review mailbox while that feature is enabled.
    if email == "appreview@vfvid.com" and os.environ.get(
        "VERIFYD_APP_REVIEW_LOGIN_ENABLED", ""
    ).strip().lower() in {"1", "true", "yes", "on"}:
        raise HTTPException(status_code=403, detail={
            "error": "review_password_required",
            "message": "Use the App Review password sign-in instead of an email code.",
        })

    # Dedicated SLP demonstration identity uses its own password endpoint.
    # All ordinary users, including clinical individual accounts, still use OTP.
    if email == "slpdemo@vfvid.com" and os.environ.get(
        "VERIFYD_SLP_DEMO_LOGIN_ENABLED", ""
    ).strip().lower() in {"1", "true", "yes", "on"}:
        raise HTTPException(status_code=403, detail={
            "error": "slp_demo_password_required",
            "message": "Use the demo password instead of requesting an email code.",
        })

    if _recent_otp_request(
        email
    ):
        raise HTTPException(
            status_code=429,
            detail={
                "error": (
                    "code_recently_sent"
                ),
                "message": (
                    "Please wait before "
                    "requesting another code."
                ),
            },
        )

    code = create_otp(
        email
    )

    if not send_otp_email(
        email,
        code,
    ):
        raise HTTPException(
            status_code=503,
            detail={
                "error": (
                    "email_send_failed"
                ),
                "message": (
                    "Verification email could "
                    "not be sent."
                ),
            },
        )

    return {
        "ok": True,
        "email": email,
        "expires_minutes": 10,
        "purpose": (
            "trust_voice_session"
        ),
    }


@router.post(
    "/session/verify"
)
def verify_session(
    payload: SessionVerifyRequest,
):
    _require_enabled()

    email = _normalize_email(
        payload.email
    )

    if not is_valid_email(
        email
    ):
        raise HTTPException(
            status_code=400,
            detail={
                "error": (
                    "invalid_email"
                ),
            },
        )

    ok, message = verify_otp(
        email,
        payload.code,
    )

    if not ok:
        raise HTTPException(
            status_code=400,
            detail={
                "error": (
                    "invalid_verification_code"
                ),
                "message": message,
            },
        )

    identity = _identity_by_email(
        email
    )

    if not identity:
        raise HTTPException(
            status_code=404,
            detail={
                "error": (
                    "identity_not_found"
                ),
            },
        )

    now = int(
        time.time()
    )

    session_id = (
        _create_persistent_session(
            identity["id"]
        )
    )

    token = _sign(
        {
            "purpose": (
                "trust_voice_session"
            ),
            "session_id": (
                session_id
            ),
            "identity_id": (
                identity["id"]
            ),
            "handle_lower": (
                identity.get(
                    "handle_lower"
                )
                or ""
            ).lower(),
            "iat": now,
            "nonce": (
                secrets.token_hex(8)
            ),
        }
    )

    return {
        "ok": True,
        "token": token,
        "persistent": True,
        "expires_in_seconds": None,
        "identity": (
            _public_identity(
                identity
            )
        ),
    }


@router.post(
    "/session/revoke"
)
def revoke_session(
    authorization: Optional[str] = Header(
        default=None
    ),
):
    _require_enabled()

    token = _token_from_authorization(
        authorization
    )

    payload = _verify_token(
        token
    )

    session_id = str(
        payload.get(
            "session_id"
        )
        or ""
    ).strip()

    # Legacy tokens predate persistent server-side
    # session records. They cannot be individually
    # revoked and will expire normally.
    if not session_id:
        return {
            "ok": True,
            "revoked": False,
            "legacy_session": True,
        }

    now = _now_iso()

    with get_db() as conn:
        cur = conn.cursor()

        cur.execute(
            """
            UPDATE verifyd_voice_sessions
            SET
                revoked_at = %s,
                revoked_reason = 'user_sign_out'
            WHERE session_id = %s
              AND revoked_at IS NULL
            """,
            (
                now,
                session_id,
            ),
        )

        revoked = bool(
            cur.rowcount
        )

    return {
        "ok": True,
        "revoked": revoked,
        "session_id": (
            session_id
        ),
    }


@router.post(
    "/session/revoke-all"
)
def revoke_all_sessions(
    authorization: Optional[str] = Header(
        default=None
    ),
):
    _require_enabled()

    token = _token_from_authorization(
        authorization
    )

    payload = _verify_token(
        token
    )

    identity_id = str(
        payload.get(
            "identity_id"
        )
        or ""
    ).strip()

    if not identity_id:
        raise HTTPException(
            status_code=401,
            detail={
                "error": (
                    "invalid_session"
                ),
            },
        )

    now = _now_iso()

    ensure_persistent_session_schema()

    with get_db() as conn:
        cur = conn.cursor()

        cur.execute(
            """
            UPDATE verifyd_voice_sessions
            SET
                revoked_at = %s,
                revoked_reason = 'revoke_all'
            WHERE identity_id = %s
              AND revoked_at IS NULL
            """,
            (
                now,
                identity_id,
            ),
        )

        revoked_count = int(
            cur.rowcount
            or 0
        )

    return {
        "ok": True,
        "revoked": True,
        "revoked_count": (
            revoked_count
        ),
    }



def _identity_from_http_authorization(
    authorization: Optional[str],
) -> dict:
    token = _token_from_authorization(authorization)
    payload = _verify_token(token)

    identity = _identity_by_id(
        str(payload.get("identity_id") or "")
    )

    if not identity:
        raise HTTPException(
            status_code=401,
            detail={"error": "identity_not_found"},
        )

    return identity


def _public_call_identity(identity: Optional[dict]) -> Optional[dict]:
    if not identity:
        return None

    public = _public_identity(identity)
    public["profile_media_url"] = (
        f"/trust-voice/profile-media/{identity['id']}"
    )
    return public


@router.get("/calls/summary")
def call_summary(
    authorization: Optional[str] = Header(default=None),
):
    _require_enabled()

    identity = _identity_from_http_authorization(
        authorization
    )
    ensure_call_history_schema()

    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT COUNT(*) AS unread_missed_count
            FROM verifyd_voice_calls
            WHERE callee_identity_id = %s
              AND status = 'missed'
              AND missed_seen_at IS NULL
            """,
            (
                identity["id"],
            ),
        )
        row = cur.fetchone() or {}

    return {
        "unread_missed_count": int(
            row.get("unread_missed_count") or 0
        ),
    }


@router.get("/calls/missed")
def list_missed_calls(
    authorization: Optional[str] = Header(default=None),
    limit: int = Query(default=20, ge=1, le=100),
):
    _require_enabled()

    identity = _identity_from_http_authorization(
        authorization
    )
    ensure_call_history_schema()

    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT *
            FROM verifyd_voice_calls
            WHERE callee_identity_id = %s
              AND status = 'missed'
            ORDER BY missed_at DESC NULLS LAST
            LIMIT %s
            """,
            (
                identity["id"],
                limit,
            ),
        )
        rows = [
            dict(row)
            for row in (cur.fetchall() or [])
        ]

        cur.execute(
            """
            SELECT COUNT(*) AS unread_missed_count
            FROM verifyd_voice_calls
            WHERE callee_identity_id = %s
              AND status = 'missed'
              AND missed_seen_at IS NULL
            """,
            (
                identity["id"],
            ),
        )
        unread_row = cur.fetchone() or {}

    calls = []

    for row in rows:
        caller = _identity_by_id(
            str(row.get("caller_identity_id") or "")
        )

        calls.append(
            {
                "call_id": row.get("call_id"),
                "call_type": row.get("call_type") or "audio",
                "status": "missed",
                "caller": _public_call_identity(caller),
                "started_at": row.get("started_at") or "",
                "ended_at": row.get("ended_at"),
                "missed_at": row.get("missed_at"),
                "seen": bool(row.get("missed_seen_at")),
            }
        )

    return {
        "count": len(calls),
        "unread_missed_count": int(
            unread_row.get("unread_missed_count") or 0
        ),
        "calls": calls,
    }


@router.post("/calls/{call_id}/read")
def mark_missed_call_read(
    call_id: str,
    authorization: Optional[str] = Header(default=None),
):
    _require_enabled()

    identity = _identity_from_http_authorization(
        authorization
    )
    ensure_call_history_schema()
    now = _now_iso()

    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            UPDATE verifyd_voice_calls
            SET missed_seen_at = COALESCE(missed_seen_at, %s)
            WHERE call_id = %s
              AND callee_identity_id = %s
              AND status = 'missed'
            """,
            (
                now,
                call_id,
                identity["id"],
            ),
        )

        if cur.rowcount == 0:
            raise HTTPException(
                status_code=404,
                detail={"error": "missed_call_not_found"},
            )

    return {
        "ok": True,
        "call_id": call_id,
        "read": True,
    }


@router.post("/calls/missed/read-all")
def mark_all_missed_calls_read(
    authorization: Optional[str] = Header(default=None),
):
    _require_enabled()

    identity = _identity_from_http_authorization(
        authorization
    )
    ensure_call_history_schema()
    now = _now_iso()

    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            UPDATE verifyd_voice_calls
            SET missed_seen_at = %s
            WHERE callee_identity_id = %s
              AND status = 'missed'
              AND missed_seen_at IS NULL
            """,
            (
                now,
                identity["id"],
            ),
        )
        updated = int(cur.rowcount or 0)

    return {
        "ok": True,
        "updated": updated,
        "unread_missed_count": 0,
    }


@router.get(
    "/presence/{handle}"
)
async def get_presence(
    handle: str,
):
    _require_enabled()

    handle_lower = (
        _normalize_handle(
            handle
        )
    )

    identity = (
        _identity_by_handle(
            handle_lower
        )
    )

    if not identity:
        raise HTTPException(
            status_code=404,
            detail={
                "error": (
                    "handle_not_found"
                ),
            },
        )

    return {
        "handle": (
            f"@{identity['handle']}"
        ),
        "online": (
            await _presence(
                identity["id"]
            )
        ),
    }


async def native_call_snapshot(call_id: str, identity_id: str) -> dict:
    async with _state_lock:
        call = dict(_active_calls.get(call_id) or {})
    if not call or identity_id not in {call.get("caller_id"), call.get("callee_id")}:
        raise HTTPException(404, detail={"error": "call_not_found"})
    if call.get("state") == "ringing" and time.time() >= call["created_at"] + INVITE_TTL_SECONDS:
        raise HTTPException(410, detail={"error": "call_expired"})
    from trust_voice_blocking import identities_blocked
    if await asyncio.to_thread(identities_blocked, call["caller_id"], call["callee_id"]):
        raise HTTPException(403, detail={"error": "not_allowed"})
    return {"ok": True, "call_id": call_id, "state": call["state"],
            "call_type": call["call_type"],
            "expires_at": int(call["created_at"] + INVITE_TTL_SECONDS),
            "is_callee": identity_id == call["callee_id"]}


async def relay_call_media(call_id: str, identity_id: str, message: dict, *,
                           android_socket: Optional[WebSocket] = None) -> None:
    """Relay only SDP/ICE for an accepted call; a native socket owns one callee leg."""
    if not _webrtc_enabled():
        raise HTTPException(409, detail={"error": "webrtc_not_enabled"})
    if not call_id:
        raise HTTPException(400, detail={"error": "call_id_required"})

    async with _state_lock:
        call = dict(_active_calls.get(call_id) or {})
    if not call:
        raise HTTPException(404, detail={"error": "call_not_found"})
    caller_id, callee_id = call["caller_id"], call["callee_id"]
    if identity_id not in {caller_id, callee_id}:
        raise HTTPException(403, detail={"error": "not_call_participant"})
    if call["state"] != "answered":
        raise HTTPException(409, detail={"error": "invalid_call_state"})

    native = call.get("android_media_socket")
    if (identity_id == callee_id and native is not android_socket
            or android_socket is not None and identity_id != callee_id):
        raise HTTPException(403, detail={"error": "call_media_owned_by_another_device"})

    kind = message.get("type")
    if kind in {"webrtc_offer", "webrtc_answer"}:
        offer = kind == "webrtc_offer"
        role = "caller" if offer else "callee"
        sdp_type = "offer" if offer else "answer"
        if identity_id != (caller_id if offer else callee_id):
            raise HTTPException(403, detail={"error": f"{sdp_type}_must_come_from_{role}"})
        field, content = "sdp", message.get("sdp")
        if (not isinstance(content, dict)
                or str(content.get("type", "")).strip().lower() != sdp_type
                or not isinstance(content.get("sdp"), str) or not content["sdp"].strip()):
            raise HTTPException(400, detail={"error": "invalid_webrtc_sdp"})
        if _json_size_bytes(content) > MAX_WEBRTC_SDP_BYTES:
            raise HTTPException(413, detail={"error": "webrtc_sdp_too_large"})
    elif kind == "webrtc_ice":
        field, content = "candidate", message.get("candidate")
        if not isinstance(content, dict) or not isinstance(content.get("candidate"), str):
            raise HTTPException(400, detail={"error": "invalid_webrtc_candidate"})
        if _json_size_bytes(content) > MAX_WEBRTC_ICE_BYTES:
            raise HTTPException(413, detail={"error": "webrtc_candidate_too_large"})
    else:
        raise HTTPException(400, detail={"error": "unsupported_message_type"})

    event = {"type": kind, "call_id": call_id, field: content}
    if identity_id == caller_id and native is not None:
        try:
            await native.send_json(event)
        except Exception:
            raise HTTPException(409, detail={"error": "webrtc_peer_unavailable"}) from None
    elif not await _send_to_identity(callee_id if identity_id == caller_id else caller_id, event):
        raise HTTPException(409, detail={"error": "webrtc_peer_unavailable"})


async def apply_call_action(call_id: str, identity_id: str, action: str, *,
                            android_socket: Optional[WebSocket] = None,
                            transport: str = "native_http") -> dict:
    """One atomic transition for native HTTP and existing WebSocket clients."""
    if action not in {"call_answer", "call_decline", "call_end"}:
        raise HTTPException(400, detail={"error": "invalid_action"})
    if not call_id:
        raise HTTPException(400, detail={"error": "call_id_required"})
    async with _state_lock:
        snapshot = dict(_active_calls.get(call_id) or {})
    if not snapshot or identity_id not in {snapshot.get("caller_id"), snapshot.get("callee_id")}:
        raise HTTPException(404, detail={"error": "call_not_found"})
    if action == "call_answer":
        from trust_voice_blocking import identities_blocked
        blocked = await asyncio.to_thread(identities_blocked, snapshot["caller_id"], snapshot["callee_id"])
        caller = await asyncio.to_thread(_identity_by_id, snapshot["caller_id"])
        callee = await asyncio.to_thread(_identity_by_id, snapshot["callee_id"])
        allowed = False
        if caller and callee:
            allowed, _ = await asyncio.to_thread(call_permission_decision, caller=caller, callee=callee)
        if blocked or not allowed:
            raise HTTPException(403, detail={"error": "not_allowed"})
    async with _state_lock:
        call = _active_calls.get(call_id)
        if not call or identity_id not in {call.get("caller_id"), call.get("callee_id")}:
            raise HTTPException(404, detail={"error": "call_not_found"})
        if action in {"call_answer", "call_decline"}:
            if identity_id != call["callee_id"]:
                raise HTTPException(403, detail={"error": "not_callee"})
            if call["state"] != "ringing":
                raise HTTPException(409, detail={"error": "invalid_call_state"})
            if time.time() >= call["created_at"] + INVITE_TTL_SECONDS:
                raise HTTPException(410, detail={"error": "call_expired"})
        native = call.get("android_media_socket")
        if action == "call_end" and (
            (android_socket is not None and native is not android_socket)
            or (native is not None and identity_id == call["callee_id"] and android_socket is None)
        ):
            raise HTTPException(409, detail={"error": "call_media_owned_by_another_device"})
        if android_socket is not None and action == "call_answer":
            if call["call_type"] != "audio" or not _webrtc_enabled():
                raise HTTPException(409, detail={"error": "native_audio_unavailable"})
            if not _connections.get(call["caller_id"]):
                raise HTTPException(409, detail={"error": "webrtc_peer_unavailable"})
            call["android_media_socket"] = android_socket
        call = dict(call)
        if action == "call_answer":
            _active_calls[call_id]["state"] = "answered"
        else:
            _active_calls.pop(call_id, None)
    _cancel_call_expiry(call_id)
    if action == "call_answer":
        _record_call_answered(call_id)
        event = {"type": "call_answered", "call_id": call_id}
    elif action == "call_decline":
        _record_call_terminal(call_id, "declined", "declined")
        event = {"type": "call_declined", "call_id": call_id}
    else:
        ringing = call["state"] == "ringing"
        _record_call_terminal(call_id, "canceled" if ringing else "ended",
                              "ended_before_answer" if ringing else "ended")
        event = {"type": "call_ended", "call_id": call_id, "reason": "ended"}
        log.info("call_trace event=end call_id=%s actor=%s transport=%s state=%s",
                 call_id, "caller" if identity_id == call["caller_id"] else "callee",
                 "android_media" if android_socket is not None else transport, call["state"])
    native = call.get("android_media_socket")
    if native is not None and action == "call_answer":
        # Release any duplicate ring on the web callee. The caller's existing
        # browser starts its offer as soon as it sees call_answered.
        await native.send_json(event)
        await _send_to_identity(call["callee_id"], {
            "type": "call_ended", "call_id": call_id, "reason": "answered_on_another_device",
        })
        async with _state_lock:
            if _active_calls.get(call_id, {}).get("android_media_socket") is not native:
                return {"ok": False, "error": "call_not_found"}
        await _send_to_identity(call["caller_id"], event)
    else:
        await _send_to_identity(call["caller_id"], event)
        await _send_to_identity(call["callee_id"], event)
        if native is not None:
            await _send(native, event)
    return {"ok": True, **event}


# VERIFYD_WS_AUTH_HEADER_V1
# Subprotocol authentication keeps credentials out of URLs logged by proxies.
# Legacy query-token clients remain supported throughout the migration.
_WS_AUTH_PROTOCOL = "verifyd-auth-v1"
_WS_TOKEN_PREFIX = "verifyd-session."


def _websocket_auth_from_scope(websocket: WebSocket, legacy_query_token: str) -> tuple[str, Optional[str]]:
    offered = websocket.scope.get("subprotocols") or []
    tokens = [str(p)[len(_WS_TOKEN_PREFIX):] for p in offered
              if isinstance(p, str) and p.startswith(_WS_TOKEN_PREFIX)]
    if tokens or _WS_AUTH_PROTOCOL in offered:
        # Never silently downgrade a malformed protocol-auth request to a URL token.
        if len(tokens) != 1 or not tokens[0] or _WS_AUTH_PROTOCOL not in offered:
            return "", None
        return tokens[0], _WS_AUTH_PROTOCOL
    return legacy_query_token, None


@router.websocket(
    "/ws"
)
async def signaling_ws(
    websocket: WebSocket,
    token: str = Query(
        default=""
    ),
):
    if not _enabled():
        await websocket.close(
            code=4404
        )
        return

    # Browser clients can authenticate by WebSocket subprotocol without
    # transmitting session credentials in URL query strings.
    token, negotiated_protocol = _websocket_auth_from_scope(websocket, token)

    try:
        # Do not run blocking PostgreSQL session queries on the event loop.
        token_payload = await run_in_threadpool(_verify_token, token)
        identity = await run_in_threadpool(
            _identity_by_id, token_payload["identity_id"]
        )

        if not identity:
            await websocket.close(
                code=4401
            )
            return

    except HTTPException as exc:
        # Only a definitively INVALID token should log the user out.
        # A transient unavailable session DB must not produce 4401.
        await websocket.close(
            code=4401 if exc.status_code == 401 else 1013
        )
        return

    except (psycopg2.Error, psycopg2.pool.PoolError):
        # Database overload is temporary, not an authentication failure.
        await websocket.close(code=1013)
        return

    except Exception:
        # Unexpected infrastructure problems also must not revoke sessions.
        await websocket.close(code=1013)
        return

    identity_id = (
        identity["id"]
    )

    # Only echo the fixed protocol identifier; never echo the token subprotocol.
    await websocket.accept(subprotocol=negotiated_protocol)

    await _register(
        identity_id,
        websocket,
        token_payload,
    )

    await _send(
        websocket,
        {
            "type": (
                "session_ready"
            ),
            "identity": (
                _public_identity(
                    identity
                )
            ),
            "audio_transport": (
                "webrtc_peer_to_peer_beta"
                if _webrtc_enabled()
                else "not_enabled"
            ),
            "webrtc_signaling": (
                "enabled"
                if _webrtc_enabled()
                else "disabled"
            ),
            "supported_call_types": [
                "audio",
                "video",
            ],
            "call_control_events": [
                "call_media_state",
                "recording_request",
                "recording_response",
                "recording_started",
                "recording_stop",
                "recording_stopped",
            ],
        },
    )

    # If the recipient opened a locked-screen push during an active pilot
    # ringing window, replay the invitation to the authenticated socket.
    # Never replay terminal or expired calls.
    from trust_voice_android_calls import pilot_allowed as android_pilot_allowed
    from trust_voice_native_calls import pilot_allowed as native_pilot_allowed
    replayed_call_ids = set()
    if (
        _incoming_call_push_pilot_enabled(identity)
        or native_pilot_allowed(identity)
        or android_pilot_allowed(identity)
    ):
        replayed_call_ids = await _replay_pilot_ringing_calls(str(identity_id), websocket)

    try:
        while True:
            message = (
                await websocket.receive_json()
            )

            if not isinstance(
                message,
                dict,
            ):
                await _send(
                    websocket,
                    {
                        "type": (
                            "error"
                        ),
                        "error": (
                            "invalid_message"
                        ),
                    },
                )
                continue

            # A revoked device must not keep an established websocket alive.
            # Transient database failures should not be mistaken for revocation.
            try:
                await run_in_threadpool(_verify_token, token)
            except HTTPException as exc:
                if exc.status_code == 401:
                    await websocket.close(code=4401)
                    break
                # An infrastructure error is not proof of revocation.
                # Do not clear device credentials or forcibly sign out.
                await _send(websocket, {"type": "error", "error": "session_check_unavailable"})
                continue

            msg_type = (
                message.get(
                    "type"
                )
            )

            # Re-check blocks for already established call signaling, not just invitations.
            # Reject stale queued call messages after a new block has been recorded.
            if msg_type in {
                "call_answer", "call_decline", "call_end",
                "webrtc_offer", "webrtc_answer", "webrtc_ice",
                "call_media_state", "recording_request", "recording_response",
                "recording_started", "recording_stop", "recording_stopped",
            }:
                pending_call_id = str(message.get("call_id") or "")
                if pending_call_id:
                    async with _state_lock:
                        pending_call = dict(_active_calls.get(pending_call_id) or {})
                    if pending_call and identity_id in {
                        str(pending_call.get("caller_id") or ""),
                        str(pending_call.get("callee_id") or ""),
                    }:
                        from trust_voice_blocking import identities_blocked
                        if identities_blocked(
                            str(pending_call.get("caller_id") or ""),
                            str(pending_call.get("callee_id") or ""),
                        ):
                            await _send(websocket, {"type": "error", "error": "not_allowed"})
                            continue

            if msg_type == "ping":
                await _send(
                    websocket,
                    {
                        "type": (
                            "pong"
                        ),
                        "ts": int(
                            time.time()
                        ),
                    },
                )
                continue

            if msg_type == "call_invite":
                target_handle = (
                    _normalize_handle(
                        str(
                            message.get(
                                "target_handle",
                                "",
                            )
                        )
                    )
                )

                call_type = str(
                    message.get(
                        "call_type",
                        "audio",
                    )
                    or "audio"
                ).strip().lower()

                if call_type not in ALLOWED_CALL_TYPES:
                    await _send(
                        websocket,
                        {
                            "type": "error",
                            "error": "invalid_call_type",
                            "allowed_call_types": [
                                "audio",
                                "video",
                            ],
                        },
                    )
                    continue

                if not target_handle:
                    await _send(
                        websocket,
                        {
                            "type": (
                                "error"
                            ),
                            "error": (
                                "target_handle_required"
                            ),
                        },
                    )
                    continue

                if (
                    target_handle
                    == (
                        identity.get(
                            "handle_lower"
                        )
                        or ""
                    ).lower()
                ):
                    await _send(
                        websocket,
                        {
                            "type": (
                                "error"
                            ),
                            "error": (
                                "cannot_call_self"
                            ),
                        },
                    )
                    continue

                now = time.time()

                previous = (
                    _last_invite_at.get(
                        identity_id,
                        0,
                    )
                )

                if (
                    now
                    - previous
                    < CALL_COOLDOWN_SECONDS
                ):
                    await _send(
                        websocket,
                        {
                            "type": (
                                "error"
                            ),
                            "error": (
                                "call_rate_limited"
                            ),
                        },
                    )
                    continue

                _last_invite_at[
                    identity_id
                ] = now

                target = (
                    _identity_by_handle(
                        target_handle
                    )
                )

                if not target:
                    await _send(
                        websocket,
                        {
                            "type": (
                                "call_failed"
                            ),
                            "reason": (
                                "handle_not_found"
                            ),
                        },
                    )
                    continue

                (
                    allowed,
                    permission_reason,
                ) = (
                    call_permission_decision(
                        caller=identity,
                        callee=target,
                    )
                )

                if not allowed:
                    log.info(
                        (
                            "Trust Voice call denied "
                            "caller_id=%s "
                            "callee_id=%s "
                            "reason=%s"
                        ),
                        identity_id,
                        target["id"],
                        permission_reason,
                    )

                    await _send(
                        websocket,
                        {
                            "type": (
                                "call_failed"
                            ),
                            "reason": (
                                "not_allowed"
                            ),
                        },
                    )

                    continue

                recipient_online = await _presence(target["id"])
                pilot_push_reachable = False
                if _incoming_call_push_pilot_enabled(target):
                    pilot_push_reachable = await asyncio.to_thread(
                        _incoming_call_push_subscription_ready, str(target["id"])
                    )

                from trust_voice_native_calls import reachable as native_reachable
                native_push_reachable = await asyncio.to_thread(native_reachable, target)
                from trust_voice_android_calls import reachable as android_reachable
                android_push_reachable = await asyncio.to_thread(android_reachable, target, call_type)
                if not any((
                    recipient_online, pilot_push_reachable,
                    native_push_reachable, android_push_reachable,
                )):
                    await _send(
                        websocket,
                        {
                            "type": (
                                "call_failed"
                            ),
                            "target_handle": (
                                f"@{target['handle']}"
                            ),
                            "reason": (
                                "offline"
                            ),
                        },
                    )

                    continue

                call_id = (
                    "tvcall_"
                    + uuid.uuid4().hex
                )

                call = {
                    "call_id": (
                        call_id
                    ),
                    "caller_id": (
                        identity_id
                    ),
                    "callee_id": (
                        target["id"]
                    ),
                    "state": (
                        "ringing"
                    ),
                    "call_type": (
                        call_type
                    ),
                    "created_at": (
                        time.time()
                    ),
                    "pilot_incoming_push": pilot_push_reachable,
                    "native_incoming_push": native_push_reachable,
                    "android_incoming_push": android_push_reachable,
                }

                # Persist history before exposing the call to a fast answer.
                _create_call_history(
                    call_id=call_id,
                    caller_identity_id=identity_id,
                    callee_identity_id=str(target["id"]),
                    call_type=call_type,
                )
                async with _state_lock:
                    _active_calls[
                        call_id
                    ] = call
                _schedule_call_expiry(call_id)

                delivered = (
                    await _send_to_identity(
                        target["id"],
                        {
                            "type": (
                                "incoming_call"
                            ),
                            "call_id": (
                                call_id
                            ),
                            "caller": (
                                _public_identity(
                                    identity
                                )
                            ),
                            "expires_in_seconds": (
                                INVITE_TTL_SECONDS
                            ),
                            "call_type": (
                                call_type
                            ),
                        },
                    )
                )

                # Pilot push is sent in addition to live WebSocket signaling.
                # An open socket can exist while a locked browser is suspended.
                push_sent = False
                if pilot_push_reachable:
                    push_sent = await _send_incoming_call_pilot_push(
                        str(target["id"]), call_id, call_type
                    )

                if native_push_reachable or android_push_reachable:
                    async def still_ringing():
                        async with _state_lock:
                            current = _active_calls.get(call_id)
                            return bool(current and current.get("state") == "ringing"
                                and time.time() < current["created_at"] + INVITE_TTL_SECONDS)

                if native_push_reachable:
                    from trust_voice_native_calls import dispatch_invite
                    native_sent = await dispatch_invite(target, call, still_ringing)
                    push_sent = push_sent or native_sent

                if android_push_reachable:
                    from trust_voice_android_calls import dispatch_invite as dispatch_android_invite
                    android_sent = await dispatch_android_invite(target, call, still_ringing)
                    push_sent = push_sent or android_sent

                # Reconcile answers/cancellation/expiry received during delivery.
                # A stale ringing event must not follow its terminal event.
                async with _state_lock:
                    live = dict(_active_calls.get(call_id) or {})
                if not live or live.get("state") != "ringing":
                    continue
                if time.time() >= call["created_at"] + INVITE_TTL_SECONDS:
                    continue

                # Preserve the original offline failure unless a subscriber's
                # push service accepted the pilot invitation, or the device
                # authenticated again during the best-effort push attempt.
                if delivered == 0 and not push_sent and not await _presence(str(target["id"])):
                    async with _state_lock:
                        _active_calls.pop(
                            call_id,
                            None,
                        )

                    _cancel_call_expiry(call_id)
                    _record_call_terminal(call_id, "failed", "offline")
                    await _send(
                        websocket,
                        {
                            "type": (
                                "call_failed"
                            ),
                            "reason": (
                                "offline"
                            ),
                        },
                    )

                    continue


                await _send(
                    websocket,
                    {
                        "type": (
                            "call_ringing"
                        ),
                        "call_id": (
                            call_id
                        ),
                        "target": (
                            _public_identity(
                                target
                            )
                        ),
                        "call_type": (
                            call_type
                        ),
                    },
                )

                continue

            if msg_type in {"webrtc_offer", "webrtc_answer", "webrtc_ice"}:
                call_id = str(message.get("call_id") or "").strip()
                try:
                    await relay_call_media(call_id, identity_id, message)
                except HTTPException as exc:
                    await _send(websocket, {"type": "error", **exc.detail})
                continue

            if msg_type in {
                "call_media_state",
                "recording_request",
                "recording_response",
                "recording_started",
                "recording_stop",
                "recording_stopped",
            }:
                call_id = str(
                    message.get(
                        "call_id",
                        "",
                    )
                ).strip()

                if not call_id:
                    await _send(
                        websocket,
                        {
                            "type": "error",
                            "error": "call_id_required",
                        },
                    )
                    continue

                async with _state_lock:
                    call = dict(
                        _active_calls.get(
                            call_id,
                            {},
                        )
                    )

                if not call:
                    await _send(
                        websocket,
                        {
                            "type": "error",
                            "error": "call_not_found",
                        },
                    )
                    continue

                caller_id = str(
                    call.get("caller_id")
                    or ""
                )
                callee_id = str(
                    call.get("callee_id")
                    or ""
                )

                if identity_id not in {
                    caller_id,
                    callee_id,
                }:
                    await _send(
                        websocket,
                        {
                            "type": "error",
                            "error": "not_call_participant",
                        },
                    )
                    continue

                if call.get("state") != "answered":
                    await _send(
                        websocket,
                        {
                            "type": "error",
                            "error": "invalid_call_state",
                        },
                    )
                    continue

                peer_id = (
                    callee_id
                    if identity_id == caller_id
                    else caller_id
                )

                if msg_type == "call_media_state":
                    relay = {
                        "type": "call_media_state",
                        "call_id": call_id,
                    }

                    state_supplied = False

                    if "video_enabled" in message:
                        video_enabled = message.get(
                            "video_enabled"
                        )
                        if not isinstance(
                            video_enabled,
                            bool,
                        ):
                            await _send(
                                websocket,
                                {
                                    "type": "error",
                                    "error": (
                                        "invalid_media_state"
                                    ),
                                },
                            )
                            continue

                        relay["video_enabled"] = (
                            video_enabled
                        )
                        state_supplied = True

                    if "audio_enabled" in message:
                        audio_enabled = message.get(
                            "audio_enabled"
                        )
                        if not isinstance(
                            audio_enabled,
                            bool,
                        ):
                            await _send(
                                websocket,
                                {
                                    "type": "error",
                                    "error": (
                                        "invalid_media_state"
                                    ),
                                },
                            )
                            continue

                        relay["audio_enabled"] = (
                            audio_enabled
                        )
                        state_supplied = True

                    # VeriFYD Circle screen share v1.
                    #
                    # The actual display video remains peer-to-peer over the
                    # existing WebRTC connection. The backend relays only the
                    # screen-share UI/control state to the other participant.
                    if "screen_share_enabled" in message:
                        screen_share_enabled = message.get(
                            "screen_share_enabled"
                        )
                        if not isinstance(
                            screen_share_enabled,
                            bool,
                        ):
                            await _send(
                                websocket,
                                {
                                    "type": "error",
                                    "error": (
                                        "invalid_media_state"
                                    ),
                                },
                            )
                            continue

                        relay["screen_share_enabled"] = (
                            screen_share_enabled
                        )
                        state_supplied = True

                    if "screen_share_transport" in message:
                        screen_share_transport = message.get(
                            "screen_share_transport"
                        )
                        if screen_share_transport not in {
                            "webrtc_replace_track_v1",
                            "native_livekit_v1",
                        }:
                            await _send(
                                websocket,
                                {
                                    "type": "error",
                                    "error": (
                                        "invalid_screen_share_transport"
                                    ),
                                },
                            )
                            continue

                        relay["screen_share_transport"] = (
                            screen_share_transport
                        )
                        state_supplied = True
                    if not state_supplied:
                        await _send(
                            websocket,
                            {
                                "type": "error",
                                "error": (
                                    "media_state_required"
                                ),
                            },
                        )
                        continue

                    delivered = (
                        await _send_to_identity(
                            peer_id,
                            relay,
                        )
                    )

                    if delivered == 0:
                        await _send(
                            websocket,
                            {
                                "type": "error",
                                "error": (
                                    "control_peer_unavailable"
                                ),
                            },
                        )

                    continue

                if msg_type == "recording_request":
                    request_error = None

                    async with _state_lock:
                        live_call = (
                            _active_calls.get(
                                call_id
                            )
                        )

                        if (
                            not live_call
                            or live_call.get(
                                "state"
                            )
                            != "answered"
                        ):
                            request_error = (
                                "invalid_call_state"
                            )
                        elif live_call.get(
                            "recording_active_by_id"
                        ):
                            request_error = (
                                "recording_already_active"
                            )
                        else:
                            pending_requester = (
                                live_call.get(
                                    "recording_requester_id"
                                )
                            )

                            if (
                                pending_requester
                                and pending_requester
                                != identity_id
                            ):
                                request_error = (
                                    "recording_request_pending"
                                )
                            else:
                                live_call[
                                    "recording_requester_id"
                                ] = identity_id
                                live_call[
                                    "recording_consent_by_id"
                                ] = None
                                live_call[
                                    "recording_consent_granted"
                                ] = False

                    if request_error:
                        await _send(
                            websocket,
                            {
                                "type": "error",
                                "error": request_error,
                            },
                        )
                        continue

                    await _send_to_identity(
                        peer_id,
                        {
                            "type": "recording_request",
                            "call_id": call_id,
                            "requester": (
                                _public_identity(
                                    identity
                                )
                            ),
                        },
                    )

                    continue

                if msg_type == "recording_response":
                    allowed = message.get(
                        "allowed"
                    )

                    if not isinstance(
                        allowed,
                        bool,
                    ):
                        await _send(
                            websocket,
                            {
                                "type": "error",
                                "error": (
                                    "recording_response_required"
                                ),
                            },
                        )
                        continue

                    response_error = None
                    requester_id = ""

                    async with _state_lock:
                        live_call = (
                            _active_calls.get(
                                call_id
                            )
                        )

                        if (
                            not live_call
                            or live_call.get(
                                "state"
                            )
                            != "answered"
                        ):
                            response_error = (
                                "invalid_call_state"
                            )
                        else:
                            requester_id = str(
                                live_call.get(
                                    "recording_requester_id"
                                )
                                or ""
                            )

                            if (
                                not requester_id
                                or requester_id
                                == identity_id
                            ):
                                response_error = (
                                    "invalid_recording_response"
                                )
                            elif allowed:
                                live_call[
                                    "recording_consent_by_id"
                                ] = identity_id
                                live_call[
                                    "recording_consent_granted"
                                ] = True
                            else:
                                live_call[
                                    "recording_requester_id"
                                ] = None
                                live_call[
                                    "recording_consent_by_id"
                                ] = None
                                live_call[
                                    "recording_consent_granted"
                                ] = False

                    if response_error:
                        await _send(
                            websocket,
                            {
                                "type": "error",
                                "error": response_error,
                            },
                        )
                        continue

                    await _send_to_identity(
                        requester_id,
                        {
                            "type": (
                                "recording_response"
                            ),
                            "call_id": call_id,
                            "allowed": allowed,
                            "responder": (
                                _public_identity(
                                    identity
                                )
                            ),
                        },
                    )

                    continue

                if msg_type == "recording_started":
                    start_error = None

                    async with _state_lock:
                        live_call = (
                            _active_calls.get(
                                call_id
                            )
                        )

                        if (
                            not live_call
                            or live_call.get(
                                "state"
                            )
                            != "answered"
                        ):
                            start_error = (
                                "invalid_call_state"
                            )
                        elif (
                            live_call.get(
                                "recording_requester_id"
                            )
                            != identity_id
                            or not live_call.get(
                                "recording_consent_granted"
                            )
                        ):
                            start_error = (
                                "recording_consent_required"
                            )
                        elif live_call.get(
                            "recording_active_by_id"
                        ):
                            start_error = (
                                "recording_already_active"
                            )
                        else:
                            live_call[
                                "recording_active_by_id"
                            ] = identity_id

                    if start_error:
                        await _send(
                            websocket,
                            {
                                "type": "error",
                                "error": start_error,
                            },
                        )
                        continue

                    await _send_to_identity(
                        peer_id,
                        {
                            "type": "recording_started",
                            "call_id": call_id,
                            "recorder": (
                                _public_identity(
                                    identity
                                )
                            ),
                        },
                    )

                    continue

                if msg_type == "recording_stop":
                    stop_error = None
                    active_recorder_id = ""

                    async with _state_lock:
                        live_call = (
                            _active_calls.get(
                                call_id
                            )
                        )

                        if (
                            not live_call
                            or live_call.get(
                                "state"
                            )
                            != "answered"
                        ):
                            stop_error = (
                                "invalid_call_state"
                            )
                        else:
                            active_recorder_id = str(
                                live_call.get(
                                    "recording_active_by_id"
                                )
                                or ""
                            )

                            if not active_recorder_id:
                                stop_error = (
                                    "recording_not_active"
                                )
                            elif (
                                active_recorder_id
                                == identity_id
                            ):
                                live_call[
                                    "recording_active_by_id"
                                ] = None
                                live_call[
                                    "recording_requester_id"
                                ] = None
                                live_call[
                                    "recording_consent_by_id"
                                ] = None
                                live_call[
                                    "recording_consent_granted"
                                ] = False

                    if stop_error:
                        await _send(
                            websocket,
                            {
                                "type": "error",
                                "error": stop_error,
                            },
                        )
                        continue

                    if (
                        active_recorder_id
                        == identity_id
                    ):
                        await _send_to_identity(
                            peer_id,
                            {
                                "type": (
                                    "recording_stopped"
                                ),
                                "call_id": call_id,
                                "stopped_by": (
                                    _public_identity(
                                        identity
                                    )
                                ),
                            },
                        )
                    else:
                        await _send_to_identity(
                            active_recorder_id,
                            {
                                "type": "recording_stop",
                                "call_id": call_id,
                                "requested_by": (
                                    _public_identity(
                                        identity
                                    )
                                ),
                            },
                        )

                    continue

                if msg_type == "recording_stopped":
                    stopped_error = None

                    async with _state_lock:
                        live_call = (
                            _active_calls.get(
                                call_id
                            )
                        )

                        if (
                            not live_call
                            or live_call.get(
                                "state"
                            )
                            != "answered"
                        ):
                            stopped_error = (
                                "invalid_call_state"
                            )
                        elif (
                            live_call.get(
                                "recording_active_by_id"
                            )
                            != identity_id
                        ):
                            stopped_error = (
                                "not_recording_owner"
                            )
                        else:
                            live_call[
                                "recording_active_by_id"
                            ] = None
                            live_call[
                                "recording_requester_id"
                            ] = None
                            live_call[
                                "recording_consent_by_id"
                            ] = None
                            live_call[
                                "recording_consent_granted"
                            ] = False

                    if stopped_error:
                        await _send(
                            websocket,
                            {
                                "type": "error",
                                "error": stopped_error,
                            },
                        )
                        continue

                    await _send_to_identity(
                        peer_id,
                        {
                            "type": "recording_stopped",
                            "call_id": call_id,
                            "stopped_by": (
                                _public_identity(
                                    identity
                                )
                            ),
                        },
                    )

                    continue

            if msg_type == "call_replay_receipt":
                call_id = message.get("call_id")
                result = message.get("result")
                if not isinstance(call_id, str) or not isinstance(result, str) or result not in {
                    "presented", "already_active", "other_call_active", "native_owns_call", "expired"
                } or call_id not in replayed_call_ids:
                    continue
                replayed_call_ids.remove(call_id)
                log.info("call_trace event=browser_receipt call_id=%s result=%s", call_id, result)
                continue

            if msg_type in {"call_answer", "call_decline", "call_end"}:
                call_id = str(message.get("call_id") or "").strip()
                try:
                    await apply_call_action(call_id, str(identity_id), msg_type, transport="websocket")
                except HTTPException as exc:
                    detail = exc.detail if isinstance(exc.detail, dict) else {"error": str(exc.detail)}
                    await _send(websocket, {"type": "error", "call_id": call_id, **detail})
                continue

            await _send(
                websocket,
                {
                    "type": (
                        "error"
                    ),
                    "error": (
                        "unsupported_message_type"
                    ),
                    "allowed": [
                        "ping",
                        "call_invite",
                        "call_answer",
                        "call_decline",
                        "call_end",
                        "webrtc_offer",
                        "webrtc_answer",
                        "webrtc_ice",
                        "call_media_state",
                        "recording_request",
                        "recording_response",
                        "recording_started",
                        "recording_stop",
                        "recording_stopped",
                    ],
                },
            )

    except WebSocketDisconnect:
        pass

    except Exception:
        log.exception(
            "Trust Voice signaling websocket error"
        )

    finally:
        await _unregister(
            identity_id,
            websocket,
        )
