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

from database import create_otp, get_db, is_valid_email, verify_otp
from emailer import send_otp_email
from trust_voice_permissions import call_permission_decision

log = logging.getLogger("verifyd.trust_voice.signaling")

router = APIRouter(prefix="/trust-voice", tags=["Trust Voice Signaling"])

FEATURE_VERSION = "0.5.0"

# Trust Voice sessions are persistent and server-revocable.
# New sessions do not expire based on elapsed time.
INVITE_TTL_SECONDS = 30
CALL_COOLDOWN_SECONDS = 3
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
        await asyncio.sleep(INVITE_TTL_SECONDS)
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
            row = _persistent_session_row(
                session_id
            )

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
) -> None:
    try:
        await ws.send_json(
            payload
        )
    except Exception:
        pass


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
    # Best-effort realtime notification only.
    # PostgreSQL/HTTP remains authoritative and a delivery
    # failure must never fail the stored message.
    safe_message_type = (
        message_type
        if message_type
        in {
            "text",
            "attachment",
        }
        else "text"
    )

    return await _send_to_identity(
        recipient_identity_id,
        {
            "type": (
                "message_created"
            ),
            "conversation_id": (
                conversation_id
            ),
            "message_id": (
                message_id
            ),
            "message_type": (
                safe_message_type
            ),
            "sender": sender,
            "created_at": (
                created_at
            ),
        },
    )


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

        await _send_to_identity(
            peer_id,
            {
                "type": "call_ended",
                "call_id": call_id,
                "reason": "peer_disconnected",
            },
        )


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

    try:
        token_payload = (
            _verify_token(
                token
            )
        )

        identity = (
            _identity_by_id(
                token_payload[
                    "identity_id"
                ]
            )
        )

        if not identity:
            await websocket.close(
                code=4401
            )
            return

    except HTTPException:
        await websocket.close(
            code=4401
        )
        return

    except Exception:
        await websocket.close(
            code=4401
        )
        return

    identity_id = (
        identity["id"]
    )

    await websocket.accept()

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
        },
    )

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

            msg_type = (
                message.get(
                    "type"
                )
            )

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

                if not await _presence(
                    target["id"]
                ):
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
                    "created_at": (
                        time.time()
                    ),
                }

                async with _state_lock:
                    _active_calls[
                        call_id
                    ] = call

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
                        },
                    )
                )

                if delivered == 0:
                    async with _state_lock:
                        _active_calls.pop(
                            call_id,
                            None,
                        )

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

                _create_call_history(
                    call_id=call_id,
                    caller_identity_id=identity_id,
                    callee_identity_id=str(target["id"]),
                    call_type="audio",
                )
                _schedule_call_expiry(call_id)

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
                    },
                )

                continue

            if msg_type in {
                "webrtc_offer",
                "webrtc_answer",
                "webrtc_ice",
            }:
                if not _webrtc_enabled():
                    await _send(
                        websocket,
                        {
                            "type": (
                                "error"
                            ),
                            "error": (
                                "webrtc_not_enabled"
                            ),
                        },
                    )

                    continue

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
                            "type": (
                                "error"
                            ),
                            "error": (
                                "call_id_required"
                            ),
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
                            "type": (
                                "error"
                            ),
                            "error": (
                                "call_not_found"
                            ),
                        },
                    )

                    continue

                caller_id = (
                    call.get(
                        "caller_id"
                    )
                )

                callee_id = (
                    call.get(
                        "callee_id"
                    )
                )

                if (
                    identity_id
                    not in {
                        caller_id,
                        callee_id,
                    }
                ):
                    await _send(
                        websocket,
                        {
                            "type": (
                                "error"
                            ),
                            "error": (
                                "not_call_participant"
                            ),
                        },
                    )

                    continue

                if (
                    call.get(
                        "state"
                    )
                    != "answered"
                ):
                    await _send(
                        websocket,
                        {
                            "type": (
                                "error"
                            ),
                            "error": (
                                "invalid_call_state"
                            ),
                        },
                    )

                    continue

                if (
                    msg_type
                    == "webrtc_offer"
                    and identity_id
                    != caller_id
                ):
                    await _send(
                        websocket,
                        {
                            "type": (
                                "error"
                            ),
                            "error": (
                                "offer_must_come_from_caller"
                            ),
                        },
                    )

                    continue

                if (
                    msg_type
                    == "webrtc_answer"
                    and identity_id
                    != callee_id
                ):
                    await _send(
                        websocket,
                        {
                            "type": (
                                "error"
                            ),
                            "error": (
                                "answer_must_come_from_callee"
                            ),
                        },
                    )

                    continue

                peer_id = (
                    callee_id
                    if (
                        identity_id
                        == caller_id
                    )
                    else caller_id
                )

                # --------------------------------------------
                # SDP OFFER / ANSWER
                # --------------------------------------------

                if msg_type in {
                    "webrtc_offer",
                    "webrtc_answer",
                }:
                    sdp = (
                        message.get(
                            "sdp"
                        )
                    )

                    expected_type = (
                        "offer"
                        if (
                            msg_type
                            == "webrtc_offer"
                        )
                        else "answer"
                    )

                    if not isinstance(
                        sdp,
                        dict,
                    ):
                        await _send(
                            websocket,
                            {
                                "type": (
                                    "error"
                                ),
                                "error": (
                                    "invalid_webrtc_sdp"
                                ),
                            },
                        )

                        continue

                    sdp_type = str(
                        sdp.get(
                            "type",
                            "",
                        )
                    ).strip().lower()

                    sdp_body = (
                        sdp.get(
                            "sdp"
                        )
                    )

                    if (
                        sdp_type
                        != expected_type
                        or not isinstance(
                            sdp_body,
                            str,
                        )
                        or not sdp_body.strip()
                    ):
                        await _send(
                            websocket,
                            {
                                "type": (
                                    "error"
                                ),
                                "error": (
                                    "invalid_webrtc_sdp"
                                ),
                            },
                        )

                        continue

                    if (
                        _json_size_bytes(
                            sdp
                        )
                        > MAX_WEBRTC_SDP_BYTES
                    ):
                        await _send(
                            websocket,
                            {
                                "type": (
                                    "error"
                                ),
                                "error": (
                                    "webrtc_sdp_too_large"
                                ),
                            },
                        )

                        continue

                    delivered = (
                        await _send_to_identity(
                            peer_id,
                            {
                                "type": (
                                    msg_type
                                ),
                                "call_id": (
                                    call_id
                                ),
                                "sdp": (
                                    sdp
                                ),
                            },
                        )
                    )

                    if delivered == 0:
                        await _send(
                            websocket,
                            {
                                "type": (
                                    "error"
                                ),
                                "error": (
                                    "webrtc_peer_unavailable"
                                ),
                            },
                        )

                    continue

                # --------------------------------------------
                # ICE CANDIDATE
                # --------------------------------------------

                candidate = (
                    message.get(
                        "candidate"
                    )
                )

                if not isinstance(
                    candidate,
                    dict,
                ):
                    await _send(
                        websocket,
                        {
                            "type": (
                                "error"
                            ),
                            "error": (
                                "invalid_webrtc_candidate"
                            ),
                        },
                    )

                    continue

                candidate_line = (
                    candidate.get(
                        "candidate"
                    )
                )

                if not isinstance(
                    candidate_line,
                    str,
                ):
                    await _send(
                        websocket,
                        {
                            "type": (
                                "error"
                            ),
                            "error": (
                                "invalid_webrtc_candidate"
                            ),
                        },
                    )

                    continue

                if (
                    _json_size_bytes(
                        candidate
                    )
                    > MAX_WEBRTC_ICE_BYTES
                ):
                    await _send(
                        websocket,
                        {
                            "type": (
                                "error"
                            ),
                            "error": (
                                "webrtc_candidate_too_large"
                            ),
                        },
                    )

                    continue

                delivered = (
                    await _send_to_identity(
                        peer_id,
                        {
                            "type": (
                                "webrtc_ice"
                            ),
                            "call_id": (
                                call_id
                            ),
                            "candidate": (
                                candidate
                            ),
                        },
                    )
                )

                if delivered == 0:
                    await _send(
                        websocket,
                        {
                            "type": (
                                "error"
                            ),
                            "error": (
                                "webrtc_peer_unavailable"
                            ),
                        },
                    )

                continue

            if msg_type in {
                "call_answer",
                "call_decline",
                "call_end",
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
                            "type": (
                                "error"
                            ),
                            "error": (
                                "call_id_required"
                            ),
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
                            "type": (
                                "error"
                            ),
                            "error": (
                                "call_not_found"
                            ),
                        },
                    )

                    continue

                if (
                    identity_id
                    not in {
                        call.get(
                            "caller_id"
                        ),
                        call.get(
                            "callee_id"
                        ),
                    }
                ):
                    await _send(
                        websocket,
                        {
                            "type": (
                                "error"
                            ),
                            "error": (
                                "not_call_participant"
                            ),
                        },
                    )

                    continue

                caller_id = (
                    call[
                        "caller_id"
                    ]
                )

                callee_id = (
                    call[
                        "callee_id"
                    ]
                )

                if (
                    msg_type
                    == "call_answer"
                ):
                    if (
                        identity_id
                        != callee_id
                        or call.get(
                            "state"
                        )
                        != "ringing"
                    ):
                        await _send(
                            websocket,
                            {
                                "type": (
                                    "error"
                                ),
                                "error": (
                                    "invalid_call_state"
                                ),
                            },
                        )

                        continue

                    async with _state_lock:
                        if (
                            call_id
                            in _active_calls
                        ):
                            _active_calls[
                                call_id
                            ][
                                "state"
                            ] = (
                                "answered"
                            )

                    _cancel_call_expiry(call_id)
                    _record_call_answered(call_id)

                    await _send_to_identity(
                        caller_id,
                        {
                            "type": (
                                "call_answered"
                            ),
                            "call_id": (
                                call_id
                            ),
                        },
                    )

                    await _send_to_identity(
                        callee_id,
                        {
                            "type": (
                                "call_answered"
                            ),
                            "call_id": (
                                call_id
                            ),
                        },
                    )

                    continue

                if (
                    msg_type
                    == "call_decline"
                ):
                    if (
                        identity_id
                        != callee_id
                        or call.get(
                            "state"
                        )
                        != "ringing"
                    ):
                        await _send(
                            websocket,
                            {
                                "type": (
                                    "error"
                                ),
                                "error": (
                                    "invalid_call_state"
                                ),
                            },
                        )

                        continue

                    async with _state_lock:
                        _active_calls.pop(
                            call_id,
                            None,
                        )

                    _cancel_call_expiry(call_id)
                    _record_call_terminal(
                        call_id,
                        "declined",
                        "declined",
                    )

                    await _send_to_identity(
                        caller_id,
                        {
                            "type": (
                                "call_declined"
                            ),
                            "call_id": (
                                call_id
                            ),
                        },
                    )

                    await _send_to_identity(
                        callee_id,
                        {
                            "type": (
                                "call_declined"
                            ),
                            "call_id": (
                                call_id
                            ),
                        },
                    )

                    continue

                if (
                    msg_type
                    == "call_end"
                ):
                    async with _state_lock:
                        _active_calls.pop(
                            call_id,
                            None,
                        )

                    _cancel_call_expiry(call_id)

                    if call.get("state") == "ringing":
                        _record_call_terminal(
                            call_id,
                            "canceled",
                            "ended_before_answer",
                        )
                    else:
                        _record_call_terminal(
                            call_id,
                            "ended",
                            "ended",
                        )

                    await _send_to_identity(
                        caller_id,
                        {
                            "type": (
                                "call_ended"
                            ),
                            "call_id": (
                                call_id
                            ),
                            "reason": (
                                "ended"
                            ),
                        },
                    )

                    await _send_to_identity(
                        callee_id,
                        {
                            "type": (
                                "call_ended"
                            ),
                            "call_id": (
                                call_id
                            ),
                            "reason": (
                                "ended"
                            ),
                        },
                    )

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
