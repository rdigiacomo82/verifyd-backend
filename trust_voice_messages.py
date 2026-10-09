# ============================================================
# VeriFYD Trust Voice — Direct Messaging Foundation
# VERIFYD_TRUST_VOICE_MESSAGES_V1
# VERIFYD_TRUST_VOICE_MESSAGE_NOTIFY_V1
#
# Authenticated 1:1 Trust Voice messaging:
#   - isolated messaging tables
#   - message privacy settings
#   - direct conversations between Trust Voice identities
#   - durable text and attachment messages
#   - lightweight conversation summaries
#   - cursor-paginated message history
#   - batched sender + attachment metadata hydration
#   - mark-read support
#   - best-effort WebSocket message notifications
#
# IMPORTANT:
#   - Disabled unless VERIFYD_TRUST_VOICE_MESSAGES_ENABLED=1
#   - Uses the existing Trust Voice bearer/session token
#   - Reuses Trust Circle membership for "trust_circle" privacy
#   - Does NOT expose email addresses
#   - Does NOT fetch attachment file bytes from R2 while listing messages
# ============================================================

from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone
from typing import Literal, Optional, Tuple

from fastapi import APIRouter, BackgroundTasks, Header, HTTPException, Query
from pydantic import BaseModel, Field

from database import get_db
from trust_voice_permissions import trust_circle_contains
from trust_voice_signaling import (
    _identity_by_handle,
    _identity_by_id,
    _normalize_handle,
    _public_identity,
    _verify_token,
    notify_message_created,
)


router = APIRouter(
    prefix="/trust-voice",
    tags=["Trust Voice Messages"],
)

FEATURE_VERSION = "0.3.0"

ALLOWED_MESSAGE_PRIVACY = {
    "anyone",
    "verified_users",
    "trust_circle",
    "nobody",
}

DEFAULT_MESSAGE_PRIVACY = "trust_circle"
MAX_MESSAGE_LENGTH = 4000
_message_schema_ready = False


class DirectConversationCreate(BaseModel):
    target_handle: str = Field(min_length=3, max_length=25)


class TextMessageCreate(BaseModel):
    body: str = Field(min_length=1, max_length=MAX_MESSAGE_LENGTH)


class MessageSettingsUpdate(BaseModel):
    message_privacy: Literal[
        "anyone",
        "verified_users",
        "trust_circle",
        "nobody",
    ]


def _enabled() -> bool:
    value = (
        os.environ.get(
            "VERIFYD_TRUST_VOICE_MESSAGES_ENABLED",
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


def _require_enabled() -> None:
    if not _enabled():
        raise HTTPException(
            status_code=404,
            detail={"error": "not_found"},
        )


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_iso(value: Optional[str]) -> Optional[datetime]:
    raw = str(value or "").strip()
    if not raw:
        return None

    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except Exception:
        return None


def _validate_before_cursor(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None

    raw = str(value).strip()
    if not raw:
        return None

    if not _parse_iso(raw):
        raise HTTPException(
            status_code=400,
            detail={
                "error": "invalid_message_cursor",
                "message": "The message pagination cursor is invalid.",
            },
        )

    return raw


def _timestamp_gte(left: Optional[str], right: Optional[str]) -> bool:
    left_dt = _parse_iso(left)
    right_dt = _parse_iso(right)
    return bool(left_dt and right_dt and left_dt >= right_dt)


def _clean_message_body(value: str) -> str:
    body = (value or "").replace("\r\n", "\n").replace("\r", "\n").strip()

    if not body:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "message_required",
                "message": "Enter a message before sending.",
            },
        )

    if len(body) > MAX_MESSAGE_LENGTH:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "message_too_long",
                "message": f"Messages may contain up to {MAX_MESSAGE_LENGTH} characters.",
            },
        )

    return body


def _identity_from_bearer(
    authorization: Optional[str],
) -> dict:
    if (
        not authorization
        or not authorization.lower().startswith("bearer ")
    ):
        raise HTTPException(
            status_code=401,
            detail={"error": "session_required"},
        )

    token = authorization.split(" ", 1)[1].strip()

    if not token:
        raise HTTPException(
            status_code=401,
            detail={"error": "session_required"},
        )

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


def ensure_message_schema() -> None:
    """
    Create only the isolated Trust Voice messaging tables and indexes.

    Safe and idempotent. The DDL is performed once per application process
    instead of on every message request.
    """

    global _message_schema_ready

    if _message_schema_ready:
        return

    with get_db() as conn:
        cur = conn.cursor()

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS verifyd_voice_message_settings (
                identity_id      TEXT PRIMARY KEY,
                message_privacy  TEXT NOT NULL DEFAULT 'trust_circle',
                created_at       TEXT NOT NULL,
                updated_at       TEXT NOT NULL
            )
            """
        )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS verifyd_voice_conversations (
                conversation_id   TEXT PRIMARY KEY,
                conversation_type TEXT NOT NULL DEFAULT 'direct',
                direct_key        TEXT UNIQUE NOT NULL,
                created_at        TEXT NOT NULL,
                updated_at        TEXT NOT NULL
            )
            """
        )

        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_verifyd_voice_conversations_updated
            ON verifyd_voice_conversations (updated_at)
            """
        )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS verifyd_voice_conversation_members (
                conversation_id TEXT NOT NULL,
                identity_id     TEXT NOT NULL,
                joined_at       TEXT NOT NULL,
                last_read_at    TEXT,
                PRIMARY KEY (
                    conversation_id,
                    identity_id
                )
            )
            """
        )

        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_verifyd_voice_members_identity
            ON verifyd_voice_conversation_members (
                identity_id,
                conversation_id
            )
            """
        )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS verifyd_voice_messages (
                message_id          TEXT PRIMARY KEY,
                conversation_id     TEXT NOT NULL,
                sender_identity_id  TEXT NOT NULL,
                message_type        TEXT NOT NULL DEFAULT 'text',
                body                TEXT NOT NULL DEFAULT '',
                created_at          TEXT NOT NULL,
                edited_at           TEXT,
                deleted_at          TEXT
            )
            """
        )

        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_verifyd_voice_messages_conversation
            ON verifyd_voice_messages (
                conversation_id,
                created_at
            )
            """
        )

        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_verifyd_voice_messages_sender
            ON verifyd_voice_messages (
                sender_identity_id,
                created_at
            )
            """
        )

        # Performance indexes for active conversation history/unread lookups.
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS
            idx_verifyd_voice_messages_active_created
            ON verifyd_voice_messages (
                conversation_id,
                created_at DESC
            )
            WHERE deleted_at IS NULL
            """
        )

        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS
            idx_verifyd_voice_messages_active_sender_created
            ON verifyd_voice_messages (
                conversation_id,
                sender_identity_id,
                created_at DESC
            )
            WHERE deleted_at IS NULL
            """
        )

    _message_schema_ready = True

def _direct_key(
    identity_a: str,
    identity_b: str,
) -> str:
    ids = sorted([
        str(identity_a or ""),
        str(identity_b or ""),
    ])
    return "::".join(ids)


def _settings_for_identity(
    identity_id: str,
) -> dict:
    ensure_message_schema()

    now = _now_iso()

    with get_db() as conn:
        cur = conn.cursor()

        cur.execute(
            """
            INSERT INTO verifyd_voice_message_settings (
                identity_id,
                message_privacy,
                created_at,
                updated_at
            )
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (identity_id) DO NOTHING
            """,
            (
                identity_id,
                DEFAULT_MESSAGE_PRIVACY,
                now,
                now,
            ),
        )

        cur.execute(
            """
            SELECT *
            FROM verifyd_voice_message_settings
            WHERE identity_id = %s
            LIMIT 1
            """,
            (identity_id,),
        )

        row = cur.fetchone()

    return dict(row) if row else {
        "identity_id": identity_id,
        "message_privacy": DEFAULT_MESSAGE_PRIVACY,
        "created_at": now,
        "updated_at": now,
    }


def _message_permission_decision(
    sender: dict,
    recipient: dict,
) -> Tuple[bool, str]:
    """
    Server-side permission decision for sending a new Trust Voice message.

    The detailed reason is for backend diagnostics only. Caller-facing API
    responses intentionally use the generic "not_allowed" reason so the
    recipient's exact privacy setting is not exposed.
    """

    settings = _settings_for_identity(
        str(recipient.get("id") or "")
    )

    privacy = (
        settings.get("message_privacy")
        or DEFAULT_MESSAGE_PRIVACY
    ).strip().lower()

    if privacy not in ALLOWED_MESSAGE_PRIVACY:
        privacy = DEFAULT_MESSAGE_PRIVACY

    if privacy == "anyone":
        return True, "allowed_anyone"

    if privacy == "nobody":
        return False, "blocked_by_privacy"

    if privacy == "verified_users":
        allowed = bool(sender.get("email_verified"))
        return allowed, "verified_user_required"

    if privacy == "trust_circle":
        allowed = trust_circle_contains(
            str(recipient.get("id") or ""),
            str(sender.get("id") or ""),
        )
        return allowed, "trust_circle_required"

    return False, "blocked_by_privacy"


def _require_message_allowed(
    sender: dict,
    recipient: dict,
) -> None:
    allowed, _reason = _message_permission_decision(
        sender=sender,
        recipient=recipient,
    )

    if not allowed:
        raise HTTPException(
            status_code=403,
            detail={
                "error": "not_allowed",
                "message": "This person is not currently accepting messages from your account.",
            },
        )


def _conversation_row(
    conversation_id: str,
) -> Optional[dict]:
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT *
            FROM verifyd_voice_conversations
            WHERE conversation_id = %s
            LIMIT 1
            """,
            (conversation_id,),
        )
        row = cur.fetchone()

    return dict(row) if row else None


def _conversation_members(
    conversation_id: str,
) -> list[dict]:
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT *
            FROM verifyd_voice_conversation_members
            WHERE conversation_id = %s
            ORDER BY joined_at ASC
            """,
            (conversation_id,),
        )
        rows = cur.fetchall() or []

    return [dict(row) for row in rows]


def _require_conversation_member(
    conversation_id: str,
    identity_id: str,
) -> dict:
    ensure_message_schema()

    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT *
            FROM verifyd_voice_conversation_members
            WHERE conversation_id = %s
              AND identity_id = %s
            LIMIT 1
            """,
            (
                conversation_id,
                identity_id,
            ),
        )
        row = cur.fetchone()

    if not row:
        raise HTTPException(
            status_code=404,
            detail={"error": "conversation_not_found"},
        )

    return dict(row)


def _other_direct_member(
    conversation_id: str,
    identity_id: str,
) -> Tuple[dict, dict]:
    members = _conversation_members(conversation_id)

    if len(members) != 2:
        raise HTTPException(
            status_code=409,
            detail={"error": "invalid_direct_conversation"},
        )

    other_member = next(
        (
            member
            for member in members
            if member.get("identity_id") != identity_id
        ),
        None,
    )

    if not other_member:
        raise HTTPException(
            status_code=409,
            detail={"error": "invalid_direct_conversation"},
        )

    other_identity = _identity_by_id(
        str(other_member.get("identity_id") or "")
    )

    if not other_identity:
        raise HTTPException(
            status_code=404,
            detail={"error": "identity_not_found"},
        )

    return other_member, other_identity


def _public_contact(identity: dict) -> dict:
    public = _public_identity(identity)
    public["profile_media_url"] = (
        f"/trust-voice/profile-media/{identity['id']}"
    )
    return public


def _last_message(
    conversation_id: str,
) -> Optional[dict]:
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT *
            FROM verifyd_voice_messages
            WHERE conversation_id = %s
              AND deleted_at IS NULL
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (conversation_id,),
        )
        row = cur.fetchone()

    return dict(row) if row else None


def _unread_count(
    conversation_id: str,
    identity_id: str,
    last_read_at: Optional[str],
) -> int:
    with get_db() as conn:
        cur = conn.cursor()

        if last_read_at:
            cur.execute(
                """
                SELECT COUNT(*) AS unread_count
                FROM verifyd_voice_messages
                WHERE conversation_id = %s
                  AND sender_identity_id <> %s
                  AND deleted_at IS NULL
                  AND created_at::timestamptz > %s::timestamptz
                """,
                (
                    conversation_id,
                    identity_id,
                    last_read_at,
                ),
            )
        else:
            cur.execute(
                """
                SELECT COUNT(*) AS unread_count
                FROM verifyd_voice_messages
                WHERE conversation_id = %s
                  AND sender_identity_id <> %s
                  AND deleted_at IS NULL
                """,
                (
                    conversation_id,
                    identity_id,
                ),
            )

        row = cur.fetchone() or {}

    return int(row.get("unread_count") or 0)


def _attachments_enabled() -> bool:
    value = (
        os.environ.get(
            "VERIFYD_TRUST_VOICE_ATTACHMENTS_ENABLED",
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


def _public_attachment_metadata(
    row: dict,
) -> dict:
    """
    Public, lightweight attachment metadata only.

    No R2 storage key or file bytes are returned here.
    """
    attachment_id = str(
        row.get("attachment_id")
        or ""
    )

    return {
        "attachment_id": attachment_id,
        "message_id": row.get("message_id"),
        "conversation_id": row.get("conversation_id"),
        "filename": row.get("filename") or "",
        "extension": row.get("extension") or "",
        "media_category": row.get("media_category") or "unknown",
        "content_type": (
            row.get("content_type")
            or "application/octet-stream"
        ),
        "size_bytes": int(
            row.get("size_bytes")
            or 0
        ),
        "sha256": row.get("sha256") or "",
        "analysis_status": (
            row.get("analysis_status")
            or "not_started"
        ),
        "malware_status": (
            row.get("malware_status")
            or "not_scanned"
        ),
        "download_url": (
            f"/trust-voice/attachments/"
            f"{attachment_id}/download"
        ),
        "created_at": row.get("created_at") or "",
    }


def _attachment_metadata_by_message_ids(
    message_ids: list[str],
) -> dict[str, dict]:
    """
    Batch-fetch attachment metadata for one message page.

    The actual private R2 object is never retrieved by this helper.
    """
    if (
        not message_ids
        or not _attachments_enabled()
    ):
        return {}

    with get_db() as conn:
        cur = conn.cursor()

        # Attachment transport may be enabled before its schema has been
        # initialized. Avoid turning message history into a server error.
        cur.execute(
            """
            SELECT to_regclass(
                'verifyd_voice_attachments'
            ) AS table_name
            """
        )
        table_check = cur.fetchone() or {}

        if not table_check.get("table_name"):
            return {}

        cur.execute(
            """
            SELECT
                attachment_id,
                message_id,
                conversation_id,
                filename,
                extension,
                media_category,
                content_type,
                size_bytes,
                sha256,
                analysis_status,
                malware_status,
                created_at
            FROM verifyd_voice_attachments
            WHERE message_id = ANY(%s)
            """,
            (message_ids,),
        )

        rows = cur.fetchall() or []

    return {
        str(row["message_id"]): _public_attachment_metadata(
            dict(row)
        )
        for row in rows
    }


def _message_payload_fast(
    row: dict,
    viewer_identity: dict,
    other_identity: dict,
    other_last_read_at: Optional[str] = None,
    attachment: Optional[dict] = None,
) -> dict:
    """
    Serialize one direct-conversation message without querying the sender.

    Both possible sender identities are already known for a 1:1 conversation,
    eliminating the old identity-query-per-message pattern.
    """
    viewer_identity_id = str(
        viewer_identity.get("id")
        or ""
    )

    sender_identity_id = str(
        row.get("sender_identity_id")
        or ""
    )

    is_mine = (
        sender_identity_id
        == viewer_identity_id
    )

    sender_identity = (
        viewer_identity
        if is_mine
        else other_identity
    )

    sender_public = (
        _public_contact(sender_identity)
        if sender_identity
        else {
            "identity_id": sender_identity_id,
            "handle": "",
            "display_name": "",
            "display_emoji": "",
            "display_label": "",
            "verification": {
                "email_verified": False,
                "identity_verified": False,
                "organization_verified": False,
                "level": "unknown",
            },
            "profile_media_url": "",
        }
    )

    read_by_recipient = None

    if is_mine:
        created_at = str(
            row.get("created_at")
            or ""
        )

        read_by_recipient = _timestamp_gte(
            other_last_read_at,
            created_at,
        )

    payload = {
        "message_id": row.get("message_id"),
        "conversation_id": row.get("conversation_id"),
        "message_type": row.get("message_type") or "text",
        "body": row.get("body") or "",
        "sender": sender_public,
        "is_mine": is_mine,
        "read_by_recipient": read_by_recipient,
        "created_at": row.get("created_at") or "",
        "edited_at": row.get("edited_at"),
    }

    if attachment:
        payload["attachment"] = attachment

    return payload


def _message_payload(
    row: dict,
    viewer_identity_id: str,
    other_last_read_at: Optional[str] = None,
) -> dict:
    sender = _identity_by_id(
        str(row.get("sender_identity_id") or "")
    )

    sender_public = (
        _public_contact(sender)
        if sender
        else {
            "identity_id": row.get("sender_identity_id"),
            "handle": "",
            "display_name": "",
            "display_emoji": "",
            "display_label": "",
            "verification": {
                "email_verified": False,
                "identity_verified": False,
                "organization_verified": False,
                "level": "unknown",
            },
            "profile_media_url": "",
        }
    )

    is_mine = (
        str(row.get("sender_identity_id") or "")
        == str(viewer_identity_id or "")
    )

    read_by_recipient = None
    if is_mine:
        created_at = str(row.get("created_at") or "")
        read_by_recipient = _timestamp_gte(
            other_last_read_at,
            created_at,
        )

    return {
        "message_id": row.get("message_id"),
        "conversation_id": row.get("conversation_id"),
        "message_type": row.get("message_type") or "text",
        "body": row.get("body") or "",
        "sender": sender_public,
        "is_mine": is_mine,
        "read_by_recipient": read_by_recipient,
        "created_at": row.get("created_at") or "",
        "edited_at": row.get("edited_at"),
    }


def _conversation_payload(
    conversation: dict,
    viewer_identity_id: str,
) -> dict:
    viewer_member = _require_conversation_member(
        str(conversation.get("conversation_id") or ""),
        viewer_identity_id,
    )

    other_member, other_identity = _other_direct_member(
        str(conversation.get("conversation_id") or ""),
        viewer_identity_id,
    )

    last_message = _last_message(
        str(conversation.get("conversation_id") or "")
    )

    return {
        "conversation_id": conversation.get("conversation_id"),
        "conversation_type": conversation.get("conversation_type") or "direct",
        "other": _public_contact(other_identity),
        "unread_count": _unread_count(
            str(conversation.get("conversation_id") or ""),
            viewer_identity_id,
            viewer_member.get("last_read_at"),
        ),
        "last_message": (
            _message_payload(
                last_message,
                viewer_identity_id,
                other_member.get("last_read_at"),
            )
            if last_message
            else None
        ),
        "created_at": conversation.get("created_at") or "",
        "updated_at": conversation.get("updated_at") or "",
    }


@router.get("/messages/health")
def messages_health():
    attachments_enabled = _attachments_enabled()

    return {
        "status": "ok",
        "feature": "trust_voice_messages",
        "version": FEATURE_VERSION,
        "enabled": _enabled(),
        "message_types": (
            ["text", "attachment"]
            if attachments_enabled
            else ["text"]
        ),
        "attachments": (
            "private_transport_beta"
            if attachments_enabled
            else "not_enabled"
        ),
        "realtime_notifications": "websocket_best_effort_beta",
        "conversation_summaries": "batched_v1",
        "message_pagination": "cursor_v1",
        "attachment_hydration": "inline_metadata_v1",
        "message_sender_hydration": "two_party_cached_v1",
        "schema_init": "once_per_process_v1",
    }


@router.get("/message-settings")
def get_message_settings(
    authorization: Optional[str] = Header(default=None),
):
    _require_enabled()

    identity = _identity_from_bearer(authorization)
    settings = _settings_for_identity(identity["id"])

    return {
        "message_privacy": (
            settings.get("message_privacy")
            or DEFAULT_MESSAGE_PRIVACY
        ),
        "options": [
            {
                "value": "anyone",
                "label": "Anyone",
            },
            {
                "value": "verified_users",
                "label": "Email-verified VeriFYD users",
            },
            {
                "value": "trust_circle",
                "label": "My Trust Circle",
            },
            {
                "value": "nobody",
                "label": "Nobody",
            },
        ],
    }


@router.put("/message-settings")
def update_message_settings(
    payload: MessageSettingsUpdate,
    authorization: Optional[str] = Header(default=None),
):
    _require_enabled()

    identity = _identity_from_bearer(authorization)
    ensure_message_schema()

    now = _now_iso()

    with get_db() as conn:
        cur = conn.cursor()

        cur.execute(
            """
            INSERT INTO verifyd_voice_message_settings (
                identity_id,
                message_privacy,
                created_at,
                updated_at
            )
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (identity_id)
            DO UPDATE SET
                message_privacy = EXCLUDED.message_privacy,
                updated_at = EXCLUDED.updated_at
            """,
            (
                identity["id"],
                payload.message_privacy,
                now,
                now,
            ),
        )

    return {
        "ok": True,
        "message_privacy": payload.message_privacy,
    }


@router.post("/conversations/direct")
def open_direct_conversation(
    payload: DirectConversationCreate,
    authorization: Optional[str] = Header(default=None),
):
    _require_enabled()

    identity = _identity_from_bearer(authorization)
    ensure_message_schema()

    target_handle = _normalize_handle(payload.target_handle)

    if not target_handle:
        raise HTTPException(
            status_code=400,
            detail={"error": "target_handle_required"},
        )

    if target_handle == str(identity.get("handle_lower") or "").lower():
        raise HTTPException(
            status_code=400,
            detail={"error": "cannot_message_self"},
        )

    target = _identity_by_handle(target_handle)

    if not target:
        raise HTTPException(
            status_code=404,
            detail={"error": "handle_not_found"},
        )

    direct_key = _direct_key(
        identity["id"],
        target["id"],
    )

    with get_db() as conn:
        cur = conn.cursor()

        cur.execute(
            """
            SELECT *
            FROM verifyd_voice_conversations
            WHERE direct_key = %s
            LIMIT 1
            """,
            (direct_key,),
        )
        existing = cur.fetchone()

        # Existing participants may always reopen/read their existing
        # conversation history. Current recipient privacy is enforced when
        # sending a new message. Privacy is checked here only before the
        # first conversation is created.
        if not existing:
            _require_message_allowed(
                sender=identity,
                recipient=target,
            )

            now = _now_iso()
            conversation_id = "tvconv_" + uuid.uuid4().hex

            cur.execute(
                """
                INSERT INTO verifyd_voice_conversations (
                    conversation_id,
                    conversation_type,
                    direct_key,
                    created_at,
                    updated_at
                )
                VALUES (%s, 'direct', %s, %s, %s)
                ON CONFLICT (direct_key) DO NOTHING
                """,
                (
                    conversation_id,
                    direct_key,
                    now,
                    now,
                ),
            )

            cur.execute(
                """
                SELECT *
                FROM verifyd_voice_conversations
                WHERE direct_key = %s
                LIMIT 1
                """,
                (direct_key,),
            )
            existing = cur.fetchone()

        if not existing:
            raise HTTPException(
                status_code=500,
                detail={"error": "conversation_create_failed"},
            )

        conversation = dict(existing)
        actual_conversation_id = conversation["conversation_id"]
        joined_at = _now_iso()

        # Repair-safe/idempotent membership creation. This also protects
        # against a partially-created direct conversation.
        for member_identity_id in (
            identity["id"],
            target["id"],
        ):
            cur.execute(
                """
                INSERT INTO verifyd_voice_conversation_members (
                    conversation_id,
                    identity_id,
                    joined_at,
                    last_read_at
                )
                VALUES (%s, %s, %s, NULL)
                ON CONFLICT (
                    conversation_id,
                    identity_id
                ) DO NOTHING
                """,
                (
                    actual_conversation_id,
                    member_identity_id,
                    joined_at,
                ),
            )

    return {
        "ok": True,
        "conversation": _conversation_payload(
            conversation,
            identity["id"],
        ),
    }


@router.get("/conversations")
def list_conversations(
    authorization: Optional[str] = Header(default=None),
):
    """
    Lightweight inbox/conversation summaries.

    Performance v0.3:
      - fixed number of database queries instead of N+1 hydration
      - empty conversations are excluded from the Messages inbox
      - unread totals are calculated in one grouped query
      - only latest attachment metadata is hydrated
    """
    _require_enabled()

    identity = _identity_from_bearer(authorization)
    ensure_message_schema()

    viewer_identity_id = str(identity["id"])

    with get_db() as conn:
        cur = conn.cursor()

        # Conversations for this viewer that actually contain at least
        # one non-deleted message. This prevents empty Trust Circle
        # conversations from appearing as "No messages yet".
        cur.execute(
            """
            SELECT
                c.*,
                vm.last_read_at AS viewer_last_read_at
            FROM verifyd_voice_conversations c
            INNER JOIN verifyd_voice_conversation_members vm
                ON vm.conversation_id = c.conversation_id
            WHERE vm.identity_id = %s
              AND EXISTS (
                  SELECT 1
                  FROM verifyd_voice_messages msg
                  WHERE msg.conversation_id = c.conversation_id
                    AND msg.deleted_at IS NULL
              )
            ORDER BY c.updated_at DESC
            """,
            (viewer_identity_id,),
        )

        conversation_rows = [
            dict(row)
            for row in (cur.fetchall() or [])
        ]

        if not conversation_rows:
            return {
                "count": 0,
                "total_unread": 0,
                "unread_conversation_count": 0,
                "conversations": [],
            }

        conversation_ids = [
            str(row["conversation_id"])
            for row in conversation_rows
        ]

        # Fetch every member for all returned conversations in one query.
        cur.execute(
            """
            SELECT
                conversation_id,
                identity_id,
                joined_at,
                last_read_at
            FROM verifyd_voice_conversation_members
            WHERE conversation_id = ANY(%s)
            """,
            (conversation_ids,),
        )

        member_rows = [
            dict(row)
            for row in (cur.fetchall() or [])
        ]

        other_member_by_conversation = {}

        for member in member_rows:
            if (
                str(member.get("identity_id") or "")
                != viewer_identity_id
            ):
                other_member_by_conversation[
                    str(member["conversation_id"])
                ] = member

        other_identity_ids = list({
            str(member.get("identity_id") or "")
            for member in other_member_by_conversation.values()
            if member.get("identity_id")
        })

        # Fetch all other participant identities in one query.
        identity_by_id = {}

        if other_identity_ids:
            cur.execute(
                """
                SELECT *
                FROM verifyd_identities
                WHERE id = ANY(%s)
                  AND status = 'active'
                """,
                (other_identity_ids,),
            )

            identity_by_id = {
                str(row["id"]): dict(row)
                for row in (cur.fetchall() or [])
            }

        # Fetch the latest message for every conversation in one query.
        cur.execute(
            """
            SELECT DISTINCT ON (conversation_id)
                *
            FROM verifyd_voice_messages
            WHERE conversation_id = ANY(%s)
              AND deleted_at IS NULL
            ORDER BY
                conversation_id,
                created_at DESC
            """,
            (conversation_ids,),
        )

        latest_by_conversation = {
            str(row["conversation_id"]): dict(row)
            for row in (cur.fetchall() or [])
        }

        # Calculate unread counts for all conversations in one query.
        cur.execute(
            """
            SELECT
                msg.conversation_id,
                COUNT(*) AS unread_count
            FROM verifyd_voice_messages msg
            INNER JOIN verifyd_voice_conversation_members vm
                ON vm.conversation_id = msg.conversation_id
               AND vm.identity_id = %s
            WHERE msg.conversation_id = ANY(%s)
              AND msg.sender_identity_id <> %s
              AND msg.deleted_at IS NULL
              AND (
                  vm.last_read_at IS NULL
                  OR msg.created_at::timestamptz >
                     vm.last_read_at::timestamptz
              )
            GROUP BY msg.conversation_id
            """,
            (
                viewer_identity_id,
                conversation_ids,
                viewer_identity_id,
            ),
        )

        unread_by_conversation = {
            str(row["conversation_id"]): int(
                row.get("unread_count")
                or 0
            )
            for row in (cur.fetchall() or [])
        }

    # Only latest-message attachment metadata is needed for the inbox.
    latest_message_ids = [
        str(row["message_id"])
        for row in latest_by_conversation.values()
        if row.get("message_id")
    ]

    attachment_by_message_id = (
        _attachment_metadata_by_message_ids(
            latest_message_ids
        )
    )

    conversations = []

    for conversation in conversation_rows:
        conversation_id = str(
            conversation["conversation_id"]
        )

        other_member = (
            other_member_by_conversation.get(
                conversation_id
            )
        )

        if not other_member:
            continue

        other_identity = identity_by_id.get(
            str(
                other_member.get("identity_id")
                or ""
            )
        )

        if not other_identity:
            continue

        latest_message = (
            latest_by_conversation.get(
                conversation_id
            )
        )

        unread_count = (
            unread_by_conversation.get(
                conversation_id,
                0,
            )
        )

        last_message_payload = None

        if latest_message:
            message_id = str(
                latest_message.get("message_id")
                or ""
            )

            last_message_payload = (
                _message_payload_fast(
                    latest_message,
                    identity,
                    other_identity,
                    other_member.get("last_read_at"),
                    attachment=(
                        attachment_by_message_id.get(
                            message_id
                        )
                    ),
                )
            )

        conversations.append(
            {
                "conversation_id": conversation_id,
                "conversation_type": (
                    conversation.get(
                        "conversation_type"
                    )
                    or "direct"
                ),
                "other": _public_contact(
                    other_identity
                ),
                "unread_count": unread_count,
                "last_message": last_message_payload,
                "created_at": (
                    conversation.get("created_at")
                    or ""
                ),
                "updated_at": (
                    conversation.get("updated_at")
                    or ""
                ),
            }
        )

    total_unread = sum(
        int(
            conversation.get("unread_count")
            or 0
        )
        for conversation in conversations
    )

    unread_conversation_count = sum(
        1
        for conversation in conversations
        if int(
            conversation.get("unread_count")
            or 0
        ) > 0
    )

    return {
        "count": len(conversations),
        "total_unread": total_unread,
        "unread_conversation_count": (
            unread_conversation_count
        ),
        "conversations": conversations,
    }


@router.get("/conversations/{conversation_id}")
def get_conversation(
    conversation_id: str,
    authorization: Optional[str] = Header(default=None),
):
    _require_enabled()

    identity = _identity_from_bearer(authorization)
    ensure_message_schema()

    _require_conversation_member(
        conversation_id,
        identity["id"],
    )

    conversation = _conversation_row(conversation_id)

    if not conversation:
        raise HTTPException(
            status_code=404,
            detail={"error": "conversation_not_found"},
        )

    return {
        "conversation": _conversation_payload(
            conversation,
            identity["id"],
        ),
    }


@router.get("/conversations/{conversation_id}/messages")
def list_messages(
    conversation_id: str,
    authorization: Optional[str] = Header(default=None),
    limit: int = Query(default=50, ge=1, le=100),
    before: Optional[str] = Query(default=None, max_length=80),
):
    """
    Cursor-paginated conversation history.

    Performance v0.3:
      - no identity query per message
      - attachment metadata is batch-hydrated for the page
      - one extra row is fetched to calculate has_more accurately
      - actual R2 file bytes are never fetched here
    """
    _require_enabled()

    identity = _identity_from_bearer(authorization)
    ensure_message_schema()

    _require_conversation_member(
        conversation_id,
        identity["id"],
    )

    other_member, other_identity = _other_direct_member(
        conversation_id,
        identity["id"],
    )

    before = _validate_before_cursor(before)
    fetch_limit = limit + 1

    with get_db() as conn:
        cur = conn.cursor()

        if before:
            cur.execute(
                """
                SELECT *
                FROM verifyd_voice_messages
                WHERE conversation_id = %s
                  AND deleted_at IS NULL
                  AND created_at::timestamptz < %s::timestamptz
                ORDER BY created_at DESC
                LIMIT %s
                """,
                (
                    conversation_id,
                    before,
                    fetch_limit,
                ),
            )
        else:
            cur.execute(
                """
                SELECT *
                FROM verifyd_voice_messages
                WHERE conversation_id = %s
                  AND deleted_at IS NULL
                ORDER BY created_at DESC
                LIMIT %s
                """,
                (
                    conversation_id,
                    fetch_limit,
                ),
            )

        rows = [
            dict(row)
            for row in (cur.fetchall() or [])
        ]

    has_more = len(rows) > limit
    page_rows = rows[:limit]

    message_ids = [
        str(row["message_id"])
        for row in page_rows
        if row.get("message_id")
    ]

    attachment_by_message_id = (
        _attachment_metadata_by_message_ids(
            message_ids
        )
    )

    messages_desc = []

    for row in page_rows:
        message_id = str(
            row.get("message_id")
            or ""
        )

        messages_desc.append(
            _message_payload_fast(
                row,
                identity,
                other_identity,
                other_member.get("last_read_at"),
                attachment=(
                    attachment_by_message_id.get(
                        message_id
                    )
                ),
            )
        )

    # Preserve the historical response order for frontend compatibility.
    # page_rows is newest -> oldest; the response remains oldest -> newest.
    messages = list(
        reversed(messages_desc)
    )

    next_before = None

    if has_more and page_rows:
        # The final row in the DESC query is the oldest row in this page
        # and therefore becomes the cursor for the next older page.
        next_before = str(
            page_rows[-1].get("created_at")
            or ""
        )

    return {
        "conversation_id": conversation_id,
        "count": len(messages),
        "messages": messages,
        "has_more": has_more,
        "next_before": next_before,
        "page_size": limit,
    }


@router.post("/conversations/{conversation_id}/messages")
def send_text_message(
    conversation_id: str,
    payload: TextMessageCreate,
    background_tasks: BackgroundTasks,
    authorization: Optional[str] = Header(default=None),
):
    _require_enabled()

    identity = _identity_from_bearer(authorization)
    ensure_message_schema()

    _require_conversation_member(
        conversation_id,
        identity["id"],
    )

    _other_member, other_identity = _other_direct_member(
        conversation_id,
        identity["id"],
    )

    _require_message_allowed(
        sender=identity,
        recipient=other_identity,
    )

    body = _clean_message_body(payload.body)
    message_id = "tvmsg_" + uuid.uuid4().hex
    now = _now_iso()

    with get_db() as conn:
        cur = conn.cursor()

        cur.execute(
            """
            INSERT INTO verifyd_voice_messages (
                message_id,
                conversation_id,
                sender_identity_id,
                message_type,
                body,
                created_at,
                edited_at,
                deleted_at
            )
            VALUES (%s, %s, %s, 'text', %s, %s, NULL, NULL)
            """,
            (
                message_id,
                conversation_id,
                identity["id"],
                body,
                now,
            ),
        )

        cur.execute(
            """
            UPDATE verifyd_voice_conversations
            SET updated_at = %s
            WHERE conversation_id = %s
            """,
            (
                now,
                conversation_id,
            ),
        )

        cur.execute(
            """
            SELECT *
            FROM verifyd_voice_messages
            WHERE message_id = %s
            LIMIT 1
            """,
            (message_id,),
        )
        row = dict(cur.fetchone())

    message_payload = _message_payload(
        row,
        identity["id"],
        _other_member.get("last_read_at"),
    )

    # Realtime delivery is intentionally best-effort. The database write above
    # is authoritative; an offline recipient receives the message on the next
    # HTTP inbox refresh. Starlette runs this async BackgroundTask after the
    # response path completes, keeping WebSocket delivery off the sync DB path.
    background_tasks.add_task(
        notify_message_created,
        recipient_identity_id=str(other_identity["id"]),
        conversation_id=conversation_id,
        message_id=message_id,
        sender=_public_contact(identity),
        created_at=now,
    )

    return {
        "ok": True,
        "message": message_payload,
    }


@router.post("/messages/{message_id}/read")
def mark_message_read(
    message_id: str,
    authorization: Optional[str] = Header(default=None),
):
    _require_enabled()

    identity = _identity_from_bearer(authorization)
    ensure_message_schema()

    with get_db() as conn:
        cur = conn.cursor()

        cur.execute(
            """
            SELECT *
            FROM verifyd_voice_messages
            WHERE message_id = %s
              AND deleted_at IS NULL
            LIMIT 1
            """,
            (message_id,),
        )
        message = cur.fetchone()

        if not message:
            raise HTTPException(
                status_code=404,
                detail={"error": "message_not_found"},
            )

        message = dict(message)
        conversation_id = str(message["conversation_id"])

        cur.execute(
            """
            SELECT 1
            FROM verifyd_voice_conversation_members
            WHERE conversation_id = %s
              AND identity_id = %s
            LIMIT 1
            """,
            (
                conversation_id,
                identity["id"],
            ),
        )

        if not cur.fetchone():
            raise HTTPException(
                status_code=404,
                detail={"error": "message_not_found"},
            )

        if str(message.get("sender_identity_id") or "") == str(identity["id"]):
            return {
                "ok": True,
                "message_id": message_id,
                "read": True,
            }

        created_at = str(message.get("created_at") or _now_iso())

        cur.execute(
            """
            SELECT last_read_at
            FROM verifyd_voice_conversation_members
            WHERE conversation_id = %s
              AND identity_id = %s
            LIMIT 1
            """,
            (
                conversation_id,
                identity["id"],
            ),
        )
        member = cur.fetchone() or {}
        existing_last_read = member.get("last_read_at")

        new_last_read = created_at
        if _timestamp_gte(existing_last_read, created_at):
            new_last_read = str(existing_last_read)

        cur.execute(
            """
            UPDATE verifyd_voice_conversation_members
            SET last_read_at = %s
            WHERE conversation_id = %s
              AND identity_id = %s
            """,
            (
                new_last_read,
                conversation_id,
                identity["id"],
            ),
        )

    return {
        "ok": True,
        "message_id": message_id,
        "read": True,
        "last_read_at": new_last_read,
    }
