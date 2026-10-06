# ============================================================
# VeriFYD Trust Voice — Private Attachment Transport Foundation
# VERIFYD_TRUST_VOICE_ATTACHMENTS_V1
#
# Phase 1 attachment transport:
#   - authenticated conversation-member upload
#   - recipient message-privacy enforcement before send
#   - private Cloudflare R2 storage (never PUBLIC_URL)
#   - SHA-256 + metadata persistence
#   - authenticated metadata/download endpoints
#   - durable attachment message + realtime message_created event
#
# IMPORTANT:
#   - Disabled unless VERIFYD_TRUST_VOICE_ATTACHMENTS_ENABLED=1
#   - Authenticity analysis is NOT enabled in this phase
#   - Malware scanning is NOT enabled in this phase
#   - An attachment being accepted/stored does NOT mean it is safe or authentic
# ============================================================

from __future__ import annotations

import hashlib
import logging
import mimetypes
import os
import re
import tempfile
import uuid

import boto3
from botocore.config import Config
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, File, Header, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from database import get_db
from trust_voice_messages import (
    _identity_from_bearer,
    _other_direct_member,
    _public_contact,
    _require_conversation_member,
    _require_message_allowed,
    ensure_message_schema,
)
from trust_voice_signaling import notify_message_created


log = logging.getLogger("verifyd.trust_voice.attachments")

router = APIRouter(
    prefix="/trust-voice",
    tags=["Trust Voice Attachments"],
)

FEATURE_VERSION = "0.1.1"

DEFAULT_MAX_ATTACHMENT_BYTES = 25 * 1024 * 1024
ABSOLUTE_MAX_ATTACHMENT_BYTES = 100 * 1024 * 1024

# Dedicated Trust Voice R2 credentials.
# The account ID may reuse the existing R2_ACCOUNT_ID when the private
# Trust Voice bucket lives in the same Cloudflare account.
TRUST_VOICE_ACCOUNT_ID = (
    os.environ.get("R2_TRUST_VOICE_ACCOUNT_ID", "")
    or os.environ.get("R2_ACCOUNT_ID", "")
    or ""
).strip()
TRUST_VOICE_ACCESS_KEY_ID = (
    os.environ.get("R2_TRUST_VOICE_ACCESS_KEY_ID", "")
    or ""
).strip()
TRUST_VOICE_SECRET_KEY = (
    os.environ.get("R2_TRUST_VOICE_SECRET_KEY", "")
    or ""
).strip()
TRUST_VOICE_BUCKET = (
    os.environ.get("R2_TRUST_VOICE_BUCKET", "")
    or ""
).strip()
TRUST_VOICE_ENDPOINT = (
    f"https://{TRUST_VOICE_ACCOUNT_ID}.r2.cloudflarestorage.com"
    if TRUST_VOICE_ACCOUNT_ID
    else ""
)

PHOTO_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif"}
VIDEO_EXTENSIONS = {
    ".mp4", ".mov", ".m4v", ".webm", ".avi", ".mkv",
    ".mpg", ".mpeg", ".3gp", ".3g2", ".mts", ".m2ts",
    ".ts", ".ogv", ".flv", ".wmv",
}
AUDIO_EXTENSIONS = {
    ".mp3", ".wav", ".m4a", ".aac", ".flac",
    ".ogg", ".oga", ".opus", ".webm",
}
DOCUMENT_EXTENSIONS = {
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    ".odt", ".ods", ".odp", ".txt", ".md", ".csv", ".rtf",
    ".eml", ".msg", ".html", ".htm", ".mhtml", ".mht",
    ".xml", ".json", ".svg", ".vsdx", ".yaml", ".yml",
    ".toml", ".env", ".ini", ".properties", ".conf", ".cfg",
    ".config", ".cnf", ".log", ".sql",
}

ALLOWED_EXTENSIONS = (
    PHOTO_EXTENSIONS
    | VIDEO_EXTENSIONS
    | AUDIO_EXTENSIONS
    | DOCUMENT_EXTENSIONS
)

_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._()\- ]+")


def _enabled() -> bool:
    value = (
        os.environ.get(
            "VERIFYD_TRUST_VOICE_ATTACHMENTS_ENABLED",
            "",
        )
        or ""
    ).strip().lower()
    return value in {"1", "true", "yes", "on"}


def _require_enabled() -> None:
    if not _enabled():
        raise HTTPException(
            status_code=404,
            detail={"error": "not_found"},
        )


def _storage_configured() -> bool:
    return bool(
        TRUST_VOICE_ACCOUNT_ID
        and TRUST_VOICE_ACCESS_KEY_ID
        and TRUST_VOICE_SECRET_KEY
        and TRUST_VOICE_BUCKET
        and TRUST_VOICE_ENDPOINT
    )


def _get_attachment_client():
    if not _storage_configured():
        raise RuntimeError(
            "Trust Voice private R2 storage is not fully configured."
        )

    return boto3.client(
        "s3",
        endpoint_url=TRUST_VOICE_ENDPOINT,
        aws_access_key_id=TRUST_VOICE_ACCESS_KEY_ID,
        aws_secret_access_key=TRUST_VOICE_SECRET_KEY,
        config=Config(signature_version="s3v4"),
        region_name="auto",
    )


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _max_attachment_bytes() -> int:
    raw = (
        os.environ.get(
            "VERIFYD_TRUST_VOICE_ATTACHMENT_MAX_BYTES",
            str(DEFAULT_MAX_ATTACHMENT_BYTES),
        )
        or ""
    ).strip()
    try:
        value = int(raw)
    except Exception:
        value = DEFAULT_MAX_ATTACHMENT_BYTES

    return max(
        1024,
        min(value, ABSOLUTE_MAX_ATTACHMENT_BYTES),
    )


def _safe_filename(value: str) -> str:
    base = os.path.basename(value or "").strip()
    if not base:
        base = "attachment"

    cleaned = _SAFE_NAME_RE.sub("_", base)
    cleaned = cleaned.strip(" .")
    if not cleaned:
        cleaned = "attachment"

    root, ext = os.path.splitext(cleaned)
    ext = ext.lower()
    root = root[:140] or "attachment"

    return f"{root}{ext}"[:180]


def _extension(value: str) -> str:
    return os.path.splitext(value or "")[1].lower()


def _media_category(
    ext: str,
    supplied_content_type: str = "",
) -> str:
    # .webm is the only currently ambiguous allowed extension because it
    # may contain audio-only or video. For other extensions, use the
    # extension classification so a spoofed client MIME type cannot change
    # future analysis routing.
    if ext == ".webm":
        mime = (supplied_content_type or "").strip().lower()
        if mime.startswith("audio/"):
            return "audio"
        return "video"

    if ext in PHOTO_EXTENSIONS:
        return "photo"
    if ext in AUDIO_EXTENSIONS:
        return "audio"
    if ext in VIDEO_EXTENSIONS:
        return "video"
    if ext in DOCUMENT_EXTENSIONS:
        return "document"
    return "unknown"


def _content_type(filename: str, supplied: Optional[str]) -> str:
    supplied = (supplied or "").strip().lower()
    guessed, _ = mimetypes.guess_type(filename)

    if supplied and supplied != "application/octet-stream":
        return supplied[:120]

    return (guessed or "application/octet-stream")[:120]


def ensure_attachment_schema() -> None:
    ensure_message_schema()

    with get_db() as conn:
        cur = conn.cursor()

        cur.execute(
            '''
            CREATE TABLE IF NOT EXISTS verifyd_voice_attachments (
                attachment_id          TEXT PRIMARY KEY,
                message_id             TEXT UNIQUE NOT NULL,
                conversation_id        TEXT NOT NULL,
                uploader_identity_id   TEXT NOT NULL,
                filename               TEXT NOT NULL,
                extension              TEXT NOT NULL,
                media_category         TEXT NOT NULL,
                content_type           TEXT NOT NULL,
                size_bytes             BIGINT NOT NULL,
                sha256                 TEXT NOT NULL,
                storage_key            TEXT UNIQUE NOT NULL,
                analysis_status        TEXT NOT NULL DEFAULT 'not_started',
                analysis_result        TEXT,
                malware_status         TEXT NOT NULL DEFAULT 'not_scanned',
                created_at             TEXT NOT NULL
            )
            '''
        )

        cur.execute(
            '''
            CREATE INDEX IF NOT EXISTS idx_verifyd_voice_attachments_conversation
            ON verifyd_voice_attachments (
                conversation_id,
                created_at
            )
            '''
        )

        cur.execute(
            '''
            CREATE INDEX IF NOT EXISTS idx_verifyd_voice_attachments_message
            ON verifyd_voice_attachments (
                message_id
            )
            '''
        )


def _attachment_row(attachment_id: str) -> Optional[dict]:
    ensure_attachment_schema()

    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            '''
            SELECT *
            FROM verifyd_voice_attachments
            WHERE attachment_id = %s
            LIMIT 1
            ''',
            (attachment_id,),
        )
        row = cur.fetchone()

    return dict(row) if row else None


def _public_attachment(row: dict) -> dict:
    attachment_id = str(row.get("attachment_id") or "")
    return {
        "attachment_id": attachment_id,
        "message_id": row.get("message_id"),
        "conversation_id": row.get("conversation_id"),
        "filename": row.get("filename") or "",
        "extension": row.get("extension") or "",
        "media_category": row.get("media_category") or "unknown",
        "content_type": row.get("content_type") or "application/octet-stream",
        "size_bytes": int(row.get("size_bytes") or 0),
        "sha256": row.get("sha256") or "",
        "analysis_status": row.get("analysis_status") or "not_started",
        "malware_status": row.get("malware_status") or "not_scanned",
        "download_url": f"/trust-voice/attachments/{attachment_id}/download",
        "created_at": row.get("created_at") or "",
    }


def _require_attachment_member(
    attachment_id: str,
    identity_id: str,
) -> dict:
    row = _attachment_row(attachment_id)

    if not row:
        raise HTTPException(
            status_code=404,
            detail={"error": "attachment_not_found"},
        )

    _require_conversation_member(
        str(row["conversation_id"]),
        identity_id,
    )

    return row


def _delete_r2_quietly(storage_key: str) -> None:
    try:
        if storage_key and _storage_configured():
            _get_attachment_client().delete_object(
                Bucket=TRUST_VOICE_BUCKET,
                Key=storage_key,
            )
    except Exception as exc:
        log.warning(
            "Trust Voice attachment cleanup failed key=%s error=%s",
            storage_key,
            exc,
        )


@router.get("/attachments/health")
def attachments_health():
    return {
        "status": "ok",
        "feature": "trust_voice_attachments",
        "version": FEATURE_VERSION,
        "enabled": _enabled(),
        "storage": (
            "private_r2_configured"
            if _storage_configured()
            else "not_configured"
        ),
        "credential_mode": (
            "dedicated_bucket_token"
            if _storage_configured()
            else "not_configured"
        ),
        "max_attachment_bytes": _max_attachment_bytes(),
        "authenticity_analysis": "not_enabled",
        "malware_scanning": "not_enabled",
    }


@router.post("/conversations/{conversation_id}/attachments")
async def send_attachment(
    conversation_id: str,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    authorization: Optional[str] = Header(default=None),
):
    _require_enabled()

    if not _storage_configured():
        raise HTTPException(
            status_code=503,
            detail={
                "error": "attachment_storage_unavailable",
                "message": "Attachment storage is temporarily unavailable.",
            },
        )

    identity = _identity_from_bearer(authorization)
    ensure_attachment_schema()

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

    safe_name = _safe_filename(file.filename or "attachment")
    ext = _extension(safe_name)

    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail={
                "error": "unsupported_attachment_type",
                "message": "This file type is not currently supported for Trust Voice attachments.",
            },
        )

    content_type = _content_type(safe_name, file.content_type)
    category = _media_category(
        ext,
        file.content_type or "",
    )
    max_bytes = _max_attachment_bytes()

    attachment_id = "tvatt_" + uuid.uuid4().hex
    message_id = "tvmsg_" + uuid.uuid4().hex
    now = _now_iso()

    fd, temp_path = tempfile.mkstemp(
        prefix=f"{attachment_id}_",
        suffix=ext,
    )
    os.close(fd)

    size_bytes = 0
    digest = hashlib.sha256()
    storage_key = ""

    try:
        with open(temp_path, "wb") as out:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break

                size_bytes += len(chunk)
                if size_bytes > max_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail={
                            "error": "attachment_too_large",
                            "message": "Attachment exceeds the current Trust Voice size limit.",
                            "max_bytes": max_bytes,
                        },
                    )

                digest.update(chunk)
                out.write(chunk)

        if size_bytes <= 0:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "empty_attachment",
                    "message": "The attachment is empty.",
                },
            )

        sha256 = digest.hexdigest()

        storage_key = (
            f"trust-voice/attachments/"
            f"{conversation_id}/"
            f"{attachment_id}{ext}"
        )

        try:
            client = _get_attachment_client()
            client.upload_file(
                temp_path,
                TRUST_VOICE_BUCKET,
                storage_key,
                ExtraArgs={
                    "ContentType": content_type,
                    "ContentDisposition": f'attachment; filename="{safe_name}"',
                    "Metadata": {
                        "type": "trust_voice_attachment",
                        "attachment_id": attachment_id,
                        "conversation_id": conversation_id,
                        "sha256": sha256,
                    },
                },
            )
        except Exception:
            log.exception(
                "Trust Voice attachment upload failed attachment_id=%s",
                attachment_id,
            )
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "attachment_storage_unavailable",
                    "message": "Attachment storage is temporarily unavailable.",
                },
            )

        # Privacy is checked again after the upload finishes so a recipient
        # changing settings during a large upload still takes effect before
        # the attachment becomes a durable message.
        try:
            _require_message_allowed(
                sender=identity,
                recipient=other_identity,
            )
        except HTTPException:
            _delete_r2_quietly(storage_key)
            raise

        try:
            with get_db() as conn:
                cur = conn.cursor()

                cur.execute(
                    '''
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
                    VALUES (%s, %s, %s, 'attachment', %s, %s, NULL, NULL)
                    ''',
                    (
                        message_id,
                        conversation_id,
                        identity["id"],
                        safe_name,
                        now,
                    ),
                )

                cur.execute(
                    '''
                    INSERT INTO verifyd_voice_attachments (
                        attachment_id,
                        message_id,
                        conversation_id,
                        uploader_identity_id,
                        filename,
                        extension,
                        media_category,
                        content_type,
                        size_bytes,
                        sha256,
                        storage_key,
                        analysis_status,
                        analysis_result,
                        malware_status,
                        created_at
                    )
                    VALUES (
                        %s, %s, %s, %s, %s,
                        %s, %s, %s, %s, %s,
                        %s, 'not_started', NULL, 'not_scanned', %s
                    )
                    ''',
                    (
                        attachment_id,
                        message_id,
                        conversation_id,
                        identity["id"],
                        safe_name,
                        ext,
                        category,
                        content_type,
                        size_bytes,
                        sha256,
                        storage_key,
                        now,
                    ),
                )

                cur.execute(
                    '''
                    UPDATE verifyd_voice_conversations
                    SET updated_at = %s
                    WHERE conversation_id = %s
                    ''',
                    (
                        now,
                        conversation_id,
                    ),
                )
        except Exception:
            _delete_r2_quietly(storage_key)
            raise

        background_tasks.add_task(
            notify_message_created,
            recipient_identity_id=str(other_identity["id"]),
            conversation_id=conversation_id,
            message_id=message_id,
            sender=_public_contact(identity),
            created_at=now,
            message_type="attachment",
        )

        return {
            "ok": True,
            "message": {
                "message_id": message_id,
                "conversation_id": conversation_id,
                "message_type": "attachment",
                "body": safe_name,
                "sender": _public_contact(identity),
                "is_mine": True,
                "read_by_recipient": False,
                "created_at": now,
                "edited_at": None,
                "attachment": _public_attachment(
                    {
                        "attachment_id": attachment_id,
                        "message_id": message_id,
                        "conversation_id": conversation_id,
                        "filename": safe_name,
                        "extension": ext,
                        "media_category": category,
                        "content_type": content_type,
                        "size_bytes": size_bytes,
                        "sha256": sha256,
                        "analysis_status": "not_started",
                        "malware_status": "not_scanned",
                        "created_at": now,
                    }
                ),
            },
        }

    finally:
        try:
            await file.close()
        except Exception:
            pass
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        except Exception:
            pass


@router.get("/conversations/{conversation_id}/attachments")
def list_conversation_attachments(
    conversation_id: str,
    authorization: Optional[str] = Header(default=None),
):
    _require_enabled()

    identity = _identity_from_bearer(authorization)
    ensure_attachment_schema()

    _require_conversation_member(
        conversation_id,
        identity["id"],
    )

    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT * FROM verifyd_voice_attachments "
            "WHERE conversation_id = %s "
            "ORDER BY created_at ASC",
            (conversation_id,),
        )
        rows = cur.fetchall() or []

    attachments = [
        _public_attachment(dict(row))
        for row in rows
    ]

    return {
        "conversation_id": conversation_id,
        "count": len(attachments),
        "attachments": attachments,
    }


@router.get("/attachments/{attachment_id}")
def get_attachment_metadata(
    attachment_id: str,
    authorization: Optional[str] = Header(default=None),
):
    _require_enabled()

    identity = _identity_from_bearer(authorization)
    row = _require_attachment_member(
        attachment_id,
        identity["id"],
    )

    return {
        "attachment": _public_attachment(row),
    }


@router.get("/attachments/{attachment_id}/download")
def download_attachment(
    attachment_id: str,
    authorization: Optional[str] = Header(default=None),
):
    _require_enabled()

    identity = _identity_from_bearer(authorization)
    row = _require_attachment_member(
        attachment_id,
        identity["id"],
    )

    if not _storage_configured():
        raise HTTPException(
            status_code=503,
            detail={"error": "attachment_storage_unavailable"},
        )

    try:
        obj = _get_attachment_client().get_object(
            Bucket=TRUST_VOICE_BUCKET,
            Key=row["storage_key"],
        )
    except Exception:
        raise HTTPException(
            status_code=404,
            detail={"error": "attachment_file_not_found"},
        )

    body = obj["Body"]
    filename = row.get("filename") or "attachment"
    media_type = (
        row.get("content_type")
        or "application/octet-stream"
    )

    def stream():
        try:
            for chunk in body.iter_chunks(
                chunk_size=1024 * 1024
            ):
                if chunk:
                    yield chunk
        finally:
            try:
                body.close()
            except Exception:
                pass

    return StreamingResponse(
        stream(),
        media_type=media_type,
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Cache-Control": "private, no-store",
            "X-Content-Type-Options": "nosniff",
        },
    )
