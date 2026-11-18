"""Optional private voice/text greeting before a Trust Voice voicemail.
Feature flag VERIFYD_VOICEMAIL_GREETINGS_ENABLED defaults OFF.
No changes to active call signaling, Web Push, or existing voicemail uploads.
"""
from __future__ import annotations

import logging
import os
import tempfile
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Header, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from database import get_db
from trust_voice_signaling import (
    _identity_from_http_authorization, _identity_by_handle, _normalize_handle,
)
from trust_voice_messages import _require_message_allowed
from trust_voice_attachments import (
    TRUST_VOICE_BUCKET, _get_attachment_client, _storage_configured,
)
from trust_voice_attachment_validation import (
    AttachmentContentError, validate_attachment_content,
)

router = APIRouter(prefix="/trust-voice", tags=["Trust Voice Voicemail Greetings"])
log = logging.getLogger("verifyd.trust_voice.voicemail_greetings")
_schema_lock = threading.Lock()
_schema_ready = False
MAX_AUDIO_BYTES = 3 * 1024 * 1024
MAX_TEXT_CHARS = 400
AUDIO_EXTENSIONS = {".webm", ".m4a", ".mp3", ".ogg", ".wav"}
DEFAULT_GREETING = "Hi, I can't answer right now. Please leave a message when you're ready."


def _enabled():
    return os.getenv("VERIFYD_VOICEMAIL_GREETINGS_ENABLED", "0").strip().lower() in {"1", "true", "yes", "on"}


def _require_enabled():
    if not _enabled():
        raise HTTPException(status_code=404, detail={"error": "greetings_not_enabled"})


def _now():
    return datetime.now(timezone.utc).isoformat()


def _ensure_schema():
    global _schema_ready
    if _schema_ready:
        return
    with _schema_lock:
        if _schema_ready:
            return
        with get_db() as conn:
            conn.cursor().execute("""CREATE TABLE IF NOT EXISTS verifyd_voice_voicemail_greetings (
                identity_id TEXT PRIMARY KEY REFERENCES verifyd_identities(id) ON DELETE CASCADE,
                mode TEXT NOT NULL DEFAULT 'default' CHECK(mode IN ('default','text','audio')),
                greeting_text TEXT NOT NULL DEFAULT '',
                storage_key TEXT,
                content_type TEXT,
                updated_at TEXT NOT NULL
            )""")
        _schema_ready = True


def _read(identity_id: str):
    _ensure_schema()
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("""SELECT mode, greeting_text, storage_key, content_type, updated_at
            FROM verifyd_voice_voicemail_greetings WHERE identity_id=%s LIMIT 1""", (identity_id,))
        row = cur.fetchone()
    return dict(row) if row else {
        "mode": "default", "greeting_text": "", "storage_key": None,
        "content_type": None, "updated_at": None,
    }


def _public(row: dict):
    mode = row.get("mode") or "default"
    return {
        "mode": mode,
        "text": row.get("greeting_text") if mode == "text" else DEFAULT_GREETING if mode == "default" else "",
        "has_audio": bool(mode == "audio" and row.get("storage_key")),
        "updated_at": row.get("updated_at"),
    }


def _owner(authorization: Optional[str]):
    _require_enabled()
    return _identity_from_http_authorization(authorization)


def _caller_and_target(handle: str, authorization: Optional[str]):
    _require_enabled()
    caller = _identity_from_http_authorization(authorization)
    target = _identity_by_handle(_normalize_handle(handle))
    if not target:
        raise HTTPException(status_code=404, detail={"error": "not_found"})
    if str(caller["id"]) != str(target["id"]):
        # Privacy, Circle membership, block lists checked on EVERY read.
        _require_message_allowed(sender=caller, recipient=target)
    return caller, target


def _upsert(identity_id: str, mode: str, greeting_text: str, storage_key=None, content_type=None):
    with get_db() as conn:
        conn.cursor().execute("""INSERT INTO verifyd_voice_voicemail_greetings
          (identity_id,mode,greeting_text,storage_key,content_type,updated_at)
          VALUES (%s,%s,%s,%s,%s,%s)
          ON CONFLICT (identity_id) DO UPDATE SET
          mode=EXCLUDED.mode, greeting_text=EXCLUDED.greeting_text,
          storage_key=EXCLUDED.storage_key, content_type=EXCLUDED.content_type,
          updated_at=EXCLUDED.updated_at""",
          (identity_id, mode, greeting_text, storage_key, content_type, _now()))


def _delete_old(storage_key):
    if not storage_key or not _storage_configured():
        return
    try:
        _get_attachment_client().delete_object(Bucket=TRUST_VOICE_BUCKET, Key=storage_key)
    except Exception:
        log.warning("Voicemail greeting cleanup failed", exc_info=True)


class GreetingUpdate(BaseModel):
    mode: str = Field(pattern="^(default|text|audio)$")
    text: str = Field(default="", max_length=MAX_TEXT_CHARS)


@router.get("/voicemail-greetings/health")
def greeting_health():
    """Sanitized readiness; never expose R2 bucket, key or caller data."""
    return {
        "feature": "trust_voice_voicemail_greetings_v1",
        "enabled": _enabled(),
        "private_audio_storage_configured": _storage_configured(),
    }


@router.get("/voicemail-greetings/me")
def get_my_greeting(authorization: Optional[str] = Header(default=None)):
    identity = _owner(authorization)
    return {"greeting": _public(_read(str(identity["id"])))}


@router.put("/voicemail-greetings/me")
def update_my_greeting(payload: GreetingUpdate, authorization: Optional[str] = Header(default=None)):
    identity = _owner(authorization)
    old = _read(str(identity["id"]))
    greeting_text = payload.text.strip()
    if payload.mode == "text" and not greeting_text:
        raise HTTPException(status_code=422, detail={"error": "greeting_text_required"})
    if payload.mode == "audio" and not old.get("storage_key"):
        raise HTTPException(status_code=409, detail={"error": "recording_required"})
    key = old.get("storage_key") if payload.mode == "audio" else None
    mime = old.get("content_type") if payload.mode == "audio" else None
    _upsert(str(identity["id"]), payload.mode, greeting_text if payload.mode == "text" else "", key, mime)
    if old.get("storage_key") and key != old["storage_key"]:
        _delete_old(old["storage_key"])
    return {"ok": True, "greeting": _public(_read(str(identity["id"])))}


@router.post("/voicemail-greetings/me/audio")
async def upload_my_greeting_audio(
    file: UploadFile = File(...),
    authorization: Optional[str] = Header(default=None),
):
    identity = _owner(authorization)
    if not _storage_configured():
        raise HTTPException(status_code=503, detail={"error": "private_audio_storage_unavailable"})
    ext = Path(file.filename or "").suffix.lower()
    if ext not in AUDIO_EXTENSIONS:
        raise HTTPException(status_code=415, detail={"error": "unsupported_greeting_audio"})
    blob = await file.read(MAX_AUDIO_BYTES + 1)
    if not blob:
        raise HTTPException(status_code=400, detail={"error": "empty_audio"})
    if len(blob) > MAX_AUDIO_BYTES:
        raise HTTPException(status_code=413, detail={"error": "greeting_audio_too_large"})
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as temp:
        temp.write(blob)
        temp_path = temp.name
    try:
        try:
            validated = validate_attachment_content(
                temp_path, extension=ext, supplied_content_type=file.content_type or "",
            )
        except AttachmentContentError:
            raise HTTPException(status_code=415, detail={"error": "invalid_audio_content"})
    finally:
        os.unlink(temp_path)
    if validated.get("media_category") != "audio":
        raise HTTPException(status_code=415, detail={"error": "not_audio"})
    mime = validated.get("content_type") or "application/octet-stream"
    key = f"trust-voice/voicemail-greetings/{identity['id']}/{uuid.uuid4().hex}{ext}"
    try:
        _get_attachment_client().put_object(
            Bucket=TRUST_VOICE_BUCKET, Key=key, Body=blob, ContentType=mime,
            ContentDisposition="inline",
            Metadata={"type": "trust_voice_voicemail_greeting"},
        )
    except Exception:
        log.exception("Private voicemail greeting upload failed")
        raise HTTPException(status_code=503, detail={"error": "greeting_upload_unavailable"})
    old = _read(str(identity["id"]))
    try:
        _upsert(str(identity["id"]), "audio", "", key, mime)
    except Exception:
        _delete_old(key)
        raise
    if old.get("storage_key") and old["storage_key"] != key:
        _delete_old(old["storage_key"])
    return {"ok": True, "greeting": _public(_read(str(identity["id"])))}


@router.get("/voicemail-greetings/for/{handle}")
def get_caller_greeting(handle: str, authorization: Optional[str] = Header(default=None)):
    _caller, target = _caller_and_target(handle, authorization)
    return {"greeting": _public(_read(str(target["id"])))}


@router.get("/voicemail-greetings/me/audio")
def play_my_greeting_audio(authorization: Optional[str] = Header(default=None)):
    owner = _owner(authorization)
    return _stream_audio(_read(str(owner["id"])))


@router.get("/voicemail-greetings/for/{handle}/audio")
def play_caller_greeting_audio(handle: str, authorization: Optional[str] = Header(default=None)):
    _caller, target = _caller_and_target(handle, authorization)
    return _stream_audio(_read(str(target["id"])))


def _stream_audio(row):
    if row.get("mode") != "audio" or not row.get("storage_key"):
        raise HTTPException(status_code=404, detail={"error": "recording_not_found"})
    if not _storage_configured():
        raise HTTPException(status_code=503, detail={"error": "private_audio_storage_unavailable"})
    try:
        obj = _get_attachment_client().get_object(Bucket=TRUST_VOICE_BUCKET, Key=row["storage_key"])
    except Exception:
        log.warning("Private voicemail greeting file unavailable")
        raise HTTPException(status_code=404, detail={"error": "recording_not_found"})
    body = obj["Body"]

    def stream():
        try:
            for chunk in body.iter_chunks(chunk_size=256 * 1024):
                if chunk:
                    yield chunk
        finally:
            body.close()

    return StreamingResponse(stream(), media_type=row.get("content_type") or "audio/webm", headers={
        "Cache-Control": "private, no-store", "X-Content-Type-Options": "nosniff",
        "Content-Disposition": 'inline; filename="VeriFYD_Greeting"',
    })
