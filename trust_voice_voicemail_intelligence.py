"""Voicemail intelligence: private, recipient-opt-in analysis and per-recipient inbox deletion.

API is intentionally limited to existing verified-network audio attachments.
No PSTN ingestion or public business endpoint is enabled in Phase 1.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel
from database import get_db
from trust_voice_messages import _identity_from_bearer
from trust_voice_attachments import (
    _require_attachment_member, _storage_configured,
    _get_attachment_client, TRUST_VOICE_BUCKET,
)
from trust_voice_attachment_validation import validate_attachment_content
from trust_voice_voicemail_risk import review_voicemail_content

log = logging.getLogger("verifyd.voice.voicemail_review")
router = APIRouter(prefix="/trust-voice/voicemail", tags=["Trust Voice Voicemail Intelligence"])
MAX_BYTES = 25 * 1024 * 1024
PREFIX = "VeriFYD_Voicemail_"


def _enabled() -> bool:
    return os.environ.get("VERIFYD_VOICEMAIL_INTELLIGENCE_ENABLED", "0").lower().strip() in {"1","true","yes","on"}


def _require_enabled() -> None:
    if not _enabled():
        raise HTTPException(404, detail={"error": "voicemail_analysis_unavailable"})


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _schema() -> None:
    # Separate from existing messaging/attachments tables; no existing rows changed.
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS verifyd_voice_voicemail_reviews (
                attachment_id TEXT NOT NULL,
                recipient_identity_id TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'not_started',
                result_json TEXT,
                deleted_at TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (attachment_id, recipient_identity_id)
            )
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_verifyd_voice_vm_reviews_recipient
            ON verifyd_voice_voicemail_reviews (recipient_identity_id, updated_at)
        """)


def _owner(authorization: Optional[str]) -> dict:
    return _identity_from_bearer(authorization)


def _incoming_voicemail(attachment_id: str, owner_id: str) -> dict:
    row = _require_attachment_member(attachment_id, owner_id)
    if not str(row.get("filename") or "").startswith(PREFIX) or str(row.get("media_category")) != "audio":
        raise HTTPException(404, detail={"error": "voicemail_not_found"})
    if str(row.get("uploader_identity_id")) == str(owner_id):
        raise HTTPException(403, detail={"error": "recipient_only"})
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT deleted_at, sender_identity_id FROM verifyd_voice_messages
            WHERE message_id=%s AND conversation_id=%s LIMIT 1
        """, (row["message_id"], row["conversation_id"]))
        msg = cur.fetchone()
    if not msg or msg.get("deleted_at") or str(msg["sender_identity_id"]) == str(owner_id):
        raise HTTPException(404, detail={"error": "voicemail_not_found"})
    return row


def _review_state(attachment_id: str, owner_id: str) -> dict | None:
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT status,result_json,deleted_at,updated_at
            FROM verifyd_voice_voicemail_reviews
            WHERE attachment_id=%s AND recipient_identity_id=%s
        """, (attachment_id, owner_id))
        row = cur.fetchone()
    return dict(row) if row else None


def _public(row: dict | None) -> dict:
    if not row:
        return {"status": "not_started", "analysis": None}
    result = None
    if row.get("result_json"):
        try:
            result = json.loads(row["result_json"])
        except (TypeError, ValueError):
            result = None
    return {
        "status": str(row.get("status") or "not_started"),
        "analysis": result,
        "updated_at": row.get("updated_at"),
    }


def _set_state(attachment_id: str, owner_id: str, status: str, result: dict | None = None) -> bool:
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("""
            UPDATE verifyd_voice_voicemail_reviews
            SET status=%s,result_json=%s,updated_at=%s
            WHERE attachment_id=%s AND recipient_identity_id=%s
              AND deleted_at IS NULL
        """, (
            status, json.dumps(result, ensure_ascii=False) if result is not None else None,
            _now(), attachment_id, owner_id,
        ))
        return bool(cur.rowcount)


def _transcribe(path: str) -> str:
    """Transcribe selected message only; never persist generated WAV."""
    from openai import OpenAI

    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("transcription_not_configured")

    fd, wav_path = tempfile.mkstemp(prefix="verifyd_voicemail_transcribe_", suffix=".wav")
    os.close(fd)
    try:
        command = [
            os.environ.get("FFMPEG_BIN", "ffmpeg"), "-nostdin", "-y",
            "-hide_banner", "-loglevel", "error", "-i", path,
            "-vn", "-t", "125", "-ac", "1", "-ar", "16000", wav_path,
        ]
        run = subprocess.run(command, capture_output=True, timeout=100)
        if run.returncode != 0 or os.path.getsize(wav_path) < 200:
            raise RuntimeError("audio_conversion_failed")
        with open(wav_path, "rb") as fh:
            text = OpenAI(timeout=120).audio.transcriptions.create(
                model=os.environ.get("VERIFYD_VOICEMAIL_TRANSCRIPTION_MODEL", "gpt-transcribe"),
                file=fh,
            )
        return str(getattr(text, "text", "") or "").strip()[:12000]
    finally:
        try:
            os.unlink(wav_path)
        except OSError:
            pass


def process_voicemail_review(attachment_id: str, owner_id: str) -> dict:
    """Worker entry point. An earlier 'Delete' invalidates pending jobs."""
    temp_path = ""
    try:
        review = _review_state(attachment_id, owner_id)
        if not review or review.get("deleted_at") or review.get("status") not in {"queued", "analyzing"}:
            return {"ok": False, "reason": "not_pending"}
        row = _incoming_voicemail(attachment_id, owner_id)
        if not _set_state(attachment_id, owner_id, "analyzing"):
            return {"ok": False, "reason": "deleted"}
        if not _storage_configured():
            raise RuntimeError("private_storage_unavailable")
        if int(row.get("size_bytes") or 0) < 200 or int(row.get("size_bytes") or 0) > MAX_BYTES:
            raise RuntimeError("voicemail_size_invalid")
        ext = str(row.get("extension") or "")
        if ext.lower() not in {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".oga", ".opus", ".webm"}:
            raise RuntimeError("voicemail_media_unsupported")
        fd, temp_path = tempfile.mkstemp(prefix="verifyd_private_vm_", suffix=ext)
        os.close(fd)
        _get_attachment_client().download_file(TRUST_VOICE_BUCKET, str(row["storage_key"]), temp_path)
        digest = hashlib.sha256()
        with open(temp_path, "rb") as file:
            for chunk in iter(lambda: file.read(1024*1024), b""):
                digest.update(chunk)
        if digest.hexdigest().lower() != str(row.get("sha256") or "").lower():
            raise RuntimeError("attachment_integrity_failed")
        checked = validate_attachment_content(temp_path, extension=ext, supplied_content_type="")
        if checked.get("media_category") != "audio":
            raise RuntimeError("voicemail_media_invalid")

        transcript = ""
        transcript_status = "available"
        try:
            transcript = _transcribe(temp_path)
            if not transcript:
                transcript_status = "unavailable"
        except Exception as e:
            log.warning("Voicemail transcription unavailable, attachment_id=%s reason=%s",
                        attachment_id, type(e).__name__)
            transcript_status = "unavailable"

        from audio_detector import analyze_audio
        audio = {}
        try:
            audio = analyze_audio(temp_path) or {}
        except Exception as e:
            log.warning("Voicemail heuristic unavailable attachment_id=%s reason=%s",
                        attachment_id, type(e).__name__)
        risk = review_voicemail_content(transcript, audio)
        result = {
            "schema_version": "verifyd_voicemail_review_v1",
            "attachment_id": attachment_id,
            "source_type": "verifyd_network",
            "transcript": transcript if transcript_status == "available" else None,
            "transcript_status": transcript_status,
            "risk": risk,
            "analyzed_at": _now(),
            "advisory": "AI and spam detection are estimates, not proof of safety, identity or synthetic speech.",
        }
        # Delete is idempotent and wins against an in-flight analysis.
        if not _set_state(attachment_id, owner_id, "completed", result):
            return {"ok": False, "reason": "deleted_during_analysis"}
        return {"ok": True, "status": "completed"}
    except Exception:
        log.exception("Private voicemail review failed attachment_id=%s", attachment_id)
        _set_state(attachment_id, owner_id, "failed", {
            "message": "Voicemail analysis was unavailable. Retry later; do not assume the message is safe."
        })
        return {"ok": False, "reason": "analysis_unavailable"}
    finally:
        if temp_path:
            try:
                os.unlink(temp_path)
            except OSError:
                pass


class AnalyzeConsent(BaseModel):
    consent_to_external_transcription: bool = False


@router.get("/health")
def voicemail_intelligence_health():
    return {
        "feature": "voicemail_intelligence",
        "enabled": _enabled(),
        "mode": "recipient_initiated",
        "transcription_configured": bool(os.environ.get("OPENAI_API_KEY")),
        "public_business_ingestion": False,
    }


@router.get("/inbox-state")
def voicemail_inbox_state(authorization: Optional[str] = Header(default=None)):
    _require_enabled()
    owner = _owner(authorization)
    _schema()
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT attachment_id,status,deleted_at FROM verifyd_voice_voicemail_reviews
            WHERE recipient_identity_id=%s ORDER BY updated_at DESC LIMIT 500
        """, (str(owner["id"]),))
        rows = cur.fetchall() or []
    return {
        "deleted_attachment_ids": [str(x["attachment_id"]) for x in rows if x["deleted_at"]],
        "review_statuses": {str(x["attachment_id"]): x["status"] for x in rows if not x["deleted_at"]},
    }


@router.get("/{attachment_id}/review")
def get_voicemail_review(attachment_id: str, authorization: Optional[str] = Header(default=None)):
    _require_enabled()
    owner = _owner(authorization)
    _incoming_voicemail(attachment_id, str(owner["id"]))
    _schema()
    record = _review_state(attachment_id, str(owner["id"]))
    if record and record.get("deleted_at"):
        raise HTTPException(404, detail={"error": "voicemail_not_found"})
    return {"attachment_id": attachment_id, **_public(record)}


@router.post("/{attachment_id}/review")
def request_voicemail_review(
    attachment_id: str, consent: AnalyzeConsent,
    authorization: Optional[str] = Header(default=None),
):
    _require_enabled()
    owner = _owner(authorization)
    owner_id = str(owner["id"])
    _incoming_voicemail(attachment_id, owner_id)
    if not consent.consent_to_external_transcription:
        raise HTTPException(400, detail={"error": "transcription_consent_required"})
    if not os.environ.get("OPENAI_API_KEY"):
        raise HTTPException(503, detail={"error": "transcription_not_configured"})
    _schema()
    now = _now()
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO verifyd_voice_voicemail_reviews
                (attachment_id,recipient_identity_id,status,created_at,updated_at)
            VALUES (%s,%s,'not_started',%s,%s)
            ON CONFLICT (attachment_id,recipient_identity_id) DO NOTHING
        """, (attachment_id, owner_id, now, now))
        cur.execute("""
            SELECT status,deleted_at FROM verifyd_voice_voicemail_reviews
            WHERE attachment_id=%s AND recipient_identity_id=%s FOR UPDATE
        """, (attachment_id, owner_id))
        existing = cur.fetchone()
        if not existing or existing.get("deleted_at"):
            raise HTTPException(404, detail={"error": "voicemail_not_found"})
        if existing["status"] in {"queued", "analyzing", "completed"}:
            return {"ok": True, "queued": False, "status": existing["status"]}
        cur.execute("""
            UPDATE verifyd_voice_voicemail_reviews
            SET status='queued',result_json=NULL,updated_at=%s
            WHERE attachment_id=%s AND recipient_identity_id=%s AND deleted_at IS NULL
        """, (now, attachment_id, owner_id))
    try:
        import redis
        from rq import Queue
        queue = Queue(
            os.environ.get("VERIFYD_TRUST_VOICE_ANALYSIS_QUEUE", "verifyd"),
            connection=redis.from_url(os.environ["REDIS_URL"]),
        )
        queue.enqueue(process_voicemail_review, attachment_id, owner_id,
                      job_timeout=600, result_ttl=600, failure_ttl=3600)
    except Exception:
        log.exception("Voicemail analysis queue unavailable attachment_id=%s", attachment_id)
        _set_state(attachment_id, owner_id, "failed", {
            "message": "Voicemail analysis queue unavailable. You can retry later."
        })
        raise HTTPException(503, detail={"error": "analysis_queue_unavailable"})
    return {"ok": True, "queued": True, "status": "queued"}


@router.delete("/{attachment_id}")
def delete_voicemail_for_recipient(
    attachment_id: str, authorization: Optional[str] = Header(default=None),
):
    """Remove from THIS user's inbox across devices, preserving the sender's copy.

    This is not a claim of physical erasure from R2; retention/purge needs its
    own separately audited operation.
    """
    _require_enabled()
    owner = _owner(authorization)
    owner_id = str(owner["id"])
    _incoming_voicemail(attachment_id, owner_id)
    _schema()
    now = _now()
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO verifyd_voice_voicemail_reviews
            (attachment_id,recipient_identity_id,status,result_json,deleted_at,created_at,updated_at)
            VALUES (%s,%s,'deleted',NULL,%s,%s,%s)
            ON CONFLICT (attachment_id,recipient_identity_id) DO UPDATE SET
               status='deleted',result_json=NULL,deleted_at=EXCLUDED.deleted_at,
               updated_at=EXCLUDED.updated_at
        """, (attachment_id, owner_id, now, now, now))
    return {"ok": True, "removed_from_recipient_inbox": True, "audio_deleted_from_storage": False}


@router.post("/{attachment_id}/restore")
def restore_voicemail_for_recipient(
    attachment_id: str, authorization: Optional[str] = Header(default=None),
):
    _require_enabled()
    owner = _owner(authorization)
    owner_id = str(owner["id"])
    _incoming_voicemail(attachment_id, owner_id)
    _schema()
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("""
            UPDATE verifyd_voice_voicemail_reviews SET status='not_started',
                result_json=NULL,deleted_at=NULL,updated_at=%s
            WHERE attachment_id=%s AND recipient_identity_id=%s
        """, (_now(), attachment_id, owner_id))
    return {"ok": True, "restored": True}
