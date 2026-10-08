# ============================================================
# VeriFYD Trust Voice — Isolated Attachment Authenticity Analysis
# VERIFYD_TRUST_VOICE_ATTACHMENT_ANALYSIS_V1
#
# Phase 1:
#   - manual/authenticated analysis trigger
#   - RQ execution on existing VeriFYD worker queue
#   - private Trust Voice R2 object retrieval
#   - SHA-256 integrity re-check before detector execution
#   - re-validation of actual file content
#   - direct use of low-level VeriFYD detectors only
#   - durable analysis_status / analysis_result persistence
#
# IMPORTANT:
#   - Does NOT increment VeriFYD usage
#   - Does NOT create certificates
#   - Does NOT create certified media
#   - Does NOT send email
#   - Does NOT perform malware scanning
#   - Does NOT automatically analyze every attachment yet
# ============================================================

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from typing import Any, Optional

from fastapi import APIRouter, Header, HTTPException

from database import get_db
from trust_voice_messages import _identity_from_bearer
from trust_voice_attachments import (
    TRUST_VOICE_BUCKET,
    _get_attachment_client,
    _require_attachment_member,
    _storage_configured,
)
from trust_voice_attachment_validation import (
    AttachmentContentError,
    validate_attachment_content,
)


log = logging.getLogger("verifyd.trust_voice.attachment_analysis")

router = APIRouter(
    prefix="/trust-voice",
    tags=["Trust Voice Attachment Analysis"],
)

FEATURE_VERSION = "0.1.0"


def _enabled() -> bool:
    value = (
        os.environ.get(
            "VERIFYD_TRUST_VOICE_ATTACHMENT_ANALYSIS_ENABLED",
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


def _queue_name() -> str:
    value = (
        os.environ.get(
            "VERIFYD_TRUST_VOICE_ANALYSIS_QUEUE",
            "verifyd",
        )
        or "verifyd"
    ).strip()
    return value or "verifyd"


def _job_timeout_seconds() -> int:
    raw = (
        os.environ.get(
            "VERIFYD_TRUST_VOICE_ANALYSIS_TIMEOUT_SECONDS",
            "1200",
        )
        or "1200"
    ).strip()
    try:
        value = int(raw)
    except Exception:
        value = 1200
    return max(60, min(value, 3600))


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_json(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, dict):
        return {
            str(key): _safe_json(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple, set)):
        return [_safe_json(item) for item in value]

    item_method = getattr(value, "item", None)
    if callable(item_method):
        try:
            return _safe_json(item_method())
        except Exception:
            pass

    return str(value)


def _analysis_row(attachment_id: str) -> Optional[dict]:
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT *
            FROM verifyd_voice_attachments
            WHERE attachment_id = %s
            LIMIT 1
            """,
            (attachment_id,),
        )
        row = cur.fetchone()

    return dict(row) if row else None


def _set_analysis_state(
    attachment_id: str,
    status: str,
    result: Optional[dict] = None,
) -> None:
    encoded = (
        json.dumps(
            _safe_json(result),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        if result is not None
        else None
    )

    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            UPDATE verifyd_voice_attachments
            SET analysis_status = %s,
                analysis_result = %s
            WHERE attachment_id = %s
            """,
            (
                status,
                encoded,
                attachment_id,
            ),
        )


def _decoded_result(row: dict) -> Optional[dict]:
    raw = row.get("analysis_result")
    if not raw:
        return None
    if isinstance(raw, dict):
        return _safe_json(raw)

    try:
        parsed = json.loads(str(raw))
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        return None


def _public_analysis(row: dict) -> dict:
    return {
        "attachment_id": row.get("attachment_id"),
        "conversation_id": row.get("conversation_id"),
        "analysis_status": row.get("analysis_status") or "not_started",
        "analysis": _decoded_result(row),
        "malware_status": row.get("malware_status") or "not_scanned",
    }


def _label_from_authenticity(authenticity: int) -> str:
    score = max(0, min(100, int(authenticity)))
    if score >= 55:
        return "REAL"
    if score >= 40:
        return "UNDETERMINED"
    return "AI"


def _compact_result(
    *,
    attachment_id: str,
    media_category: str,
    detector_name: str,
    authenticity: int,
    label: str,
    detail: dict,
) -> dict:
    safe_detail = _safe_json(detail or {})

    try:
        ai_score = int(
            round(
                float(
                    safe_detail.get(
                        "ai_score",
                        100 - int(authenticity),
                    )
                )
            )
        )
    except Exception:
        ai_score = 100 - int(authenticity)

    ai_score = max(0, min(100, ai_score))
    authenticity = max(0, min(100, int(round(float(authenticity)))))

    reasoning = str(
        safe_detail.get("gpt_reasoning")
        or safe_detail.get("reasoning")
        or ""
    )[:6000]

    flags = safe_detail.get("gpt_flags")
    if not isinstance(flags, list):
        flags = []

    result = {
        "schema_version": "trust_voice_attachment_analysis_v1",
        "attachment_id": attachment_id,
        "media_category": media_category,
        "detector": detector_name,
        "label": str(label or "UNDETERMINED"),
        "authenticity_score": authenticity,
        "ai_score": ai_score,
        "reasoning": reasoning,
        "flags": flags[:20],
        "certified": False,
        "usage_incremented": False,
        "malware_scanned": False,
    }

    selected = {}
    for key in (
        "signal_ai_score",
        "gpt_ai_score",
        "blend_mode",
        "content_type",
        "audio_ai_score",
        "audio_confidence",
        "audio_evidence",
        "gpt_audio_score",
        "gpt_audio_reasoning",
        "document_type",
        "overall_risk",
        "risk_score",
        "metadata_integrity",
        "document_risk_report",
        "risk_report",
        "ai_source_detected",
        "ai_source_generator",
        "ai_source_confidence",
        "provenance_override",
    ):
        if key in safe_detail:
            selected[key] = safe_detail[key]

    if selected:
        result["detector_detail"] = selected

    return result


def _run_detector(
    path: str,
    media_category: str,
) -> tuple[int, str, dict, str]:
    if media_category == "photo":
        from photo_detection import run_photo_detection

        authenticity, label, detail = run_photo_detection(path)
        return (
            int(authenticity),
            str(label),
            detail or {},
            "photo_detection.run_photo_detection",
        )

    if media_category == "video":
        from detection import run_detection_multiclip

        authenticity, label, detail = run_detection_multiclip(path)
        return (
            int(authenticity),
            str(label),
            detail or {},
            "detection.run_detection_multiclip",
        )

    if media_category == "document":
        from document_detection import run_document_detection

        authenticity, label, detail = run_document_detection(path)
        return (
            int(authenticity),
            str(label),
            detail or {},
            "document_detection.run_document_detection",
        )

    if media_category == "audio":
        from audio_detector import analyze_audio

        detail = analyze_audio(path) or {}
        try:
            audio_ai_score = int(
                round(
                    float(
                        detail.get(
                            "audio_ai_score",
                            50,
                        )
                    )
                )
            )
        except Exception:
            audio_ai_score = 50

        audio_ai_score = max(0, min(100, audio_ai_score))
        authenticity = 100 - audio_ai_score
        label = _label_from_authenticity(authenticity)

        detail = dict(detail)
        detail["ai_score"] = audio_ai_score
        detail["audio_ai_score"] = audio_ai_score
        detail.setdefault(
            "gpt_reasoning",
            " ".join(
                str(item)
                for item in (detail.get("evidence") or [])[:8]
            ),
        )
        detail.setdefault(
            "gpt_flags",
            list(detail.get("evidence") or [])[:8],
        )

        return (
            authenticity,
            label,
            detail,
            "audio_detector.analyze_audio",
        )

    raise RuntimeError(
        f"Unsupported Trust Voice analysis category: {media_category}"
    )


def process_trust_voice_attachment_analysis(
    attachment_id: str,
) -> dict:
    """
    RQ worker entry point.

    This function deliberately does not import/call VeriFYD certification
    workers. It invokes detector entry points directly.
    """
    temp_path = ""

    try:
        row = _analysis_row(attachment_id)
        if not row:
            return {
                "ok": False,
                "error": "attachment_not_found",
            }

        _set_analysis_state(
            attachment_id,
            "analyzing",
            None,
        )

        if not _storage_configured():
            raise RuntimeError(
                "Trust Voice private attachment storage is not configured."
            )

        ext = str(row.get("extension") or "").lower()
        fd, temp_path = tempfile.mkstemp(
            prefix=f"tv_analysis_{attachment_id}_",
            suffix=ext,
        )
        os.close(fd)

        _get_attachment_client().download_file(
            TRUST_VOICE_BUCKET,
            str(row["storage_key"]),
            temp_path,
        )

        actual_sha256 = _sha256_file(temp_path)
        expected_sha256 = str(row.get("sha256") or "").lower()

        if (
            not expected_sha256
            or actual_sha256.lower() != expected_sha256
        ):
            result = {
                "schema_version": "trust_voice_attachment_analysis_v1",
                "attachment_id": attachment_id,
                "error": "integrity_mismatch",
                "message": "Stored attachment integrity verification failed.",
            }
            _set_analysis_state(
                attachment_id,
                "failed",
                result,
            )
            return {
                "ok": False,
                **result,
            }

        validated = validate_attachment_content(
            temp_path,
            extension=ext,
            supplied_content_type="",
        )

        validated_category = str(
            validated.get("media_category")
            or "unknown"
        )
        stored_category = str(
            row.get("media_category")
            or "unknown"
        )

        if (
            validated_category == "unknown"
            or validated_category != stored_category
        ):
            result = {
                "schema_version": "trust_voice_attachment_analysis_v1",
                "attachment_id": attachment_id,
                "error": "content_routing_mismatch",
                "message": "Stored attachment type no longer matches its validated analysis route.",
            }
            _set_analysis_state(
                attachment_id,
                "failed",
                result,
            )
            return {
                "ok": False,
                **result,
            }

        authenticity, label, detail, detector_name = _run_detector(
            temp_path,
            validated_category,
        )

        result = _compact_result(
            attachment_id=attachment_id,
            media_category=validated_category,
            detector_name=detector_name,
            authenticity=authenticity,
            label=label,
            detail=detail,
        )

        _set_analysis_state(
            attachment_id,
            "completed",
            result,
        )

        log.info(
            "Trust Voice attachment analysis complete attachment_id=%s category=%s label=%s authenticity=%s",
            attachment_id,
            validated_category,
            result.get("label"),
            result.get("authenticity_score"),
        )

        return {
            "ok": True,
            "analysis_status": "completed",
            "analysis": result,
        }

    except AttachmentContentError as exc:
        result = {
            "schema_version": "trust_voice_attachment_analysis_v1",
            "attachment_id": attachment_id,
            "error": exc.code,
            "message": exc.message,
        }
        try:
            _set_analysis_state(
                attachment_id,
                "failed",
                result,
            )
        except Exception:
            pass
        return {
            "ok": False,
            **result,
        }

    except Exception:
        log.exception(
            "Trust Voice attachment analysis failed attachment_id=%s",
            attachment_id,
        )
        result = {
            "schema_version": "trust_voice_attachment_analysis_v1",
            "attachment_id": attachment_id,
            "error": "analysis_failed",
            "message": "Attachment authenticity analysis could not be completed.",
        }
        try:
            _set_analysis_state(
                attachment_id,
                "failed",
                result,
            )
        except Exception:
            pass
        return {
            "ok": False,
            **result,
        }

    finally:
        if temp_path:
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            except Exception:
                pass


@router.get("/attachments/analysis/health")
def attachment_analysis_health():
    return {
        "status": "ok",
        "feature": "trust_voice_attachment_analysis",
        "version": FEATURE_VERSION,
        "enabled": _enabled(),
        "mode": "manual_rq_queue",
        "queue": _queue_name(),
        "auto_analysis": "not_enabled",
        "certification": "not_performed",
        "usage_increment": "not_performed",
        "email": "not_sent",
        "malware_scanning": "not_enabled",
    }


@router.get("/attachments/{attachment_id}/analysis")
def get_attachment_analysis(
    attachment_id: str,
    authorization: Optional[str] = Header(default=None),
):
    _require_enabled()

    identity = _identity_from_bearer(authorization)
    row = _require_attachment_member(
        attachment_id,
        identity["id"],
    )

    return _public_analysis(row)


@router.post("/attachments/{attachment_id}/analysis")
def queue_attachment_analysis(
    attachment_id: str,
    authorization: Optional[str] = Header(default=None),
):
    _require_enabled()

    identity = _identity_from_bearer(authorization)
    row = _require_attachment_member(
        attachment_id,
        identity["id"],
    )

    current = str(
        row.get("analysis_status")
        or "not_started"
    )

    if current in {"queued", "analyzing"}:
        return {
            "ok": True,
            "attachment_id": attachment_id,
            "analysis_status": current,
            "queued": False,
        }

    if current == "completed":
        return {
            "ok": True,
            "attachment_id": attachment_id,
            "analysis_status": "completed",
            "queued": False,
            "analysis": _decoded_result(row),
        }

    _set_analysis_state(
        attachment_id,
        "queued",
        None,
    )

    try:
        import redis
        from rq import Queue

        connection = redis.from_url(
            os.environ.get(
                "REDIS_URL",
                "redis://localhost:6379",
            ),
            decode_responses=False,
        )

        queue = Queue(
            _queue_name(),
            connection=connection,
        )

        job = queue.enqueue(
            process_trust_voice_attachment_analysis,
            attachment_id,
            job_timeout=_job_timeout_seconds(),
            result_ttl=600,
            failure_ttl=3600,
        )

    except Exception:
        log.exception(
            "Trust Voice analysis enqueue failed attachment_id=%s",
            attachment_id,
        )
        _set_analysis_state(
            attachment_id,
            "failed",
            {
                "schema_version": "trust_voice_attachment_analysis_v1",
                "attachment_id": attachment_id,
                "error": "analysis_queue_unavailable",
                "message": "Attachment analysis is temporarily unavailable.",
            },
        )
        raise HTTPException(
            status_code=503,
            detail={
                "error": "analysis_queue_unavailable",
                "message": "Attachment analysis is temporarily unavailable.",
            },
        )

    return {
        "ok": True,
        "attachment_id": attachment_id,
        "analysis_status": "queued",
        "queued": True,
        "job_id": job.id,
    }
