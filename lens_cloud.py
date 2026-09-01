# VeriFYD Lens Cloud Router v0.4
# Additive router: sends quarantined Lens files through the existing VeriFYD
# worker/detection pipelines without changing detection.py, worker.py, or queue_helper.py.

from fastapi import APIRouter, UploadFile, File, Header, HTTPException
from fastapi.responses import JSONResponse
from pathlib import Path
import os
import tempfile
import uuid

from queue_helper import (
    enqueue_upload,
    enqueue_photo_upload,
    enqueue_audio_upload,
    enqueue_document_upload,
    get_job_result,
)

from database import get_lens_entitlement

router = APIRouter(prefix="/lens", tags=["VeriFYD Lens"])

LENS_API_KEY = os.environ.get("VERIFYD_LENS_API_KEY", "").strip()
MAX_BYTES = int(os.environ.get("VERIFYD_LENS_MAX_BYTES", str(250 * 1024 * 1024)))

VIDEO_EXTS = {
    ".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm", ".mpg", ".mpeg",
    ".3gp", ".3g2", ".mts", ".m2ts", ".ts", ".ogv", ".flv", ".wmv",
}
PHOTO_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif"}
AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".oga", ".opus"}
DOCUMENT_EXTS = {
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    ".txt", ".rtf", ".csv", ".odt", ".ods", ".odp", ".yaml", ".yml",
}

# VERIFYD_LENS_ENTITLEMENT_AUTH_V1
def _authorize(
    x_verifyd_lens_key: str | None,
    x_verifyd_lens_entitlement: str | None = None,
):
    entitlement = (x_verifyd_lens_entitlement or "").strip()
    if entitlement:
        record = get_lens_entitlement(entitlement)
        if record and str(record.get("status") or "").upper() == "COMPLETED":
            return {
                "auth_type": "entitlement",
                "buyer_email": str(record.get("buyer_email") or ""),
            }
        raise HTTPException(status_code=401, detail="Invalid Lens entitlement.")

    shared_key = (x_verifyd_lens_key or "").strip()
    if shared_key and LENS_API_KEY and shared_key == LENS_API_KEY:
        return {
            "auth_type": "shared_key",
            "buyer_email": "",
        }

    raise HTTPException(status_code=401, detail="Lens authentication is required.")

def _media_type(filename: str) -> str:
    ext = Path(filename or "").suffix.lower()
    if ext in VIDEO_EXTS:
        return "video"
    if ext in PHOTO_EXTS:
        return "photo"
    if ext in AUDIO_EXTS:
        return "audio"
    if ext in DOCUMENT_EXTS:
        return "document"
    raise HTTPException(status_code=415, detail=f"Unsupported Lens file type: {ext or 'unknown'}")

@router.get("/health")
def lens_health(
    x_verifyd_lens_key: str | None = Header(default=None),
    x_verifyd_lens_entitlement: str | None = Header(default=None),
):
    _authorize(x_verifyd_lens_key, x_verifyd_lens_entitlement)
    return {
        "ok": True,
        "product": "VeriFYD Lens Cloud",
        "version": "0.4.0",
        "pipelines": ["video", "photo", "audio", "document"],
    }

@router.post("/analyze")
async def lens_analyze(
    file: UploadFile = File(...),
    x_verifyd_lens_key: str | None = Header(default=None),
    x_verifyd_lens_entitlement: str | None = Header(default=None),
):
    auth_context = _authorize(x_verifyd_lens_key, x_verifyd_lens_entitlement)

    filename = os.path.basename(file.filename or "lens-upload.bin")
    media_type = _media_type(filename)
    job_id = str(uuid.uuid4())

    suffix = Path(filename).suffix
    fd, raw_path = tempfile.mkstemp(prefix=f"lens_{job_id[:8]}_", suffix=suffix)
    os.close(fd)

    written = 0
    try:
        with open(raw_path, "wb") as out:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                written += len(chunk)
                if written > MAX_BYTES:
                    raise HTTPException(status_code=413, detail="Lens cloud upload exceeds configured size limit.")
                out.write(chunk)

        tracking_email = (
            auth_context.get("buyer_email")
            if auth_context.get("auth_type") == "entitlement" and auth_context.get("buyer_email")
            else f"lens_{job_id[:12]}@verifyd-enterprise.com"
        )

        if media_type == "video":
            enqueue_upload(job_id, raw_path, filename, tracking_email, suppress_email=True)
        elif media_type == "photo":
            enqueue_photo_upload(job_id, raw_path, filename, tracking_email, suppress_email=True)
        elif media_type == "audio":
            enqueue_audio_upload(job_id, raw_path, filename, tracking_email, suppress_email=True)
        elif media_type == "document":
            enqueue_document_upload(job_id, raw_path, filename, tracking_email, suppress_email=True)

        # queue_helper stores the file in R2/Redis then removes raw_path.
        return JSONResponse({
            "ok": True,
            "job_id": job_id,
            "job_status": "queued",
            "media_type": media_type,
            "filename": filename,
        })

    except HTTPException:
        try:
            if os.path.exists(raw_path):
                os.remove(raw_path)
        except Exception:
            pass
        raise
    except Exception as exc:
        try:
            if os.path.exists(raw_path):
                os.remove(raw_path)
        except Exception:
            pass
        raise HTTPException(status_code=500, detail=f"Lens analysis could not be queued: {type(exc).__name__}")

@router.get("/job/{job_id}")
def lens_job(
    job_id: str,
    x_verifyd_lens_key: str | None = Header(default=None),
    x_verifyd_lens_entitlement: str | None = Header(default=None),
):
    _authorize(x_verifyd_lens_key, x_verifyd_lens_entitlement)
    result = get_job_result(job_id) or {}

    lifecycle = str(
        result.get("job_status")
        or result.get("job_state")
        or result.get("status")
        or "processing"
    ).lower()

    if lifecycle == "not_found":
        return JSONResponse(
            {"job_id": job_id, "job_status": "not_found"},
            status_code=404,
        )

    # Return only fields useful to Lens. Do not leak internal tracebacks/paths.
    safe = {
        "job_id": job_id,
        "job_status": lifecycle,
        "media_type": result.get("media_type"),
        "label": result.get("label"),
        "authenticity_score": result.get("authenticity_score"),
        "ai_score": result.get("ai_score"),
        "gpt_reasoning": result.get("gpt_reasoning"),
        "gpt_flags": result.get("gpt_flags"),
        "signal_score": result.get("signal_score"),
        "gpt_score": result.get("gpt_score"),
        "audio_score": result.get("audio_score"),
        "document_status": result.get("document_status"),
        "status": result.get("status"),
    }

    if lifecycle == "error":
        safe["error"] = "VeriFYD cloud analysis failed."
    return safe
