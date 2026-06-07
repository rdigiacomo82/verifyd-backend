# ============================================================
#  VeriFYD — storage.py
#
#  Unified storage layer: Cloudflare R2 (S3-compatible) for
#  video files and certified outputs.
#
#  Falls back to Redis if R2 env vars are not set, so the
#  existing system keeps working during transition.
#
#  Required env vars (set on Render):
#    R2_ACCOUNT_ID      — Cloudflare account ID
#    R2_ACCESS_KEY_ID   — R2 API token access key
#    R2_SECRET_KEY      — R2 API token secret key
#    R2_BUCKET          — bucket name (e.g. verifyd-videos)
#    R2_PUBLIC_URL      — public bucket URL (optional, for direct serving)
# ============================================================

import os
import logging
import mimetypes
import boto3
from botocore.config import Config

log = logging.getLogger("verifyd.storage")

# ── Configuration ─────────────────────────────────────────────
ACCOUNT_ID  = os.environ.get("R2_ACCOUNT_ID", "")
ACCESS_KEY  = os.environ.get("R2_ACCESS_KEY_ID", "")
SECRET_KEY  = os.environ.get("R2_SECRET_KEY", "")
BUCKET      = os.environ.get("R2_BUCKET", "verifyd-videos")
PUBLIC_URL  = os.environ.get("R2_PUBLIC_URL", "")  # optional CDN URL

# R2 endpoint follows this pattern
R2_ENDPOINT = f"https://{ACCOUNT_ID}.r2.cloudflarestorage.com" if ACCOUNT_ID else ""

# TTLs for presigned URLs (seconds)
UPLOAD_URL_TTL = 3600        # 1 hour — raw upload files
CERT_URL_TTL   = 86400 * 6   # 6 days - R2 presigned URL max is 7 days (604800s)

def _is_configured() -> bool:
    return bool(ACCOUNT_ID and ACCESS_KEY and SECRET_KEY and BUCKET)

def _get_client():
    """Return boto3 S3 client pointed at Cloudflare R2."""
    return boto3.client(
        "s3",
        endpoint_url=R2_ENDPOINT,
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=SECRET_KEY,
        config=Config(signature_version="s3v4"),
        region_name="auto",
    )


def _content_type_for_path(path: str, default: str = "application/octet-stream") -> str:
    guessed, _ = mimetypes.guess_type(path or "")
    return guessed or default

# ── Upload raw video/file upload ──────────────────────────────
def upload_video(job_id: str, file_path: str, filename: str) -> str:
    """
    Upload a raw file to R2.

    Historical name is upload_video because video was the first VeriFYD media
    type. This helper is also used by document/photo upload staging, so Phase 9A
    now stores a safer content type based on the uploaded filename.
    Returns the R2 object key.
    """
    if not _is_configured():
        raise RuntimeError("R2 not configured — missing env vars")

    suffix = os.path.splitext(filename)[1] or ".mp4"
    key    = f"uploads/{job_id}{suffix}"

    client = _get_client()
    client.upload_file(
        file_path,
        BUCKET,
        key,
        ExtraArgs={"ContentType": _content_type_for_path(filename, "application/octet-stream")},
    )
    log.info("storage: uploaded %s → r2://%s/%s", filename, BUCKET, key)
    return key

# ── Download raw video/file upload ────────────────────────────
def download_video(key: str, dest_path: str) -> None:
    """Download a staged upload from R2 to dest_path."""
    client = _get_client()
    client.download_file(BUCKET, key, dest_path)
    log.info("storage: downloaded r2://%s/%s → %s", BUCKET, key, dest_path)

# ── Delete raw video/file upload ──────────────────────────────
def delete_video(key: str) -> None:
    """Delete a staged upload from R2 (call after worker finishes)."""
    try:
        client = _get_client()
        client.delete_object(Bucket=BUCKET, Key=key)
        log.info("storage: deleted r2://%s/%s", BUCKET, key)
    except Exception as e:
        log.warning("storage: delete failed for %s: %s", key, e)

# ── Upload certified video ────────────────────────────────────
# Plan-based retention in days
CERT_RETENTION_DAYS = {
    "free":       1,    # 24 hours
    "creator":    3,    # 72 hours
    "pro":        7,    # 7 days
    "enterprise": 30,   # 30 days
}

_PLAN_ORDER = ["free", "creator", "pro", "enterprise"]
_LOOKUP_PLAN_ORDER = ["pro", "creator", "enterprise", "free"]

def upload_certified(job_id: str, cert_path: str, plan: str = "free") -> str:
    """Upload a certified (stamped) video to R2."""
    key = f"certified/{plan}/{job_id}.mp4"
    client = _get_client()
    client.upload_file(
        cert_path,
        BUCKET,
        key,
        ExtraArgs={
            "ContentType": "video/mp4",
            "Metadata": {"plan": plan, "job_id": job_id},
        },
    )
    log.info("storage: uploaded certified video → r2://%s/%s (plan=%s)", BUCKET, key, plan)
    return key

# ── Generate presigned download URL ──────────────────────────
def get_download_url(job_id: str, expires: int = CERT_URL_TTL) -> str:
    key = get_certified_key(job_id)
    if PUBLIC_URL:
        return f"{PUBLIC_URL.rstrip('/')}/{key}"
    client = _get_client()
    return client.generate_presigned_url(
        "get_object",
        Params={"Bucket": BUCKET, "Key": key},
        ExpiresIn=expires,
    )

# ── Check if certified video exists ──────────────────────────
def certified_exists(job_id: str) -> bool:
    client = _get_client()
    for plan in _PLAN_ORDER:
        try:
            client.head_object(Bucket=BUCKET, Key=f"certified/{plan}/{job_id}.mp4")
            return True
        except Exception:
            continue
    return False


def get_certified_key(job_id: str) -> str:
    client = _get_client()
    for plan in _PLAN_ORDER:
        key = f"certified/{plan}/{job_id}.mp4"
        try:
            client.head_object(Bucket=BUCKET, Key=key)
            return key
        except Exception:
            continue
    return f"certified/free/{job_id}.mp4"

def upload_certified_photo(job_id: str, cert_path: str,
                           plan: str = "free", ext: str = ".jpg") -> str:
    """Upload a certified (stamped) photo to R2."""
    content_types = {
        ".jpg":  "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png":  "image/png",
        ".webp": "image/webp",
    }
    content_type = content_types.get(ext.lower(), "image/jpeg")
    key = f"certified-photos/{plan}/{job_id}{ext}"

    client = _get_client()
    client.upload_file(
        cert_path,
        BUCKET,
        key,
        ExtraArgs={
            "ContentType": content_type,
            "Metadata": {"plan": plan, "job_id": job_id, "type": "photo"},
        },
    )
    log.info("storage: uploaded certified photo → r2://%s/%s (plan=%s)", BUCKET, key, plan)
    return key



def upload_certified_document(job_id: str, cert_path: str, plan: str = "free") -> str:
    """Upload a certified/stamped document PDF to R2."""
    key = f"certified-documents/{plan}/{job_id}.pdf"
    client = _get_client()
    client.upload_file(
        cert_path,
        BUCKET,
        key,
        ExtraArgs={
            "ContentType": "application/pdf",
            "Metadata": {"plan": plan, "job_id": job_id, "type": "document"},
        },
    )
    log.info("storage: uploaded certified document → r2://%s/%s (plan=%s)", BUCKET, key, plan)
    return key


def certified_document_exists(job_id: str) -> bool:
    """Check whether a certified document exists in R2 (any plan subfolder)."""
    client = _get_client()
    for plan in _PLAN_ORDER:
        try:
            client.head_object(Bucket=BUCKET, Key=f"certified-documents/{plan}/{job_id}.pdf")
            return True
        except Exception:
            continue
    return False


def get_certified_document_key(job_id: str) -> str:
    """Find the R2 key for a certified document across plan subfolders."""
    client = _get_client()
    for plan in _PLAN_ORDER:
        key = f"certified-documents/{plan}/{job_id}.pdf"
        try:
            client.head_object(Bucket=BUCKET, Key=key)
            return key
        except Exception:
            continue
    return f"certified-documents/free/{job_id}.pdf"


def get_document_download_url(job_id: str, expires: int = CERT_URL_TTL) -> str:
    """Generate a presigned URL for downloading a certified document PDF."""
    key = get_certified_document_key(job_id)
    if PUBLIC_URL:
        return f"{PUBLIC_URL.rstrip('/')}/{key}"
    client = _get_client()
    return client.generate_presigned_url(
        "get_object",
        Params={"Bucket": BUCKET, "Key": key},
        ExpiresIn=expires,
    )



# ─────────────────────────────────────────────
# Certified audio storage
# ─────────────────────────────────────────────

def upload_certified_audio(job_id: str, cert_path: str, plan: str = "free", ext: str = ".mp3") -> str:
    """Upload a certified audio file to R2 without changing the audible content."""
    ext = (ext or ".mp3").lower()
    content_types = {
        ".mp3": "audio/mpeg",
        ".wav": "audio/wav",
        ".m4a": "audio/mp4",
        ".aac": "audio/aac",
        ".flac": "audio/flac",
        ".ogg": "audio/ogg",
        ".oga": "audio/ogg",
        ".opus": "audio/opus",
        ".webm": "audio/webm",
    }
    key = f"certified-audio/{plan}/{job_id}{ext}"
    client = _get_client()
    client.upload_file(
        cert_path,
        BUCKET,
        key,
        ExtraArgs={
            "ContentType": content_types.get(ext, "application/octet-stream"),
            "Metadata": {"plan": plan, "job_id": job_id, "type": "audio", "ext": ext.lstrip(".")},
        },
    )
    log.info("storage: uploaded certified audio → r2://%s/%s (plan=%s)", BUCKET, key, plan)
    return key

def certified_audio_exists(job_id: str) -> bool:
    client = _get_client()
    for plan in _PLAN_ORDER:
        for ext in (".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".oga", ".opus", ".webm"):
            try:
                client.head_object(Bucket=BUCKET, Key=f"certified-audio/{plan}/{job_id}{ext}")
                return True
            except Exception:
                continue
    return False

def get_certified_audio_key(job_id: str) -> str:
    client = _get_client()
    for plan in _PLAN_ORDER:
        for ext in (".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".oga", ".opus", ".webm"):
            key = f"certified-audio/{plan}/{job_id}{ext}"
            try:
                client.head_object(Bucket=BUCKET, Key=key)
                return key
            except Exception:
                continue
    return f"certified-audio/free/{job_id}.mp3"

def get_audio_download_url(job_id: str, expires: int = CERT_URL_TTL) -> str:
    """Generate a presigned certified-audio URL with download headers when possible."""
    key = get_certified_audio_key(job_id)
    ext = os.path.splitext(key)[1].lower() or ".mp3"
    content_types = {
        ".mp3": "audio/mpeg",
        ".wav": "audio/wav",
        ".m4a": "audio/mp4",
        ".aac": "audio/aac",
        ".flac": "audio/flac",
        ".ogg": "audio/ogg",
        ".oga": "audio/ogg",
        ".opus": "audio/opus",
        ".webm": "audio/webm",
    }
    if PUBLIC_URL:
        # PUBLIC_URL cannot force Content-Disposition; main.py /download-audio/{cid}
        # proxies R2 objects to guarantee attachment downloads.
        return f"{PUBLIC_URL.rstrip('/')}/{key}"
    client = _get_client()
    return client.generate_presigned_url(
        "get_object",
        Params={
            "Bucket": BUCKET,
            "Key": key,
            "ResponseContentDisposition": f'attachment; filename="VeriFYD_Certified_Audio_{job_id[:8]}{ext}"',
            "ResponseContentType": content_types.get(ext, "application/octet-stream"),
        },
        ExpiresIn=expires,
    )

# ─────────────────────────────────────────────
# Phase 9A — Universal Certified File Package
# ─────────────────────────────────────────────

def upload_certified_file_package(job_id: str, package_path: str, plan: str = "free") -> str:
    """
    Upload a universal VeriFYD certified file package to R2.
    Stored at certified-files/{plan}/{job_id}.zip.
    """
    key = f"certified-files/{plan}/{job_id}.zip"
    client = _get_client()
    client.upload_file(
        package_path,
        BUCKET,
        key,
        ExtraArgs={
            "ContentType": "application/zip",
            "Metadata": {
                "plan": plan,
                "job_id": job_id,
                "type": "universal_certified_file",
                "format": "zip",
            },
        },
    )
    log.info("storage: uploaded certified file package → r2://%s/%s (plan=%s)", BUCKET, key, plan)
    return key


def certified_file_package_exists(job_id: str) -> bool:
    """Check whether a universal certified file package exists in R2."""
    client = _get_client()
    for plan in _PLAN_ORDER:
        try:
            client.head_object(Bucket=BUCKET, Key=f"certified-files/{plan}/{job_id}.zip")
            return True
        except Exception:
            continue
    return False


def get_certified_file_package_key(job_id: str) -> str:
    """Find the R2 key for a universal certified file package."""
    client = _get_client()
    for plan in _PLAN_ORDER:
        key = f"certified-files/{plan}/{job_id}.zip"
        try:
            client.head_object(Bucket=BUCKET, Key=key)
            return key
        except Exception:
            continue
    return f"certified-files/free/{job_id}.zip"


def get_certified_file_package_download_url(job_id: str, expires: int = CERT_URL_TTL) -> str:
    """Generate a presigned URL for downloading a universal certified file package."""
    key = get_certified_file_package_key(job_id)
    if PUBLIC_URL:
        return f"{PUBLIC_URL.rstrip('/')}/{key}"
    client = _get_client()
    return client.generate_presigned_url(
        "get_object",
        Params={"Bucket": BUCKET, "Key": key},
        ExpiresIn=expires,
    )


# ── Convenience: is R2 available? ────────────────────────────
def r2_available() -> bool:
    return _is_configured()



