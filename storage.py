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
CERT_URL_TTL   = 86400 * 30  # 30 days — certified videos

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

# ── Upload raw video file ─────────────────────────────────────
def upload_video(job_id: str, file_path: str, filename: str) -> str:
    """
    Upload a raw video file to R2.
    Returns the R2 object key.
    Raises RuntimeError if R2 is not configured.
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
        ExtraArgs={"ContentType": "video/mp4"},
    )
    log.info("storage: uploaded %s → r2://%s/%s", filename, BUCKET, key)
    return key

# ── Download raw video file ───────────────────────────────────
def download_video(key: str, dest_path: str) -> None:
    """
    Download a video file from R2 to dest_path.
    """
    client = _get_client()
    client.download_file(BUCKET, key, dest_path)
    log.info("storage: downloaded r2://%s/%s → %s", BUCKET, key, dest_path)

# ── Delete raw video file ────────────────────────────────────
def delete_video(key: str) -> None:
    """Delete a video file from R2 (call after worker finishes)."""
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

def upload_certified(job_id: str, cert_path: str, plan: str = "free") -> str:
    """
    Upload a certified (stamped) video to R2.
    Tags with plan so Cloudflare lifecycle rules can expire by tier.
    Returns the R2 object key.
    """
    key = f"certified/{plan}/{job_id}.mp4"
    client = _get_client()
    client.upload_file(
        cert_path,
        BUCKET,
        key,
        ExtraArgs={
            "ContentType": "video/mp4",
            "Metadata": {
                "plan":   plan,
                "job_id": job_id,
            },
        },
    )
    log.info("storage: uploaded certified video → r2://%s/%s (plan=%s)", BUCKET, key, plan)
    return key

# ── Generate presigned download URL ──────────────────────────
def get_download_url(job_id: str, expires: int = CERT_URL_TTL) -> str:
    """
    Generate a presigned URL for downloading a certified video.
    Finds the correct key across plan subfolders.
    If R2_PUBLIC_URL is set, returns a direct public URL instead.
    """
    key = get_certified_key(job_id)

    if PUBLIC_URL:
        return f"{PUBLIC_URL.rstrip('/')}/{key}"

    client = _get_client()
    url = client.generate_presigned_url(
        "get_object",
        Params={"Bucket": BUCKET, "Key": key},
        ExpiresIn=expires,
    )
    return url

# ── Check if certified video exists ──────────────────────────
def certified_exists(job_id: str) -> bool:
    """Check whether a certified video exists in R2 (any plan subfolder)."""
    client = _get_client()
    for plan in ["free", "creator", "pro", "enterprise"]:
        try:
            client.head_object(Bucket=BUCKET, Key=f"certified/{plan}/{job_id}.mp4")
            return True
        except Exception:
            continue
    return False


def get_certified_key(job_id: str) -> str:
    """Find the R2 key for a certified video across plan subfolders."""
    client = _get_client()
    for plan in ["free", "creator", "pro", "enterprise"]:
        key = f"certified/{plan}/{job_id}.mp4"
        try:
            client.head_object(Bucket=BUCKET, Key=key)
            return key
        except Exception:
            continue
    return f"certified/free/{job_id}.mp4"  # fallback

# ── Convenience: is R2 available? ────────────────────────────
def r2_available() -> bool:
    return _is_configured()
