# ============================================================
#  VeriFYD — main.py
#
#  FastAPI application entry point.
#
#  Detection pipeline:
#    upload / analyze-link
#      → clip_first_6_seconds()
#        → run_detection()          (detection.py)
#          → detect_ai()            (detector.py)
#            → authenticity, label, detail
#              → certify / respond
# ============================================================

from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
import os, uuid, requests, tempfile, logging, hashlib
from contextlib import asynccontextmanager

from detection import run_detection   # returns (authenticity, label, detail)
from queue_helper import (enqueue_upload, enqueue_link,
                          enqueue_photo_upload, enqueue_photo_link,
                          enqueue_audio_upload,
                          enqueue_document_upload,
                          enqueue_trust_desk_zip,
                          get_job_result)
from config import (                  # single source of truth for all settings
    BASE_URL,
    UPLOAD_DIR, CERT_DIR, TMP_DIR,
)
from emailer  import send_otp_email, send_certification_email, send_enterprise_welcome_email
from database import (init_db, insert_certificate, increment_downloads,
                      get_or_create_user, get_user_status, increment_user_uses,
                      get_user_by_email, get_email_typo_suggestion,
                      is_valid_email, FREE_USES, get_certificate,
                      save_certificate_to_vault, get_vault_record, get_vault_record_by_cert_id,
                      is_email_verified, create_otp, verify_otp,
                      create_api_key, get_api_key, increment_api_key_uses,
                      revoke_api_key, list_api_keys, update_api_key_branding)
from video import clip_first_6_seconds, stamp_video, download_video_ytdlp

log = logging.getLogger("verifyd.main")

# Admin key — set ADMIN_KEY env var on Render; falls back to default for local dev
_ADMIN_KEY = os.environ.get("ADMIN_KEY", "Honda6915")

def _cleanup_old_files(max_age_hours: float = 2.0) -> None:
    """
    Remove uploaded and certified videos older than max_age_hours.
    Called automatically after each upload to prevent disk fill.
    """
    import glob, time
    now = time.time()
    max_age = max_age_hours * 3600
    for directory in (UPLOAD_DIR, CERT_DIR, TMP_DIR):
        if not os.path.isdir(directory):
            continue
        for f in glob.glob(os.path.join(directory, "*")):
            try:
                if os.path.isfile(f) and (now - os.path.getmtime(f)) > max_age:
                    os.remove(f)
            except Exception:
                pass


def _is_admin(key: str) -> bool:
    """Check admin key — supports both URL-encoded and raw forms."""
    key = (key or "").strip()
    return key == _ADMIN_KEY or key == _ADMIN_KEY.replace("#", "%23")


# ─────────────────────────────────────────────
#  Email verification / auto-verified allowlist
# ─────────────────────────────────────────────
# Preserve the original DB verification function, then shadow is_email_verified
# so all existing upload/link/audio/document/Trust Desk routes automatically
# honor the allowlist without broad route rewrites.
_db_is_email_verified = is_email_verified


def normalize_email(email: str) -> str:
    """Normalize email consistently without truncation or alias creation."""
    return (email or "").strip().lower()


def _email_typo_json(email: str):
    """Return a JSONResponse for common email-domain typos, otherwise None."""
    suggestion = get_email_typo_suggestion(email)
    if not suggestion:
        return None
    normalized = normalize_email(email)
    return JSONResponse({
        "error": "possible_email_typo",
        "message": f"Did you mean {suggestion}?",
        "email": normalized,
        "suggested_email": suggestion,
    }, status_code=400)


def _auto_verified_email_patterns() -> set:
    raw = os.environ.get("VERIFYD_AUTO_VERIFIED_EMAILS", "")
    return {normalize_email(e) for e in raw.split(",") if normalize_email(e)}


def is_auto_verified_email(email: str) -> bool:
    """Return True for exact allowlisted emails or wildcard domain entries."""
    normalized = normalize_email(email)
    if not normalized:
        return False
    patterns = _auto_verified_email_patterns()
    if normalized in patterns:
        log.info("AUTO_VERIFIED_EMAIL accepted: %s", normalized)
        return True
    if "@" in normalized:
        domain = normalized.split("@", 1)[1]
        wildcard = f"*@{domain}"
        if wildcard in patterns:
            log.info("AUTO_VERIFIED_EMAIL accepted by wildcard %s: %s", wildcard, normalized)
            return True
    return False


def is_email_verified(email: str) -> bool:
    """
    Shared server-side verification source of truth.
    This intentionally shadows the imported database.is_email_verified so existing
    route code keeps working while gaining allowlist support.
    """
    normalized = normalize_email(email)
    if is_auto_verified_email(normalized):
        return True
    try:
        return bool(_db_is_email_verified(normalized))
    except Exception as exc:
        log.warning("email verification lookup failed for %s: %s", normalized, exc)
        return False


def is_email_verified_or_allowlisted(email: str) -> dict:
    normalized = normalize_email(email)
    auto_verified = is_auto_verified_email(normalized)
    verified = True if auto_verified else is_email_verified(normalized)
    return {
        "email": normalized,
        "verified": bool(verified),
        "auto_verified": bool(auto_verified),
    }


def _verification_status_payload(email: str) -> dict:
    check = is_email_verified_or_allowlisted(email)
    normalized = check["email"]
    plan = "free"
    uses = 0
    limit = FREE_USES
    try:
        if normalized and is_valid_email(normalized):
            status = get_user_status(normalized, create=False)
            plan = status.get("plan", plan)
            uses = int(status.get("uses", status.get("used", status.get("period_uses", uses))) or 0)
            limit = int(status.get("limit", limit) or limit)
    except Exception as exc:
        log.warning("verification-status usage lookup failed for %s: %s", normalized, exc)
    return {
        "email": normalized,
        "verified": bool(check["verified"]),
        "plan": plan,
        "uses": uses,
        "limit": limit,
        "auto_verified": bool(check["auto_verified"]),
    }

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    init_db()
    log.info("VeriFYD startup complete")

    # ── Keepalive scheduler ─────────────────────────────────
    # Enqueues a lightweight no-op job every 8 minutes to prevent
    # Render from suspending the worker process between real jobs.
    # Cold starts add ~15-20s (model pre-warm) — this eliminates them.
    import threading as _threading
    import time as _time

    def _keepalive_scheduler():
        """Background thread that enqueues keepalive pings every 8 minutes."""
        _time.sleep(60)  # Wait 1 minute after startup before first ping
        while True:
            try:
                import redis as _redis
                import rq as _rq
                _r = _redis.from_url(
                    os.environ.get("REDIS_URL", "redis://localhost:6379"),
                    decode_responses=False
                )
                _q = _rq.Queue("verifyd", connection=_r)
                from worker import keepalive_ping
                _q.enqueue(keepalive_ping, job_timeout=60, result_ttl=60)
                log.info("Keepalive: enqueued ping to keep worker warm")
            except Exception as _ke:
                log.debug("Keepalive: enqueue failed (%s) — worker may be restarting", _ke)
            _time.sleep(480)  # 8 minutes

    _t = _threading.Thread(target=_keepalive_scheduler, daemon=True, name="keepalive")
    _t.start()
    log.info("Keepalive scheduler started — pinging worker every 8 minutes")

    yield
    # Shutdown (add cleanup here if needed)

app = FastAPI(title="VeriFYD", lifespan=lifespan)
# Directories are created on import inside config.py — no makedirs needed here

# ─────────────────────────────────────────────
#  File upload size — no hard limit on backend.
#  FastAPI streams the file directly to disk so
#  memory usage stays flat regardless of size.
#  Frontend should accept any size too (no JS
#  file size validation that blocks large files).
# ─────────────────────────────────────────────
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

class NoSizeLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request._body_size_limit = None   # disable Starlette's default cap
        return await call_next(request)

# ─────────────────────────────────────────────
#  Label → UI display mapping
#  Keep presentation logic here, detection
#  thresholds stay in detection.py.
# ─────────────────────────────────────────────
# VERIFYD_VIDEO_FORMAT_EXPANSION_PATCH: accepted public video containers.
# ffprobe/video.is_valid_video remains the source of truth that a file actually contains video.
VIDEO_ALLOWED_EXTENSIONS = {
    ".mp4", ".mov", ".m4v", ".avi", ".webm", ".mkv",
    ".mpg", ".mpeg", ".3gp", ".3g2", ".mts", ".m2ts",
    ".ts", ".ogv", ".flv", ".wmv",
}
VIDEO_ACCEPT_STRING = "video/*,.mp4,.mov,.m4v,.avi,.webm,.mkv,.mpg,.mpeg,.3gp,.3g2,.mts,.m2ts,.ts,.ogv,.flv,.wmv"
VIDEO_FORMAT_LABEL = "MP4, MOV, M4V, AVI, WEBM, MKV, MPEG/MPG, 3GP/3G2, MTS/M2TS, TS, OGV, FLV, WMV"

LABEL_UI = {
    "REAL":          ("REAL VIDEO VERIFIED", "green",  True),
    "UNDETERMINED":  ("VIDEO UNDETERMINED",  "blue",   False),
    "AI":            ("AI DETECTED",         "red",    False),
}

# ─────────────────────────────────────────────
#  CORS
# ─────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://vfvid.com",
        "https://www.vfvid.com",
        "http://localhost:5173",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
#  Health
# ─────────────────────────────────────────────
@app.get("/")
def home():
    return {"status": "VeriFYD API LIVE"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/debug-proxy")
def debug_proxy():
    import re
    raw = os.environ.get("RESIDENTIAL_PROXY_URL", "NOT SET")
    masked = re.sub(r':(.*?)@', ':***@', raw)
    rapidapi_key = os.environ.get("RAPIDAPI_KEY", "")
    return {
        "proxy_raw_length":   len(raw),
        "proxy_masked":       masked,
        "rapidapi_key_set":   bool(rapidapi_key),
        "rapidapi_key_length": len(rapidapi_key),
        "rapidapi_key_preview": rapidapi_key[:8] + "..." if rapidapi_key else "NOT SET",
    }

# ─────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────


def _sha256(path: str) -> str:
    """Return the SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _run_analysis(source_path: str) -> tuple:
    """
    Clip → run_detection() (detection.py handles signal+GPT in parallel)
    → return (authenticity, label, detail, ui_text, color, certify, clip_path).
    clip_path is preserved for stamping; caller must delete it after use.
    """
    clip_path = clip_first_6_seconds(source_path)
    try:
        authenticity, label, detail = run_detection(clip_path)
    except Exception:
        # Clean up clip on detection error
        if os.path.exists(clip_path):
            os.remove(clip_path)
        raise

    ui_text, color, certify = LABEL_UI.get(label, ("UNKNOWN", "grey", False))
    log.info("Analysis: label=%s authenticity=%d ai_score=%d",
             label, authenticity, detail["ai_score"])
    # Return clip_path — caller stamps from it then cleans up
    return authenticity, label, detail, ui_text, color, certify, clip_path


def _stamp_video_background(
    raw_path: str, certified_path: str, cid: str,
    email: str = "", authenticity: int = 0,
    original_filename: str = "", download_url: str = ""
):
    """
    Stamp video in background thread, then send certification email.
    Cleans up clip file always. If stamp fails, removes any partial
    certified_path so download endpoint returns a clean 404 instead
    of serving a corrupt/empty file.
    """
    import traceback
    stamp_ok = False
    try:
        log.info("Background stamp starting: input=%s exists=%s", raw_path, os.path.exists(raw_path))
        if not os.path.exists(raw_path):
            log.error("Background stamp FAILED — clip not found: %s", raw_path)
            return
        stamp_video(raw_path, certified_path, cid)
        exists = os.path.exists(certified_path)
        size   = os.path.getsize(certified_path) if exists else 0
        log.info("Background stamp complete: %s  output_exists=%s  size=%d", cid, exists, size)
        if not exists or size < 1000:
            log.error("Stamp produced empty/missing output for %s — removing", cid)
            if exists:
                os.remove(certified_path)
            return
        stamp_ok = True
    except Exception:
        log.error("Background stamp EXCEPTION for %s:\n%s", cid, traceback.format_exc())
        # Remove any partial output so download returns 404 not corrupt data
        if os.path.exists(certified_path):
            os.remove(certified_path)
        return
    finally:
        # Always clean up the clip file
        if os.path.exists(raw_path):
            os.remove(raw_path)
            log.info("Cleaned up clip: %s", raw_path)

    # Send email only after confirmed successful stamp
    if stamp_ok and email:
        try:
            send_certification_email(email, cid, authenticity, original_filename, download_url)
            log.info("Certification email sent to %s for cert %s", email, cid)
        except Exception as e:
            log.error("Certification email failed for %s: %s", cid, e)


# ─────────────────────────────────────────────
#  Abstract API Email Validation
# ─────────────────────────────────────────────
import urllib.request as _urllib_req
import urllib.parse   as _urllib_parse

_ABSTRACT_KEY = os.environ.get("ABSTRACT_EMAIL_KEY", "") or os.environ.get("ABSTRACT_API_KEY", "")
_email_cache: dict = {}   # cache results to avoid duplicate API calls

def _verify_email_deliverable(email: str) -> tuple:
    """
    Check if email is real and deliverable using Abstract API.
    Returns (is_valid: bool, reason: str)
    Caches results to avoid repeated API calls for same email.
    """
    email_lower = normalize_email(email)

    # Approved demo/admin emails are trusted by backend allowlist.
    if is_auto_verified_email(email_lower):
        return True, "auto_verified"

    # Return cached result
    if email_lower in _email_cache:
        return _email_cache[email_lower]

    # Whitelist major trusted email providers — never block these
    # Gmail plus addresses (user+tag@gmail.com) are also covered by gmail.com domain
    TRUSTED_DOMAINS = {
        "gmail.com", "yahoo.com", "yahoo.co.uk", "yahoo.ca",
        "outlook.com", "hotmail.com", "hotmail.co.uk",
        "icloud.com", "me.com", "mac.com",
        "protonmail.com", "proton.me",
        "aol.com", "msn.com", "live.com",
    }
    domain = email_lower.split("@")[-1] if "@" in email_lower else ""
    if domain in TRUSTED_DOMAINS:
        log.info("Email deliverability: %s → trusted domain, skipping API check", email_lower)
        return True, "trusted_domain"

    # No API key — skip verification
    if not _ABSTRACT_KEY:
        log.warning("ABSTRACT_EMAIL_KEY not set — skipping deliverability check")
        return True, "unchecked"

    try:
        params = _urllib_parse.urlencode({"api_key": _ABSTRACT_KEY, "email": email_lower})
        url = f"https://emailreputation.abstractapi.com/v1/?{params}"
        req = _urllib_req.Request(url, headers={"User-Agent": "VeriFYD/1.0"})
        with _urllib_req.urlopen(req, timeout=5) as resp:
            import json as _json
            data = _json.loads(resp.read().decode("utf-8"))

        log.info("Abstract email reputation check: %s → %s", email_lower, data)

        # Email Reputation API response fields:
        # status: "DELIVERABLE", "UNDELIVERABLE", "RISKY", "UNKNOWN"
        # is_disposable_email: true/false (direct boolean)
        # is_valid_format: true/false (direct boolean)
        # quality_score: float 0.0-1.0
        status        = str(data.get("status", "UNKNOWN")).upper()
        is_disposable = bool(data.get("is_disposable_email", False))
        is_valid_fmt  = bool(data.get("is_valid_format", True))
        quality_score = float(data.get("quality_score", 0.70) or 0.70)

        log.info("Email reputation result: status=%s disposable=%s valid_fmt=%s quality=%.2f",
                 status, is_disposable, is_valid_fmt, quality_score)

        if not is_valid_fmt:
            result = (False, "Invalid email format.")
        elif is_disposable:
            result = (False, "Disposable or temporary email addresses are not allowed.")
        elif status == "UNDELIVERABLE":
            result = (False, "This email address does not appear to exist. Please use a real email.")
        elif quality_score < 0.40:
            result = (False, "This email address could not be verified. Please use a valid email.")
        else:
            result = (True, "ok")

        _email_cache[email_lower] = result
        return result

    except Exception as e:
        log.warning("Abstract email validation error: %s — allowing email", e)
        # On API error, allow the email (don't block users due to API issues)
        return True, "unchecked"


# ─────────────────────────────────────────────
#  Routes
# ─────────────────────────────────────────────


@app.get("/verification-status/")
def verification_status(email: str = ""):
    """Return backend email verification status for the frontend before upload."""
    email = normalize_email(email)
    if not email:
        return JSONResponse({"detail": "email_required"}, status_code=400)
    if not is_valid_email(email):
        return JSONResponse({"detail": "invalid_email", "email": email}, status_code=400)
    typo_response = _email_typo_json(email)
    if typo_response:
        return typo_response
    return JSONResponse(_verification_status_payload(email))

@app.post("/upload/")
async def upload(file: UploadFile = File(...), email: str = Form(...)):
    # ── Email format validation ───────────────────────────────
    if not is_valid_email(email):
        return JSONResponse({"error": "Invalid email address."}, status_code=400)
    typo_response = _email_typo_json(email)
    if typo_response:
        return typo_response

    # ── Email deliverability check ────────────────────────────
    is_deliverable, reason = _verify_email_deliverable(email)
    if not is_deliverable:
        return JSONResponse({"error": reason}, status_code=400)

    # ── Email verification check ──────────────────────────────
    if not is_email_verified(email):
        return JSONResponse({
            "error":            "email_not_verified",
            "message":          "Please verify your email address before uploading.",
        }, status_code=403)

    # ── Proactive disk cleanup ───────────────────────────────
    # Run before each upload to prevent disk-full errors
    try:
        _cleanup_old_files(max_age_hours=2.0)
    except Exception as _ce:
        log.warning("Disk cleanup error: %s", _ce)

    # ── Usage limit check ─────────────────────────────────────
    status = get_user_status(email)
    if not status["allowed"]:
        return JSONResponse({
            "error":      "limit_reached",
            "plan":       status["plan"],
            "uses_left":  0,
            "limit":      status["limit"],
        }, status_code=402)

    # ── Plan-based file size limit ───────────────────────────
    PLAN_SIZE_LIMITS = {
        "free":        50 * 1024 * 1024,
        "creator":    150 * 1024 * 1024,
        "pro":        500 * 1024 * 1024,
        "enterprise": 2048 * 1024 * 1024,
    }
    user_plan  = status["plan"]
    max_bytes  = PLAN_SIZE_LIMITS.get(user_plan, PLAN_SIZE_LIMITS["free"])
    max_mb     = max_bytes // (1024 * 1024)
    plan_label = {"free": "Free", "creator": "Creator",
                  "pro": "Pro", "enterprise": "Enterprise"}.get(user_plan, user_plan.title())

    # Check Content-Length header if present (fast path)
    content_length = file.size
    if content_length and content_length > max_bytes:
        return JSONResponse({
            "error":   "file_too_large",
            "message": f"File exceeds the {max_mb}MB limit for your {plan_label} plan.",
            "max_mb":  max_mb,
            "plan":    user_plan,
        }, status_code=413)

    # ── Save file to disk then enqueue ────────────────────────
    job_id   = str(uuid.uuid4())
    raw_path = f"{UPLOAD_DIR}/{job_id}_{file.filename}"

    # Stream to disk while enforcing size limit
    bytes_written = 0
    with open(raw_path, "wb") as f:
        while chunk := await file.read(1024 * 1024):
            bytes_written += len(chunk)
            if bytes_written > max_bytes:
                try:
                    os.remove(raw_path)
                except Exception:
                    pass
                return JSONResponse({
                    "error":   "file_too_large",
                    "message": f"File exceeds the {max_mb}MB limit for your {plan_label} plan.",
                    "max_mb":  max_mb,
                    "plan":    user_plan,
                }, status_code=413)
            f.write(chunk)

    log.info("upload: saved %dMB file for %s (plan=%s limit=%dMB)",
             bytes_written // (1024*1024), email, user_plan, max_mb)

    try:
        enqueue_upload(job_id, raw_path, file.filename, email)
        log.info("upload: queued job %s for %s file=%s", job_id, email, file.filename)
    except Exception as e:
        # Redis unavailable — fall back to synchronous processing
        log.warning("Queue unavailable (%s) — falling back to sync processing", e)
        try:
            sha256 = _sha256(raw_path)
            authenticity, label, detail, ui_text, color, certify, clip_path = _run_analysis(raw_path)
            increment_user_uses(email)
            cid = job_id
            insert_certificate(
                cert_id=cid, email=email, original_file=file.filename,
                label=label, authenticity=authenticity,
                ai_score=detail["ai_score"], sha256=sha256,
            )
            if certify:
                certified_path = f"{CERT_DIR}/{cid}.mp4"
                download_url   = f"{BASE_URL}/download/{cid}"
                import threading
                threading.Thread(
                    target=_stamp_video_background,
                    kwargs=dict(raw_path=clip_path, certified_path=certified_path,
                                cid=cid, email=email, authenticity=authenticity,
                                original_filename=file.filename, download_url=download_url),
                    daemon=True
                ).start()
                return {"status": ui_text, "authenticity_score": authenticity,
                        "certificate_id": cid, "download_url": download_url,
                        "color": color, "gpt_reasoning": detail.get("gpt_reasoning",""),
                        "gpt_flags": detail.get("gpt_flags",[]),
                        "signal_score": detail.get("signal_ai_score",0),
                        "gpt_score": detail.get("gpt_ai_score",0),
                        "audio_score": detail.get("audio_ai_score",50),
                        "audio_confidence": detail.get("audio_confidence","unavailable"),
                        "audio_contribution": detail.get("audio_contribution",0)}
            if os.path.exists(clip_path):
                os.remove(clip_path)
            return {"status": ui_text, "authenticity_score": authenticity, "color": color,
                    "gpt_reasoning": detail.get("gpt_reasoning",""),
                    "gpt_flags": detail.get("gpt_flags",[]),
                    "signal_score": detail.get("signal_ai_score",0),
                    "gpt_score": detail.get("gpt_ai_score",0),
                    "audio_score": detail.get("audio_ai_score",50),
                    "audio_confidence": detail.get("audio_confidence","unavailable"),
                    "audio_contribution": detail.get("audio_contribution",0)}
        except Exception as e2:
            log.exception("Sync fallback also failed for %s", raw_path)
            return JSONResponse({"error": str(e2)}, status_code=500)

    # ── Return immediately — frontend polls /job-status/{job_id} ──────────
    # This allows the worker optimization to take effect: the worker can store
    # an early detection result with video_ready=False, and the UI can display
    # the authenticity score while stamping / R2 upload / email finish after.
    # IMPORTANT: this does not change detection or stamping; it only stops the
    # upload request from blocking until the whole RQ job completes.
    log.info("upload: returning job_id=%s for frontend polling", job_id)
    return JSONResponse({"job_id": job_id, "status": "queued"})



# ─────────────────────────────────────────────
#  Redis helper (shared by photo endpoints)
# ─────────────────────────────────────────────
def _get_redis():
    import redis as _redis
    return _redis.from_url(
        os.environ.get("REDIS_URL", "redis://localhost:6379"),
        decode_responses=False,
    )


# ─────────────────────────────────────────────
#  Photo upload endpoint
# ─────────────────────────────────────────────
PHOTO_ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif"}
PHOTO_SIZE_LIMITS = {
    "free":        10 * 1024 * 1024,   # 10MB
    "creator":     25 * 1024 * 1024,   # 25MB
    "pro":         50 * 1024 * 1024,   # 50MB
    "enterprise":  200 * 1024 * 1024,  # 200MB
}

PHOTO_LABEL_UI = {
    "REAL":         ("REAL PHOTO VERIFIED", "green",  True),
    "UNDETERMINED": ("PHOTO UNDETERMINED",  "blue",   False),
    "AI":           ("AI DETECTED",         "red",    False),
}


@app.post("/upload-photo/")
async def upload_photo(file: UploadFile = File(...), email: str = Form(...)):
    """
    Photo upload endpoint. Mirrors /upload/ but for still images.
    Accepts: JPEG, PNG, WebP, HEIC/HEIF.
    """
    # ── Email validation ──────────────────────────────────────
    if not is_valid_email(email):
        return JSONResponse({"error": "Invalid email address."}, status_code=400)
    typo_response = _email_typo_json(email)
    if typo_response:
        return typo_response

    is_deliverable, reason = _verify_email_deliverable(email)
    if not is_deliverable:
        return JSONResponse({"error": reason}, status_code=400)

    if not is_email_verified(email):
        return JSONResponse({
            "error":   "email_not_verified",
            "message": "Please verify your email address before uploading.",
        }, status_code=403)

    # ── File type check ───────────────────────────────────────
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in PHOTO_ALLOWED_EXTENSIONS:
        return JSONResponse({
            "error":   "unsupported_format",
            "message": f"Unsupported image format '{ext}'. "
                       f"Accepted formats: JPEG, PNG, WebP, HEIC.",
        }, status_code=415)

    # ── Disk cleanup ──────────────────────────────────────────
    try:
        _cleanup_old_files(max_age_hours=2.0)
    except Exception:
        pass

    # ── Usage limit check ─────────────────────────────────────
    status = get_user_status(email)
    if not status["allowed"]:
        return JSONResponse({
            "error":     "limit_reached",
            "plan":      status["plan"],
            "uses_left": 0,
            "limit":     status["limit"],
        }, status_code=402)

    # ── Plan-based size limit ─────────────────────────────────
    user_plan = status["plan"]
    max_bytes = PHOTO_SIZE_LIMITS.get(user_plan, PHOTO_SIZE_LIMITS["free"])
    max_mb    = max_bytes // (1024 * 1024)
    plan_label = {"free": "Free", "creator": "Creator",
                  "pro": "Pro", "enterprise": "Enterprise"}.get(user_plan, user_plan.title())

    content_length = file.size
    if content_length and content_length > max_bytes:
        return JSONResponse({
            "error":   "file_too_large",
            "message": f"Image exceeds the {max_mb}MB limit for your {plan_label} plan.",
            "max_mb":  max_mb,
            "plan":    user_plan,
        }, status_code=413)

    # ── Save to disk ──────────────────────────────────────────
    job_id   = str(uuid.uuid4())
    raw_path = f"{UPLOAD_DIR}/{job_id}_{file.filename}"

    bytes_written = 0
    with open(raw_path, "wb") as f_out:
        while chunk := await file.read(1024 * 1024):
            bytes_written += len(chunk)
            if bytes_written > max_bytes:
                try:
                    os.remove(raw_path)
                except Exception:
                    pass
                return JSONResponse({
                    "error":   "file_too_large",
                    "message": f"Image exceeds the {max_mb}MB limit for your {plan_label} plan.",
                    "max_mb":  max_mb,
                    "plan":    user_plan,
                }, status_code=413)
            f_out.write(chunk)

    log.info("photo_upload: saved %dKB for %s (plan=%s)",
             bytes_written // 1024, email, user_plan)

    # ── Enqueue ───────────────────────────────────────────────
    try:
        enqueue_photo_upload(job_id, raw_path, file.filename, email)
        log.info("photo_upload: queued job %s for %s file=%s", job_id, email, file.filename)
    except Exception as e:
        log.warning("Queue unavailable (%s) — falling back to sync", e)
        from photo_detection import run_photo_detection
        try:
            authenticity, label, detail = run_photo_detection(raw_path)
            ui_text, color, _ = PHOTO_LABEL_UI.get(label, ("PHOTO UNDETERMINED", "blue", False))
            return JSONResponse({
                "status":             ui_text,
                "authenticity_score": authenticity,
                "color":              color,
                "label":              label,
                "gpt_reasoning":      detail.get("gpt_reasoning", ""),
                "gpt_flags":          detail.get("gpt_flags", []),
                "signal_score":       detail.get("signal_ai_score", 0),
                "gpt_score":          detail.get("gpt_ai_score", 0),
                "media_type":         "photo",
            })
        except Exception as sync_e:
            return JSONResponse({"error": str(sync_e)[:200]}, status_code=500)
        finally:
            if os.path.exists(raw_path):
                os.remove(raw_path)

    # ── Poll for result ───────────────────────────────────────
    import asyncio
    for _ in range(60):  # up to 3 minutes — photos are faster than video
        await asyncio.sleep(3)
        result = get_job_result(job_id)
        if result and result.get("job_status") == "complete":
            result.pop("job_status", None)
            return JSONResponse(result)
        if result and result.get("job_status") == "error":
            raw_error = result.get("error", "")
            is_tb = "Traceback" in raw_error or "File /opt" in raw_error or len(raw_error) > 200
            safe_error = (
                "Photo analysis failed. Please try again."
                if is_tb else raw_error or "Photo analysis failed."
            )
            return JSONResponse({"error": safe_error}, status_code=500)

    return JSONResponse({"error": "Photo analysis timed out. Please try again."}, status_code=504)


@app.get("/analyze-photo-link/")
async def analyze_photo_link(request: Request, image_url: str = "", email: str = ""):
    """
    Photo URL analysis endpoint. Mirrors /analyze-link-json/ for images.
    """
    import asyncio

    if not image_url:
        return JSONResponse({"error": "image_url is required"}, status_code=400)

    if not email:
        # Try session cookie fallback
        session_token = request.cookies.get("vfy_session", "")
        if session_token and session_token in _session_store:
            email = _session_store[session_token]["email"]
        else:
            return JSONResponse({"error": "email is required"}, status_code=400)

    if not is_valid_email(email):
        return JSONResponse({"error": "Invalid email address."}, status_code=400)

    # Check URL cache first
    try:
        import hashlib as _hl, json as _jc
        r_cache = _get_redis()
        _key = "urlcache:photo:v1:" + _hl.md5(image_url.strip().encode()).hexdigest()
        cached = r_cache.get(_key)
        if cached:
            result = _jc.loads(cached)
            log.info("photo_link: cache hit for %s", image_url[:60])
            return JSONResponse(result)
    except Exception:
        pass

    # Usage check
    status = get_user_status(email)
    if not status["allowed"]:
        return JSONResponse({
            "error":     "limit_reached",
            "plan":      status["plan"],
            "uses_left": 0,
        }, status_code=402)

    job_id = str(uuid.uuid4())

    try:
        enqueue_photo_link(job_id, image_url, email)
    except Exception as e:
        return JSONResponse({"error": f"Queue unavailable: {str(e)[:100]}"}, status_code=503)

    # Return job_id immediately for polling (mirrors analyze-link-json)
    log.info("analyze-photo-link: returning job_id=%s for frontend polling", job_id)
    return JSONResponse({"job_id": job_id, "status": "queued"})


@app.get("/download-photo/{cid}")
async def download_photo(cid: str):
    """
    Serve certified photo — checks R2 first, falls back to Redis.
    Mirrors /download/{cid} for videos.
    """
    from fastapi.responses import Response

    # Try R2 first
    try:
        from storage import _get_client, BUCKET
        client = _get_client()
        for plan in ("pro", "creator", "free", "enterprise"):
            for ext in (".jpg", ".png", ".webp"):
                key = f"certified-photos/{plan}/{cid}{ext}"
                try:
                    obj = client.get_object(Bucket=BUCKET, Key=key)
                    data = obj["Body"].read()
                    content_type = {
                        ".jpg": "image/jpeg",
                        ".png": "image/png",
                        ".webp": "image/webp",
                    }.get(ext, "image/jpeg")
                    return Response(
                        content=data,
                        media_type=content_type,
                        headers={
                            "Content-Disposition": f'attachment; filename="verifyd_cert_{cid[:8]}{ext}"',
                            "Cache-Control": "private, max-age=3600",
                        }
                    )
                except Exception:
                    continue
    except Exception as e:
        log.warning("R2 photo download lookup failed for %s: %s", cid, e)

    # Redis fallback
    try:
        r = _get_redis()
        data = r.get(f"cert:{cid}")
        if data:
            return Response(
                content=data,
                media_type="image/jpeg",
                headers={
                    "Content-Disposition": f'attachment; filename="verifyd_cert_{cid[:8]}.jpg"',
                }
            )
    except Exception:
        pass

    return JSONResponse({"error": "Certified photo not found or expired."}, status_code=404)




# ─────────────────────────────────────────────
#  Audio upload endpoint
# ─────────────────────────────────────────────
AUDIO_ALLOWED_EXTENSIONS = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".oga", ".opus", ".webm"}
AUDIO_SIZE_LIMITS = {
    "free":        25 * 1024 * 1024,    # 25MB
    "creator":     75 * 1024 * 1024,    # 75MB
    "pro":        150 * 1024 * 1024,    # 150MB
    "enterprise": 500 * 1024 * 1024,    # 500MB
}


@app.post("/upload-audio/")
async def upload_audio(file: UploadFile = File(...), email: str = Form(...)):
    """
    Standalone audio upload endpoint.
    Accepts MP3, WAV, M4A, AAC, FLAC, OGG/OGA, OPUS, and audio-only WebM.
    Returns a job_id for the same /job-status/{job_id} polling flow used by video uploads.
    """
    if not is_valid_email(email):
        return JSONResponse({"error": "Invalid email address."}, status_code=400)
    typo_response = _email_typo_json(email)
    if typo_response:
        return typo_response

    is_deliverable, reason = _verify_email_deliverable(email)
    if not is_deliverable:
        return JSONResponse({"error": reason}, status_code=400)

    if not is_email_verified(email):
        return JSONResponse({
            "error":   "email_not_verified",
            "message": "Please verify your email address before uploading.",
        }, status_code=403)

    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in AUDIO_ALLOWED_EXTENSIONS:
        return JSONResponse({
            "error":   "unsupported_format",
            "message": "Unsupported audio format. Accepted formats: MP3, WAV, M4A, AAC, FLAC, OGG, OGA, OPUS, and WebM audio.",
        }, status_code=415)

    try:
        _cleanup_old_files(max_age_hours=2.0)
    except Exception:
        pass

    status = get_user_status(email)
    if not status["allowed"]:
        return JSONResponse({
            "error":     "limit_reached",
            "plan":      status["plan"],
            "uses_left": 0,
            "limit":     status["limit"],
        }, status_code=402)

    user_plan = status["plan"]
    max_bytes = AUDIO_SIZE_LIMITS.get(user_plan, AUDIO_SIZE_LIMITS["free"])
    max_mb = max_bytes // (1024 * 1024)
    plan_label = {"free": "Free", "creator": "Creator", "pro": "Pro", "enterprise": "Enterprise"}.get(user_plan, user_plan.title())

    if file.size and file.size > max_bytes:
        return JSONResponse({
            "error":   "file_too_large",
            "message": f"Audio file exceeds the {max_mb}MB limit for your {plan_label} plan.",
            "max_mb":  max_mb,
            "plan":    user_plan,
        }, status_code=413)

    job_id = str(uuid.uuid4())
    safe_name = os.path.basename(file.filename or f"audio{ext}")
    raw_path = f"{UPLOAD_DIR}/{job_id}_{safe_name}"

    bytes_written = 0
    with open(raw_path, "wb") as f_out:
        while chunk := await file.read(1024 * 1024):
            bytes_written += len(chunk)
            if bytes_written > max_bytes:
                try:
                    os.remove(raw_path)
                except Exception:
                    pass
                return JSONResponse({
                    "error":   "file_too_large",
                    "message": f"Audio file exceeds the {max_mb}MB limit for your {plan_label} plan.",
                    "max_mb":  max_mb,
                    "plan":    user_plan,
                }, status_code=413)
            f_out.write(chunk)

    log.info("audio_upload: saved %dKB for %s (plan=%s file=%s)", bytes_written // 1024, email, user_plan, safe_name)

    try:
        enqueue_audio_upload(job_id, raw_path, safe_name, email)
        log.info("audio_upload: queued job %s for %s file=%s", job_id, email, safe_name)
    except Exception as e:
        log.warning("Audio queue unavailable (%s) — falling back to sync", e)
        try:
            from audio_detector import analyze_audio
            detail = analyze_audio(raw_path)
            audio_score = int(detail.get("audio_ai_score", 50))
            authenticity = max(0, min(100, 100 - audio_score))
            label = "REAL" if authenticity >= 55 else "UNDETERMINED" if authenticity >= 40 else "AI"
            AUDIO_LABEL_UI = {
                "REAL": ("REAL AUDIO VERIFIED", "green"),
                "UNDETERMINED": ("AUDIO UNDETERMINED", "blue"),
                "AI": ("AI AUDIO DETECTED", "red"),
            }
            ui_text, color = AUDIO_LABEL_UI.get(label, ("AUDIO UNDETERMINED", "blue"))
            increment_user_uses(email)
            sha256 = _sha256(raw_path)
            insert_certificate(
                cert_id=job_id,
                email=email,
                original_file=safe_name,
                label=label,
                authenticity=authenticity,
                ai_score=audio_score,
                sha256=sha256,
            )
            result = {
                "status": ui_text,
                "authenticity_score": authenticity,
                "color": color,
                "label": label,
                "media_type": "audio",
                "job_status": "complete",
                "audio_ready": False,
                "certified_audio_ready": False,
                "audio_score": audio_score,
                "base_audio_score": detail.get("base_audio_ai_score", audio_score),
                "audio_confidence": detail.get("confidence", "low"),
                "audio_evidence": detail.get("evidence", []),
                "audio_duration": detail.get("audio_duration", 0),
                "duration_mismatch": detail.get("duration_mismatch", 0),
                "stereo_corr": detail.get("stereo_corr"),
                "gpt_audio_score": detail.get("gpt_audio_score", 0),
                "gpt_audio_available": detail.get("gpt_audio_available", False),
                "gpt_audio_adjustment": detail.get("gpt_audio_adjustment", 0),
                "gpt_audio_reasoning": detail.get("gpt_audio_reasoning", ""),
                "gpt_audio_flags": detail.get("gpt_audio_flags", []),
                "signal_score": audio_score,
                "gpt_score": detail.get("gpt_audio_score", 0),
                "gpt_reasoning": "; ".join(str(x) for x in detail.get("evidence", [])[:3]) or "Audio forensic analysis complete.",
                "gpt_flags": detail.get("evidence", [])[:5],
                "certificate_id": job_id,
                "sha256": sha256,
            }

            if label == "REAL":
                import shutil as _shutil
                cert_audio_path = os.path.join(tempfile.gettempdir(), f"cert_audio_{job_id}{ext}")
                download_url = f"{BASE_URL}/download-audio/{job_id}"
                try:
                    log.info(
                        "audio_upload sync: creating certified audio: job=%s src=%s dest=%s label=%s auth=%s",
                        job_id, raw_path, cert_audio_path, label, authenticity,
                    )
                    _shutil.copyfile(raw_path, cert_audio_path)
                    cert_exists = os.path.exists(cert_audio_path)
                    cert_size = os.path.getsize(cert_audio_path) if cert_exists else 0
                    log.info(
                        "audio_upload sync: certified audio created: job=%s exists=%s size=%d path=%s",
                        job_id, cert_exists, cert_size, cert_audio_path,
                    )
                    if not cert_exists or cert_size < 256:
                        raise RuntimeError("Certified audio output missing or too small")

                    stored = False
                    try:
                        from storage import r2_available, upload_certified_audio
                        if r2_available():
                            log.info(
                                "audio_upload sync: uploading certified audio to R2: job=%s path=%s size=%d plan=%s",
                                job_id, cert_audio_path, os.path.getsize(cert_audio_path), user_plan,
                            )
                            upload_certified_audio(job_id, cert_audio_path, user_plan, ext)
                            stored = True
                            log.info("audio_upload sync: certified audio stored in R2: job=%s plan=%s", job_id, user_plan)
                    except Exception as r2e:
                        log.warning("audio_upload sync: R2 audio upload failed, falling back to Redis: %s", r2e)

                    if not stored:
                        cert_ttl = {"free": 86400, "creator": 259200, "pro": 604800, "enterprise": 2592000}.get(user_plan, 86400)
                        with open(cert_audio_path, "rb") as af:
                            cert_bytes = af.read()
                        rr = _get_redis()
                        rr.setex(f"audiocert:{job_id}", cert_ttl, cert_bytes)
                        rr.setex(f"audiocert:{job_id}:ext", cert_ttl, ext)
                        log.info("audio_upload sync: certified audio stored in Redis: job=%s ttl=%s", job_id, cert_ttl)

                    if email and "@" in email:
                        try:
                            sent = send_certification_email(email, job_id, authenticity, safe_name, download_url, is_audio=True)
                            log.info("audio_upload sync: audio certification email sent=%s job=%s email=%s", sent, job_id, email)
                        except Exception as em:
                            log.warning("audio_upload sync: audio certification email failed for %s: %s", job_id, em)

                    result["audio_ready"] = True
                    result["certified_audio_ready"] = True
                    result["certification_status"] = "ready"
                    result["download_url"] = download_url
                    result["share_url"] = download_url
                    result["download_type"] = "certified_audio"
                except Exception as cert_e:
                    log.warning("audio_upload sync: certified audio generation failed for %s: %s", job_id, cert_e)
                    result["audio_ready"] = False
                    result["certified_audio_ready"] = False
                    result["certification_status"] = "failed"
                    result["certification_error"] = "Certified audio generation failed, but analysis completed."
                finally:
                    try:
                        if os.path.exists(cert_audio_path):
                            os.remove(cert_audio_path)
                    except Exception:
                        pass
            else:
                result["audio_ready"] = True

            return JSONResponse(result)
        except Exception as sync_e:
            log.exception("Audio sync fallback failed")
            return JSONResponse({"error": str(sync_e)[:200]}, status_code=500)
        finally:
            if os.path.exists(raw_path):
                os.remove(raw_path)

    return JSONResponse({"job_id": job_id, "status": "queued", "media_type": "audio"})



@app.get("/download-audio/{cid}")
async def download_audio(cid: str):
    """Serve certified audio — checks R2 first, falls back to Redis."""
    from fastapi.responses import Response, StreamingResponse

    audio_exts = (".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".oga", ".opus", ".webm")
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

    # R2 first. Look directly across plan folders and extensions.
    try:
        from storage import _get_client, BUCKET, PUBLIC_URL, CERT_URL_TTL, r2_available

        if r2_available():
            client = _get_client()
            for plan in ("pro", "creator", "enterprise", "free"):
                for ext in audio_exts:
                    key = f"certified-audio/{plan}/{cid}{ext}"
                    try:
                        client.head_object(Bucket=BUCKET, Key=key)
                        # Proxy the object through FastAPI so browsers download it
                        # instead of opening an inline audio player. This also works
                        # when PUBLIC_URL is configured and cannot carry response
                        # header overrides.
                        obj = client.get_object(Bucket=BUCKET, Key=key)
                        media_type = content_types.get(ext.lower(), "application/octet-stream")
                        filename = f"VeriFYD_Certified_Audio_{cid[:8]}{ext}"
                        log.info("download-audio: R2 hit job=%s key=%s", cid, key)
                        return StreamingResponse(
                            obj["Body"].iter_chunks(chunk_size=1024 * 1024),
                            media_type=media_type,
                            headers={
                                "Content-Disposition": f'attachment; filename="{filename}"',
                                "Cache-Control": "private, max-age=3600",
                            },
                        )
                    except Exception:
                        continue
    except Exception as e:
        log.warning("R2 audio download lookup failed for %s: %s", cid, e)

    # Redis fallback.
    try:
        r = _get_redis()
        data = r.get(f"audiocert:{cid}")
        ext_raw = r.get(f"audiocert:{cid}:ext")
        ext = ".mp3"
        if ext_raw:
            ext = ext_raw.decode("utf-8", errors="replace") if isinstance(ext_raw, bytes) else str(ext_raw)
            if not ext.startswith("."):
                ext = "." + ext
        if data:
            return Response(
                content=data,
                media_type=content_types.get(ext.lower(), "application/octet-stream"),
                headers={
                    "Content-Disposition": f'attachment; filename="VeriFYD_Certified_Audio_{cid[:8]}{ext}"',
                    "Cache-Control": "private, max-age=3600",
                },
            )
    except Exception as e:
        log.warning("Redis audio download lookup failed for %s: %s", cid, e)

    return JSONResponse({"error": "Certified audio not found or expired."}, status_code=404)


# ─────────────────────────────────────────────
#  Document upload endpoint — VeriFYD Docs MVP
# ─────────────────────────────────────────────
DOCUMENT_ALLOWED_EXTENSIONS = {
    ".pdf", ".docx", ".doc", ".xlsx", ".xls", ".pptx", ".ppt",
    ".odt", ".ods", ".odp",
    ".txt", ".md", ".csv", ".rtf", ".eml", ".msg",
    ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tif", ".tiff", ".webp", ".heic", ".heif",
    ".html", ".htm", ".mhtml", ".mht", ".xml", ".json", ".svg", ".vsdx",
    ".yaml", ".yml", ".toml", ".env", ".ini", ".properties", ".conf", ".cfg", ".config", ".cnf", ".log", ".sql",
    ".pst", ".ost", ".dwg", ".dxf",
    ".zip",
}
DOCUMENT_SIZE_LIMITS = {
    "free":        10 * 1024 * 1024,    # 10MB
    "creator":     25 * 1024 * 1024,    # 25MB
    "pro":         75 * 1024 * 1024,    # 75MB
    "enterprise":  250 * 1024 * 1024,   # 250MB
}


TRUST_DESK_ALLOWED_EXTENSIONS = {".zip"}
TRUST_DESK_SIZE_LIMITS = {
    "free":        50 * 1024 * 1024,
    "creator":    150 * 1024 * 1024,
    "pro":        500 * 1024 * 1024,
    "enterprise": 2048 * 1024 * 1024,
}

DOCUMENT_LABEL_UI = {
    "REAL":         ("REAL DOCUMENT VERIFIED", "green", True),
    "UNDETERMINED": ("DOCUMENT UNDETERMINED",  "blue",  False),
    "AI":           ("AI / TAMPERING DETECTED", "red",  False),
}


def _uploaded_pdf_has_verifyd_seal(path: str) -> tuple[bool, str]:
    """Return whether an uploaded PDF already contains a VeriFYD secure seal."""
    try:
        if not path or not path.lower().endswith(".pdf") or not os.path.exists(path):
            return False, ""
        from pypdf import PdfReader
        reader = PdfReader(path)
        md = reader.metadata or {}
        seal = str(md.get("/VeriFYD_Secure_Seal") or "").strip().upper()
        payload = str(md.get("/VeriFYD_Seal_Payload_B64") or "").strip()
        cid = str(md.get("/VeriFYD_Seal_Certificate_ID") or md.get("/VeriFYD_Certificate_ID") or "").strip()
        if seal == "PRESENT" or payload:
            return True, cid
    except Exception as e:
        log.debug("already-certified PDF seal check failed: %s", e)
    return False, ""


@app.post("/trust-desk/upload-zip/")
async def upload_trust_desk_zip(
    file: UploadFile = File(...),
    email: str = Form(...),
    organization: str = Form(""),
    organization_name: str = Form(""),
    submitter_name: str = Form(""),
    case_number: str = Form(""),
    notes: str = Form(""),
):
    """
    Trust Desk ZIP intake endpoint.

    Phase 1 queues a ZIP for safe extraction, file inventory, SHA-256 hashing,
    manifest creation, and Trust Desk return-package assembly. Individual
    child-file certification routing will be added in the next phase.
    """
    if not is_valid_email(email):
        return JSONResponse({"error": "Invalid email address."}, status_code=400)
    typo_response = _email_typo_json(email)
    if typo_response:
        return typo_response

    # Support both frontend field names: organization and organization_name.
    # The Trust Desk intake page uses organization_name; the original backend
    # skeleton expected organization. Normalize here so the manifest/email/logs
    # always receive the company name.
    organization = (organization or organization_name or "").strip()
    submitter_name = (submitter_name or "").strip()
    case_number = (case_number or "").strip()
    notes = (notes or "").strip()


    is_deliverable, reason = _verify_email_deliverable(email)
    if not is_deliverable:
        return JSONResponse({"error": reason}, status_code=400)

    if not is_email_verified(email):
        return JSONResponse({
            "error": "email_not_verified",
            "message": "Please verify your email address before submitting a Trust Desk package.",
        }, status_code=403)

    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in TRUST_DESK_ALLOWED_EXTENSIONS:
        return JSONResponse({
            "error": "unsupported_format",
            "message": "Trust Desk intake currently accepts ZIP packages only.",
        }, status_code=415)

    try:
        _cleanup_old_files(max_age_hours=2.0)
    except Exception:
        pass

    status = get_user_status(email)
    if not status["allowed"]:
        return JSONResponse({
            "error": "limit_reached",
            "plan": status["plan"],
            "uses_left": 0,
            "limit": status["limit"],
        }, status_code=402)

    user_plan = status["plan"]
    max_bytes = TRUST_DESK_SIZE_LIMITS.get(user_plan, TRUST_DESK_SIZE_LIMITS["free"])
    max_mb = max_bytes // (1024 * 1024)
    plan_label = {"free": "Free", "creator": "Creator", "pro": "Pro", "enterprise": "Enterprise"}.get(user_plan, user_plan.title())

    if file.size and file.size > max_bytes:
        return JSONResponse({
            "error": "file_too_large",
            "message": f"Trust Desk ZIP exceeds the {max_mb}MB limit for your {plan_label} plan.",
            "max_mb": max_mb,
            "plan": user_plan,
        }, status_code=413)

    job_id = str(uuid.uuid4())
    safe_name = os.path.basename(file.filename or "trust-desk-package.zip")
    raw_path = f"{UPLOAD_DIR}/{job_id}_{safe_name}"

    bytes_written = 0
    with open(raw_path, "wb") as fh:
        while chunk := await file.read(1024 * 1024):
            bytes_written += len(chunk)
            if bytes_written > max_bytes:
                try:
                    os.remove(raw_path)
                except Exception:
                    pass
                return JSONResponse({
                    "error": "file_too_large",
                    "message": f"Trust Desk ZIP exceeds the {max_mb}MB limit for your {plan_label} plan.",
                    "max_mb": max_mb,
                    "plan": user_plan,
                }, status_code=413)
            fh.write(chunk)

    log.info(
        "trust_desk_upload: saved %dKB for %s (plan=%s org=%s case=%s file=%s)",
        bytes_written // 1024, email, user_plan, organization, case_number, safe_name
    )

    try:
        enqueue_trust_desk_zip(
            job_id=job_id,
            raw_path=raw_path,
            filename=safe_name,
            email=email,
            organization=organization,
            submitter_name=submitter_name,
            case_number=case_number,
            notes=notes,
        )
        return JSONResponse({
            "trust_desk_job_id": job_id,
            "job_id": job_id,
            "status": "queued",
            "message": "Trust Desk ZIP package received and queued for intake processing.",
        })
    except Exception as exc:
        log.exception("trust_desk_upload: queue failed")
        try:
            if os.path.exists(raw_path):
                os.remove(raw_path)
        except Exception:
            pass
        return JSONResponse({
            "error": "queue_unavailable",
            "message": "Trust Desk package could not be queued. Please try again.",
            "detail": str(exc)[:200],
        }, status_code=500)


@app.post("/upload-document/")
async def upload_document(file: UploadFile = File(...), email: str = Form(...)):
    """
    Document upload endpoint for VeriFYD Docs MVP.
    Accepts PDF, DOCX/DOC, XLSX/XLS, PPTX/PPT, ODT/ODS/ODP, TXT/MD/CSV/RTF/EML/MSG, HTML/MHTML/XML/JSON/SVG/VSDX, YAML/TOML/ENV/INI/PROPERTIES/CONF/CFG/CONFIG/CNF/LOG/SQL config files, JPG/JPEG/PNG/GIF/BMP/TIF/TIFF/WEBP/HEIC/HEIF images, PST/OST, DWG/DXF, and ZIP evidence packages and returns a job_id for polling
    through /job-status/{job_id}, matching video/photo behavior.
    """
    if not is_valid_email(email):
        return JSONResponse({"error": "Invalid email address."}, status_code=400)
    typo_response = _email_typo_json(email)
    if typo_response:
        return typo_response

    is_deliverable, reason = _verify_email_deliverable(email)
    if not is_deliverable:
        return JSONResponse({"error": reason}, status_code=400)

    if not is_email_verified(email):
        return JSONResponse({
            "error":   "email_not_verified",
            "message": "Please verify your email address before uploading.",
        }, status_code=403)

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in DOCUMENT_ALLOWED_EXTENSIONS:
        return JSONResponse({
            "error":   "unsupported_format",
            "message": "Unsupported document format. Accepted formats: PDF, DOC, DOCX, XLS, XLSX, PPT, PPTX, ODT, ODS, ODP, TXT, MD, CSV, RTF, EML, MSG, HTML, HTM, MHTML, MHT, XML, JSON, SVG, VSDX, YAML, YML, TOML, ENV, INI, PROPERTIES, CONF, CFG, CONFIG, CNF, LOG, SQL, JPG, JPEG, PNG, GIF, BMP, TIF, TIFF, WEBP, HEIC, HEIF, PST, OST, DWG, DXF, ZIP.",
        }, status_code=415)

    try:
        _cleanup_old_files(max_age_hours=2.0)
    except Exception:
        pass

    status = get_user_status(email)
    if not status["allowed"]:
        return JSONResponse({
            "error":     "limit_reached",
            "plan":      status["plan"],
            "uses_left": 0,
            "limit":     status["limit"],
        }, status_code=402)

    user_plan = status["plan"]
    max_bytes = DOCUMENT_SIZE_LIMITS.get(user_plan, DOCUMENT_SIZE_LIMITS["free"])
    max_mb = max_bytes // (1024 * 1024)
    plan_label = {"free": "Free", "creator": "Creator", "pro": "Pro", "enterprise": "Enterprise"}.get(user_plan, user_plan.title())

    if file.size and file.size > max_bytes:
        return JSONResponse({
            "error":   "file_too_large",
            "message": f"Document exceeds the {max_mb}MB limit for your {plan_label} plan.",
            "max_mb":  max_mb,
            "plan":    user_plan,
        }, status_code=413)

    job_id = str(uuid.uuid4())
    safe_name = os.path.basename(file.filename or f"document{ext}")
    raw_path = f"{UPLOAD_DIR}/{job_id}_{safe_name}"

    bytes_written = 0
    with open(raw_path, "wb") as f_out:
        while chunk := await file.read(1024 * 1024):
            bytes_written += len(chunk)
            if bytes_written > max_bytes:
                try:
                    os.remove(raw_path)
                except Exception:
                    pass
                return JSONResponse({
                    "error":   "file_too_large",
                    "message": f"Document exceeds the {max_mb}MB limit for your {plan_label} plan.",
                    "max_mb":  max_mb,
                    "plan":    user_plan,
                }, status_code=413)
            f_out.write(chunk)

    log.info("document_upload: saved %dKB for %s (plan=%s file=%s)",
             bytes_written // 1024, email, user_plan, safe_name)

    if ext == ".pdf":
        already_certified, sealed_cid = _uploaded_pdf_has_verifyd_seal(raw_path)
        if already_certified:
            try:
                os.remove(raw_path)
            except Exception:
                pass
            return JSONResponse({
                "error": "already_verifyd_certified",
                "message": "This PDF already contains a VeriFYD secure seal. Use Verify Certified PDF / Verify Secure Seal instead of re-certifying it.",
                "certificate_id": sealed_cid,
                "verify_url": f"{BASE_URL}/verify-certificate/{sealed_cid}" if sealed_cid else "",
                "recommended_action": "verify_secure_seal",
            }, status_code=409)

    try:
        enqueue_document_upload(job_id, raw_path, safe_name, email)
        log.info("document_upload: queued job %s for %s file=%s", job_id, email, safe_name)
    except Exception as e:
        log.warning("Document queue unavailable (%s) — falling back to sync", e)
        try:
            from document_detection import run_document_detection
            authenticity, label, detail = run_document_detection(raw_path)
            increment_user_uses(email)
            insert_certificate(
                cert_id=job_id, email=email, original_file=safe_name,
                label=label, authenticity=authenticity,
                ai_score=detail["ai_score"], sha256=detail.get("sha256"),
            )
            ui_text, color, _ = DOCUMENT_LABEL_UI.get(label, ("DOCUMENT UNDETERMINED", "blue", False))
            return JSONResponse({
                "status":             ui_text,
                "authenticity_score": authenticity,
                "color":              color,
                "label":              label,
                "gpt_reasoning":      detail.get("gpt_reasoning", ""),
                "gpt_flags":          detail.get("gpt_flags", []),
                "signal_score":       detail.get("signal_ai_score", 0),
                "gpt_score":          detail.get("gpt_ai_score", 0),
                "metadata_score":     detail.get("metadata_score", 0),
                "text_score":         detail.get("text_score", 0),
                "document_risk_report": detail.get("document_risk_report") or detail.get("risk_report", {}),
                "risk_report":        detail.get("document_risk_report") or detail.get("risk_report", {}),
                "overall_risk":       (detail.get("document_risk_report") or detail.get("risk_report", {})).get("overall_risk") if isinstance(detail.get("document_risk_report") or detail.get("risk_report", {}), dict) else None,
                "risk_score":         (detail.get("document_risk_report") or detail.get("risk_report", {})).get("risk_score") if isinstance(detail.get("document_risk_report") or detail.get("risk_report", {}), dict) else None,
                "metadata_integrity": (detail.get("document_risk_report") or detail.get("risk_report", {})).get("metadata_integrity") if isinstance(detail.get("document_risk_report") or detail.get("risk_report", {}), dict) else None,
                "document_type":      detail.get("document_type", ext.lstrip(".")),
                "pages":              detail.get("pages", 0),
                "embedded_images":    detail.get("embedded_images", 0),
                "sha256":             detail.get("sha256", ""),
                "media_type":         "document",
            })
        except Exception as sync_e:
            log.exception("Document sync fallback failed")
            return JSONResponse({"error": str(sync_e)[:200]}, status_code=500)
        finally:
            if os.path.exists(raw_path):
                os.remove(raw_path)

    return JSONResponse({"job_id": job_id, "status": "queued", "media_type": "document"})


@app.get("/download-trust-desk/{cid}")
def download_trust_desk_package(cid: str):
    """Download the Trust Desk return package ZIP."""
    try:
        from storage import get_trust_desk_download_url, trust_desk_package_exists
        if not trust_desk_package_exists(cid):
            return JSONResponse({"error": "Trust Desk package not found or expired."}, status_code=404)
        increment_downloads(cid)
        url = get_trust_desk_download_url(cid)
        log.info("download-trust-desk: R2 hit job=%s", cid)
        return RedirectResponse(url)
    except Exception as exc:
        log.warning("download-trust-desk failed for %s: %s", cid, exc)
        return JSONResponse({"error": "Unable to prepare Trust Desk package download."}, status_code=500)


@app.get("/download-document/{cid}")
async def download_document(cid: str):
    """
    Serve certified/stamped document PDF.

    R2 is the source of truth for certified documents. This route checks every
    plan folder directly instead of relying only on the helper existence check.
    That makes large certified PDFs, such as HEIC-generated document
    certificates, more reliable immediately after upload.
    """
    from fastapi.responses import Response, StreamingResponse

    fname = f"VeriFYD_Certified_Document_{cid[:8]}.pdf"

    # Try R2 first by looking directly in each plan folder.
    try:
        from storage import _get_client, BUCKET, PUBLIC_URL, CERT_URL_TTL, r2_available

        if r2_available():
            client = _get_client()
            for plan in ("pro", "creator", "enterprise", "free"):
                key = f"certified-documents/{plan}/{cid}.pdf"
                try:
                    # A successful HEAD proves the object exists in this plan folder.
                    client.head_object(Bucket=BUCKET, Key=key)

                    if PUBLIC_URL:
                        url = f"{PUBLIC_URL.rstrip('/')}/{key}"
                    else:
                        url = client.generate_presigned_url(
                            "get_object",
                            Params={"Bucket": BUCKET, "Key": key},
                            ExpiresIn=CERT_URL_TTL,
                        )

                    log.info("download-document: R2 hit job=%s key=%s", cid, key)
                    return RedirectResponse(url=url, status_code=302)
                except Exception:
                    continue

            log.warning("download-document: R2 object not found in any plan folder for %s", cid)

    except Exception as e:
        log.warning("R2 document download lookup failed for %s: %s", cid, e)

    # Redis fallback for local/dev or old fallback jobs.
    try:
        r = _get_redis()
        data = r.get(f"doccert:{cid}")
        if data:
            log.info("download-document: Redis fallback hit job=%s size=%d", cid, len(data))
            return Response(
                content=data,
                media_type="application/pdf",
                headers={
                    "Content-Disposition": f'attachment; filename="{fname}"',
                    "Cache-Control": "private, max-age=3600",
                },
            )
    except Exception as e:
        log.warning("Redis document download lookup failed for %s: %s", cid, e)

    return JSONResponse({"error": "Certified document not found or expired."}, status_code=404)


@app.get("/download-certified-file/{cid}")
async def download_certified_file_package(cid: str):
    """
    Serve the universal certified file package ZIP.

    For Pro/Enterprise ZIP evidence uploads, this package contains the parent
    summary PDF plus individually certified reports for supported internal files.
    """
    from fastapi.responses import Response, StreamingResponse

    fname = f"VeriFYD_Certified_File_Package_{cid[:8]}.zip"

    try:
        from storage import _get_client, BUCKET, PUBLIC_URL, CERT_URL_TTL, r2_available

        if r2_available():
            client = _get_client()
            for plan in ("pro", "creator", "enterprise", "free"):
                key = f"certified-files/{plan}/{cid}.zip"
                try:
                    client.head_object(Bucket=BUCKET, Key=key)

                    if PUBLIC_URL:
                        url = f"{PUBLIC_URL.rstrip('/')}/{key}"
                    else:
                        url = client.generate_presigned_url(
                            "get_object",
                            Params={"Bucket": BUCKET, "Key": key},
                            ExpiresIn=CERT_URL_TTL,
                        )

                    log.info("download-certified-file: R2 hit job=%s key=%s", cid, key)
                    return RedirectResponse(url=url, status_code=302)
                except Exception:
                    continue

            log.warning("download-certified-file: R2 object not found in any plan folder for %s", cid)

    except Exception as e:
        log.warning("R2 certified file package lookup failed for %s: %s", cid, e)

    try:
        r = _get_redis()
        data = r.get(f"filecert:{cid}")
        if data:
            log.info("download-certified-file: Redis fallback hit job=%s size=%d", cid, len(data))
            return Response(
                content=data,
                media_type="application/zip",
                headers={
                    "Content-Disposition": f'attachment; filename="{fname}"',
                    "Cache-Control": "private, max-age=3600",
                },
            )
    except Exception as e:
        log.warning("Redis certified file package lookup failed for %s: %s", cid, e)

    return JSONResponse({"error": "Certified file package not found or expired."}, status_code=404)

@app.post("/verify-document-seal/")
async def verify_document_seal(file: UploadFile = File(...)):
    """
    Verify the hidden VeriFYD Secure Seal embedded in a certified PDF.

    This supports the enterprise submission workflow: organizations can require
    a file to be VeriFYD certified, then submit the PDF back here to confirm the
    hidden cryptographic seal is present and valid.
    """
    filename = file.filename or "certified.pdf"
    ext = os.path.splitext(filename)[1].lower()
    if ext != ".pdf":
        return JSONResponse({
            "verified": False,
            "status": "unsupported_format",
            "message": "Upload a VeriFYD certified PDF to verify its secure seal.",
        }, status_code=415)

    tmp_path = os.path.join(tempfile.gettempdir(), f"seal_verify_{uuid.uuid4()}.pdf")
    try:
        bytes_written = 0
        with open(tmp_path, "wb") as f_out:
            while chunk := await file.read(1024 * 1024):
                bytes_written += len(chunk)
                if bytes_written > 150 * 1024 * 1024:
                    return JSONResponse({
                        "verified": False,
                        "status": "file_too_large",
                        "message": "Certified PDF exceeds the seal verification size limit.",
                    }, status_code=413)
                f_out.write(chunk)

        from doc_certifier import verify_secure_seal_pdf
        result = verify_secure_seal_pdf(tmp_path)
        result = _enrich_certificate_verification_result(result, uploaded_pdf_path=tmp_path)
        status_code = 200 if result.get("verified") else 400
        return JSONResponse(result, status_code=status_code)
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        except Exception:
            pass



def _sha256_bytes_for_verify(data: bytes) -> str:
    """Return SHA-256 for bytes used during certified document integrity comparison."""
    return hashlib.sha256(data or b"").hexdigest()


def _lookup_stored_certified_document_hash(cid: str) -> dict:
    """
    Locate the issued certified PDF and return its SHA-256.

    This is the key tamper-detection comparison:
    uploaded certified PDF hash vs VeriFYD's stored certified PDF hash.
    """
    cid = (cid or "").strip()
    result = {
        "available": False,
        "source": "",
        "key": "",
        "sha256": "",
        "size_bytes": 0,
        "error": "",
    }
    if not cid:
        result["error"] = "missing_certificate_id"
        return result

    # R2 is the source of truth for production certified documents.
    try:
        from storage import _get_client, BUCKET, r2_available

        if r2_available():
            client = _get_client()
            for plan in ("pro", "creator", "enterprise", "free"):
                key = f"certified-documents/{plan}/{cid}.pdf"
                try:
                    obj = client.get_object(Bucket=BUCKET, Key=key)
                    body = obj.get("Body")
                    data = body.read() if body is not None else b""
                    if data:
                        result.update({
                            "available": True,
                            "source": "r2",
                            "key": key,
                            "sha256": _sha256_bytes_for_verify(data),
                            "size_bytes": len(data),
                            "error": "",
                        })
                        return result
                except Exception:
                    continue
    except Exception as e:
        result["error"] = f"r2_lookup_failed:{str(e)[:120]}"

    # Redis fallback for local/dev or older fallback jobs.
    try:
        r = _get_redis()
        data = r.get(f"doccert:{cid}")
        if data:
            result.update({
                "available": True,
                "source": "redis",
                "key": f"doccert:{cid}",
                "sha256": _sha256_bytes_for_verify(data),
                "size_bytes": len(data),
                "error": "",
            })
            return result
    except Exception as e:
        result["error"] = (result.get("error") + "; " if result.get("error") else "") + f"redis_lookup_failed:{str(e)[:120]}"

    if not result.get("error"):
        result["error"] = "stored_certified_document_not_found"
    return result



def _classify_post_certification_pdf_change(
    uploaded_pdf_path: str,
    stored_doc: dict | None = None,
    seal_result: dict | None = None,
) -> dict:
    """
    Classify the likely reason a certified PDF changed after VeriFYD certification.

    This does not decide whether tampering occurred. The tamper decision remains
    deterministic and hash-based. This helper only explains the likely type of
    post-certification PDF change using PDF structure clues.
    """
    stored_doc = stored_doc or {}
    seal_result = seal_result or {}

    classification = {
        "tamper_reason_code": "PDF_CHANGED_AFTER_CERTIFICATION",
        "tamper_reason": "PDF changed after certification",
        "tamper_confidence": "LOW",
        "tamper_evidence": [],
        "tamper_explanation": (
            "The VeriFYD seal is present, but this PDF no longer matches the exact certified copy "
            "stored by VeriFYD. The file was changed after certification, but the specific type "
            "of PDF change could not be confidently identified from the available structure."
        ),
    }

    evidence = classification["tamper_evidence"]

    if seal_result.get("seal_valid"):
        evidence.append("VeriFYD secure seal signature is valid")
    if str(seal_result.get("seal_version") or "").upper():
        evidence.append(f"Seal version: {seal_result.get('seal_version')}")
    if seal_result.get("certified_pdf_sha256"):
        evidence.append("Uploaded certified PDF hash differs from the VeriFYD stored certified PDF hash")

    try:
        uploaded_size = os.path.getsize(uploaded_pdf_path) if uploaded_pdf_path and os.path.exists(uploaded_pdf_path) else 0
    except Exception:
        uploaded_size = 0

    stored_size = 0
    try:
        stored_size = int(stored_doc.get("size_bytes") or 0)
    except Exception:
        stored_size = 0

    if uploaded_size and stored_size:
        delta = uploaded_size - stored_size
        evidence.append(f"Uploaded file size delta versus issued copy: {delta:+d} bytes")

    try:
        with open(uploaded_pdf_path, "rb") as fh:
            raw = fh.read()
    except Exception as e:
        evidence.append(f"PDF structure scan unavailable: {str(e)[:100]}")
        return classification

    # Decode as latin-1 so PDF object names survive without requiring valid text encoding.
    text = raw.decode("latin-1", errors="ignore")
    lower = text.lower()

    eof_count = text.count("%%EOF")
    startxref_count = lower.count("startxref")
    incremental_update = eof_count > 1 or startxref_count > 1
    if incremental_update:
        evidence.append(f"PDF appears to contain incremental updates (%%EOF={eof_count}, startxref={startxref_count})")

    has_signature_markers = any(marker in text for marker in ("/Type /Sig", "/FT /Sig", "/SubFilter", "/ByteRange"))
    has_byte_range = "/ByteRange" in text
    has_sig_type = "/Type /Sig" in text or "/FT /Sig" in text
    has_signature_contents = "/Contents" in text and ("/SubFilter" in text or "/ByteRange" in text)

    if has_byte_range:
        evidence.append("PDF contains /ByteRange signature data")
    if has_sig_type:
        evidence.append("PDF contains a /Sig signature object or signature form field")
    if has_signature_contents:
        evidence.append("PDF contains signature-related /Contents and /SubFilter fields")

    # Digital signatures are frequently stored as incremental PDF updates. This is the
    # exact case seen when a user signs a VeriFYD-certified PDF after issuance.
    if has_signature_markers:
        classification.update({
            "tamper_reason_code": "DIGITAL_SIGNATURE_ADDED_AFTER_CERTIFICATION",
            "tamper_reason": "Digital signature added after certification",
            "tamper_confidence": "HIGH" if has_byte_range and (has_sig_type or incremental_update) else "MEDIUM",
            "tamper_explanation": (
                "A digital signature or signature-related PDF object appears to have been added "
                "after VeriFYD certification. The VeriFYD seal itself remains cryptographically valid, "
                "but the submitted PDF is no longer byte-for-byte identical to the certified copy "
                "stored by VeriFYD. This is commonly caused by signing the certified PDF after it was issued."
            ),
        })
        return classification

    has_annots = "/Annots" in text or "/Annot" in text
    has_acroform = "/AcroForm" in text
    has_widget = "/Widget" in text
    has_xfa = "/XFA" in text
    if has_annots:
        evidence.append("PDF contains annotation objects")
    if has_acroform:
        evidence.append("PDF contains an AcroForm form structure")
    if has_widget:
        evidence.append("PDF contains widget/form-field objects")
    if has_xfa:
        evidence.append("PDF contains XFA form data")

    if has_annots or has_acroform or has_widget or has_xfa:
        classification.update({
            "tamper_reason_code": "ANNOTATION_OR_FORM_FIELD_ADDED_AFTER_CERTIFICATION",
            "tamper_reason": "Annotation or form field added after certification",
            "tamper_confidence": "MEDIUM",
            "tamper_explanation": (
                "The certified PDF appears to contain annotation or form-field structures that are consistent "
                "with a post-certification edit, such as adding a comment, marking up the file, filling a form, "
                "or saving an interactive field. The VeriFYD seal remains valid, but the submitted file no longer "
                "matches the issued certified copy stored by VeriFYD."
            ),
        })
        return classification

    has_moddate = "/ModDate" in text
    has_producer = "/Producer" in text
    has_creator = "/Creator" in text
    has_xmp_modify = "xmp:modifydate" in lower or "pdf:producer" in lower
    if has_moddate:
        evidence.append("PDF contains /ModDate metadata")
    if has_producer:
        evidence.append("PDF contains /Producer metadata")
    if has_creator:
        evidence.append("PDF contains /Creator metadata")
    if has_xmp_modify:
        evidence.append("PDF contains XMP modification/producer metadata")

    if incremental_update and (has_moddate or has_producer or has_creator or has_xmp_modify):
        classification.update({
            "tamper_reason_code": "PDF_RESAVED_OR_METADATA_CHANGED_AFTER_CERTIFICATION",
            "tamper_reason": "PDF re-saved or metadata changed after certification",
            "tamper_confidence": "MEDIUM",
            "tamper_explanation": (
                "The certified PDF appears to have been re-saved or updated after VeriFYD certification. "
                "The change may be metadata-only or caused by a PDF editor rewriting the file. The VeriFYD seal "
                "may still be valid, but the submitted file is no longer the exact issued certified copy."
            ),
        })
        return classification

    # If the size changed substantially but we did not find signature/annotation markers,
    # call it a significant post-certification modification rather than guessing the cause.
    if uploaded_size and stored_size:
        delta_abs = abs(uploaded_size - stored_size)
        if delta_abs > max(4096, int(stored_size * 0.02)):
            evidence.append("File size changed substantially without clear signature or annotation markers")
            classification.update({
                "tamper_reason_code": "CONTENT_OR_STRUCTURE_CHANGED_AFTER_CERTIFICATION",
                "tamper_reason": "Content or PDF structure changed after certification",
                "tamper_confidence": "MEDIUM",
                "tamper_explanation": (
                    "The certified PDF changed substantially compared with the copy stored by VeriFYD, but the "
                    "structure scan did not identify a simple digital signature, annotation, or metadata-only update. "
                    "This may indicate that page content, embedded objects, or the internal PDF structure changed after certification."
                ),
            })
            return classification

    if incremental_update:
        classification.update({
            "tamper_reason_code": "INCREMENTAL_UPDATE_AFTER_CERTIFICATION",
            "tamper_reason": "Incremental PDF update after certification",
            "tamper_confidence": "MEDIUM",
            "tamper_explanation": (
                "The PDF appears to contain an incremental update after the original certified file was issued. "
                "This usually means the file was opened and saved, signed, annotated, or otherwise updated after VeriFYD certification."
            ),
        })
        return classification

    return classification



def _download_stored_certified_document_to_path(cid: str, dest_path: str) -> dict:
    """
    Download VeriFYD's issued certified PDF to a temp path for visual comparison.

    This helper is only used after deterministic hash comparison already shows the
    uploaded certified PDF differs from the stored issued copy.
    """
    result = {
        "available": False,
        "source": "",
        "key": "",
        "path": "",
        "sha256": "",
        "size_bytes": 0,
        "error": "",
    }
    cid = (cid or "").strip()
    if not cid:
        result["error"] = "missing_certificate_id"
        return result

    try:
        from storage import _get_client, BUCKET, r2_available
        if r2_available():
            client = _get_client()
            for plan in ("pro", "creator", "enterprise", "free"):
                key = f"certified-documents/{plan}/{cid}.pdf"
                try:
                    client.download_file(BUCKET, key, dest_path)
                    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
                        result.update({
                            "available": True,
                            "source": "r2",
                            "key": key,
                            "path": dest_path,
                            "sha256": _sha256(dest_path),
                            "size_bytes": os.path.getsize(dest_path),
                            "error": "",
                        })
                        return result
                except Exception:
                    continue
    except Exception as e:
        result["error"] = f"r2_download_failed:{str(e)[:120]}"

    try:
        r = _get_redis()
        data = r.get(f"doccert:{cid}")
        if data:
            with open(dest_path, "wb") as fh:
                fh.write(data)
            result.update({
                "available": True,
                "source": "redis",
                "key": f"doccert:{cid}",
                "path": dest_path,
                "sha256": _sha256(dest_path),
                "size_bytes": os.path.getsize(dest_path),
                "error": "",
            })
            return result
    except Exception as e:
        result["error"] = (result.get("error") + "; " if result.get("error") else "") + f"redis_download_failed:{str(e)[:120]}"

    if not result.get("error"):
        result["error"] = "stored_certified_document_not_found"
    return result


def _render_pdf_pages_for_change_report(pdf_path: str, out_dir: str, prefix: str, max_pages: int = 5) -> dict:
    """Render PDF pages to PNG images for VeriFYD Change Report comparison."""
    info = {"available": False, "page_count": 0, "pages": [], "error": ""}
    try:
        import fitz  # PyMuPDF
    except Exception as e:
        info["error"] = f"pymupdf_unavailable:{str(e)[:120]}"
        return info

    try:
        doc = fitz.open(pdf_path)
        info["page_count"] = int(doc.page_count)
        render_count = min(int(doc.page_count), max(1, int(max_pages or 5)))
        matrix = fitz.Matrix(1.35, 1.35)
        for idx in range(render_count):
            page = doc.load_page(idx)
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            out_path = os.path.join(out_dir, f"{prefix}_page_{idx + 1}.png")
            pix.save(out_path)
            info["pages"].append({"page": idx + 1, "path": out_path, "width": pix.width, "height": pix.height})
        info["available"] = bool(info["pages"])
        doc.close()
    except Exception as e:
        info["error"] = f"render_failed:{str(e)[:160]}"
    return info


def _compare_rendered_pdf_pages(stored_render: dict, uploaded_render: dict) -> list:
    """Return visual page difference summaries for rendered PDF pages."""
    findings = []
    try:
        from PIL import Image, ImageChops, ImageStat
    except Exception as e:
        return [{
            "page": 0,
            "changed": False,
            "difference_score": 0,
            "summary": f"Visual image comparison unavailable: {str(e)[:100]}",
        }]

    stored_pages = {int(p.get("page")): p for p in stored_render.get("pages", [])}
    uploaded_pages = {int(p.get("page")): p for p in uploaded_render.get("pages", [])}
    all_pages = sorted(set(stored_pages) | set(uploaded_pages))

    for page_no in all_pages:
        sp = stored_pages.get(page_no)
        up = uploaded_pages.get(page_no)
        if not sp or not up:
            findings.append({
                "page": page_no,
                "changed": True,
                "difference_score": 100,
                "summary": "Page exists in one copy but not the other.",
                "region": "page_missing_or_added",
            })
            continue

        try:
            a = Image.open(sp["path"]).convert("RGB")
            b = Image.open(up["path"]).convert("RGB")
            if a.size != b.size:
                b = b.resize(a.size)
            diff = ImageChops.difference(a, b)
            stat = ImageStat.Stat(diff)
            mean = sum(stat.mean) / len(stat.mean)
            # Scale roughly to 0-100. Small antialiasing/rendering shifts should stay low.
            score = round(min(100.0, (mean / 255.0) * 100.0), 3)
            bbox = diff.getbbox()
            changed = bool(score >= 0.15 or bbox)
            region = "none"
            if bbox:
                w, h = a.size
                x1, y1, x2, y2 = bbox
                cx = (x1 + x2) / 2.0 / max(1, w)
                cy = (y1 + y2) / 2.0 / max(1, h)
                vertical = "top" if cy < 0.33 else "middle" if cy < 0.66 else "bottom"
                horizontal = "left" if cx < 0.33 else "center" if cx < 0.66 else "right"
                region = f"{vertical}-{horizontal}"
            findings.append({
                "page": page_no,
                "changed": changed,
                "difference_score": score,
                "region": region,
                "summary": (
                    f"Visual difference detected on page {page_no}"
                    + (f" near the {region} area." if region != "none" else ".")
                    if changed else f"No meaningful visual difference detected on rendered page {page_no}."
                ),
            })
        except Exception as e:
            findings.append({
                "page": page_no,
                "changed": False,
                "difference_score": 0,
                "summary": f"Page comparison failed: {str(e)[:120]}",
            })

    return findings


def _make_change_report_side_by_side(stored_image: str, uploaded_image: str, out_path: str) -> str:
    """Create a labeled side-by-side PNG for GPT Vision review."""
    from PIL import Image, ImageDraw

    left = Image.open(stored_image).convert("RGB")
    right = Image.open(uploaded_image).convert("RGB")

    max_w = 900
    for img_name in ("left", "right"):
        img = left if img_name == "left" else right
        if img.width > max_w:
            ratio = max_w / float(img.width)
            resized = img.resize((int(img.width * ratio), int(img.height * ratio)))
            if img_name == "left":
                left = resized
            else:
                right = resized

    h = max(left.height, right.height)
    pad = 28
    header = 42
    canvas = Image.new("RGB", (left.width + right.width + pad * 3, h + header + pad), "white")
    canvas.paste(left, (pad, header))
    canvas.paste(right, (left.width + pad * 2, header))

    draw = ImageDraw.Draw(canvas)
    draw.text((pad, 12), "Stored VeriFYD certified PDF", fill=(0, 0, 0))
    draw.text((left.width + pad * 2, 12), "Uploaded PDF being verified", fill=(0, 0, 0))
    draw.line((left.width + int(pad * 1.5), 0, left.width + int(pad * 1.5), canvas.height), fill=(160, 160, 160), width=2)
    canvas.save(out_path, format="PNG")
    return out_path


def _gpt_vision_change_report(side_by_side_png: str, deterministic_context: dict) -> dict:
    """
    Ask GPT Vision to explain visible differences. This is advisory only:
    deterministic hash comparison remains the source of truth.
    """
    out = {"available": False, "summary": "", "findings": [], "likely_change_type": "", "confidence": "low", "error": ""}
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        out["error"] = "OPENAI_API_KEY not configured"
        return out

    try:
        import base64
        from openai import OpenAI
        client = OpenAI(api_key=api_key, timeout=25)
        with open(side_by_side_png, "rb") as fh:
            b64 = base64.b64encode(fh.read()).decode("ascii")

        prompt = (
            "You are reviewing a side-by-side comparison for a VeriFYD Change Report. "
            "The left image is the stored VeriFYD certified PDF. The right image is the uploaded PDF being verified. "
            "The backend has already determined by cryptographic hash comparison that the uploaded PDF changed after certification. "
            "Your job is only to describe visible differences in plain language. Do not decide authenticity. "
            "Focus on pages/regions, likely visible change type such as signature, form-field update, annotation, date/text change, page replacement, or no obvious visible change. "
            "Return strict JSON with keys: summary, likely_change_type, confidence, findings. findings must be an array of short strings."
        )

        response = client.chat.completions.create(
            model=os.environ.get("VERIFYD_CHANGE_REPORT_MODEL", "gpt-4o-mini"),
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt + "\nContext: " + str(deterministic_context)[:1200]},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                ],
            }],
            response_format={"type": "json_object"},
            max_tokens=450,
        )
        raw = response.choices[0].message.content or "{}"
        import json as _json
        parsed = _json.loads(raw)
        out.update({
            "available": True,
            "summary": str(parsed.get("summary") or "")[:900],
            "likely_change_type": str(parsed.get("likely_change_type") or "")[:120],
            "confidence": str(parsed.get("confidence") or "low")[:40],
            "findings": [str(x)[:220] for x in (parsed.get("findings") or [])[:6]],
            "error": "",
        })
    except Exception as e:
        out["error"] = str(e)[:180]
    return out


def _create_verifyd_change_report(
    cid: str,
    uploaded_pdf_path: str,
    stored_doc: dict | None,
    tamper_detail: dict | None,
    seal_result: dict | None,
) -> dict:
    """
    Build a VeriFYD Change Report for certified PDFs modified after certification.

    The report is explanatory only. Hash mismatch remains the tamper decision.
    """
    tamper_detail = tamper_detail or {}
    seal_result = seal_result or {}
    report = {
        "title": "VeriFYD Change Report",
        "status": "MODIFIED_AFTER_CERTIFICATION",
        "summary": "The uploaded certified PDF no longer matches the exact VeriFYD-issued certified document.",
        "deterministic_basis": "Uploaded certified PDF SHA-256 differs from VeriFYD stored certified PDF SHA-256.",
        "likely_change_type": tamper_detail.get("tamper_reason", "PDF changed after certification"),
        "confidence": tamper_detail.get("tamper_confidence", "LOW"),
        "changed_pages": [],
        "page_findings": [],
        "gpt_visual_analysis_available": False,
        "gpt_visual_summary": "",
        "gpt_visual_findings": [],
        "recommendation": (
            "Use the original VeriFYD certified file from the Vault or evidence package as the authoritative copy. "
            "Treat the uploaded version as modified after certification."
        ),
        "technical_findings": list(tamper_detail.get("tamper_evidence") or []),
        "errors": [],
    }

    if not cid or not uploaded_pdf_path or not os.path.exists(uploaded_pdf_path):
        report["errors"].append("uploaded_pdf_unavailable")
        return report

    tmp_dir = tempfile.mkdtemp(prefix=f"verifyd_change_{cid[:8]}_")
    stored_path = os.path.join(tmp_dir, "stored_certified.pdf")
    try:
        stored_file = _download_stored_certified_document_to_path(cid, stored_path)
        if not stored_file.get("available"):
            report["errors"].append(stored_file.get("error") or "stored_certified_document_unavailable")
            return report

        try:
            if stored_file.get("sha256"):
                report["technical_findings"].append(f"Stored certified PDF SHA-256: {stored_file.get('sha256')}")
            if seal_result.get("certified_pdf_sha256"):
                report["technical_findings"].append(f"Uploaded certified PDF SHA-256: {seal_result.get('certified_pdf_sha256')}")
        except Exception:
            pass

        stored_render = _render_pdf_pages_for_change_report(stored_path, tmp_dir, "stored", max_pages=5)
        uploaded_render = _render_pdf_pages_for_change_report(uploaded_pdf_path, tmp_dir, "uploaded", max_pages=5)
        if not stored_render.get("available") or not uploaded_render.get("available"):
            report["errors"].append(stored_render.get("error") or uploaded_render.get("error") or "visual_render_unavailable")
            return report

        if stored_render.get("page_count") != uploaded_render.get("page_count"):
            report["technical_findings"].append(
                f"Page count changed: stored={stored_render.get('page_count')} uploaded={uploaded_render.get('page_count')}"
            )

        page_findings = _compare_rendered_pdf_pages(stored_render, uploaded_render)
        changed_pages = [p.get("page") for p in page_findings if p.get("changed") and p.get("page")]
        report["changed_pages"] = changed_pages[:20]
        report["page_findings"] = page_findings[:20]

        if changed_pages:
            first_changed = int(changed_pages[0])
            report["summary"] = f"The uploaded certified PDF differs visually on page {first_changed} and no longer matches the VeriFYD-issued copy."
        elif page_findings:
            report["summary"] = (
                "The PDF hash changed after certification, but rendered page comparison did not find an obvious visible page-content change. "
                "The change may be metadata, signature structure, embedded object, or PDF-internal data."
            )

        # GPT Vision advisory explanation for the first changed page, when available.
        try:
            first_page = int(changed_pages[0]) if changed_pages else 1
            sp = next((p for p in stored_render.get("pages", []) if int(p.get("page")) == first_page), None)
            up = next((p for p in uploaded_render.get("pages", []) if int(p.get("page")) == first_page), None)
            if sp and up:
                side_by_side = _make_change_report_side_by_side(
                    sp["path"], up["path"], os.path.join(tmp_dir, f"change_page_{first_page}_side_by_side.png")
                )
                gpt = _gpt_vision_change_report(side_by_side, {
                    "certificate_id": cid,
                    "tamper_reason": tamper_detail.get("tamper_reason"),
                    "tamper_evidence": tamper_detail.get("tamper_evidence", [])[:6],
                    "changed_pages": report["changed_pages"],
                })
                report["gpt_visual_analysis_available"] = bool(gpt.get("available"))
                report["gpt_visual_summary"] = gpt.get("summary", "")
                report["gpt_visual_findings"] = gpt.get("findings", [])
                if gpt.get("likely_change_type"):
                    report["likely_change_type"] = gpt.get("likely_change_type")
                if gpt.get("confidence"):
                    report["confidence"] = gpt.get("confidence")
                if gpt.get("summary"):
                    report["summary"] = gpt.get("summary")
                if gpt.get("error") and not gpt.get("available"):
                    report["errors"].append("gpt_visual_review_unavailable:" + str(gpt.get("error"))[:120])
        except Exception as e:
            report["errors"].append("gpt_visual_review_failed:" + str(e)[:120])

    finally:
        try:
            import shutil as _shutil
            _shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception:
            pass

    return report


def _enrich_certificate_verification_result(result: dict, uploaded_pdf_path: str = "") -> dict:
    """Add database, storage, and tamper-detection context to a secure-seal verification response."""
    result = dict(result or {})
    cid = str(result.get("certificate_id") or "").strip()
    db_record = None
    r2_available_for_cert = False
    stored_doc = {}
    uploaded_cert_hash = str(result.get("certified_pdf_sha256") or "").strip().lower()

    if cid:
        try:
            db_record = get_certificate(cid)
        except Exception as e:
            log.warning("verify-certificate: database lookup failed for %s: %s", cid, e)
            db_record = None

        try:
            from storage import r2_available, certified_document_exists
            if r2_available():
                r2_available_for_cert = bool(certified_document_exists(cid))
        except Exception:
            r2_available_for_cert = False

        # Strong integrity check: compare the uploaded certified PDF against
        # VeriFYD's stored issued copy. This catches post-certification edits
        # where the hidden seal metadata was left intact.
        try:
            stored_doc = _lookup_stored_certified_document_hash(cid)
            if stored_doc.get("available"):
                r2_available_for_cert = True
        except Exception as e:
            stored_doc = {"available": False, "error": str(e)[:120]}

    stored_hash = str((stored_doc or {}).get("sha256") or "").strip().lower()
    storage_record_match = bool((stored_doc or {}).get("available"))
    database_record_match = bool(db_record)
    # Some production records may be confirmed by the stored issued PDF even
    # when the certificate DB lookup is temporarily unavailable or delayed.
    # Expose both the strict DB lookup and the practical record-confirmed value
    # so the frontend does not show "Database Match: No" next to VERIFIED.
    record_confirmed = database_record_match or storage_record_match

    result["certificate_database_match"] = record_confirmed
    result["database_record_match"] = database_record_match
    result["storage_record_match"] = storage_record_match
    result["record_match_source"] = "database" if database_record_match else "stored_certified_document" if storage_record_match else "none"
    result["certified_document_available"] = r2_available_for_cert or storage_record_match

    if stored_doc:
        result["stored_certified_document"] = {
            "available": bool(stored_doc.get("available")),
            "source": stored_doc.get("source", ""),
            "key": stored_doc.get("key", ""),
            "sha256": stored_doc.get("sha256", ""),
            "size_bytes": stored_doc.get("size_bytes", 0),
            "error": stored_doc.get("error", ""),
        }

    certified_pdf_hash_match = "NOT_CHECKED"
    if uploaded_cert_hash and stored_hash:
        certified_pdf_hash_match = "YES" if uploaded_cert_hash == stored_hash else "NO"
    elif uploaded_cert_hash and cid:
        certified_pdf_hash_match = "STORED_CERTIFIED_PDF_NOT_AVAILABLE"
    elif cid:
        certified_pdf_hash_match = "UPLOADED_CERTIFIED_PDF_HASH_NOT_AVAILABLE"

    result["certified_pdf_hash_match"] = certified_pdf_hash_match
    result["tamper_status"] = "UNKNOWN"

    if db_record:
        db_sha = str(db_record.get("sha256") or db_record.get("file_sha256") or db_record.get("original_sha256") or "")
        seal_sha = str(result.get("original_sha256") or "")
        if seal_sha and db_sha:
            result["database_original_hash_match"] = "YES" if seal_sha.lower() == db_sha.lower() else "NO"
        elif seal_sha:
            result["database_original_hash_match"] = "DATABASE_HASH_NOT_AVAILABLE"
        else:
            result["database_original_hash_match"] = "NO_SEAL_ORIGINAL_HASH"

        result["database_record"] = {
            "certificate_id": db_record.get("cert_id", cid),
            "label": db_record.get("label", ""),
            "authenticity": db_record.get("authenticity", ""),
            "ai_score": db_record.get("ai_score", ""),
            "original_file": db_record.get("original_file", ""),
            "upload_time": db_record.get("upload_time", ""),
            "download_count": db_record.get("download_count", 0),
            "certified_to": db_record.get("email", ""),
            "original_sha256": db_sha,
        }

    report = dict(result.get("verification_report") or {})
    report["database_match"] = "YES" if record_confirmed else "NO"
    report["database_record_match"] = "YES" if database_record_match else "NO"
    report["storage_record_match"] = "YES" if storage_record_match else "NO"
    report["record_match_source"] = result.get("record_match_source", "none")
    report["certified_document_available"] = "YES" if (r2_available_for_cert or storage_record_match) else "UNKNOWN" if cid else "NO"
    report["certified_pdf_hash_match"] = certified_pdf_hash_match
    if stored_hash:
        report["stored_certified_pdf_sha256"] = stored_hash
    if uploaded_cert_hash:
        report["uploaded_certified_pdf_sha256"] = uploaded_cert_hash
    if "database_original_hash_match" in result:
        report["original_hash_match"] = result.get("database_original_hash_match")
    if result.get("trust_level"):
        report["trust_level"] = result.get("trust_level")

    # Integrity/tamper interpretation.
    # Phase 8: hard-stop forged/tampered V2 seals before database/storage enrichment
    # can accidentally downgrade the result to a generic NOT_VERIFIED.
    forged_or_tampered_seal = (
        str(result.get("verification_status", "")).upper() == "FORGED_OR_TAMPERED_SEAL"
        or str(result.get("status", "")).lower() == "forged_or_tampered_seal"
        or (
            str(result.get("seal_version", "")).upper() == "VERIFYD-SEAL-V2"
            and str(result.get("signature_status", "")).upper() in ("INVALID", "MISSING", "ERROR")
        )
    )

    if forged_or_tampered_seal:
        result["verified"] = False
        result["status"] = "forged_or_tampered_seal"
        result["seal_status"] = "invalid"
        result["verification_status"] = "FORGED_OR_TAMPERED_SEAL"
        result["integrity_status"] = "FAILED"
        result["tamper_status"] = "FORGED_OR_TAMPERED_SEAL"
        result["trust_level"] = "LOW"

        report["status"] = "FORGED OR TAMPERED"
        report["seal"] = "INVALID SIGNATURE"
        report["integrity"] = "FAILED"
        report["tamper_status"] = "FORGED_OR_TAMPERED_SEAL"
        report["trust_level"] = "LOW"
        report["message"] = (
            "This PDF contains a VeriFYD seal payload, but the cryptographic "
            "signature does not verify. The seal may have been forged or altered."
        )

        result["verification_report"] = report
        return result

    seal_ok = bool(result.get("seal_valid") and result.get("verified"))
    original_hash_ok = str(result.get("database_original_hash_match", "")).upper() in (
        "YES",
        "DATABASE_HASH_NOT_AVAILABLE",
        "NO_SEAL_ORIGINAL_HASH",
    )
    stored_hash_available = bool(stored_hash)
    stored_hash_matches = certified_pdf_hash_match == "YES"
    stored_hash_mismatch = certified_pdf_hash_match == "NO"

    if stored_hash_mismatch:
        tamper_detail = _classify_post_certification_pdf_change(
            uploaded_pdf_path=uploaded_pdf_path,
            stored_doc=stored_doc,
            seal_result=result,
        ) if uploaded_pdf_path else {
            "tamper_reason_code": "PDF_CHANGED_AFTER_CERTIFICATION",
            "tamper_reason": "PDF changed after certification",
            "tamper_confidence": "LOW",
            "tamper_evidence": ["Uploaded certified PDF hash differs from the VeriFYD stored certified PDF hash"],
            "tamper_explanation": (
                "The VeriFYD seal is present, but this PDF no longer matches the exact certified copy "
                "stored by VeriFYD. The file was changed after certification."
            ),
        }

        change_report = _create_verifyd_change_report(
            cid=cid,
            uploaded_pdf_path=uploaded_pdf_path,
            stored_doc=stored_doc,
            tamper_detail=tamper_detail,
            seal_result=result,
        ) if uploaded_pdf_path else {}
        if change_report:
            result["verifyd_change_report"] = change_report

        result["verified"] = False
        result["status"] = "modified_after_certification"
        result["seal_status"] = "valid" if result.get("seal_valid") else "invalid"
        result["verification_status"] = "DOCUMENT_MODIFIED_AFTER_CERTIFICATION"
        result["integrity_status"] = "FAILED"
        result["tamper_status"] = "MODIFIED_AFTER_CERTIFICATION"
        result["tamper_reason_code"] = tamper_detail.get("tamper_reason_code", "PDF_CHANGED_AFTER_CERTIFICATION")
        result["tamper_reason"] = tamper_detail.get("tamper_reason", "PDF changed after certification")
        result["tamper_confidence"] = tamper_detail.get("tamper_confidence", "LOW")
        result["tamper_evidence"] = tamper_detail.get("tamper_evidence", [])
        result["tamper_explanation"] = tamper_detail.get("tamper_explanation", "The certified PDF changed after VeriFYD certification.")
        result["trust_level"] = "LOW"

        report["status"] = "TAMPERED"
        report["seal"] = "VALID" if result.get("seal_valid") else "INVALID"
        report["seal_status"] = result.get("seal_status")
        report["integrity"] = "FAILED"
        report["tamper_status"] = "MODIFIED_AFTER_CERTIFICATION"
        report["tamper_reason_code"] = result["tamper_reason_code"]
        report["tamper_reason"] = result["tamper_reason"]
        report["tamper_confidence"] = result["tamper_confidence"]
        report["tamper_evidence"] = result["tamper_evidence"]
        if result.get("verifyd_change_report"):
            report["verifyd_change_report"] = result.get("verifyd_change_report")
        report["trust_level"] = "LOW"
        report["message"] = result["tamper_explanation"]
    elif seal_ok and record_confirmed and stored_hash_available and stored_hash_matches and original_hash_ok:
        result["status"] = "authentic"
        result["seal_status"] = "valid"
        result["verification_status"] = "AUTHENTIC_VERIFYD_DOCUMENT"
        result["integrity_status"] = "VERIFIED"
        result["tamper_status"] = "NOT_MODIFIED"
        result["trust_level"] = "HIGH"
        report["status"] = "VALID"
        report["seal"] = "VALID"
        report["integrity"] = "VERIFIED"
        report["tamper_status"] = "NOT_MODIFIED"
        report["trust_level"] = "HIGH"
        report["message"] = (
            "This document contains a valid VeriFYD secure seal, has a confirmed "
            "VeriFYD record, and matches the certified PDF stored by VeriFYD."
        )
    elif seal_ok and record_confirmed:
        result["status"] = "seal_valid_storage_hash_not_confirmed"
        result["seal_status"] = "valid"
        result["verification_status"] = "AUTHENTIC_VERIFYD_DOCUMENT_STORAGE_HASH_NOT_CONFIRMED"
        result["integrity_status"] = "SEAL_VALID_STORAGE_HASH_NOT_CONFIRMED"
        result["tamper_status"] = "NOT_CONFIRMED"
        report["status"] = "VALID"
        report["seal"] = "VALID"
        report["integrity"] = "SEAL VALID - STORAGE HASH NOT CONFIRMED"
        report["tamper_status"] = "NOT_CONFIRMED"
        report["message"] = (
            "This document contains a valid VeriFYD secure seal and matches a VeriFYD "
            "certificate record. The stored certified PDF hash was not available for "
            "full tamper comparison."
        )
    elif seal_ok:
        result["status"] = "valid_seal_database_not_confirmed"
        result["seal_status"] = "valid"
        result["verification_status"] = "VALID_SEAL_DATABASE_NOT_CONFIRMED"
        result["integrity_status"] = "SEAL_VALID_DATABASE_NOT_CONFIRMED"
        result["tamper_status"] = "NOT_CONFIRMED"
        report["status"] = "VALID SEAL"
        report["seal"] = "VALID"
        report["integrity"] = "SEAL VALID - DATABASE NOT CONFIRMED"
        report["message"] = "This document contains a valid VeriFYD secure seal, but no matching certificate record was confirmed."
    else:
        result["status"] = "not_verified"
        result["seal_status"] = "invalid" if not result.get("seal_valid") else "unknown"
        result["verification_status"] = "NOT_VERIFIED"
        result["tamper_status"] = "NOT_VERIFIED"
        report["status"] = report.get("status") or "NOT VERIFIED"
        report["integrity"] = result.get("integrity_status", "NOT VERIFIED")

    result["verification_report"] = report
    return result

@app.post("/verify-certificate/")
async def verify_certificate_upload(file: UploadFile = File(...)):
    """Enterprise-friendly alias for verifying a certified VeriFYD PDF upload."""
    return await verify_document_seal(file)


@app.get("/verify-certificate/{cid}")
def verify_certificate_by_id(cid: str):
    """Verify a certificate ID against the VeriFYD certificate database and R2 storage.

    Certificate-ID lookup is a record/storage lookup. Full hidden secure-seal and
    byte-for-byte tamper verification still requires uploading the certified PDF
    through /verify-certificate/ or /verify-document-seal/.

    Phase 3B-1 addition:
    Certified Web Capture records are stored using the existing certified-document
    and certified-file-package artifact paths, so this lookup detects those records
    and returns web-capture-specific metadata/URLs instead of generic document
    wording for the frontend.
    """
    import io as _io
    import json as _json
    import zipfile as _zipfile

    def _is_public_http_url(value: str) -> bool:
        value = str(value or "").strip().lower()
        return value.startswith("http://") or value.startswith("https://")

    def _load_web_capture_package_metadata(_cid: str) -> dict:
        """Best-effort read of metadata.json and hashes.json from the web capture ZIP."""
        out = {"metadata": {}, "hashes": {}, "package_found": False, "source": "", "key": ""}

        def _read_zip_bytes(data: bytes, source: str = "", key: str = "") -> dict:
            if not data:
                return out
            try:
                with _zipfile.ZipFile(_io.BytesIO(data), "r") as zf:
                    names = set(zf.namelist())
                    metadata = {}
                    hashes = {}
                    if "metadata.json" in names:
                        metadata = _json.loads(zf.read("metadata.json").decode("utf-8", errors="replace") or "{}")
                    if "hashes.json" in names:
                        hashes = _json.loads(zf.read("hashes.json").decode("utf-8", errors="replace") or "{}")
                    if metadata or hashes:
                        return {
                            "metadata": metadata if isinstance(metadata, dict) else {},
                            "hashes": hashes if isinstance(hashes, dict) else {},
                            "package_found": True,
                            "source": source,
                            "key": key,
                        }
            except Exception as exc:
                log.debug("verify-certificate: web capture package metadata read failed for %s: %s", _cid, exc)
            return out

        # R2 first: web capture evidence packages are stored in the universal
        # certified-files/{plan}/{cid}.zip location.
        try:
            from storage import _get_client, BUCKET, r2_available
            if r2_available():
                client = _get_client()
                for plan in ("pro", "creator", "enterprise", "free"):
                    key = f"certified-files/{plan}/{_cid}.zip"
                    try:
                        obj = client.get_object(Bucket=BUCKET, Key=key)
                        body = obj.get("Body")
                        data = body.read() if body is not None else b""
                        loaded = _read_zip_bytes(data, "r2", key)
                        if loaded.get("package_found"):
                            return loaded
                    except Exception:
                        continue
        except Exception as exc:
            log.debug("verify-certificate: R2 web capture package metadata lookup failed for %s: %s", _cid, exc)

        # Redis fallback for local/dev or fallback storage jobs.
        try:
            import redis as _redis
            r = _redis.from_url(os.environ.get("REDIS_URL", "redis://localhost:6379"), decode_responses=False)
            data = r.get(f"filecert:{_cid}")
            loaded = _read_zip_bytes(data, "redis", f"filecert:{_cid}")
            if loaded.get("package_found"):
                return loaded
        except Exception as exc:
            log.debug("verify-certificate: Redis web capture package metadata lookup failed for %s: %s", _cid, exc)

        return out

    cid = (cid or "").strip()
    if not cid:
        return JSONResponse({"verified": False, "status": "missing_certificate_id"}, status_code=400)

    cert = get_certificate(cid)
    if not cert:
        return JSONResponse({
            "verified": False,
            "status": "not_found",
            "certificate_id": cid,
            "verification_status": "CERTIFICATE_NOT_FOUND",
        }, status_code=404)

    original_file = cert.get("original_file", "") or ""
    original_sha256 = (
        cert.get("original_sha256")
        or cert.get("original_hash")
        or cert.get("sha256")
        or ""
    )
    certified_document_sha256 = cert.get("certified_document_sha256") or ""
    certified_file_package_sha256 = cert.get("certified_file_package_sha256") or ""
    certified_audio_sha256 = cert.get("certified_audio_sha256") or ""
    certified_photo_sha256 = cert.get("certified_photo_sha256") or ""
    certified_file_hash = (
        cert.get("certified_file_hash")
        or certified_file_package_sha256
        or certified_document_sha256
        or certified_audio_sha256
        or certified_photo_sha256
        or ""
    )

    document_available = False
    file_package_available = False
    video_available = False
    audio_available = False
    photo_available = False

    try:
        from storage import (
            r2_available,
            certified_document_exists,
            certified_file_package_exists,
            certified_exists,
            certified_audio_exists,
        )
        if r2_available():
            document_available = bool(certified_document_exists(cid))
            file_package_available = bool(certified_file_package_exists(cid))
            video_available = bool(certified_exists(cid))
            audio_available = bool(certified_audio_exists(cid))
    except Exception as e:
        log.warning("verify-certificate: R2 availability lookup failed for %s: %s", cid, e)

    try:
        from storage import _get_client, BUCKET, r2_available
        if r2_available():
            client = _get_client()
            for plan in ("free", "creator", "pro", "enterprise"):
                for ext in (".jpg", ".jpeg", ".png", ".webp"):
                    try:
                        client.head_object(Bucket=BUCKET, Key=f"certified-photos/{plan}/{cid}{ext}")
                        photo_available = True
                        break
                    except Exception:
                        continue
                if photo_available:
                    break
    except Exception:
        pass

    if not video_available:
        try:
            import redis as _redis
            r = _redis.from_url(os.environ.get("REDIS_URL", "redis://localhost:6379"), decode_responses=False)
            video_available = bool(r.exists(f"cert:{cid}"))
        except Exception:
            pass

    document_download_url = f"{BASE_URL}/download-document/{cid}" if document_available else ""
    file_package_url = f"{BASE_URL}/download-certified-file/{cid}" if file_package_available else ""
    video_download_url = f"{BASE_URL}/download/{cid}" if video_available else ""
    audio_download_url = f"{BASE_URL}/download-audio/{cid}" if audio_available else ""
    photo_download_url = f"{BASE_URL}/download-photo/{cid}" if photo_available else ""

    # Certified Web Capture records currently reuse the certified document/PDF
    # and certified file package storage locations. They are identifiable by a
    # public URL as original_file + report/package hashes + no media artifact.
    maybe_web_capture = (
        _is_public_http_url(original_file)
        and bool(certified_document_sha256 or document_available)
        and bool(certified_file_package_sha256 or file_package_available)
        and not bool(certified_audio_sha256 or audio_available)
        and not bool(certified_photo_sha256 or photo_available)
        and not bool(video_available)
    )

    web_capture_meta = {}
    web_capture_hashes = {}
    web_capture_package_source = ""
    web_capture_package_key = ""
    if maybe_web_capture:
        loaded_capture_package = _load_web_capture_package_metadata(cid)
        web_capture_meta = loaded_capture_package.get("metadata") or {}
        web_capture_hashes = loaded_capture_package.get("hashes") or {}
        web_capture_package_source = loaded_capture_package.get("source", "")
        web_capture_package_key = loaded_capture_package.get("key", "")

    is_web_capture = bool(
        maybe_web_capture
        and (
            web_capture_meta.get("title") == "VeriFYD Certified Web Capture Metadata"
            or web_capture_meta.get("captured_url")
            or web_capture_hashes.get("screenshot_png")
            or web_capture_hashes.get("captured_html")
            # Keep existing already-created records working even if the ZIP
            # metadata cannot be read at lookup time.
            or _is_public_http_url(original_file)
        )
    )

    if is_web_capture:
        captured_url = web_capture_meta.get("captured_url") or original_file
        final_url = web_capture_meta.get("final_url") or captured_url
        page_title = web_capture_meta.get("page_title") or ""
        captured_at = web_capture_meta.get("captured_at") or cert.get("upload_time", "")
        screenshot_sha256 = (
            web_capture_meta.get("screenshot_sha256")
            or web_capture_hashes.get("screenshot_png")
            or original_sha256
        )
        html_sha256 = (
            web_capture_meta.get("html_sha256")
            or web_capture_hashes.get("captured_html")
            or ""
        )
        report_sha256 = (
            web_capture_meta.get("report_sha256")
            or web_capture_hashes.get("certified_web_capture_report_pdf")
            or certified_document_sha256
        )

        web_document_download_url = f"{BASE_URL}/download-web-capture/{cid}" if document_available else ""
        web_package_url = f"{BASE_URL}/download-web-capture-package/{cid}" if file_package_available else ""
        primary_download_url = web_document_download_url or web_package_url or ""

        verification_report = {
            "title": "VeriFYD Certified Web Capture Verification Report",
            "status": "FOUND",
            "record_type": "Certified Web Capture",
            "certificate_id": cid,
            "database_match": "YES",
            "certified_document_available": "YES" if document_available else "NO",
            "certified_file_package_available": "YES" if file_package_available else "NO",
            "video_available": "NO",
            "audio_available": "NO",
            "photo_available": "NO",
            "trust_level": "DATABASE RECORD FOUND",
            "message": (
                "This certificate ID exists in the VeriFYD certificate database. "
                "The captured webpage record, screenshot hash, HTML snapshot hash, certified report, "
                "evidence package, and timeline are shown below."
            ),
            "captured_url": captured_url,
            "final_url": final_url,
            "page_title": page_title,
            "captured_at": captured_at,
            "screenshot_sha256": screenshot_sha256,
            "html_sha256": html_sha256,
            "certified_document_sha256": report_sha256,
            "certified_file_package_sha256": certified_file_package_sha256,
        }

        return JSONResponse({
            "verified": True,
            "status": "found",
            "verification_status": "CERTIFICATE_RECORD_FOUND",
            "certificate_id": cid,
            "label": "CERTIFIED WEB CAPTURE CREATED",
            "authenticity": "",
            "ai_score": "",
            "authenticity_score": "",
            "original_file": captured_url,
            "original_url": captured_url,
            "captured_url": captured_url,
            "final_url": final_url,
            "page_title": page_title,
            "captured_at": captured_at,
            "upload_time": cert.get("upload_time", ""),
            "download_count": cert.get("download_count", 0),
            "certified_to": cert.get("email", ""),
            "media_type": "web_capture",
            "file_type": "web_capture",
            "record_type": "Certified Web Capture",
            "download_type": "certified_web_capture",
            "web_capture": True,
            "status_label": "CERTIFIED WEB CAPTURE CREATED",
            "display_status": "CERTIFIED WEB CAPTURE RECORD VERIFIED",
            "result_status": "CERTIFIED WEB CAPTURE CREATED",
            "verdict_status": "CERTIFIED WEB CAPTURE CREATED",

            "sha256": screenshot_sha256,
            "screenshot_sha256": screenshot_sha256,
            "html_sha256": html_sha256,
            "original_sha256": screenshot_sha256,
            "original_hash": screenshot_sha256,
            "certified_document_sha256": report_sha256,
            "certified_document_hash": report_sha256,
            "certified_file_package_sha256": certified_file_package_sha256,
            "certified_file_package_hash": certified_file_package_sha256,
            "certified_evidence_package_hash": certified_file_package_sha256,
            "certified_audio_sha256": "",
            "certified_photo_sha256": "",
            "certified_file_hash": certified_file_hash,

            "certified_document_available": bool(document_available),
            "certified_file_package_available": bool(file_package_available),
            "certified_evidence_package_available": bool(file_package_available),
            "video_available": False,
            "audio_available": False,
            "photo_available": False,

            "download_url": primary_download_url,
            "certified_document_url": web_document_download_url,
            "certified_file_package_url": web_package_url,
            "certified_evidence_package_url": web_package_url,
            "evidence_package_url": web_package_url,
            "download_all_certified_files_url": web_package_url,
            "video_download_url": "",
            "audio_download_url": "",
            "photo_download_url": "",

            "capture_report": {
                "title": "VeriFYD Certified Web Capture Report",
                "summary": "VeriFYD captured this public web page at the listed time and generated a certified evidence record.",
                "captured_url": captured_url,
                "final_url": final_url,
                "page_title": page_title,
                "captured_at": captured_at,
                "screenshot_sha256": screenshot_sha256,
                "html_sha256": html_sha256,
                "metadata_source": web_capture_package_source,
                "metadata_key": web_capture_package_key,
            },

            "original_hash_match": "NOT_CHECKED",
            "certified_document_hash_match": "NOT_CHECKED",
            "certified_file_package_hash_match": "NOT_CHECKED",
            "certified_evidence_package_hash_match": "NOT_CHECKED",
            "record_match_source": "certificate_lookup",
            "verification_report": verification_report,
        })

    media_type = "document" if document_available or certified_document_sha256 else (
        "audio" if audio_available or certified_audio_sha256 else
        "photo" if photo_available or certified_photo_sha256 else
        "video" if video_available else
        "file"
    )

    primary_download_url = (
        document_download_url
        or file_package_url
        or audio_download_url
        or photo_download_url
        or video_download_url
        or ""
    )

    verification_report = {
        "title": "VeriFYD Certificate Verification Report",
        "status": "FOUND",
        "certificate_id": cid,
        "database_match": "YES",
        "certified_document_available": "YES" if document_available else "NO",
        "certified_file_package_available": "YES" if file_package_available else "NO",
        "video_available": "YES" if video_available else "NO",
        "audio_available": "YES" if audio_available else "NO",
        "photo_available": "YES" if photo_available else "NO",
        "trust_level": "DATABASE RECORD FOUND",
        "message": (
            "This certificate ID exists in the VeriFYD certificate database. "
            "Upload the certified PDF to verify the hidden secure seal and hash payload."
        ),
        "original_sha256": original_sha256,
        "certified_document_sha256": certified_document_sha256,
        "certified_file_package_sha256": certified_file_package_sha256,
    }

    return JSONResponse({
        "verified": True,
        "status": "found",
        "verification_status": "CERTIFICATE_RECORD_FOUND",
        "certificate_id": cid,
        "label": cert.get("label", ""),
        "authenticity": cert.get("authenticity", ""),
        "ai_score": cert.get("ai_score", ""),
        "original_file": original_file,
        "upload_time": cert.get("upload_time", ""),
        "download_count": cert.get("download_count", 0),
        "certified_to": cert.get("email", ""),
        "media_type": media_type,
        "file_type": media_type,

        "sha256": original_sha256,
        "original_sha256": original_sha256,
        "original_hash": original_sha256,
        "certified_document_sha256": certified_document_sha256,
        "certified_document_hash": certified_document_sha256,
        "certified_file_package_sha256": certified_file_package_sha256,
        "certified_file_package_hash": certified_file_package_sha256,
        "certified_evidence_package_hash": certified_file_package_sha256,
        "certified_audio_sha256": certified_audio_sha256,
        "certified_photo_sha256": certified_photo_sha256,
        "certified_file_hash": certified_file_hash,

        "certified_document_available": bool(document_available),
        "certified_file_package_available": bool(file_package_available),
        "certified_evidence_package_available": bool(file_package_available),
        "video_available": bool(video_available),
        "audio_available": bool(audio_available),
        "photo_available": bool(photo_available),

        "download_url": primary_download_url,
        "certified_document_url": document_download_url,
        "certified_file_package_url": file_package_url,
        "certified_evidence_package_url": file_package_url,
        "video_download_url": video_download_url,
        "audio_download_url": audio_download_url,
        "photo_download_url": photo_download_url,

        "original_hash_match": "NOT_CHECKED",
        "certified_document_hash_match": "NOT_CHECKED",
        "certified_file_package_hash_match": "NOT_CHECKED",
        "certified_evidence_package_hash_match": "NOT_CHECKED",
        "record_match_source": "certificate_lookup",
        "verification_report": verification_report,
    })





# ─────────────────────────────────────────────
#  VeriFYD Vault — certificate preservation records
# ─────────────────────────────────────────────

def _vault_public_payload(record: dict) -> dict:
    """Return a user-safe Vault response payload."""
    record = dict(record or {})
    return {
        "status": "stored",
        "vault_status": "Stored in VeriFYD Vault",
        "vault_key": record.get("vault_key", ""),
        "vault_record_id": record.get("cert_id", ""),
        "certificate_id": record.get("cert_id", ""),
        "email": record.get("email", ""),
        "original_file": record.get("original_file", ""),
        "media_type": record.get("media_type", ""),
        "created_at": record.get("created_at", ""),
        "updated_at": record.get("updated_at", ""),
        "original_sha256": record.get("original_sha256", ""),
        "certified_document_sha256": record.get("certified_document_sha256", ""),
        "certified_file_package_sha256": record.get("certified_file_package_sha256", ""),
        "certified_audio_sha256": record.get("certified_audio_sha256", ""),
        "certified_photo_sha256": record.get("certified_photo_sha256", ""),
        "certified_file_hash": record.get("certified_file_hash", ""),
        "stored_evidence": record.get("stored_evidence", {}),
        "evidence_timeline": record.get("evidence_timeline", []),
        "verification_report": record.get("verification_report", {}),
        "message": "Certificate saved to VeriFYD Vault for future reference.",
    }


@app.post("/vault/save")
async def vault_save(request: Request):
    """Save an existing certificate record into VeriFYD Vault.

    Accepts JSON, form fields, or query parameters:
    - cert_id or certificate_id
    - email optional
    - notes optional
    """
    payload = {}
    try:
        content_type = (request.headers.get("content-type") or "").lower()
        if "application/json" in content_type:
            payload = await request.json()
        elif "form" in content_type:
            form = await request.form()
            payload = dict(form)
    except Exception:
        payload = {}

    cert_id = (
        str(payload.get("cert_id") or payload.get("certificate_id") or "").strip()
        or str(request.query_params.get("cert_id") or request.query_params.get("certificate_id") or "").strip()
    )
    email = (
        str(payload.get("email") or "").strip()
        or str(request.query_params.get("email") or "").strip()
    )
    notes = (
        str(payload.get("notes") or "").strip()
        or str(request.query_params.get("notes") or "").strip()
    )

    if not cert_id:
        return JSONResponse({"status": "error", "error": "missing_certificate_id"}, status_code=400)

    try:
        record = save_certificate_to_vault(cert_id=cert_id, email=email, notes=notes)
        return JSONResponse(_vault_public_payload(record))
    except ValueError as exc:
        err = str(exc)
        if err == "certificate_not_found":
            return JSONResponse({"status": "error", "error": "certificate_not_found", "certificate_id": cert_id}, status_code=404)
        return JSONResponse({"status": "error", "error": err}, status_code=400)
    except Exception as exc:
        log.exception("vault_save failed for cert_id=%s", cert_id)
        return JSONResponse({"status": "error", "error": "vault_save_failed", "detail": str(exc)[:200]}, status_code=500)


@app.get("/vault/by-certificate/{cid}")
def vault_lookup_by_certificate(cid: str):
    """Look up a Vault record by certificate ID."""
    record = get_vault_record_by_cert_id(cid)
    if not record:
        return JSONResponse({"status": "not_found", "error": "vault_record_not_found", "certificate_id": cid}, status_code=404)
    return JSONResponse(_vault_public_payload(record))


@app.get("/vault/{vault_key}")
def vault_lookup(vault_key: str):
    """Look up a Vault record by Vault Key."""
    record = get_vault_record(vault_key)
    if not record:
        return JSONResponse({"status": "not_found", "error": "vault_record_not_found", "vault_key": vault_key}, status_code=404)
    return JSONResponse(_vault_public_payload(record))


@app.get("/job-status/{job_id}")
def job_status(job_id: str):
    """Poll endpoint for async frontends.

    Backward-compatible response:
    - status / job_status / job_state = lifecycle state used for polling
    - result_status / display_status / verdict_status = visible result-card title
    """
    result = get_job_result(job_id)

    if not result or result.get("job_status") == "not_found":
        return JSONResponse({
            "status": "not_found",
            "job_status": "not_found",
            "job_state": "not_found",
        }, status_code=404)

    result_copy = dict(result)

    raw_status = str(result.get("status") or "").strip()
    raw_status_lc = raw_status.lower()
    lifecycle_values = {"queued", "processing", "complete", "error", "not_found"}

    job_st = (
        result.get("job_status")
        or result.get("job_state")
        or ("error" if raw_status_lc == "error" else "")
        or "processing"
    )

    # Preserve the visible verdict/title separately.
    if raw_status and raw_status_lc not in lifecycle_values:
        display_status = raw_status
    else:
        label = str(result.get("label") or "").upper()
        media_type = str(result.get("media_type") or "").lower()
        download_type = str(result.get("download_type") or "").lower()

        if media_type == "audio" or download_type == "certified_audio":
            display_status = (
                "REAL AUDIO VERIFIED" if label == "REAL" else
                "AUDIO UNDETERMINED" if label == "UNDETERMINED" else
                "AI AUDIO DETECTED" if label == "AI" else
                "AUDIO ANALYSIS COMPLETE"
            )
        elif media_type == "video" or result.get("video_ready") is not None:
            display_status = (
                "REAL VIDEO VERIFIED" if label == "REAL" else
                "VIDEO UNDETERMINED" if label == "UNDETERMINED" else
                "AI DETECTED" if label == "AI" else
                "VIDEO ANALYSIS COMPLETE"
            )
        elif media_type == "photo":
            display_status = (
                "REAL PHOTO VERIFIED" if label == "REAL" else
                "PHOTO UNDETERMINED" if label == "UNDETERMINED" else
                "AI DETECTED" if label == "AI" else
                "PHOTO ANALYSIS COMPLETE"
            )
        elif media_type == "document":
            display_status = (
                "REAL DOCUMENT VERIFIED" if label == "REAL" else
                "DOCUMENT UNDETERMINED" if label == "UNDETERMINED" else
                "AI / TAMPERING DETECTED" if label == "AI" else
                "DOCUMENT ANALYSIS COMPLETE"
            )
        else:
            display_status = raw_status if raw_status and raw_status_lc not in lifecycle_values else "ANALYSIS COMPLETE"

    result_copy["result_status"] = display_status
    result_copy["display_status"] = display_status
    result_copy["verdict_status"] = display_status

    # Keep lifecycle state explicit and backward-compatible.
    result_copy["job_status"] = job_st
    result_copy["job_state"] = job_st

    # Important: keep status as lifecycle so older frontend polling stops spinning.
    result_copy["status"] = job_st

    # Backend safety normalization for media download routing.
    cid = result_copy.get("certificate_id") or job_id
    media_type = str(result_copy.get("media_type") or "").lower()
    download_type = str(result_copy.get("download_type") or "").lower()

    if media_type == "audio" or download_type == "certified_audio":
        result_copy["media_type"] = "audio"
        result_copy["download_type"] = "certified_audio"
        if not result_copy.get("download_url"):
            result_copy["download_url"] = f"{BASE_URL}/download-audio/{cid}"
        if not result_copy.get("share_url"):
            result_copy["share_url"] = result_copy.get("download_url")

    elif media_type == "video" or result_copy.get("video_ready") is not None:
        result_copy["media_type"] = "video"
        if result_copy.get("label") == "REAL":
            result_copy["download_type"] = result_copy.get("download_type") or "certified_video"
            if not result_copy.get("download_url"):
                result_copy["download_url"] = f"{BASE_URL}/download/{cid}"
            if not result_copy.get("share_url"):
                result_copy["share_url"] = result_copy.get("download_url")

    return JSONResponse(result_copy)


@app.get("/upload-limits/")
def upload_limits(email: str = ""):
    """
    Return the file size limit for a given email's plan.
    Frontend calls this on load to show correct size messaging.
    """
    PLAN_SIZE_LIMITS_MB = {
        "free":       50,
        "creator":    150,
        "pro":        500,
        "enterprise": 2048,
    }
    PLAN_LABELS = {
        "free":       "Free",
        "creator":    "Creator",
        "pro":        "Pro",
        "enterprise": "Enterprise",
    }
    plan = "free"
    if email and is_valid_email(email):
        try:
            status = get_user_status(email, create=False)
            plan = status.get("plan", "free")
        except Exception:
            pass

    max_mb = PLAN_SIZE_LIMITS_MB.get(plan, 50)
    return JSONResponse({
        "plan":    plan,
        "max_mb":  max_mb,
        "max_bytes": max_mb * 1024 * 1024,
        "label":   PLAN_LABELS.get(plan, "Free"),
        "all_limits": PLAN_SIZE_LIMITS_MB,
    })


@app.get("/share/{cid}")
def share_video(cid: str):
    """
    Redirect to the direct MP4 stream for a certified video.
    This URL is what gets copied — it streams the video directly
    when pasted into social media, chat apps, or browsers.
    No certificate page — just the raw video stream.
    """
    from fastapi.responses import RedirectResponse
    return RedirectResponse(
        url=f"{BASE_URL}/download/{cid}",
        status_code=302
    )


@app.get("/download/{cid}")
def download(cid: str):
    """Serve certified video — checks R2 first, falls back to Redis."""
    from fastapi.responses import RedirectResponse, StreamingResponse
    import io

    increment_downloads(cid)
    fname = f"VeriFYD_Certified_{cid[:8]}.mp4"

    # ── Try R2 first ──────────────────────────────────────────
    try:
        from storage import get_download_url, certified_exists, r2_available
        if r2_available() and certified_exists(cid):
            url = get_download_url(cid)
            return RedirectResponse(url=url, status_code=302)
    except Exception as e:
        log.warning("R2 download lookup failed for %s: %s", cid, e)

    # ── Redis fallback (original behaviour) ───────────────────
    import redis as _redis
    try:
        r = _redis.from_url(os.environ.get("REDIS_URL", "redis://localhost:6379"), decode_responses=False)
        cert_bytes = r.get(f"cert:{cid}")
    except Exception as e:
        log.error("Redis error in /download/%s: %s", cid, e)
        cert_bytes = None
    if not cert_bytes:
        return JSONResponse({"error": "Certificate not found or expired. Please re-verify your video."}, status_code=404)
    return StreamingResponse(
        io.BytesIO(cert_bytes),
        media_type="video/mp4",
        headers={
            "Content-Disposition": f'inline; filename="{fname}"',
            "Content-Length": str(len(cert_bytes)),
            "Accept-Ranges": "bytes",
            "Cache-Control": "no-cache",
        },
    )


@app.get("/certificate/{cid}")
def certificate_lookup(cid: str):
    """Public certificate verification endpoint."""
    cert = get_certificate(cid)
    if not cert:
        return JSONResponse({"error": "Certificate not found"}, status_code=404)
    # Check R2 first, then Redis for video availability
    video_available = False
    try:
        from storage import certified_exists, r2_available
        if r2_available():
            video_available = certified_exists(cid)
    except Exception:
        pass
    if not video_available:
        import redis as _redis
        try:
            r = _redis.from_url(os.environ.get("REDIS_URL", "redis://localhost:6379"), decode_responses=False)
            video_available = bool(r.exists(f"cert:{cid}"))
        except Exception:
            pass
    certified_to = cert.get("email") or ""
    return {
        "certificate_id":  cert["cert_id"],
        "label":           cert["label"],
        "authenticity":    cert["authenticity"],
        "ai_score":        cert["ai_score"],
        "original_file":   cert["original_file"],
        "upload_time":     cert["upload_time"],
        "download_count":  cert["download_count"],
        "verified_by":     "VeriFYD",
        "video_available": video_available,
        "certified_to":    certified_to,
        "email":           certified_to,
    }


@app.get("/pro-download/{cid}")
def pro_download(cid: str, email: str = ""):
    """Re-download certified video — Pro AI and Enterprise plans only."""
    if not email:
        return JSONResponse({"error": "Email required"}, status_code=400)

    # Check plan
    user = get_user_status(email.lower().strip())
    if not user:
        return JSONResponse({"error": "User not found"}, status_code=404)
    if user.get("plan") not in ("pro", "enterprise"):
        return JSONResponse({
            "error": "Pro AI plan required",
            "upgrade_url": "https://vfvid.com/pricing"
        }, status_code=403)

    # Serve file — R2 first, Redis fallback
    from fastapi.responses import RedirectResponse, StreamingResponse
    import io
    increment_downloads(cid)
    fname = f"VeriFYD_Certified_{cid[:8]}.mp4"

    try:
        from storage import get_download_url, certified_exists, r2_available
        if r2_available() and certified_exists(cid):
            url = get_download_url(cid)
            return RedirectResponse(url=url, status_code=302)
    except Exception as e:
        log.warning("R2 pro-download lookup failed for %s: %s", cid, e)

    import redis as _redis
    try:
        r = _redis.from_url(os.environ.get("REDIS_URL", "redis://localhost:6379"), decode_responses=False)
        cert_bytes = r.get(f"cert:{cid}")
    except Exception as e:
        log.error("Redis error in /pro-download/%s: %s", cid, e)
        cert_bytes = None
    if not cert_bytes:
        return JSONResponse({"error": "Video no longer available — please re-verify"}, status_code=404)
    return StreamingResponse(
        io.BytesIO(cert_bytes),
        media_type="video/mp4",
        headers={
            "Content-Disposition": f'inline; filename="{fname}"',
            "Content-Length": str(len(cert_bytes)),
            "Accept-Ranges": "bytes",
            "Cache-Control": "no-cache",
        },
    )


def _proxy_coming_soon_html(video_url: str) -> str:
    platform = "TikTok" if "tiktok.com" in video_url else "Instagram"
    return f"""
    <html>
    <body style="background:black;color:white;text-align:center;
                 padding-top:100px;font-family:Arial;padding-left:20px;padding-right:20px">
      <h1 style="color:#f0a500;font-size:36px">&#128274; {platform} Link Analysis</h1>
      <p style="font-size:18px;max-width:560px;margin:30px auto;line-height:1.6">
        {platform} blocks downloads from cloud servers.<br>
        Full support is <strong>coming soon</strong> to VeriFYD.
      </p>
      <p style="font-size:16px;max-width:560px;margin:0 auto;color:#ccc">
        In the meantime, download the video to your device and use the
        <strong>Upload</strong> button — it works perfectly and gives you
        a full AI analysis.
      </p>
      <p style="color:#555;margin-top:60px;font-size:14px">Analyzed by VeriFYD</p>
    </body>
    </html>
    """


def _error_html(message: str, color: str = "orange") -> str:
    return f"""
    <html>
    <body style="background:black;color:white;text-align:center;
                 padding-top:120px;font-family:Arial;padding-left:20px;padding-right:20px">
      <h1 style="color:{color};font-size:36px">Cannot Analyze Link</h1>
      <p style="font-size:18px;max-width:600px;margin:auto">{message}</p>
      <p style="color:#aaa;margin-top:40px">Analyzed by VeriFYD</p>
    </body>
    </html>
    """


def _payment_required_html(email: str, plan: str) -> str:
    return f"""
    <html>
    <head>
    <style>
      body {{ background:#0a0a0a; color:white; text-align:center;
              padding:60px 20px; font-family:Georgia,serif; margin:0; }}
      .card {{ background:#111; border:1px solid #2a2a2a; border-radius:16px;
               padding:40px; max-width:500px; margin:0 auto;
               box-shadow:0 0 60px rgba(240,165,0,0.08); }}
      h1 {{ color:#f0a500; font-size:28px; margin-bottom:10px; }}
      p  {{ color:#aaa; font-size:15px; line-height:1.6; }}
      .uses {{ background:#1a1a1a; border-radius:8px; padding:12px 20px;
               margin:20px 0; font-size:14px; color:#888; }}
      .plan-btn {{ background:#f0a500; color:black; border:none; border-radius:8px;
                   padding:14px 24px; font-size:14px; font-weight:bold;
                   cursor:pointer; text-decoration:none; display:inline-block;
                   min-width:140px; }}
      .plan-btn:hover {{ background:#ffd166; }}
      small {{ color:#555; font-size:12px; }}
    </style>
    </head>
    <body>
      <div class="card">
        <h1>&#128274; Analysis Limit Reached</h1>
        <p>You have used all <strong>{FREE_USES} free analyses</strong> on your account.</p>
        <div class="uses">
          Signed in as: <strong>{email}</strong><br>
          Current plan: <strong>{plan.title()}</strong>
        </div>
        <p>Upgrade to continue analyzing videos:</p>
        <div style="display:flex;gap:12px;justify-content:center;flex-wrap:wrap;margin:20px 0">
          <a style="background:#1a1a1a;color:#f0a500;border:1px solid #f0a500;border-radius:8px;padding:12px 20px;text-decoration:none;font-size:13px;font-weight:bold;" href="https://vfvid.com/pricing">
            Creator — $19/mo<br><small style="color:#888;font-weight:normal">100 verifications</small>
          </a>
          <a style="background:#f0a500;color:black;border-radius:8px;padding:12px 20px;text-decoration:none;font-size:13px;font-weight:bold;" href="https://vfvid.com/pricing">
            Pro AI — $39/mo<br><small style="color:#333;font-weight:normal">500 verifications</small>
          </a>
        </div>
        <br><br>
        <small>Already upgraded? It may take a moment to reflect. Try again shortly.</small>
      </div>
      <p style="color:#333;margin-top:40px;font-size:13px">Analyzed by VeriFYD</p>
    </body>
    </html>
    """


def _invalid_email_html() -> str:
    return """
    <html>
    <body style="background:#0a0a0a;color:white;text-align:center;
                 padding:80px 20px;font-family:Arial,sans-serif">
      <h1 style="color:#f0a500;font-size:28px">Invalid Email Address</h1>
      <p style="color:#aaa;font-size:16px;max-width:400px;margin:20px auto">
        Please enter a valid email address to use VeriFYD.
      </p>
      <p style="color:#555;margin-top:40px;font-size:13px">Analyzed by VeriFYD</p>
    </body>
    </html>
    """


# ── Platform routing ─────────────────────────────────────────────────────────
#
# PROXY_REQUIRED: TikTok and Instagram actively block cloud/datacenter IPs.
#   These work once RESIDENTIAL_PROXY_URL is set in Render env vars.
#   During testing we return a clean "coming soon" message instead of crashing.
#
# YTDLP_DOMAINS: All other platforms yt-dlp can handle from a cloud IP.
#   YouTube, Vimeo, Reddit, Twitter/X, Facebook, Twitch, Dailymotion, etc.
#
# DIRECT_DOWNLOAD: Plain .mp4 / video file URLs — just requests.get().

YTDLP_DOMAINS = (
    "youtube.com", "youtu.be",
    "tiktok.com",
    "instagram.com",
    "facebook.com", "fb.watch",
    "twitter.com", "x.com", "t.co",
    "reddit.com", "v.redd.it",
    "twitch.tv", "clips.twitch.tv",
    "vimeo.com",
    "dailymotion.com",
    "streamable.com",
    "gfycat.com",
    "imgur.com",
    "bilibili.com",
    "rumble.com",
    "odysee.com",
    "ok.ru",
    "vk.com",
)




# ─────────────────────────────────────────────
#  Session-based email store
#  In-memory dict: session_token → email
#  Persists for the lifetime of the server process.
# ─────────────────────────────────────────────
import hashlib, time
from fastapi.responses import JSONResponse

_session_store: dict = {}  # token → {"email": str, "ts": float}

def _clean_sessions():
    """Remove sessions older than 30 days."""
    cutoff = time.time() - (30 * 86400)
    expired = [k for k, v in _session_store.items() if v["ts"] < cutoff]
    for k in expired:
        del _session_store[k]


@app.post("/register-email/")
async def register_email(request: Request):
    """
    Called by the frontend when user enters their email.
    Returns a session token stored as a cookie on the frontend.
    """
    try:
        body = await request.json()
        email = body.get("email", "").strip()
    except Exception:
        return JSONResponse({"error": "invalid body"}, status_code=400)

    if not is_valid_email(email):
        return JSONResponse({"error": "invalid email"}, status_code=400)
    typo_response = _email_typo_json(email)
    if typo_response:
        return typo_response

    # Create or reuse session token for this email
    token = hashlib.sha256(f"{email}:{int(time.time() // 3600)}".encode()).hexdigest()[:32]
    _session_store[token] = {"email": email, "ts": time.time()}
    _clean_sessions()

    get_or_create_user(email)
    log.info("register-email: %s → token %s", email, token[:8])

    response = JSONResponse({"status": "ok", "token": token})
    response.set_cookie(
        key="vfy_session",
        value=token,
        max_age=2592000,  # 30 days
        samesite="none",
        secure=True,
        httponly=False,
    )
    return response


@app.get("/user-status/")
def user_status(email: str = ""):
    """
    Check a user's current usage status.
    Used by the frontend to show remaining analyses.
    """
    if not email or not is_valid_email(email):
        return JSONResponse({"error": "Invalid email"}, status_code=400)
    typo_response = _email_typo_json(email)
    if typo_response:
        return typo_response
    status = get_user_status(email, create=False)
    return status


@app.get("/analyze-link/", response_class=HTMLResponse)
def analyze_link(request: Request, video_url: str, email: str = ""):
    if not video_url.startswith("http"):
        return HTMLResponse(_error_html("Invalid URL — must start with http."), status_code=400)

    # ── Email resolution ──────────────────────────────────────
    # Priority: 1) query param  2) session token cookie  3) anonymous
    if not email or not is_valid_email(email):
        session_token = request.cookies.get("vfy_session", "")
        if session_token and session_token in _session_store:
            email = _session_store[session_token]["email"]
            log.info("analyze-link: using email from session: %s", email)
        else:
            email = "anonymous@verifyd.com"
            log.info("analyze-link: no session found, using anonymous")

    # ── Usage limit check (skip for anonymous) ────────────────
    if email != "anonymous@verifyd.com":
        status = get_user_status(email)
        if not status["allowed"]:
            return HTMLResponse(
                _payment_required_html(email, status["plan"]),
                status_code=402
            )

    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tmp_path = tmp_file.name
    tmp_file.close()

    try:
        use_ytdlp = any(d in video_url for d in YTDLP_DOMAINS)

        if use_ytdlp:
            log.info("Using yt-dlp for: %s", video_url)
            download_video_ytdlp(video_url, tmp_path)
        else:
            # ── Direct .mp4 / CDN URL → requests ─────────────────────────────
            log.info("Using direct download for: %s", video_url)
            r = requests.get(video_url, stream=True, timeout=30, headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            })
            if r.status_code != 200:
                return HTMLResponse(_error_html(
                    f"Could not download video (HTTP {r.status_code}). "
                    "Make sure the URL points directly to a video file."
                ))
            content_type = r.headers.get("content-type", "")
            if "text/html" in content_type:
                return HTMLResponse(_error_html(
                    "The URL returned a webpage, not a video file. "
                    "Paste a direct .mp4 link, or a URL from YouTube, Vimeo, "
                    "Reddit, Twitter/X, Facebook, Twitch, or Dailymotion."
                ), status_code=400)
            with open(tmp_path, "wb") as f:
                for chunk in r.iter_content(8192):
                    f.write(chunk)

        if os.path.getsize(tmp_path) < 1024:
            return HTMLResponse(_error_html(
                "Downloaded file is too small to be a valid video. "
                "The platform may have blocked the download."
            ), status_code=400)

        authenticity, label, detail, ui_text, color, certify, clip_path = _run_analysis(tmp_path)

        # ── Count this use and store result ──────────────────
        if email != "anonymous@verifyd.com":
            increment_user_uses(email)
            log.info("analyze-link: counted use for %s", email)

        cid = str(uuid.uuid4())
        insert_certificate(
            cert_id       = cid,
            email         = email,
            original_file = video_url[:100],
            label         = label,
            authenticity  = authenticity,
            ai_score      = detail["ai_score"],
        )

        # Return JSON if client accepts it (React frontend), HTML otherwise (legacy/direct browser)
        accept = request.headers.get("accept", "")
        if "application/json" in accept or "text/html" not in accept:
            return JSONResponse({
                "status":             ui_text,
                "authenticity_score": authenticity,
                "color":              color,
                "label":              label,
                "gpt_reasoning":      detail.get("gpt_reasoning", ""),
                "gpt_flags":          detail.get("gpt_flags", []),
                "signal_score":       detail.get("signal_ai_score", 0),
                "gpt_score":          detail.get("gpt_ai_score", 0),
                "audio_score":        detail.get("audio_ai_score", 50),
                "audio_confidence":   detail.get("audio_confidence", "unavailable"),
                "audio_contribution": detail.get("audio_contribution", 0),
            })

        html = f"""
        <html>
        <body style="background:black;color:white;text-align:center;
                     padding-top:120px;font-family:Arial">
          <h1 style="color:{color};font-size:48px">{ui_text}</h1>
          <h2>Authenticity Score: {authenticity}/100</h2>
          <p style="color:#aaa">AI Signal Score: {detail["ai_score"]}/100</p>
          <p>Analyzed by VeriFYD</p>
        </body>
        </html>
        """
        return HTMLResponse(html)

    except RuntimeError as e:
        accept = request.headers.get("accept", "")
        if "application/json" in accept or "text/html" not in accept:
            return JSONResponse({"error": str(e)}, status_code=400)
        return HTMLResponse(_error_html(str(e)), status_code=400)

    except ValueError as e:
        accept = request.headers.get("accept", "")
        if "application/json" in accept or "text/html" not in accept:
            return JSONResponse({"error": str(e)}, status_code=400)
        return HTMLResponse(_error_html(str(e)), status_code=400)

    except Exception as e:
        log.exception("analyze-link failed for %s", video_url)
        accept = request.headers.get("accept", "")
        if "application/json" in accept or "text/html" not in accept:
            return JSONResponse({"error": "Could not process this video. Please try again."}, status_code=500)
        return HTMLResponse(_error_html(
            "Could not process this video. Please try again."
        ), status_code=500)

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        # clip_path is a tmp file created by _run_analysis — always clean it up
        try:
            if 'clip_path' in locals() and clip_path and os.path.exists(clip_path):
                os.remove(clip_path)
        except Exception:
            pass


@app.get("/analyze-link-json/")
async def analyze_link_json(request: Request, video_url: str, email: str = ""):
    """
    JSON-returning async version of /analyze-link/ for the React frontend.
    Uses the RQ worker queue — same pipeline as /upload/.
    """
    if not video_url.startswith("http"):
        return JSONResponse({"error": "Invalid URL — must start with http."}, status_code=400)

    # ── Email resolution ──────────────────────────────────────
    if not email or not is_valid_email(email):
        session_token = request.cookies.get("vfy_session", "")
        if session_token and session_token in _session_store:
            email = _session_store[session_token]["email"]
        else:
            email = "anonymous@verifyd.com"

    # ── Email verification check ──────────────────────────────
    if email != "anonymous@verifyd.com" and not is_email_verified(email):
        return JSONResponse({
            "error":   "email_not_verified",
            "message": "Please verify your email address before analyzing.",
        }, status_code=403)

    # ── Usage limit check ─────────────────────────────────────
    if email != "anonymous@verifyd.com":
        status = get_user_status(email)
        if not status["allowed"]:
            return JSONResponse({
                "error":     "limit_reached",
                "plan":      status["plan"],
                "uses_left": 0,
                "limit":     status["limit"],
            }, status_code=402)

    # ── URL result cache check ───────────────────────────────
    # If this exact URL was analyzed in the last hour, return cached result.
    # Saves SMVD render time (~40s for YouTube) and doesn't count against quota.
    import hashlib as _hl
    _url_key = "urlcache:v2:" + _hl.md5(video_url.strip().encode()).hexdigest()
    try:
        import redis as _redis2
        _r2 = _redis2.from_url(os.environ.get("REDIS_URL","redis://localhost:6379"), decode_responses=False)
        _cached = _r2.get(_url_key)
        if _cached:
            import json as _json
            _cached_result = _json.loads(_cached)
            log.info("analyze-link-json: URL cache HIT for %s", video_url[:80])
            return JSONResponse(_cached_result)
    except Exception as _ce:
        log.warning("analyze-link-json: cache check failed: %s", _ce)

    # ── Enqueue link job ──────────────────────────────────────
    job_id = str(uuid.uuid4())
    try:
        enqueue_link(job_id, video_url, email)
        log.info("analyze-link-json: queued job %s for %s url=%s", job_id, email, video_url[:80])
    except Exception as e:
        log.warning("analyze-link-json: queue unavailable (%s) — running sync", e)
        tmp_path = os.path.join(tempfile.gettempdir(), f"{job_id}.mp4")
        try:
            download_video_ytdlp(video_url, tmp_path)
            authenticity, label, detail, ui_text, color, certify, clip_path = _run_analysis(tmp_path)
            if email != "anonymous@verifyd.com":
                increment_user_uses(email)
            insert_certificate(cert_id=job_id, email=email, original_file=video_url[:100],
                               label=label, authenticity=authenticity,
                               ai_score=detail["ai_score"], sha256=None)
            return JSONResponse({
                "status": ui_text, "authenticity_score": authenticity,
                "color": color, "label": label,
                "gpt_reasoning": detail.get("gpt_reasoning", ""),
                "gpt_flags": detail.get("gpt_flags", []),
                "signal_score": detail.get("signal_ai_score", 0),
                "gpt_score": detail.get("gpt_ai_score", 0),
            })
        except RuntimeError as re:
            return JSONResponse({"error": str(re)}, status_code=400)
        except Exception as e2:
            return JSONResponse({"error": "Analysis failed. Please try again."}, status_code=500)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    # ── Return job_id immediately — frontend polls /job-status/ ─
    # Previously this endpoint blocked for up to 6 minutes waiting for
    # the worker. Render's 30s request timeout caused long jobs (YouTube,
    # multi-clip) to appear to fail even when the worker completed fine.
    # Now we return job_id immediately and the frontend polls job-status.
    # The URL cache key is stored so the worker can write the cache result.
    log.info("analyze-link-json: returning job_id=%s for frontend polling", job_id)
    return JSONResponse({"job_id": job_id, "status": "queued"})


# ─────────────────────────────────────────────
#  PayPal Webhook Handler
#
#  Listens for subscription events from PayPal
#  and upgrades the user's plan automatically.
#
#  Register this URL in PayPal Developer Dashboard:
#  https://verifyd-backend.onrender.com/paypal-webhook/
#
#  Events to subscribe to:
#    - BILLING.SUBSCRIPTION.ACTIVATED
#    - BILLING.SUBSCRIPTION.RENEWED
#    - BILLING.SUBSCRIPTION.CANCELLED
#    - BILLING.SUBSCRIPTION.EXPIRED
# ─────────────────────────────────────────────

from database import upgrade_user_plan, reset_period_uses

# Map PayPal plan IDs → VeriFYD plan names
def _get_plan_map() -> dict:
    return {
        os.environ.get("PAYPAL_PLAN_ID",            ""): "creator",
        os.environ.get("PAYPAL_PLAN_ID_PRO",        ""): "pro",
        os.environ.get("PAYPAL_PLAN_ID_ENTERPRISE", ""): "enterprise",
    }


async def _verify_paypal_webhook(request: Request) -> bool:
    """
    Verify the webhook came from PayPal using their verification API.
    Returns True if valid, False if suspect.
    """
    try:
        client_id  = os.environ.get("PAYPAL_CLIENT_ID", "")
        secret     = os.environ.get("PAYPAL_SECRET", "")
        mode       = os.environ.get("PAYPAL_MODE", "live")
        base_url   = "https://api.paypal.com" if mode == "live" else "https://api.sandbox.paypal.com"

        # Get access token
        token_resp = requests.post(
            f"{base_url}/v1/oauth2/token",
            auth=(client_id, secret),
            data={"grant_type": "client_credentials"},
            timeout=10,
        )
        access_token = token_resp.json().get("access_token", "")
        if not access_token:
            log.warning("PayPal webhook: could not get access token")
            return False

        body = await request.body()
        headers = request.headers

        verify_resp = requests.post(
            f"{base_url}/v1/notifications/verify-webhook-signature",
            headers={
                "Authorization": f"Bearer {access_token}",
                "Content-Type":  "application/json",
            },
            json={
                "transmission_id":   headers.get("paypal-transmission-id", ""),
                "transmission_time": headers.get("paypal-transmission-time", ""),
                "cert_url":          headers.get("paypal-cert-url", ""),
                "auth_algo":         headers.get("paypal-auth-algo", ""),
                "transmission_sig":  headers.get("paypal-transmission-sig", ""),
                "webhook_id":        os.environ.get("PAYPAL_WEBHOOK_ID", ""),
                "webhook_event":     body.decode("utf-8"),
            },
            timeout=10,
        )
        result = verify_resp.json().get("verification_status", "")
        return result == "SUCCESS"

    except Exception as e:
        log.warning("PayPal webhook verification error: %s", e)
        return False


@app.post("/paypal-webhook/")
async def paypal_webhook(request: Request):
    """
    Receives PayPal subscription lifecycle events and updates user plans.
    """
    # Verify signature (skip if PAYPAL_WEBHOOK_ID not set yet)
    webhook_id = os.environ.get("PAYPAL_WEBHOOK_ID", "")
    if webhook_id:
        valid = await _verify_paypal_webhook(request)
        if not valid:
            log.warning("PayPal webhook: invalid signature — rejected")
            return JSONResponse({"error": "invalid signature"}, status_code=400)

    try:
        payload    = await request.json()
        event_type = payload.get("event_type", "")
        resource   = payload.get("resource", {})
        plan_id    = resource.get("plan_id", "")
        sub_id     = resource.get("id", "")

        # Extract subscriber email
        subscriber = resource.get("subscriber", {})
        email      = subscriber.get("email_address", "")

        log.info("PayPal webhook: %s  plan=%s  sub=%s  email=%s",
                 event_type, plan_id, sub_id, email)

        plan_map = _get_plan_map()
        plan_name = plan_map.get(plan_id, "")

        if event_type in ("BILLING.SUBSCRIPTION.ACTIVATED", "BILLING.SUBSCRIPTION.RENEWED"):
            if not email:
                log.warning("PayPal webhook: no email in payload")
                return JSONResponse({"status": "no email"}, status_code=200)

            if not plan_name:
                log.warning("PayPal webhook: unknown plan_id %s", plan_id)
                return JSONResponse({"status": "unknown plan"}, status_code=200)

            upgrade_user_plan(email, plan_name, paypal_sub_id=sub_id)
            log.info("Upgraded %s to %s (sub: %s)", email, plan_name, sub_id)

            # ── Enterprise: auto-provision API key + send welcome email ──
            if plan_name == "enterprise" and event_type == "BILLING.SUBSCRIPTION.ACTIVATED":
                try:
                    # Check if they already have an active key (e.g. resubscribe)
                    existing_keys = list_api_keys(email)
                    active_keys   = [k for k in existing_keys if k.get("active")]

                    if active_keys:
                        # Re-activate existing key rather than creating a duplicate
                        key_record = active_keys[0]
                        log.info("Enterprise re-activation — reusing key for %s", email)
                    else:
                        # New Enterprise customer — provision fresh key
                        # Extract company name from PayPal subscriber data if available
                        company_name = (
                            resource.get("subscriber", {})
                            .get("name", {})
                            .get("full_name", "")
                            or email.split("@")[0].replace(".", " ").title()
                        )
                        key_record = create_api_key(
                            owner_email=email,
                            company_name=company_name,
                        )
                        log.info("Enterprise API key provisioned for %s: %s...",
                                 email, key_record["api_key"][:20])

                    # Send welcome email with embed code
                    send_enterprise_welcome_email(
                        to_email=email,
                        company_name=key_record.get("company_name", ""),
                        api_key=key_record["api_key"],
                    )
                    log.info("Enterprise welcome email sent to %s", email)

                except Exception as e:
                    # Don't fail the webhook — key provisioning failure is recoverable
                    # Admin can manually provision via /admin-create-api-key/
                    log.error("Enterprise key provisioning failed for %s: %s", email, e)

        elif event_type in ("BILLING.SUBSCRIPTION.CANCELLED", "BILLING.SUBSCRIPTION.EXPIRED"):
            if email:
                upgrade_user_plan(email, "free")
                log.info("Downgraded %s to free (sub cancelled/expired)", email)

                # Revoke Enterprise API key if they had one
                try:
                    existing_keys = list_api_keys(email)
                    for k in existing_keys:
                        if k.get("active"):
                            revoke_api_key(k["api_key"])
                            log.info("Enterprise API key revoked for %s on cancellation", email)
                except Exception as e:
                    log.error("Failed to revoke Enterprise key for %s: %s", email, e)

        return JSONResponse({"status": "ok"}, status_code=200)

    except Exception as e:
        log.exception("PayPal webhook processing error: %s", e)
        return JSONResponse({"error": "processing error"}, status_code=500)


@app.get("/admin-set-brand/")
def admin_set_brand(
    key:          str = "",
    api_key:      str = "",
    company_name: str = "",
    logo_url:     str = "",
    brand_color:  str = "",
):
    """Set branding on an enterprise API key. Creates columns if missing."""
    if not _is_admin(key):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    if not api_key:
        return JSONResponse({"error": "api_key param required"}, status_code=400)
    try:
        import psycopg2
        db_url = os.environ.get("DATABASE_URL", "")
        conn = psycopg2.connect(db_url)
        conn.autocommit = False
        cur = conn.cursor()
        # Add columns if they don't exist
        for col, dflt in [("company_name","''"),("logo_url","''"),("brand_color","'#f59e0b'")]:
            cur.execute(f"ALTER TABLE api_keys ADD COLUMN IF NOT EXISTS {col} TEXT DEFAULT {dflt}")
        conn.commit()
        # Update the record
        cur.execute("""
            UPDATE api_keys
            SET company_name = %s, logo_url = %s, brand_color = %s
            WHERE api_key = %s
        """, (company_name, logo_url, brand_color, api_key))
        updated = cur.rowcount
        conn.commit()
        # Fetch back
        cur.execute("SELECT api_key, company_name, logo_url, brand_color FROM api_keys WHERE api_key = %s", (api_key,))
        row = cur.fetchone()
        cur.close(); conn.close()
        if updated == 0:
            return JSONResponse({"error": f"No row found for api_key={api_key}"})
        return JSONResponse({
            "status": "updated",
            "api_key": row[0], "company_name": row[1],
            "logo_url": row[2], "brand_color": row[3],
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/admin-data/")
def admin_data(key: str = ""):
    """
    Returns all user data for the admin dashboard.
    Protected by a simple API key.
    """
    if not _is_admin(key):
        return JSONResponse({"error": "unauthorized"}, status_code=401)

    try:
        from database import get_db
        with get_db() as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT email, plan, total_uses, period_uses, period_start, created_at, last_seen
                FROM users
                ORDER BY created_at DESC
            """)
            users = [dict(row) for row in cur.fetchall()]

        # Summary stats
        total_users     = len(users)
        free_users      = sum(1 for u in users if u["plan"] == "free")
        creator_users   = sum(1 for u in users if u["plan"] == "creator")
        pro_users       = sum(1 for u in users if u["plan"] == "pro")
        enterprise_users = sum(1 for u in users if u["plan"] == "enterprise")
        total_analyses  = sum(u["total_uses"] for u in users)
        monthly_revenue = (creator_users * 19) + (pro_users * 39)

        return {
            "summary": {
                "total_users":      total_users,
                "free_users":       free_users,
                "creator_users":    creator_users,
                "pro_users":        pro_users,
                "enterprise_users": enterprise_users,
                "total_analyses":   total_analyses,
                "monthly_revenue":  monthly_revenue,
            },
            "users": users
        }
    except Exception as e:
        log.error("Admin data error: %s", e)
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/admin-user-certs/")
def admin_user_certs(key: str = "", email: str = ""):
    """
    Returns all certificates scanned by a specific user.
    Usage: /admin-user-certs/?key=ADMIN_KEY&email=user@example.com
    """
    if not _is_admin(key):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    if not email:
        return JSONResponse({"error": "email parameter required"}, status_code=400)

    try:
        from database import get_db
        with get_db() as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT cert_id, original_file, label, authenticity, ai_score, upload_time
                FROM certificates
                WHERE email = %s
                ORDER BY upload_time DESC
            """, (email,))
            rows = [dict(row) for row in cur.fetchall()]

        return {
            "email": email,
            "total": len(rows),
            "certificates": rows
        }
    except Exception as e:
        log.error("Admin user certs error: %s", e)
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/debug-db/")
def debug_db():
    """Show all tables in the database."""
    import sqlite3
    db_path = "/data/verifyd.db" if os.path.isdir("/data") else "verifyd.db"
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cur.fetchall()]
        # Try to create users table directly
        cur.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                email         TEXT    UNIQUE NOT NULL,
                email_lower   TEXT    UNIQUE NOT NULL,
                plan          TEXT    NOT NULL DEFAULT 'free',
                total_uses    INTEGER NOT NULL DEFAULT 0,
                period_uses   INTEGER NOT NULL DEFAULT 0,
                period_start  TEXT    NOT NULL,
                created_at    TEXT    NOT NULL,
                last_seen     TEXT    NOT NULL,
                paypal_sub_id TEXT,
                notes         TEXT
            )
        """)
        conn.commit()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables_after = [row[0] for row in cur.fetchall()]
        conn.close()
        return {"tables_before": tables, "tables_after": tables_after, "db_path": db_path}
    except Exception as e:
        return {"error": str(e)}


@app.get("/admin-list-apikeys/")
def admin_list_apikeys(key: str = ""):
    """List all API keys and ensure branding columns exist."""
    if not _is_admin(key):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    try:
        from database import get_db
        with get_db() as conn:
            cur = conn.cursor()
            # Ensure branding columns exist (safe to run multiple times)
            for col, default in [
                ("company_name", "''"),
                ("logo_url",     "''"),
                ("brand_color",  "'#f59e0b'"),
            ]:
                try:
                    cur.execute(f"ALTER TABLE api_keys ADD COLUMN IF NOT EXISTS {col} TEXT NOT NULL DEFAULT {default}")
                    conn.commit()
                except Exception:
                    conn.rollback()
            # Now fetch all keys
            cur.execute("SELECT api_key, company_name, logo_url, brand_color, owner_email, active FROM api_keys")
            rows = cur.fetchall()
        return JSONResponse({"keys": [
            {"api_key": r[0], "company_name": r[1], "logo_url": r[2],
             "brand_color": r[3], "owner_email": r[4], "active": r[5]}
            for r in rows
        ]})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/admin-update-apikey/")
def admin_update_apikey(
    key:          str = "",
    api_key:      str = "",
    company_name: str = "",
    logo_url:     str = "",
    brand_color:  str = "",
):
    """
    Update branding fields on an enterprise API key.
    Usage: /admin-update-apikey/?key=ADMIN_KEY&api_key=vfyd_live_...&company_name=...&logo_url=...&brand_color=...
    """
    if not _is_admin(key):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    if not api_key:
        return JSONResponse({"error": "api_key required"}, status_code=400)
    try:
        from database import get_db
        with get_db() as conn:
            cur = conn.cursor()
            cur.execute("""
                UPDATE api_keys
                SET company_name = COALESCE(NULLIF(%s, ''), company_name),
                    logo_url     = COALESCE(NULLIF(%s, ''), logo_url),
                    brand_color  = COALESCE(NULLIF(%s, ''), brand_color)
                WHERE api_key = %s
            """, (company_name, logo_url, brand_color, api_key))
            updated = cur.rowcount
            conn.commit()
            # Return updated record
            cur.execute("SELECT api_key, company_name, logo_url, brand_color FROM api_keys WHERE api_key = %s", (api_key,))
            row = cur.fetchone()
        if updated == 0:
            return JSONResponse({"error": "API key not found"}, status_code=404)
        return JSONResponse({
            "status":       "updated",
            "key":          row[0] if row else api_key,
            "company_name": row[1] if row else company_name,
            "logo_url":     row[2] if row else logo_url,
            "brand_color":  row[3] if row else brand_color,
        })
    except Exception as e:
        log.error("Admin update apikey error: %s", e)
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/admin-delete-user/")
def admin_delete_user(email: str = "", key: str = ""):
    """
    Permanently delete a user from the database.
    Used for testing — removes all traces of the user so they can re-register fresh.
    """
    if not _is_admin(key):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    if not email:
        return JSONResponse({"error": "email required"}, status_code=400)
    try:
        from database import get_db
        email_lower = email.strip().lower()
        with get_db() as conn:
            cur = conn.cursor()
            cur.execute("DELETE FROM users WHERE email_lower = %s", (email_lower,))
            deleted = cur.rowcount
            cur.execute("DELETE FROM email_otp WHERE email_lower = %s", (email_lower,))
            conn.commit()
        log.info("Admin deleted user: %s", email_lower)
        return JSONResponse({
            "status": "deleted",
            "email": email_lower,
            "rows_affected": deleted
        })
    except Exception as e:
        log.error("Admin delete user error: %s", e)
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/admin-clear-cache/")
def admin_clear_cache(url: str = "", key: str = ""):
    """
    Clear the URL result cache for a specific video URL.
    Use when a video was incorrectly cached and needs to be re-analyzed.
    Example: /admin-clear-cache/?url=https://youtube.com/shorts/XYZ&key=Honda%236915
    """
    if not _is_admin(key):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    if not url:
        return JSONResponse({"error": "url required"}, status_code=400)
    try:
        import hashlib as _hl
        import redis as _redis_cc
        r = _redis_cc.from_url(
            os.environ.get("REDIS_URL", "redis://localhost:6379"),
            decode_responses=True
        )
        # Clear the exact URL and common variations
        cleared = []
        not_found = []
        variations = [url.strip()]
        # Add www/non-www variation
        if url.startswith("https://www."):
            variations.append(url.replace("https://www.", "https://", 1))
        elif url.startswith("https://"):
            variations.append(url.replace("https://", "https://www.", 1))
        for v in variations:
            cache_key = "urlcache:v2:" + _hl.md5(v.strip().encode()).hexdigest()
            result = r.delete(cache_key)
            if result:
                cleared.append(v)
            else:
                not_found.append(v)
        log.info("Admin cleared URL cache: cleared=%s not_found=%s", cleared, not_found)
        return JSONResponse({
            "status": "ok",
            "cleared": cleared,
            "not_found": not_found,
            "message": f"Cleared {len(cleared)} cache entries. Video will be re-analyzed on next submission."
        })
    except Exception as e:
        log.error("Admin clear cache error: %s", e)
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/admin-disk-cleanup/")
def admin_disk_cleanup(key: str = "", dry_run: bool = False):
    """
    Clean up disk space by removing:
    1. Orphaned certified files on disk with no matching Redis TTL
       (files that should have expired but weren't deleted)
    2. Tmp clip files left over from crashed jobs
    3. Old raw upload files in /var/data/videos/

    Pass dry_run=true to see what would be deleted without deleting.
    Example: /admin-disk-cleanup/?key=Honda%236915
    Example: /admin-disk-cleanup/?key=Honda%236915&dry_run=true
    """
    if not _is_admin(key):
        return JSONResponse({"error": "unauthorized"}, status_code=401)

    import shutil as _shutil
    from config import CERT_DIR, UPLOAD_DIR, TMP_DIR

    results = {
        "dry_run": dry_run,
        "deleted": [],
        "skipped": [],
        "errors": [],
        "freed_bytes": 0,
    }

    try:
        import redis as _redis_dc
        r = _redis_dc.from_url(
            os.environ.get("REDIS_URL", "redis://localhost:6379"),
            decode_responses=True
        )
    except Exception as e:
        return JSONResponse({"error": f"Redis connection failed: {e}"}, status_code=500)

    # ── 1. Orphaned certified files ───────────────────────────
    # These are files in /var/data/certified/ (local disk fallback)
    # that have no matching cert: key in Redis (already expired/served)
    for _dir in [CERT_DIR, "/var/data/certified"]:
        if not os.path.isdir(_dir):
            continue
        for fname in os.listdir(_dir):
            fpath = os.path.join(_dir, fname)
            if not os.path.isfile(fpath):
                continue
            # Extract job_id from filename (cert_<job_id>.mp4 or cert_<job_id>.jpg)
            _base = fname.replace("cert_", "").replace(".mp4", "").replace(".jpg", "").replace(".png", "")
            _redis_key = f"cert:{_base}"
            _exists_in_redis = r.exists(_redis_key)
            fsize = os.path.getsize(fpath)
            if not _exists_in_redis:
                results["freed_bytes"] += fsize
                if not dry_run:
                    try:
                        os.remove(fpath)
                        results["deleted"].append(f"cert/{fname} ({fsize//1024}KB)")
                        log.info("Disk cleanup: deleted orphaned cert file %s", fpath)
                    except Exception as e:
                        results["errors"].append(f"{fname}: {e}")
                else:
                    results["deleted"].append(f"[DRY RUN] cert/{fname} ({fsize//1024}KB)")
            else:
                results["skipped"].append(f"cert/{fname} (still active in Redis)")

    # ── 2. Old tmp clip files ─────────────────────────────────
    # /var/data/tmp/ should be cleaned after each job but crashes can leave files
    # Delete anything older than 2 hours
    import time as _time
    _now = _time.time()
    _tmp_dirs = [TMP_DIR, "/data/tmp", "/var/data/tmp"]
    for _dir in _tmp_dirs:
        if not os.path.isdir(_dir):
            continue
        for fname in os.listdir(_dir):
            fpath = os.path.join(_dir, fname)
            if not os.path.isfile(fpath):
                continue
            _age_hours = (_now - os.path.getmtime(fpath)) / 3600
            if _age_hours > 2:
                fsize = os.path.getsize(fpath)
                results["freed_bytes"] += fsize
                if not dry_run:
                    try:
                        os.remove(fpath)
                        results["deleted"].append(f"tmp/{fname} ({fsize//1024}KB, {_age_hours:.1f}h old)")
                    except Exception as e:
                        results["errors"].append(f"tmp/{fname}: {e}")
                else:
                    results["deleted"].append(f"[DRY RUN] tmp/{fname} ({fsize//1024}KB, {_age_hours:.1f}h old)")

    # ── 3. Old raw upload files ───────────────────────────────
    # /var/data/videos/ — these should be cleaned after processing
    # but delete anything older than 1 hour as a safety net
    if os.path.isdir(UPLOAD_DIR):
        for fname in os.listdir(UPLOAD_DIR):
            fpath = os.path.join(UPLOAD_DIR, fname)
            if not os.path.isfile(fpath):
                continue
            _age_hours = (_now - os.path.getmtime(fpath)) / 3600
            if _age_hours > 1:
                fsize = os.path.getsize(fpath)
                results["freed_bytes"] += fsize
                if not dry_run:
                    try:
                        os.remove(fpath)
                        results["deleted"].append(f"videos/{fname} ({fsize//1024}KB, {_age_hours:.1f}h old)")
                    except Exception as e:
                        results["errors"].append(f"videos/{fname}: {e}")
                else:
                    results["deleted"].append(f"[DRY RUN] videos/{fname} ({fsize//1024}KB, {_age_hours:.1f}h old)")

    results["freed_mb"] = round(results["freed_bytes"] / (1024 * 1024), 2)
    results["summary"] = (
        f"{'Would free' if dry_run else 'Freed'} {results['freed_mb']}MB across "
        f"{len(results['deleted'])} files"
    )
    log.info("Disk cleanup: freed=%.2fMB deleted=%d errors=%d dry_run=%s",
             results["freed_mb"], len(results["deleted"]), len(results["errors"]), dry_run)
    return JSONResponse(results)


@app.get("/admin-disk-usage/")
def admin_disk_usage(key: str = ""):
    """
    Show current disk usage breakdown.
    Example: /admin-disk-usage/?key=Honda%236915
    """
    if not _is_admin(key):
        return JSONResponse({"error": "unauthorized"}, status_code=401)

    import shutil as _shutil
    from config import CERT_DIR, UPLOAD_DIR, TMP_DIR

    def _dir_size(path):
        total = 0
        count = 0
        if os.path.isdir(path):
            for f in os.listdir(path):
                fp = os.path.join(path, f)
                if os.path.isfile(fp):
                    total += os.path.getsize(fp)
                    count += 1
        return {"files": count, "mb": round(total / (1024*1024), 2)}

    # Overall disk
    try:
        _total, _used, _free = _shutil.disk_usage("/var/data" if os.path.isdir("/var/data") else "/data")
        disk_info = {
            "total_gb": round(_total / (1024**3), 2),
            "used_gb":  round(_used  / (1024**3), 2),
            "free_gb":  round(_free  / (1024**3), 2),
            "used_pct": round(_used / _total * 100, 1),
        }
    except Exception:
        disk_info = {}

    return JSONResponse({
        "disk": disk_info,
        "directories": {
            "certified":       _dir_size(CERT_DIR),
            "certified_var":   _dir_size("/var/data/certified"),
            "uploads":         _dir_size(UPLOAD_DIR),
            "tmp":             _dir_size(TMP_DIR),
            "tmp_data":        _dir_size("/data/tmp"),
            "hf_cache":        _dir_size(os.environ.get("HF_HOME", "/var/data/huggingface")),
            "database":        {"mb": round(os.path.getsize("/var/data/certificates.db") / (1024*1024), 2) if os.path.exists("/var/data/certificates.db") else 0},
        },
        "hf_home": os.environ.get("HF_HOME", "not set"),
        "persistent_disk": "/var/data" if os.path.isdir("/var/data") else "not mounted",
        "tip": "Run /admin-disk-cleanup/?key=... to free space from orphaned files"
    })


@app.get("/admin-reset-user/")
def admin_reset_user(email: str = "", key: str = ""):
    """Reset a user's period_uses to 0 for testing."""
    if not _is_admin(key):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    if not email:
        return JSONResponse({"error": "email required"}, status_code=400)
    try:
        reset_period_uses(email)
        return {"status": "reset", "email": email}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/admin-upgrade-user/")
def admin_upgrade_user(email: str = "", plan: str = "enterprise", key: str = ""):
    """Upgrade a user to a paid plan. Admin only."""
    if not _is_admin(key):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    if not email:
        return JSONResponse({"error": "email required"}, status_code=400)
    valid_plans = ["free", "creator", "pro", "enterprise"]
    if plan not in valid_plans:
        return JSONResponse({"error": f"invalid plan, must be one of {valid_plans}"}, status_code=400)
    try:
        get_or_create_user(email)
        upgrade_user_plan(email, plan)
        return {"status": "upgraded", "email": email, "plan": plan}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/admin-disk-cleanup/")
def admin_disk_cleanup(key: str = ""):
    """
    Emergency disk cleanup — removes old temp files, uploaded videos,
    and certified videos older than 2 hours from the persistent disk.
    Call when disk is full: /admin-disk-cleanup/?key=Honda%236915
    """
    if not _is_admin(key):
        return JSONResponse({"error": "unauthorized"}, status_code=401)

    import glob, time
    removed = []
    freed_bytes = 0
    now = time.time()
    errors = []

    # Directories to clean
    dirs_to_clean = [
        (UPLOAD_DIR,  2 * 3600,  "uploaded videos"),
        (CERT_DIR,    2 * 3600,  "certified videos"),
        (TMP_DIR,     1 * 3600,  "temp files"),
        ("/tmp",      1 * 3600,  "system temp"),
    ]

    for directory, max_age, label in dirs_to_clean:
        if not os.path.isdir(directory):
            continue
        try:
            for f in glob.glob(os.path.join(directory, "*")):
                try:
                    age = now - os.path.getmtime(f)
                    if age > max_age and os.path.isfile(f):
                        size = os.path.getsize(f)
                        os.remove(f)
                        freed_bytes += size
                        removed.append(f"{label}: {os.path.basename(f)} ({size//1024}KB, {age/3600:.1f}h old)")
                except Exception as fe:
                    errors.append(str(fe))
        except Exception as de:
            errors.append(f"{label}: {de}")

    # Also check disk usage after cleanup
    try:
        import shutil
        total, used, free = shutil.disk_usage("/data" if os.path.isdir("/data") else "/")
        disk_info = {
            "total_gb": round(total / 1e9, 2),
            "used_gb":  round(used  / 1e9, 2),
            "free_gb":  round(free  / 1e9, 2),
            "used_pct": round(used / total * 100, 1),
        }
    except Exception as e:
        disk_info = {"error": str(e)}

    log.info("Disk cleanup: removed %d files, freed %d MB", len(removed), freed_bytes // 1_000_000)
    return {
        "status":       "complete",
        "files_removed": len(removed),
        "freed_mb":     round(freed_bytes / 1_000_000, 2),
        "disk":         disk_info,
        "removed":      removed[:50],  # cap list
        "errors":       errors[:10],
    }


@app.get("/admin-disk-usage/")
def admin_disk_usage(key: str = ""):
    """Check disk usage and file counts."""
    if not _is_admin(key):
        return JSONResponse({"error": "unauthorized"}, status_code=401)

    import glob, shutil
    result = {}

    for name, directory in [("upload", UPLOAD_DIR), ("cert", CERT_DIR), ("tmp", TMP_DIR)]:
        if os.path.isdir(directory):
            files = glob.glob(os.path.join(directory, "*"))
            total_size = sum(os.path.getsize(f) for f in files if os.path.isfile(f))
            result[name] = {"count": len(files), "size_mb": round(total_size/1e6, 2)}
        else:
            result[name] = {"count": 0, "size_mb": 0}

    try:
        total, used, free = shutil.disk_usage("/data" if os.path.isdir("/data") else "/")
        result["disk"] = {
            "total_gb": round(total/1e9, 2),
            "used_gb":  round(used/1e9, 2),
            "free_gb":  round(free/1e9, 2),
            "used_pct": round(used/total*100, 1),
        }
    except Exception as e:
        result["disk"] = {"error": str(e)}

    return result


@app.get("/test-email/")
def test_email(email: str = "", key: str = ""):
    """Test email validation. Admin only."""
    if not _is_admin(key):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    if not email:
        return JSONResponse({"error": "email required"}, status_code=400)
    is_valid, reason = _verify_email_deliverable(email)
    return {"email": email, "valid": is_valid, "reason": reason}


# ─────────────────────────────────────────────
#  OTP Endpoints
# ─────────────────────────────────────────────
@app.post("/send-otp/")
async def send_otp(email: str = Form(...)):
    """Send a 6-digit OTP to the given email for verification."""
    if not is_valid_email(email):
        return JSONResponse({"error": "Invalid email address."}, status_code=400)
    typo_response = _email_typo_json(email)
    if typo_response:
        return typo_response

    # auto_verified_email_otp_bypass
    if is_auto_verified_email(email):
        return {"status": "already_verified", "message": "Email already verified.", "email": email, "verified": True, "auto_verified": True}

    # Check deliverability first
    is_deliverable, reason = _verify_email_deliverable(email)
    if not is_deliverable:
        return JSONResponse({"error": reason}, status_code=400)

    # Already verified — no need to send again
    if is_email_verified(email):
        return {"status": "already_verified", "message": "Email already verified."}

    # Generate and send OTP
    code = create_otp(email)
    sent = send_otp_email(email, code)

    if not sent:
        return JSONResponse(
            {"error": "Failed to send verification email. Please try again."},
            status_code=500
        )

    log.info("OTP sent to %s", email)
    return {"status": "sent", "message": f"Verification code sent to {email}"}


@app.post("/verify-otp/")
async def verify_otp_route(email: str = Form(...), code: str = Form(...)):
    """Verify the OTP code submitted by the user."""
    if not is_valid_email(email):
        return JSONResponse({"error": "Invalid email address."}, status_code=400)
    typo_response = _email_typo_json(email)
    if typo_response:
        return typo_response

    # auto_verified_email_verify_bypass
    if is_auto_verified_email(email):
        return {"status": "verified", "message": "Email already verified.", "email": email, "verified": True, "auto_verified": True}

    success, message = verify_otp(email, code)

    if not success:
        return JSONResponse({"error": message}, status_code=400)

    # Check if verified user is already at their usage limit
    # This catches users who try to re-verify to bypass the 10-use free limit
    if message == "limit_reached":
        return JSONResponse({
            "error":    "limit_reached",
            "message":  "You have used all 10 free analyses on this account. Please upgrade to continue.",
            "uses_left": 0,
        }, status_code=402)

    return {"status": "verified", "message": message}


@app.get("/test-resend/")
def test_resend(key: str = ""):
    """Test Resend API key directly."""
    if not _is_admin(key):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    import urllib.request, urllib.error, json
    resend_key = os.environ.get("RESEND_API_KEY", "")
    if not resend_key:
        return {"error": "RESEND_API_KEY not set"}
    payload = {
        "from": "onboarding@resend.dev",
        "to": ["rdigiacomo82@gmail.com"],
        "subject": "VeriFYD Test",
        "html": "<p>Test email from VeriFYD</p>"
    }
    try:
        req = urllib.request.Request(
            "https://api.resend.com/emails",
            data=json.dumps(payload).encode(),
            headers={"Authorization": f"Bearer {resend_key}", "Content-Type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            return {"success": True, "response": json.loads(resp.read())}
    except urllib.error.HTTPError as e:
        return {"error": f"HTTP {e.code}", "detail": e.read().decode()}
    except Exception as e:
        return {"error": str(e)}


# ═══════════════════════════════════════════════════════════════
#  ENTERPRISE API — Embeddable Widget
#
#  How it works:
#    1. Admin provisions an Enterprise customer via /admin-create-api-key/
#       → generates key like: vfyd_live_abc123...
#    2. Customer drops ONE script tag on their site:
#       <div id="verifyd-widget"></div>
#       <script src="https://verifyd-backend.onrender.com/widget.js
#                    ?key=vfyd_live_abc123"></script>
#    3. Script injects an iframe pointing to /widget/embed/?key=...
#    4. iframe is a fully self-contained upload UI served from this backend
#    5. Detection runs through the normal queue pipeline
#    6. Results displayed inside the iframe — no data leaves VeriFYD
#
#  Branding config (per API key, stored in DB):
#    company_name  — shown in widget header ("Verified by [company]")
#    logo_url      — customer's logo URL (shown in header)
#    brand_color   — hex color for buttons/accents (default: VeriFYD gold #f59e0b)
#    widget_domains — comma-separated allowed origins (CORS, default: *)
# ═══════════════════════════════════════════════════════════════

def _validate_api_key(key: str):
    """
    Validate an API key from query param or Authorization header.
    Returns the key record dict or None.
    """
    if not key:
        return None
    return get_api_key(key)


# ─────────────────────────────────────────────
#  Admin: provision Enterprise API key
# ─────────────────────────────────────────────
@app.post("/admin-create-api-key/")
def admin_create_api_key(
    owner_email:    str = "",
    company_name:   str = "",
    logo_url:       str = "",
    brand_color:    str = "#f59e0b",
    widget_domains: str = "*",
    key:            str = "",
):
    """
    Admin-only. Provisions a new Enterprise API key.
    The generated key is shown ONCE — store it securely.

    Example:
      POST /admin-create-api-key/?key=Honda%236915
        &owner_email=client@company.com
        &company_name=Acme+News
        &logo_url=https://acme.com/logo.png
        &brand_color=%23ff0000
        &widget_domains=acme.com,www.acme.com
    """
    if not _is_admin(key):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    if not owner_email or not is_valid_email(owner_email):
        return JSONResponse({"error": "valid owner_email required"}, status_code=400)

    record = create_api_key(
        owner_email=owner_email,
        company_name=company_name,
        logo_url=logo_url,
        brand_color=brand_color,
        widget_domains=widget_domains,
    )
    log.info("Admin provisioned Enterprise key for %s", owner_email)

    # Send welcome email with API key and embed code
    email_sent = False
    try:
        email_sent = send_enterprise_welcome_email(
            to_email=owner_email,
            company_name=company_name or owner_email.split("@")[0].title(),
            api_key=record["api_key"],
            brand_color=brand_color,
        )
        log.info("Enterprise welcome email sent to %s: %s", owner_email, email_sent)
    except Exception as e:
        log.error("Enterprise welcome email failed for %s: %s", owner_email, e)

    return {
        "status":       "created",
        "api_key":      record["api_key"],
        "owner_email":  record["owner_email"],
        "company_name": record["company_name"],
        "email_sent":   email_sent,
        "embed_script": (
            f'<div id="verifyd-widget"></div>\n'
            f'<script src="{BASE_URL}/widget.js?key={record["api_key"]}"></script>'
        ),
    }


@app.get("/admin-list-api-keys/")
def admin_list_api_keys(owner_email: str = "", key: str = ""):
    """Admin-only. List all Enterprise API keys."""
    if not _is_admin(key):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    keys = list_api_keys(owner_email if owner_email else None)
    # Mask key in list output — show first 20 chars only
    for k in keys:
        k["api_key"] = k["api_key"][:20] + "..."
    return {"keys": keys}


@app.post("/admin-revoke-api-key/")
def admin_revoke_api_key(api_key: str = "", key: str = ""):
    """Admin-only. Revoke an Enterprise API key."""
    if not _is_admin(key):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    if not api_key:
        return JSONResponse({"error": "api_key required"}, status_code=400)
    revoke_api_key(api_key)
    return {"status": "revoked"}


@app.post("/admin-update-api-key/")
def admin_update_api_key(
    api_key:        str = "",
    company_name:   str = "",
    logo_url:       str = "",
    brand_color:    str = "",
    widget_domains: str = "",
    key:            str = "",
):
    """Admin-only. Update branding config for an Enterprise key."""
    if not _is_admin(key):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    if not api_key:
        return JSONResponse({"error": "api_key required"}, status_code=400)
    updated = update_api_key_branding(
        api_key=api_key,
        company_name=company_name   or None,
        logo_url=logo_url           or None,
        brand_color=brand_color     or None,
        widget_domains=widget_domains or None,
    )
    if not updated:
        return JSONResponse({"error": "API key not found"}, status_code=404)
    return {"status": "updated", "record": updated}


# ─────────────────────────────────────────────
#  Public: widget config (used by widget.js)
# ─────────────────────────────────────────────
@app.get("/widget-config/")
def widget_config(key: str = ""):
    """
    Returns branding config for a given API key.
    Called by widget.js on load to get customer branding.
    No sensitive data exposed — only visual config.
    """
    record = _validate_api_key(key)
    if not record:
        return JSONResponse({"error": "invalid or inactive API key"}, status_code=401)

    return {
        "company_name": record["company_name"] or "VeriFYD",
        "logo_url":     record["logo_url"] or "",
        "brand_color":  record["brand_color"] or "#f59e0b",
        "powered_by":   True,   # always show "Powered by VeriFYD"
    }


# ─────────────────────────────────────────────
#  Public: serve widget.js loader script
# ─────────────────────────────────────────────
@app.get("/widget.js")
def widget_js(key: str = ""):
    """
    Loader script the Enterprise customer drops on their site.
    Validates the key, fetches branding, injects the iframe.

    Usage:
      <div id="verifyd-widget"></div>
      <script src="https://verifyd-backend.onrender.com/widget.js?key=vfyd_live_..."></script>
    """
    record = _validate_api_key(key)
    if not record:
        # Return a no-op script so the page doesn't break
        js = "console.warn('VeriFYD: invalid API key');"
        return HTMLResponse(content=js, media_type="application/javascript")

    embed_url = f"{BASE_URL}/widget/embed/?key={key}"
    brand_color = record.get("brand_color", "#f59e0b")

    js = f"""
(function() {{
  var containerId = 'verifyd-widget';
  var container = document.getElementById(containerId);
  if (!container) {{
    console.warn('VeriFYD: element #' + containerId + ' not found');
    return;
  }}

  var iframe = document.createElement('iframe');
  iframe.src = '{embed_url}';
  iframe.style.cssText = [
    'width: 100%',
    'min-height: 520px',
    'border: none',
    'border-radius: 12px',
    'box-shadow: 0 4px 24px rgba(0,0,0,0.18)',
    'background: #0a0a0a',
    'display: block',
  ].join(';');
  iframe.allow = 'clipboard-write';
  iframe.title = 'VeriFYD Video Verification';

  container.appendChild(iframe);

  // Listen for height resize messages from the widget
  window.addEventListener('message', function(e) {{
    if (e.data && e.data.type === 'verifyd-resize') {{
      iframe.style.minHeight = e.data.height + 'px';
    }}
  }});
}})();
""".strip()

    return HTMLResponse(content=js, media_type="application/javascript")


# ─────────────────────────────────────────────
#  Public: serve the embeddable widget HTML
# ─────────────────────────────────────────────
@app.get("/widget/embed/")
def widget_embed(key: str = ""):
    """
    The actual iframe content. Self-contained HTML+JS upload + URL analysis UI.
    Validates the API key, applies customer branding, runs detection
    through the normal upload/link pipeline with a special widget email.
    """
    record = _validate_api_key(key)
    if not record:
        return HTMLResponse(
            "<html><body style='background:#0a0a0a;color:#f87171;font-family:sans-serif;"
            "padding:32px;text-align:center'><p>Invalid or expired API key.</p></body></html>",
            status_code=401
        )

    company_name = record.get("company_name") or "VeriFYD"
    logo_url     = record.get("logo_url") or ""
    brand_color  = record.get("brand_color") or "#f59e0b"
    backend_url  = BASE_URL

    # Customer name badge + VeriFYD badge
    if company_name and company_name != "VeriFYD":
        logo_html = (
            f'<span style="font-weight:700;font-size:13px;color:#ffffff;margin-right:10px">{company_name}</span>'
            f'<span style="color:rgba(255,255,255,0.4);font-size:16px;margin-right:10px">|</span>'
            f'<span style="font-weight:800;font-size:14px;color:#ffffff;letter-spacing:1px">VERI<span style="color:#fbbf24">FYD</span></span>'
        )
    else:
        logo_html = (
            f'<span style="font-weight:800;font-size:16px;color:#ffffff;letter-spacing:1px">VERI<span style="color:#fbbf24">FYD</span></span>'
        )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{company_name} — Video Verification</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background: #f8faff;
    color: #1f2937;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    padding: 24px;
    min-height: 480px;
  }}
  .header {{
    display: flex;
    align-items: center;
    margin-bottom: 20px;
    padding: 14px 16px;
    background: {brand_color};
    border-radius: 10px;
    margin: -24px -24px 20px -24px;
  }}
  .header-text h2 {{ font-size: 15px; font-weight: 700; color: #ffffff; }}
  .header-text p  {{ font-size: 11px; color: rgba(255,255,255,0.75); margin-top: 2px; }}

  /* ── Tabs ── */
  .tabs {{
    display: flex;
    gap: 0;
    margin-bottom: 18px;
    border-bottom: 2px solid #e5e7eb;
  }}
  .tab {{
    flex: 1;
    padding: 10px 0;
    text-align: center;
    font-size: 13px;
    font-weight: 600;
    color: #9ca3af;
    cursor: pointer;
    border-bottom: 2px solid transparent;
    margin-bottom: -2px;
    transition: color 0.2s, border-color 0.2s;
  }}
  .tab.active {{
    color: {brand_color};
    border-bottom-color: {brand_color};
    background: rgba(255,255,255,0.08);
    border-radius: 6px 6px 0 0;
  }}
  .tab-panel {{ display: none; }}
  .tab-panel.active {{ display: block; }}

  /* ── Upload zone ── */
  .drop-zone {{
    border: 2px dashed #93c5fd;
    border-radius: 10px;
    padding: 36px 20px;
    text-align: center;
    cursor: pointer;
    transition: border-color 0.2s, background 0.2s;
    margin-bottom: 16px;
    position: relative;
    background: #f0f7ff;
    box-shadow: inset 0 0 0 1px rgba(59,130,246,0.1);
  }}
  .drop-zone:hover, .drop-zone.drag-over {{
    border-color: {brand_color};
    background: rgba(245,158,11,0.04);
  }}
  .drop-zone input {{
    position: absolute; inset: 0; opacity: 0;
    cursor: pointer; width: 100%; height: 100%;
  }}
  .drop-icon  {{ font-size: 36px; margin-bottom: 10px; }}
  .drop-label {{ font-size: 14px; color: #9ca3af; }}
  .drop-label span {{ color: {brand_color}; font-weight: 600; }}
  .drop-sub   {{ font-size: 11px; color: #4b5563; margin-top: 6px; }}

  /* ── URL input ── */
  .url-input-wrap {{ margin-bottom: 14px; }}
  .url-input {{
    width: 100%;
    padding: 12px 14px;
    background: white;
    border: 2px solid #93c5fd;
    border-radius: 8px;
    color: #1f2937;
    font-size: 14px;
    outline: none;
    transition: border-color 0.2s;
    box-shadow: 0 1px 4px rgba(59,130,246,0.08);
  }}
  .url-input:focus {{ border-color: {brand_color}; }}
  .url-input::placeholder {{ color: #4b5563; }}
  .url-disclaimer {{
    font-size: 11px;
    color: #6b7280;
    margin-top: 8px;
    padding: 8px 12px;
    background: #f3f4f6;
    border-radius: 6px;
    border-left: 3px solid #d1d5db;
    line-height: 1.5;
  }}

  /* ── File preview ── */
  .file-preview {{
    display: none;
    align-items: center;
    gap: 10px;
    background: #f0f7ff;
    border-radius: 8px;
    padding: 10px 14px;
    margin-bottom: 16px;
    font-size: 13px;
    border: 2px solid #93c5fd;
  }}
  .file-preview .fname {{ color: #1f2937; flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
  .file-preview .fsize {{ color: #6b7280; font-size: 11px; }}
  .file-preview .remove {{ color: #ef4444; cursor: pointer; font-size: 18px; line-height: 1; }}

  /* ── Buttons ── */
  .btn {{
    width: 100%; padding: 13px; border-radius: 8px; border: none;
    font-size: 15px; font-weight: 700; cursor: pointer;
    background: {brand_color}; color: #0a0a0a;
    transition: opacity 0.2s, transform 0.1s; letter-spacing: 0.3px;
  }}
  .btn:hover:not(:disabled) {{ opacity: 0.88; transform: translateY(-1px); }}
  .btn:disabled {{ opacity: 0.45; cursor: not-allowed; transform: none; }}
  .reset-btn {{
    margin-top: 14px; padding: 8px 20px;
    background: transparent; border: 1px solid #374151;
    border-radius: 6px; color: #9ca3af; font-size: 13px;
    cursor: pointer; transition: border-color 0.2s; width: 100%;
  }}
  .reset-btn:hover {{ border-color: {brand_color}; color: {brand_color}; }}

  /* ── Progress ── */
  .progress-wrap {{
    display: none; margin: 20px 0; text-align: center;
  }}
  .spinner {{
    width: 40px; height: 40px; margin: 0 auto 14px;
    border: 3px solid #1f2937;
    border-top-color: {brand_color};
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
  }}
  @keyframes spin {{ to {{ transform: rotate(360deg); }} }}
  .progress-label {{
    font-size: 13px; color: #9ca3af; margin-bottom: 4px; font-weight: 500;
  }}
  .progress-sublabel {{
    font-size: 11px; color: #4b5563; margin-top: 4px;
  }}

  /* ── Result ── */
  .result-box {{ display: none; border-radius: 10px; padding: 20px; text-align: center; margin-top: 16px; }}
  .result-box.green {{ background: rgba(16,185,129,0.10); border: 2px solid rgba(16,185,129,0.5); }}
  .result-box.red   {{ background: rgba(239,68,68,0.10);  border: 2px solid rgba(239,68,68,0.5); }}
  .result-box.blue  {{ background: rgba(59,130,246,0.10); border: 2px solid rgba(59,130,246,0.5); }}
  .result-label {{ font-size: 20px; font-weight: 800; margin-bottom: 6px; }}
  .result-label.green {{ color: #10b981; }}
  .result-label.red   {{ color: #ef4444; }}
  .result-label.blue  {{ color: #3b82f6; }}
  .result-score {{ font-size: 13px; color: #9ca3af; margin-bottom: 10px; }}
  .result-reasoning {{ font-size: 12px; color: #6b7280; line-height: 1.6; text-align: left; margin-top: 8px; }}
  .download-btn {{
    display: block; width: 100%; margin-top: 12px; padding: 13px;
    border-radius: 8px; border: none; font-size: 14px; font-weight: 700;
    cursor: pointer; background: #22c55e; color: #000;
    text-decoration: none; text-align: center; transition: opacity 0.2s;
  }}
  .download-btn:hover {{ opacity: 0.85; }}
  .copy-link-btn {{
    display: block; width: 100%; margin-top: 8px; padding: 11px;
    border-radius: 8px; font-size: 13px; font-weight: 600; cursor: pointer;
    background: transparent; color: #9ca3af; border: 1px solid #374151;
    text-align: center; transition: border-color 0.2s, color 0.2s;
  }}
  .copy-link-btn:hover {{ border-color: #22c55e; color: #22c55e; }}
  .powered-by {{ margin-top: 18px; text-align: center; font-size: 11px; color: #9ca3af; }}
  .powered-by a {{ color: #6b7280; text-decoration: none; }}
  .powered-by a:hover {{ color: {brand_color}; }}
  .error-msg {{ color: #ef4444; font-size: 13px; margin-top: 10px; text-align: center; }}
</style>
</head>
<body>

<div class="header">
  {logo_html}
  <div class="header-text">
    <h2>Video Authenticity Verification</h2>
    <p>Powered by VeriFYD AI Detection</p>
  </div>
</div>

<!-- Tabs -->
<div class="tabs">
  <div class="tab active" id="tabUpload" onclick="switchTab('upload')">⬆ Upload Video</div>
  <div class="tab" id="tabUrl" onclick="switchTab('url')">🔗 Analyze Link</div>
</div>

<!-- Upload Panel -->
<div class="tab-panel active" id="panelUpload">
  <div class="drop-zone" id="dropZone">
    <input type="file" id="fileInput" accept="video/*,.mp4,.mov,.m4v,.avi,.webm,.mkv,.mpg,.mpeg,.3gp,.3g2,.mts,.m2ts,.ts,.ogv,.flv,.wmv">
    <div class="drop-icon">🎬</div>
    <div class="drop-label">Drop video here or <span>browse</span></div>
    <div class="drop-sub">MP4, MOV, M4V, AVI, WEBM, MKV, MPEG/MPG, 3GP, MTS, TS, OGV, FLV, WMV · Max 2GB</div>
  </div>
  <div class="file-preview" id="filePreview">
    <span style="font-size:20px">🎬</span>
    <span class="fname" id="fileName"></span>
    <span class="fsize" id="fileSize"></span>
    <span class="remove" id="removeFile">×</span>
  </div>
  <button class="btn" id="analyzeBtnUpload" disabled onclick="startUpload()">Verify Video</button>
</div>

<!-- URL Panel -->
<div class="tab-panel" id="panelUrl">
  <div class="url-input-wrap">
    <input class="url-input" type="url" id="urlInput"
           placeholder="https://www.tiktok.com/... or YouTube, Instagram, Twitter"
           oninput="onUrlInput()">
    <div class="url-disclaimer">
      ⚠ <strong>Note:</strong> Analyzing a link verifies the downloaded copy of this video.
      A REAL result certifies the video content itself — not that the original source or
      account is authentic. Always verify the source independently.
    </div>
  </div>
  <button class="btn" id="analyzeBtnUrl" disabled onclick="startUrlAnalysis()">Analyze Link</button>
</div>

<!-- Shared progress + result -->
<div class="progress-wrap" id="progressWrap">
  <div class="spinner" id="spinner"></div>
  <div class="progress-label" id="progressLabel">Analyzing video…</div>
  <div class="progress-sublabel" id="progressSublabel">This can take up to 60 seconds</div>
</div>

<div class="result-box" id="resultBox">
  <div class="result-label" id="resultLabel"></div>
  <div class="result-score" id="resultScore"></div>
  <div class="result-reasoning" id="resultReasoning"></div>
  <a class="download-btn" id="downloadBtn" href="#" target="_blank" style="display:none;">
    ⬇ Download Certified Video
  </a>
  <button class="copy-link-btn" id="copyLinkBtn" onclick="copyCertLink()" style="display:none;">
    🔗 Copy Certified Link
  </button>
  <button class="reset-btn" onclick="resetWidget()">Verify another video</button>
</div>

<div class="error-msg" id="errorMsg"></div>

<div class="powered-by">
  Powered by <a href="https://vfvid.com" target="_blank" rel="noopener">VeriFYD</a>
</div>

<script>
var BACKEND      = '{backend_url}';
var API_KEY      = '{key}';
var WIDGET_EMAIL = 'widget_' + API_KEY.slice(-12) + '@verifyd-enterprise.com';
var selectedFile = null;
var currentTab   = 'upload';

// ── Tab switching ─────────────────────────────────────────────
function switchTab(tab) {{
  currentTab = tab;
  document.getElementById('tabUpload').classList.toggle('active', tab === 'upload');
  document.getElementById('tabUrl').classList.toggle('active', tab === 'url');
  document.getElementById('panelUpload').classList.toggle('active', tab === 'upload');
  document.getElementById('panelUrl').classList.toggle('active', tab === 'url');
  document.getElementById('resultBox').style.display    = 'none';
  document.getElementById('progressWrap').style.display = 'none';
  document.getElementById('errorMsg').textContent        = '';
  notifyResize();
}}

// ── URL input ─────────────────────────────────────────────────
function onUrlInput() {{
  var v = document.getElementById('urlInput').value.trim();
  document.getElementById('analyzeBtnUrl').disabled = (v.length < 10);
}}

// ── Drag & drop ───────────────────────────────────────────────
var dz = document.getElementById('dropZone');
dz.addEventListener('dragover',  function(e) {{ e.preventDefault(); dz.classList.add('drag-over'); }});
dz.addEventListener('dragleave', function()  {{ dz.classList.remove('drag-over'); }});
dz.addEventListener('drop', function(e) {{
  e.preventDefault(); dz.classList.remove('drag-over');
  var files = e.dataTransfer.files;
  if (files.length) setFile(files[0]);
}});
document.getElementById('fileInput').addEventListener('change', function(e) {{
  if (e.target.files.length) setFile(e.target.files[0]);
}});
document.getElementById('removeFile').addEventListener('click', function(e) {{
  e.stopPropagation(); resetWidget();
}});

function setFile(f) {{
  selectedFile = f;
  document.getElementById('fileName').textContent          = f.name;
  document.getElementById('fileSize').textContent          = (f.size / 1048576).toFixed(1) + ' MB';
  document.getElementById('filePreview').style.display     = 'flex';
  document.getElementById('dropZone').style.display        = 'none';
  document.getElementById('analyzeBtnUpload').disabled     = false;
  document.getElementById('errorMsg').textContent          = '';
}}

function resetWidget() {{
  selectedFile = null;
  document.getElementById('fileInput').value                = '';
  document.getElementById('urlInput').value                 = '';
  document.getElementById('filePreview').style.display      = 'none';
  document.getElementById('dropZone').style.display         = '';
  document.getElementById('analyzeBtnUpload').disabled      = true;
  document.getElementById('analyzeBtnUrl').disabled         = true;
  document.getElementById('resultBox').style.display        = 'none';
  document.getElementById('progressWrap').style.display     = 'none';
  document.getElementById('errorMsg').textContent           = '';
  document.getElementById('downloadBtn').style.display      = 'none';
  document.getElementById('downloadBtn').href               = '#';
  document.getElementById('copyLinkBtn').style.display      = 'none';
  document.getElementById('copyLinkBtn').textContent        = '🔗 Copy Certified Link';
  document.getElementById('analyzeBtnUpload').textContent   = 'Verify Video';
  document.getElementById('analyzeBtnUpload').disabled      = false;
  document.getElementById('analyzeBtnUrl').textContent      = 'Analyze Link';
  document.getElementById('analyzeBtnUrl').disabled         = true;
  window._certLink = '';
  notifyResize();
}}

function setProgress(pct, label, sublabel) {{
  document.getElementById('progressLabel').textContent    = label || 'Analyzing video…';
  document.getElementById('progressSublabel').textContent = sublabel || 'This can take up to 60 seconds';
}}

function notifyResize() {{
  try {{ window.parent.postMessage({{ type: 'verifyd-resize', height: document.body.scrollHeight + 40 }}, '*'); }} catch(e) {{}}
}}

// ── Upload analysis ───────────────────────────────────────────
function startUpload() {{
  if (!selectedFile) return;
  document.getElementById('analyzeBtnUpload').disabled    = true;
  document.getElementById('analyzeBtnUpload').textContent = 'Analyzing…';
  document.getElementById('progressWrap').style.display   = 'block';
  document.getElementById('resultBox').style.display      = 'none';
  document.getElementById('errorMsg').textContent         = '';
  setProgress(10, 'Uploading video…');
  notifyResize();

  var fd = new FormData();
  fd.append('file',  selectedFile);
  fd.append('email', WIDGET_EMAIL);

  var xhr = new XMLHttpRequest();
  xhr.open('POST', BACKEND + '/widget-upload/?key=' + API_KEY);

  xhr.upload.onprogress = function(e) {{
    if (e.lengthComputable && e.total > 0) {{
      var pct = Math.round(e.loaded / e.total * 100);
      setProgress(0, 'Uploading video… ' + pct + '%', 'Please wait while your video uploads');
    }}
  }};

  xhr.onload = function() {{
    setProgress(0, 'Running detection engines…', '5 AI engines analyzing your video');
    try {{
      var data = JSON.parse(xhr.responseText);
      if (xhr.status >= 400 || data.error) {{ showError(data.error || 'Verification failed.'); return; }}
      showResult(data);
    }} catch(e) {{ showError('Unexpected response. Please try again.'); }}
  }};

  xhr.onerror = function() {{ showError('Network error. Please check your connection.'); }};

  xhr.send(fd);
}}

// ── URL analysis ──────────────────────────────────────────────
function startUrlAnalysis() {{
  var url = document.getElementById('urlInput').value.trim();
  if (!url) return;
  document.getElementById('analyzeBtnUrl').disabled    = true;
  document.getElementById('analyzeBtnUrl').textContent = 'Analyzing…';
  document.getElementById('progressWrap').style.display = 'block';
  document.getElementById('resultBox').style.display    = 'none';
  document.getElementById('errorMsg').textContent       = '';
  setProgress(0, 'Downloading video…', 'Fetching video from the link provided');
  notifyResize();

  var xhr = new XMLHttpRequest();
  xhr.open('POST', BACKEND + '/widget-analyze-link/?key=' + API_KEY);
  xhr.setRequestHeader('Content-Type', 'application/json');

  xhr.onload = function() {{
    try {{
      var data = JSON.parse(xhr.responseText);
      if (xhr.status >= 400 || data.error) {{ showError(data.error || 'Analysis failed.'); return; }}
      if (data.job_id) {{
        setProgress(0, 'Running detection engines…', '5 AI engines analyzing — this can take up to 60 seconds');
        pollResult(data.job_id);
      }} else {{
        showResult(data);
      }}
    }} catch(e) {{ showError('Unexpected response. Please try again.'); }}
  }};

  xhr.onerror = function() {{ showError('Network error. Please check your connection.'); }};

  xhr.send(JSON.stringify({{ url: url, email: WIDGET_EMAIL }}));
}}

function pollResult(jobId) {{
  var attempts = 0;
  var labels = [
    'Examining video frames…',
    'Analyzing motion and lighting…',
    'Checking visual consistency…',
    'Running authenticity checks…',
    'Calculating final result…'
  ];
  var interval = setInterval(function() {{
    attempts++;
    // Cycle through informative labels
    var lbl = labels[Math.min(Math.floor(attempts / 4), labels.length - 1)];
    setProgress(0, lbl, 'This can take up to 60 seconds');

    if (attempts > 80) {{
      clearInterval(interval);
      showError('Analysis timed out. Please try again.');
      return;
    }}
    var xhr = new XMLHttpRequest();
    xhr.open('GET', BACKEND + '/job-status/' + jobId);
    xhr.timeout = 8000;
    xhr.onload = function() {{
      try {{
        var data = JSON.parse(xhr.responseText);
        // /job-status/ returns data.status (not data.job_status)
        var st = data.status || data.job_status || '';
        if (st === 'complete') {{
          clearInterval(interval);
          showResult(data);
        }} else if (st === 'error') {{
          clearInterval(interval);
          showError(data.error || 'Analysis failed.');
        }}
        // queued / processing / not_found → keep polling
      }} catch(e) {{}}
    }};
    xhr.ontimeout = function() {{/* keep polling */}};
    xhr.onerror  = function() {{/* keep polling — transient errors happen */}};
    xhr.send();
  }}, 3000);
}}

// ── Show result ───────────────────────────────────────────────
function showResult(data) {{
  document.getElementById('progressWrap').style.display = 'none';
  var box   = document.getElementById('resultBox');
  var color = (data.color || 'blue').toLowerCase();
  box.className = 'result-box ' + color;
  document.getElementById('resultLabel').className    = 'result-label ' + color;
  document.getElementById('resultLabel').textContent  = data.status || 'RESULT';
  document.getElementById('resultScore').textContent  = 'Authenticity Score: ' + (data.authenticity_score || 0) + ' / 100';
  var reasoning = data.gpt_reasoning || '';
  document.getElementById('resultReasoning').textContent    = reasoning;
  document.getElementById('resultReasoning').style.display  = reasoning ? '' : 'none';

  var dlBtn   = document.getElementById('downloadBtn');
  var copyBtn = document.getElementById('copyLinkBtn');
  if (data.download_url) {{
    dlBtn.href = data.download_url;
    dlBtn.style.display  = 'block';
    window._certLink     = data.download_url;
    copyBtn.style.display = 'block';
    copyBtn.textContent  = '🔗 Copy Certified Link';
  }} else {{
    dlBtn.style.display   = 'none';
    copyBtn.style.display = 'none';
  }}

  box.style.display = 'block';
  document.getElementById('analyzeBtnUpload').textContent = 'Verify Video';
  document.getElementById('analyzeBtnUpload').disabled    = false;
  document.getElementById('analyzeBtnUrl').textContent    = 'Analyze Link';
  document.getElementById('analyzeBtnUrl').disabled       = false;
  notifyResize();
}}

function copyCertLink() {{
  var link = window._certLink || '';
  if (!link) return;
  if (navigator.clipboard && navigator.clipboard.writeText) {{
    navigator.clipboard.writeText(link).then(function() {{
      var btn = document.getElementById('copyLinkBtn');
      btn.textContent = '✓ Link Copied!';
      btn.style.color = '#22c55e'; btn.style.borderColor = '#22c55e';
      setTimeout(function() {{
        btn.textContent = '🔗 Copy Certified Link';
        btn.style.color = ''; btn.style.borderColor = '';
      }}, 2500);
    }});
  }} else {{
    var ta = document.createElement('textarea');
    ta.value = link; ta.style.position = 'fixed'; ta.style.opacity = '0';
    document.body.appendChild(ta); ta.select(); document.execCommand('copy');
    document.body.removeChild(ta);
    var btn = document.getElementById('copyLinkBtn');
    btn.textContent = '✓ Link Copied!';
    setTimeout(function() {{ btn.textContent = '🔗 Copy Certified Link'; }}, 2500);
  }}
}}

function showError(msg) {{
  document.getElementById('progressWrap').style.display  = 'none';
  document.getElementById('errorMsg').textContent        = msg;
  document.getElementById('analyzeBtnUpload').textContent = 'Verify Video';
  document.getElementById('analyzeBtnUpload').disabled    = false;
  document.getElementById('analyzeBtnUrl').textContent    = 'Analyze Link';
  document.getElementById('analyzeBtnUrl').disabled       = false;
  notifyResize();
}}

notifyResize();
</script>
</body>
</html>"""

    return HTMLResponse(content=html)


# ─────────────────────────────────────────────
#  Enterprise: widget upload endpoint
#  Identical to /upload/ but authenticated by
#  API key instead of user email + OTP.
# ─────────────────────────────────────────────
@app.post("/widget-upload/")
async def widget_upload(
    request: Request,
    file:    UploadFile = File(...),
    email:   str        = Form(...),
    key:     str        = "",
):
    """
    Upload endpoint for the embedded widget.
    Authenticated by API key (query param ?key=).
    No OTP / email verification required — the Enterprise
    customer has already authenticated via their API key.
    Usage is tracked against the API key, not the end-user email.
    """
    # ── Validate API key ──────────────────────────────────────
    if not key:
        key = request.query_params.get("key", "")
    record = _validate_api_key(key)
    if not record:
        return JSONResponse({"error": "Invalid or inactive API key."}, status_code=401)

    # ── Plan-based file size limit ───────────────────────────
    # Enterprise widget always gets 2GB limit
    max_bytes = 2048 * 1024 * 1024  # 2GB for enterprise
    max_mb    = 2048

    # Check Content-Length header if present (avoids reading entire file)
    content_length = file.size
    if content_length and content_length > max_bytes:
        return JSONResponse({
            "error":   "file_too_large",
            "message": f"File exceeds the {max_mb}MB enterprise limit.",
            "max_mb":  max_mb,
        }, status_code=413)

    # ── Save file to disk then enqueue ────────────────────────
    job_id   = str(uuid.uuid4())
    raw_path = f"{UPLOAD_DIR}/{job_id}_{file.filename}"

    # Stream to disk
    bytes_written = 0
    with open(raw_path, "wb") as f:
        while chunk := await file.read(1024 * 1024):
            bytes_written += len(chunk)
            if bytes_written > max_bytes:
                try:
                    os.remove(raw_path)
                except Exception:
                    pass
                return JSONResponse({
                    "error":   "file_too_large",
                    "message": f"File exceeds the {max_mb}MB enterprise limit.",
                    "max_mb":  max_mb,
                }, status_code=413)
            f.write(chunk)

    log.info("widget-upload: saved %dMB file for key %s...",
             bytes_written // (1024*1024), key[:20])

    # Use a synthetic enterprise email for tracking
    tracking_email = f"widget_{key[-12:]}@verifyd-enterprise.com"

    try:
        enqueue_upload(job_id, raw_path, file.filename, tracking_email)
        log.info("widget-upload: queued job %s for key %s...", job_id, key[:20])
    except Exception as e:
        log.warning("Queue unavailable for widget upload (%s) — sync fallback", e)
        try:
            sha256 = _sha256(raw_path)
            authenticity, label, detail, ui_text, color, certify, clip_path = _run_analysis(raw_path)
            increment_api_key_uses(key)
            cid = job_id
            insert_certificate(
                cert_id=cid, email=tracking_email, original_file=file.filename,
                label=label, authenticity=authenticity,
                ai_score=detail["ai_score"], sha256=sha256,
            )
            if os.path.exists(clip_path):
                os.remove(clip_path)
            return {"status": ui_text, "authenticity_score": authenticity,
                    "color": color,
                    "gpt_reasoning": detail.get("gpt_reasoning", ""),
                    "gpt_flags":     detail.get("gpt_flags", []),
                    "signal_score":  detail.get("signal_ai_score", 0),
                    "gpt_score":     detail.get("gpt_ai_score", 0)}
        except Exception as e2:
            log.exception("Widget sync fallback failed: %s", e2)
            return JSONResponse({"error": str(e2)}, status_code=500)

    # ── Poll for worker result ────────────────────────────────
    import asyncio
    for _ in range(120):
        await asyncio.sleep(3)
        result = get_job_result(job_id)
        if result and result.get("job_status") == "complete":
            result.pop("job_status", None)
            increment_api_key_uses(key)
            return JSONResponse(result)
        if result and result.get("job_status") == "error":
            # Never expose raw tracebacks to widget end-users
            raw_error = result.get("error", "")
            is_traceback = ("Traceback" in raw_error or "File /opt" in raw_error or len(raw_error) > 200)
            safe_error = (
                "Analysis failed. Please try again or contact support."
                if is_traceback
                else raw_error or "Analysis failed. Please try again."
            )
            log.error("Job error: %s", raw_error[:300])
            return JSONResponse({"error": safe_error}, status_code=500)

    return JSONResponse({"error": "Analysis timed out. Please try again."}, status_code=504)

@app.post("/widget-analyze-link/")
async def widget_analyze_link(request: Request, key: str = ""):
    """
    URL analysis endpoint for the embedded widget.
    Authenticated by API key. Enqueues a link job and returns job_id
    so the widget can poll /job-status/{job_id} for the result.
    """
    if not key:
        key = request.query_params.get("key", "")
    record = _validate_api_key(key)
    if not record:
        return JSONResponse({"error": "Invalid or inactive API key."}, status_code=401)

    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON body."}, status_code=400)

    video_url = (body.get("url") or "").strip()
    if not video_url:
        return JSONResponse({"error": "URL is required."}, status_code=400)

    tracking_email = f"widget_{key[-12:]}@verifyd-enterprise.com"
    job_id = str(uuid.uuid4())

    try:
        enqueue_link(job_id, video_url, tracking_email)
        increment_api_key_uses(key)
        log.info("widget-analyze-link: queued job %s for key %s url=%s",
                 job_id, key[:20], video_url[:60])
        return JSONResponse({"job_id": job_id})
    except Exception as e:
        log.exception("widget-analyze-link enqueue failed: %s", e)
        return JSONResponse({"error": "Failed to queue analysis. Please try again."}, status_code=500)

# ─────────────────────────────────────────────
#  Phase 3B-1 — Certified Web & Social Capture MVP
# ─────────────────────────────────────────────
def _web_capture_common_response(target_url: str, email: str):
    """Validate and enqueue a public URL screenshot/evidence capture."""
    target_url = (target_url or "").strip()
    email = normalize_email(email)

    if not target_url:
        return JSONResponse({"error": "url_required", "message": "A public URL is required."}, status_code=400)
    if not target_url.lower().startswith(("http://", "https://")):
        target_url = "https://" + target_url

    if not email or not is_valid_email(email):
        return JSONResponse({"error": "Invalid email address."}, status_code=400)
    typo_response = _email_typo_json(email)
    if typo_response:
        return typo_response

    is_deliverable, reason = _verify_email_deliverable(email)
    if not is_deliverable:
        return JSONResponse({"error": reason}, status_code=400)

    if not is_email_verified(email):
        return JSONResponse({
            "error": "email_not_verified",
            "message": "Please verify your email address before capturing a web page.",
        }, status_code=403)

    status = get_user_status(email)
    if not status["allowed"]:
        return JSONResponse({
            "error": "limit_reached",
            "plan": status["plan"],
            "uses_left": 0,
            "limit": status["limit"],
        }, status_code=402)

    # Run URL normalization/SSRF validation before queueing so users get a fast error.
    try:
        from web_capture import normalize_public_url
        target_url = normalize_public_url(target_url)
    except Exception as exc:
        return JSONResponse({"error": "invalid_or_private_url", "message": str(exc)[:240]}, status_code=400)

    job_id = str(uuid.uuid4())
    try:
        from queue_helper import enqueue_web_capture
        enqueue_web_capture(job_id, target_url, email)
    except Exception as exc:
        log.warning("capture-url: queue unavailable: %s", exc)
        return JSONResponse({"error": f"Queue unavailable: {str(exc)[:120]}"}, status_code=503)

    return JSONResponse({
        "job_id": job_id,
        "status": "queued",
        "job_status": "queued",
        "media_type": "web_capture",
        "message": "Certified Web Capture queued. Poll /job-status/{job_id} for the certified capture result.",
    })


@app.post("/capture-url/")
async def capture_url(url: str = Form(""), target_url: str = Form(""), email: str = Form("")):
    """Create a certified web capture from a public URL."""
    return _web_capture_common_response(target_url or url, email)


@app.get("/capture-url/")
async def capture_url_get(url: str = "", target_url: str = "", email: str = ""):
    """GET helper for testing Certified Web Capture from a public URL."""
    return _web_capture_common_response(target_url or url, email)


@app.get("/download-web-capture/{cid}")
async def download_web_capture(cid: str):
    """Alias for downloading the Certified Web Capture PDF report."""
    return RedirectResponse(url=f"{BASE_URL}/download-document/{cid}", status_code=302)


@app.get("/download-web-capture-package/{cid}")
async def download_web_capture_package(cid: str):
    """Alias for downloading the Certified Web Capture evidence package ZIP."""
    return RedirectResponse(url=f"{BASE_URL}/download-certified-file/{cid}", status_code=302)
