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
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
import os, uuid, requests, tempfile, logging, hashlib
from contextlib import asynccontextmanager

from detection import run_detection   # returns (authenticity, label, detail)
from queue_helper import (enqueue_upload, enqueue_link,
                          enqueue_photo_upload, enqueue_photo_link,
                          get_job_result)
from config import (                  # single source of truth for all settings
    BASE_URL,
    UPLOAD_DIR, CERT_DIR, TMP_DIR,
)
from emailer  import send_otp_email, send_certification_email, send_enterprise_welcome_email
from database import (init_db, insert_certificate, increment_downloads,
                      get_or_create_user, get_user_status, increment_user_uses,
                      is_valid_email, FREE_USES, get_certificate,
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

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    init_db()
    log.info("VeriFYD startup complete")
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
    email_lower = email.lower().strip()

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
@app.post("/upload/")
async def upload(file: UploadFile = File(...), email: str = Form(...)):
    # ── Email format validation ───────────────────────────────
    if not is_valid_email(email):
        return JSONResponse({"error": "Invalid email address."}, status_code=400)

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
                        "gpt_score": detail.get("gpt_ai_score",0)}
            if os.path.exists(clip_path):
                os.remove(clip_path)
            return {"status": ui_text, "authenticity_score": authenticity, "color": color,
                    "gpt_reasoning": detail.get("gpt_reasoning",""),
                    "gpt_flags": detail.get("gpt_flags",[]),
                    "signal_score": detail.get("signal_ai_score",0),
                    "gpt_score": detail.get("gpt_ai_score",0)}
        except Exception as e2:
            log.exception("Sync fallback also failed for %s", raw_path)
            return JSONResponse({"error": str(e2)}, status_code=500)

    # ── Transparent polling — wait for worker result ──────────
    # Frontend sees same response format as before — no job_id exposed.
    import asyncio
    for _ in range(120):   # poll up to 6 minutes
        await asyncio.sleep(3)
        result = get_job_result(job_id)
        if result and result.get("job_status") == "complete":
            result.pop("job_status", None)
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


@app.get("/job-status/{job_id}")
def job_status(job_id: str):
    """Poll endpoint for direct async frontends."""
    result = get_job_result(job_id)
    if not result or result.get("job_status") == "not_found":
        return JSONResponse({"status": "not_found"}, status_code=404)
    # Normalize job_status → status so frontend can use data.status consistently
    job_st = result.get("job_status", "")
    result_copy = {k: v for k, v in result.items() if k != "job_status"}
    if job_st == "complete":
        result_copy["status"] = "complete"
    elif job_st == "error":
        result_copy["status"] = "error"
    else:
        result_copy["status"] = job_st or "processing"
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
            status = get_user_status(email)
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
    status = get_user_status(email)
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
    <input type="file" id="fileInput" accept="video/*,.mp4,.mov,.avi,.webm,.mkv">
    <div class="drop-icon">🎬</div>
    <div class="drop-label">Drop video here or <span>browse</span></div>
    <div class="drop-sub">MP4, MOV, AVI, WEBM · Max 2GB</div>
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