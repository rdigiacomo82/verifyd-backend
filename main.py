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
from queue_helper import enqueue_upload, enqueue_link, get_job_result
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
    allow_origins=["*"],
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

    # ── Usage limit check ─────────────────────────────────────
    status = get_user_status(email)
    if not status["allowed"]:
        return JSONResponse({
            "error":      "limit_reached",
            "plan":       status["plan"],
            "uses_left":  0,
            "limit":      status["limit"],
        }, status_code=402)

    # ── Save file to disk then enqueue ────────────────────────
    job_id   = str(uuid.uuid4())
    raw_path = f"{UPLOAD_DIR}/{job_id}_{file.filename}"

    with open(raw_path, "wb") as f:
        while chunk := await file.read(1024 * 1024):
            f.write(chunk)

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


@app.get("/job-status/{job_id}")
def job_status(job_id: str):
    """Poll endpoint for direct async frontends."""
    result = get_job_result(job_id)
    if not result or result.get("job_status") == "not_found":
        return JSONResponse({"job_status": "not_found"}, status_code=404)
    return JSONResponse(result)


@app.get("/download/{cid}")
def download(cid: str):
    """Serve certified video stored in Redis by the worker.
    Uses StreamingResponse with Content-Disposition: inline so mobile
    browsers play the video directly rather than triggering a download."""
    import redis as _redis
    from fastapi.responses import StreamingResponse
    import io
    redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379")
    try:
        r = _redis.from_url(redis_url, decode_responses=False)
        cert_bytes = r.get(f"cert:{cid}")
    except Exception as e:
        log.error("Redis error in /download/%s: %s", cid, e)
        cert_bytes = None
    if not cert_bytes:
        return JSONResponse({"error": "Certificate not found or expired. Videos are available for 1 hour after verification."}, status_code=404)
    increment_downloads(cid)
    fname = f"VeriFYD_Certified_{cid[:8]}.mp4"
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
    # Check Redis for video availability (worker stores certified video there)
    import redis as _redis
    try:
        r = _redis.from_url(os.environ.get("REDIS_URL", "redis://localhost:6379"), decode_responses=False)
        video_available = bool(r.exists(f"cert:{cid}"))
    except Exception:
        video_available = False
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

    # Serve file from Redis
    import redis as _redis
    import tempfile
    try:
        r = _redis.from_url(os.environ.get("REDIS_URL", "redis://localhost:6379"), decode_responses=False)
        cert_bytes = r.get(f"cert:{cid}")
    except Exception as e:
        log.error("Redis error in /pro-download/%s: %s", cid, e)
        cert_bytes = None
    if not cert_bytes:
        return JSONResponse({"error": "Video no longer available — please re-verify"}, status_code=404)
    increment_downloads(cid)
    from fastapi.responses import StreamingResponse
    import io
    fname = f"VeriFYD_Certified_{cid[:8]}.mp4"
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
        return HTMLResponse(_error_html(str(e)), status_code=400)

    except ValueError as e:
        return HTMLResponse(_error_html(str(e)), status_code=400)

    except Exception as e:
        log.exception("analyze-link failed for %s", video_url)
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

    # ── Poll for worker result ────────────────────────────────
    import asyncio
    for _ in range(120):   # up to 6 minutes
        await asyncio.sleep(3)
        result = get_job_result(job_id)
        if result and result.get("job_status") == "complete":
            result.pop("job_status", None)
            return JSONResponse(result)
        if result and result.get("job_status") == "error":
            raw_error = result.get("error", "")
            is_tb = "Traceback" in raw_error or "File /opt" in raw_error or len(raw_error) > 200
            safe  = "Analysis failed. Please try again." if is_tb else (raw_error or "Analysis failed.")
            log.error("analyze-link-json job error: %s", raw_error[:300])
            return JSONResponse({"error": safe}, status_code=500)

    return JSONResponse({"error": "Analysis timed out. Please try again."}, status_code=504)


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


@app.get("/admin-data/")
def admin_data(key: str = ""):
    """
    Returns all user data for the admin dashboard.
    Protected by a simple API key.
    """
    if not _is_admin(key):
        return JSONResponse({"error": "unauthorized"}, status_code=401)

    try:
        import sqlite3
        db_path = "/data/verifyd.db" if os.path.isdir("/data") else "verifyd.db"
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()

        # Get all users
        cur.execute("""
            SELECT email, plan, total_uses, period_uses, period_start, created_at, last_seen
            FROM users
            ORDER BY created_at DESC
        """)
        users = [dict(row) for row in cur.fetchall()]

        # Summary stats
        total_users   = len(users)
        free_users    = sum(1 for u in users if u["plan"] == "free")
        creator_users = sum(1 for u in users if u["plan"] == "creator")
        pro_users     = sum(1 for u in users if u["plan"] == "pro")
        total_analyses  = sum(u["total_uses"] for u in users)
        monthly_revenue = (creator_users * 19) + (pro_users * 39)

        conn.close()

        return {
            "summary": {
                "total_users": total_users,
                "free_users": free_users,
                "creator_users": creator_users,
                "pro_users": pro_users,
                "total_analyses": total_analyses,
                "monthly_revenue": monthly_revenue,
            },
            "users": users
        }
    except Exception as e:
        log.error("Admin data error: %s", e)
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
    The actual iframe content. Self-contained HTML+JS upload UI.
    Validates the API key, applies customer branding, runs detection
    through the normal upload pipeline with a special widget email.
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

    logo_html = (
        f'<img src="{logo_url}" alt="{company_name}" '
        f'style="height:32px;max-width:160px;object-fit:contain;margin-right:10px">'
        if logo_url else ""
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
    background: #0a0a0a;
    color: #e5e7eb;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    padding: 24px;
    min-height: 480px;
  }}
  .header {{
    display: flex;
    align-items: center;
    margin-bottom: 20px;
    padding-bottom: 16px;
    border-bottom: 1px solid #1f2937;
  }}
  .header-text h2 {{
    font-size: 16px;
    font-weight: 700;
    color: #f9fafb;
  }}
  .header-text p {{
    font-size: 12px;
    color: #6b7280;
    margin-top: 2px;
  }}
  .drop-zone {{
    border: 2px dashed #374151;
    border-radius: 10px;
    padding: 36px 20px;
    text-align: center;
    cursor: pointer;
    transition: border-color 0.2s, background 0.2s;
    margin-bottom: 16px;
    position: relative;
  }}
  .drop-zone:hover, .drop-zone.drag-over {{
    border-color: {brand_color};
    background: rgba(245,158,11,0.04);
  }}
  .drop-zone input {{
    position: absolute;
    inset: 0;
    opacity: 0;
    cursor: pointer;
    width: 100%;
    height: 100%;
  }}
  .drop-icon {{ font-size: 36px; margin-bottom: 10px; }}
  .drop-label {{ font-size: 14px; color: #9ca3af; }}
  .drop-label span {{ color: {brand_color}; font-weight: 600; }}
  .drop-sub {{ font-size: 11px; color: #4b5563; margin-top: 6px; }}
  .file-preview {{
    display: none;
    align-items: center;
    gap: 10px;
    background: #111827;
    border-radius: 8px;
    padding: 10px 14px;
    margin-bottom: 16px;
    font-size: 13px;
  }}
  .file-preview .fname {{ color: #e5e7eb; flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
  .file-preview .fsize {{ color: #6b7280; font-size: 11px; }}
  .file-preview .remove {{ color: #ef4444; cursor: pointer; font-size: 18px; line-height: 1; }}
  .btn {{
    width: 100%;
    padding: 13px;
    border-radius: 8px;
    border: none;
    font-size: 15px;
    font-weight: 700;
    cursor: pointer;
    background: {brand_color};
    color: #0a0a0a;
    transition: opacity 0.2s, transform 0.1s;
    letter-spacing: 0.3px;
  }}
  .btn:hover:not(:disabled) {{ opacity: 0.88; transform: translateY(-1px); }}
  .btn:disabled {{ opacity: 0.45; cursor: not-allowed; transform: none; }}
  .progress-wrap {{
    display: none;
    margin: 16px 0;
  }}
  .progress-bar {{
    height: 4px;
    background: #1f2937;
    border-radius: 2px;
    overflow: hidden;
  }}
  .progress-fill {{
    height: 100%;
    background: {brand_color};
    width: 0%;
    transition: width 0.3s;
    border-radius: 2px;
  }}
  .progress-label {{ font-size: 12px; color: #6b7280; margin-top: 6px; text-align: center; }}
  .result-box {{
    display: none;
    border-radius: 10px;
    padding: 20px;
    text-align: center;
    margin-top: 16px;
  }}
  .result-box.green {{ background: rgba(16,185,129,0.10); border: 1px solid rgba(16,185,129,0.3); }}
  .result-box.red   {{ background: rgba(239,68,68,0.10);  border: 1px solid rgba(239,68,68,0.3); }}
  .result-box.blue  {{ background: rgba(59,130,246,0.10); border: 1px solid rgba(59,130,246,0.3); }}
  .result-label {{ font-size: 20px; font-weight: 800; margin-bottom: 6px; }}
  .result-label.green {{ color: #10b981; }}
  .result-label.red   {{ color: #ef4444; }}
  .result-label.blue  {{ color: #3b82f6; }}
  .result-score {{ font-size: 13px; color: #9ca3af; margin-bottom: 10px; }}
  .result-reasoning {{ font-size: 12px; color: #6b7280; line-height: 1.6; text-align: left; margin-top: 8px; }}
  .reset-btn {{
    margin-top: 14px;
    padding: 8px 20px;
    background: transparent;
    border: 1px solid #374151;
    border-radius: 6px;
    color: #9ca3af;
    font-size: 13px;
    cursor: pointer;
    transition: border-color 0.2s;
  }}
  .reset-btn:hover {{ border-color: {brand_color}; color: {brand_color}; }}
  .powered-by {{
    margin-top: 18px;
    text-align: center;
    font-size: 11px;
    color: #374151;
  }}
  .powered-by a {{ color: #4b5563; text-decoration: none; }}
  .powered-by a:hover {{ color: {brand_color}; }}
  .error-msg {{ color: #ef4444; font-size: 13px; margin-top: 10px; text-align: center; }}
  .download-btn {{
    display: block; width: 100%; margin-top: 12px;
    padding: 13px; border-radius: 8px; border: none;
    font-size: 14px; font-weight: 700; cursor: pointer;
    background: #22c55e; color: #000;
    text-decoration: none; text-align: center;
    transition: opacity 0.2s;
  }}
  .download-btn:hover {{ opacity: 0.85; }}
  .copy-link-btn {{
    display: block; width: 100%; margin-top: 8px;
    padding: 11px; border-radius: 8px;
    font-size: 13px; font-weight: 600; cursor: pointer;
    background: transparent; color: #9ca3af;
    border: 1px solid #374151; text-align: center;
    transition: border-color 0.2s, color 0.2s;
  }}
  .copy-link-btn:hover {{ border-color: #22c55e; color: #22c55e; }}
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

<div class="drop-zone" id="dropZone">
  <input type="file" id="fileInput" accept="video/*,.mp4,.mov,.avi,.webm,.mkv">
  <div class="drop-icon">🎬</div>
  <div class="drop-label">Drop video here or <span>browse</span></div>
  <div class="drop-sub">MP4, MOV, AVI, WEBM · Max 500MB</div>
</div>

<div class="file-preview" id="filePreview">
  <span style="font-size:20px">🎬</span>
  <span class="fname" id="fileName"></span>
  <span class="fsize" id="fileSize"></span>
  <span class="remove" id="removeFile">×</span>
</div>

<button class="btn" id="analyzeBtn" disabled onclick="startAnalysis()">
  Verify Video
</button>

<div class="progress-wrap" id="progressWrap">
  <div class="progress-bar"><div class="progress-fill" id="progressFill"></div></div>
  <div class="progress-label" id="progressLabel">Uploading…</div>
</div>

<div class="result-box" id="resultBox">
  <div class="result-label" id="resultLabel"></div>
  <div class="result-score" id="resultScore"></div>
  <div class="result-reasoning" id="resultReasoning"></div>
  <a class="download-btn" id="downloadBtn" href="#" target="_blank"
     style="display:none;">
    ⬇ Download Certified Video
  </a>
  <button class="copy-link-btn" id="copyLinkBtn" onclick="copyCertLink()"
     style="display:none;">
    🔗 Copy Certified Link
  </button>
  <button class="reset-btn" onclick="resetWidget()" style="margin-top:10px;">Verify another video</button>
</div>

<div class="error-msg" id="errorMsg"></div>

<div class="powered-by">
  Powered by <a href="https://vfvid.com" target="_blank" rel="noopener">VeriFYD</a>
</div>

<script>
var BACKEND = '{backend_url}';
var API_KEY  = '{key}';
// Widget uses a synthetic enterprise email keyed to the API key
// so uses are tracked per Enterprise account, not per end-user
var WIDGET_EMAIL = 'widget_' + API_KEY.slice(-12) + '@verifyd-enterprise.com';

var selectedFile = null;

// ── Drag & drop ──────────────────────────────────────────────
var dz = document.getElementById('dropZone');
dz.addEventListener('dragover',  function(e) {{ e.preventDefault(); dz.classList.add('drag-over'); }});
dz.addEventListener('dragleave', function()  {{ dz.classList.remove('drag-over'); }});
dz.addEventListener('drop', function(e) {{
  e.preventDefault();
  dz.classList.remove('drag-over');
  var files = e.dataTransfer.files;
  if (files.length) setFile(files[0]);
}});
document.getElementById('fileInput').addEventListener('change', function(e) {{
  if (e.target.files.length) setFile(e.target.files[0]);
}});
document.getElementById('removeFile').addEventListener('click', function(e) {{
  e.stopPropagation();
  resetWidget();
}});

function setFile(f) {{
  selectedFile = f;
  document.getElementById('fileName').textContent = f.name;
  document.getElementById('fileSize').textContent = (f.size / 1048576).toFixed(1) + ' MB';
  document.getElementById('filePreview').style.display = 'flex';
  document.getElementById('dropZone').style.display    = 'none';
  document.getElementById('analyzeBtn').disabled        = false;
  document.getElementById('errorMsg').textContent       = '';
}}

function resetWidget() {{
  selectedFile = null;
  document.getElementById('fileInput').value           = '';
  document.getElementById('filePreview').style.display = 'none';
  document.getElementById('dropZone').style.display    = '';
  document.getElementById('analyzeBtn').disabled        = true;
  document.getElementById('resultBox').style.display    = 'none';
  document.getElementById('progressWrap').style.display = 'none';
  document.getElementById('errorMsg').textContent        = '';
  document.getElementById('downloadBtn').style.display    = 'none';
  document.getElementById('downloadBtn').href              = '#';
  document.getElementById('copyLinkBtn').style.display    = 'none';
  document.getElementById('copyLinkBtn').textContent      = '🔗 Copy Certified Link';
  window._certLink = '';
  document.getElementById('analyzeBtn').textContent      = 'Verify Video';
  document.getElementById('analyzeBtn').disabled         = false;
  notifyResize();
}}

function setProgress(pct, label) {{
  document.getElementById('progressFill').style.width  = pct + '%';
  document.getElementById('progressLabel').textContent = label;
}}

function notifyResize() {{
  try {{ window.parent.postMessage({{ type: 'verifyd-resize', height: document.body.scrollHeight + 40 }}, '*'); }} catch(e) {{}}
}}

function startAnalysis() {{
  if (!selectedFile) return;
  document.getElementById('analyzeBtn').disabled        = true;
  document.getElementById('analyzeBtn').textContent     = 'Analyzing…';
  document.getElementById('progressWrap').style.display = 'block';
  document.getElementById('resultBox').style.display    = 'none';
  document.getElementById('errorMsg').textContent        = '';
  setProgress(10, 'Uploading video…');
  notifyResize();

  var fd = new FormData();
  fd.append('file',  selectedFile);
  fd.append('email', WIDGET_EMAIL);

  var xhr = new XMLHttpRequest();
  xhr.open('POST', BACKEND + '/widget-upload/?key=' + API_KEY);

  xhr.upload.onprogress = function(e) {{
    if (e.lengthComputable) {{
      var pct = Math.round((e.loaded / e.total) * 60);
      setProgress(10 + pct, 'Uploading… ' + Math.round(e.loaded/e.total*100) + '%');
    }}
  }};

  xhr.onload = function() {{
    setProgress(95, 'Finalizing result…');
    try {{
      var data = JSON.parse(xhr.responseText);
      if (xhr.status >= 400 || data.error) {{
        showError(data.error || 'Verification failed. Please try again.');
        return;
      }}
      showResult(data);
    }} catch(e) {{
      showError('Unexpected response. Please try again.');
    }}
  }};

  xhr.onerror = function() {{
    showError('Network error. Please check your connection and try again.');
  }};

  // Simulate analysis progress while waiting
  var pct = 70;
  var progressTimer = setInterval(function() {{
    if (pct < 92) {{ pct += 2; setProgress(pct, 'Running AI analysis…'); }}
    else clearInterval(progressTimer);
  }}, 1200);

  xhr.send(fd);
}}

function showResult(data) {{
  document.getElementById('progressWrap').style.display = 'none';
  var box = document.getElementById('resultBox');
  var color = (data.color || 'blue').toLowerCase();
  box.className = 'result-box ' + color;
  document.getElementById('resultLabel').className = 'result-label ' + color;
  document.getElementById('resultLabel').textContent = data.status || 'RESULT';
  document.getElementById('resultScore').textContent =
    'Authenticity Score: ' + (data.authenticity_score || 0) + ' / 100';
  var reasoning = data.gpt_reasoning || '';
  document.getElementById('resultReasoning').textContent = reasoning;
  document.getElementById('resultReasoning').style.display = reasoning ? '' : 'none';

  // Show download + copy link buttons if certified video is available
  var dlBtn   = document.getElementById('downloadBtn');
  var copyBtn = document.getElementById('copyLinkBtn');
  if (data.download_url) {{
    dlBtn.href = data.download_url;
    dlBtn.style.display = 'block';
    // Build the public certificate URL
    // Copy link = direct video stream so it plays inline on social media
    window._certLink = data.download_url || '';
    copyBtn.style.display = 'block';
    copyBtn.textContent = '🔗 Copy Certified Link';
  }} else {{
    dlBtn.style.display  = 'none';
    copyBtn.style.display = 'none';
  }}

  box.style.display = 'block';
  document.getElementById('analyzeBtn').textContent = 'Verify Video';
  document.getElementById('analyzeBtn').disabled    = false;
  notifyResize();
}}

function copyCertLink() {{
  var link = window._certLink || '';
  if (!link) return;
  if (navigator.clipboard && navigator.clipboard.writeText) {{
    navigator.clipboard.writeText(link).then(function() {{
      var btn = document.getElementById('copyLinkBtn');
      btn.textContent = '✓ Link Copied!';
      btn.style.color = '#22c55e';
      btn.style.borderColor = '#22c55e';
      setTimeout(function() {{
        btn.textContent = '🔗 Copy Certified Link';
        btn.style.color = '';
        btn.style.borderColor = '';
      }}, 2500);
    }});
  }} else {{
    // Fallback for older browsers
    var ta = document.createElement('textarea');
    ta.value = link;
    ta.style.position = 'fixed';
    ta.style.opacity = '0';
    document.body.appendChild(ta);
    ta.select();
    document.execCommand('copy');
    document.body.removeChild(ta);
    var btn = document.getElementById('copyLinkBtn');
    btn.textContent = '✓ Link Copied!';
    setTimeout(function() {{ btn.textContent = '🔗 Copy Certified Link'; }}, 2500);
  }}
}}

function showError(msg) {{
  document.getElementById('progressWrap').style.display = 'none';
  document.getElementById('errorMsg').textContent        = msg;
  document.getElementById('analyzeBtn').textContent      = 'Verify Video';
  document.getElementById('analyzeBtn').disabled          = false;
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

    # ── Save file to disk then enqueue ────────────────────────
    job_id   = str(uuid.uuid4())
    raw_path = f"{UPLOAD_DIR}/{job_id}_{file.filename}"

    with open(raw_path, "wb") as f:
        while chunk := await file.read(1024 * 1024):
            f.write(chunk)

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








