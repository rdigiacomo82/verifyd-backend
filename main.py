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
from pydantic import BaseModel
import os, uuid, requests, tempfile, logging, hashlib

class EmailRequest(BaseModel):
    email: str
from contextlib import asynccontextmanager

from detection import run_detection   # returns (authenticity, label, detail)
from config import (                  # single source of truth for all settings
    BASE_URL,
    UPLOAD_DIR, CERT_DIR, TMP_DIR,
)
from emailer  import send_otp_email, send_certification_email, send_magic_link_email
from database import (init_db, insert_certificate, increment_downloads,
                      get_or_create_user, get_user_status, increment_user_uses,
                      is_valid_email, FREE_USES, get_certificate,
                      is_email_verified, create_otp, verify_otp,
                      create_magic_link, verify_magic_link,
                      create_session, get_session_email, delete_session,
                      clean_expired_sessions)
from video import clip_first_6_seconds, stamp_video, download_video_ytdlp
from queue_helper import enqueue_upload, enqueue_link, get_job_result

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
            "error":   "email_not_verified",
            "message": "Please verify your email address before uploading.",
        }, status_code=403)

    # ── Usage limit check ─────────────────────────────────────
    status = get_user_status(email)
    if not status["allowed"]:
        return JSONResponse({
            "error":     "limit_reached",
            "plan":      status["plan"],
            "uses_left": 0,
            "limit":     status["limit"],
        }, status_code=402)

    # ── Save file to disk then enqueue ────────────────────────
    job_id   = str(uuid.uuid4())
    raw_path = f"{UPLOAD_DIR}/{job_id}_{file.filename}"

    with open(raw_path, "wb") as f:
        while chunk := await file.read(1024 * 1024):
            f.write(chunk)

    enqueue_upload(job_id, raw_path, file.filename, email)
    log.info("upload: queued job %s for %s file=%s", job_id, email, file.filename)

    # ── Poll Redis until worker completes (transparent to frontend) ──
    # Frontend never sees job_id — gets same response format as before.
    import asyncio
    for _ in range(200):   # poll up to 200 times = ~10 minutes max
        await asyncio.sleep(3)
        result = get_job_result(job_id)
        if result and result.get("job_status") == "complete":
            # Strip internal job_status field before returning
            result.pop("job_status", None)
            result.pop("label", None)
            return JSONResponse(result)
        if result and result.get("job_status") == "error":
            return JSONResponse(
                {"error": result.get("error", "Analysis failed.")},
                status_code=500
            )

    return JSONResponse({"error": "Analysis timed out. Please try again."}, status_code=504)


@app.get("/job-status/{job_id}")
def job_status(job_id: str):
    """Poll endpoint for frontends that handle async directly."""
    result = get_job_result(job_id)
    if not result or result.get("job_status") == "not_found":
        return JSONResponse({"job_status": "not_found"}, status_code=404)
    return JSONResponse(result)


@app.get("/download/{cid}")
def download(cid: str):
    # First check local disk (legacy / direct web service stamping)
    path = f"{CERT_DIR}/{cid}.mp4"
    if os.path.exists(path):
        increment_downloads(cid)
        return FileResponse(path, media_type="video/mp4")

    # Check Redis — worker stores certified videos here
    try:
        from queue_helper import get_redis
        r = get_redis()
        video_bytes = r.get(f"certified:{cid}")
        if video_bytes:
            increment_downloads(cid)
            # Cache locally for faster subsequent downloads
            with open(path, "wb") as f:
                f.write(video_bytes)
            return FileResponse(path, media_type="video/mp4")
    except Exception as e:
        log.warning("download: Redis check failed for %s: %s", cid, e)

    return JSONResponse({"error": "Certificate not found"}, status_code=404)


@app.get("/certificate/{cid}")
def certificate_lookup(cid: str):
    """Public certificate verification endpoint."""
    cert = get_certificate(cid)
    if not cert:
        return JSONResponse({"error": "Certificate not found"}, status_code=404)
    video_available = os.path.exists(f"{CERT_DIR}/{cid}.mp4")
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

    # Serve file
    path = f"{CERT_DIR}/{cid}.mp4"
    if not os.path.exists(path):
        return JSONResponse({"error": "Video no longer available — please re-verify"}, status_code=404)

    increment_downloads(cid)
    return FileResponse(path, media_type="video/mp4")


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
#  Session management (persistent — database backed)
# ─────────────────────────────────────────────
import time
from fastapi.responses import JSONResponse


def _set_session_cookie(response, token: str):
    """Helper to set the vfy_session cookie consistently."""
    response.set_cookie(
        key="vfy_session",
        value=token,
        max_age=2592000,  # 30 days
        samesite="none",
        secure=True,
        httponly=False,
    )


def _get_session_from_request(request: Request) -> str | None:
    """Extract and validate session token from request cookie."""
    token = request.cookies.get("vfy_session", "")
    if not token:
        return None
    return get_session_email(token)


@app.post("/register-email/")
async def register_email(request: Request):
    """
    Called by the frontend when user enters their email.
    Creates a persistent database session.
    """
    try:
        body = await request.json()
        email = body.get("email", "").strip()
    except Exception:
        return JSONResponse({"error": "invalid body"}, status_code=400)

    if not is_valid_email(email):
        return JSONResponse({"error": "invalid email"}, status_code=400)

    get_or_create_user(email)
    token = create_session(email)
    log.info("register-email: %s → session %s", email, token[:8])

    response = JSONResponse({"status": "ok", "token": token})
    _set_session_cookie(response, token)
    return response


# ─────────────────────────────────────────────
#  Magic Link endpoints
# ─────────────────────────────────────────────
BASE_URL = os.environ.get("BASE_URL", "https://verifyd-backend.onrender.com")

@app.post("/send-magic-link/")
async def send_magic_link(body: EmailRequest):
    """
    Send a magic link login email to the given address.
    User clicks the link → authenticated for 30 days.
    """
    email = body.email.strip()

    if not is_valid_email(email):
        return JSONResponse({"error": "invalid email"}, status_code=400)

    get_or_create_user(email)
    token = create_magic_link(email)
    url   = f"{BASE_URL}/auth/{token}"
    sent  = send_magic_link_email(email, url)

    if not sent:
        log.error("send-magic-link: email failed for %s", email)
        return JSONResponse({"error": "Failed to send email. Please try again."}, status_code=500)

    log.info("send-magic-link: sent to %s", email)
    return JSONResponse({"status": "ok", "message": "Check your email for a sign-in link."})


@app.get("/auth/{token}")
async def magic_link_auth(token: str, request: Request):
    """
    Magic link callback — user clicks link from email.
    Verifies token, creates session, redirects to app.
    """
    email, success, message = verify_magic_link(token)

    if not success:
        # Redirect to app with error
        return HTMLResponse(f"""
        <html><head>
        <meta http-equiv="refresh" content="3;url=https://vfvid.com/app">
        </head><body style="background:#0a0a0a;color:#fff;font-family:sans-serif;
                             display:flex;align-items:center;justify-content:center;
                             height:100vh;margin:0;">
        <div style="text-align:center;">
            <p style="font-size:48px;margin:0 0 16px;">⚠️</p>
            <h2 style="color:#f59e0b;">Link expired or already used</h2>
            <p style="color:#888;">{message}</p>
            <p style="color:#555;font-size:13px;">Redirecting you back...</p>
        </div>
        </body></html>
        """, status_code=400)

    # Create persistent session
    session_token = create_session(email)
    log.info("magic-link auth: session created for %s", email)

    # Redirect to app with session cookie set
    response = HTMLResponse(f"""
    <html><head>
    <meta http-equiv="refresh" content="1;url=https://vfvid.com/app">
    </head><body style="background:#0a0a0a;color:#fff;font-family:sans-serif;
                         display:flex;align-items:center;justify-content:center;
                         height:100vh;margin:0;">
    <div style="text-align:center;">
        <p style="font-size:48px;margin:0 0 16px;">✅</p>
        <h2 style="color:#f59e0b;">You're signed in!</h2>
        <p style="color:#888;">Redirecting you to VeriFYD...</p>
    </div>
    </body></html>
    """)
    _set_session_cookie(response, session_token)
    return response


@app.post("/logout/")
async def logout(request: Request):
    """Clear the session cookie and delete from database."""
    token = request.cookies.get("vfy_session", "")
    if token:
        delete_session(token)
    response = JSONResponse({"status": "ok"})
    response.delete_cookie("vfy_session")
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
def analyze_link(request: Request, video_url: str, email: str = "",
                 double_count: bool = False):
    if not video_url.startswith("http"):
        return HTMLResponse(_error_html("Invalid URL — must start with http."), status_code=400)

    # ── Email resolution ──────────────────────────────────────
    if not email or not is_valid_email(email):
        session_email = _get_session_from_request(request)
        if session_email:
            email = session_email
            log.info("analyze-link: using email from session: %s", email)
        else:
            email = "anonymous@verifyd.com"
            log.info("analyze-link: no session found, using anonymous")

    # ── Usage limit check ────────────────────────────────────
    if email != "anonymous@verifyd.com":
        status    = get_user_status(email)
        required  = 2 if double_count else 1
        uses_left = status.get("uses_left", 0)
        if not status["allowed"] or uses_left < required:
            if double_count and uses_left < 2:
                return HTMLResponse(_error_html(
                    "You need at least 2 analyses remaining to scan a link "
                    "without downloading your certified video first."
                ), status_code=402)
            return HTMLResponse(
                _payment_required_html(email, status["plan"]),
                status_code=402
            )

    # ── Enqueue job — return polling page immediately ─────────
    job_id = str(uuid.uuid4())
    enqueue_link(job_id, video_url, email, double_count)
    log.info("analyze-link: queued job %s for %s url=%s", job_id, email, video_url)

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
      <meta charset="utf-8">
      <title>VeriFYD — Analyzing...</title>
      <style>
        body {{ background:#0a0a0a; color:white; text-align:center;
                padding-top:120px; font-family:Arial,sans-serif; margin:0; }}
        .spinner {{ width:64px; height:64px; border:6px solid #333;
                    border-top:6px solid #f59e0b; border-radius:50%;
                    animation:spin 1s linear infinite; margin:0 auto 30px; }}
        @keyframes spin {{ to {{ transform:rotate(360deg); }} }}
        h2 {{ color:#f59e0b; }}
        #status {{ color:#aaa; font-size:15px; margin-top:10px; }}
      </style>
    </head>
    <body>
      <div id="loading">
        <div class="spinner"></div>
        <h2>Analyzing your video...</h2>
        <p id="status">Thank you for your patience, this may take up to 1 minute.</p>
      </div>
      <div id="result" style="display:none"></div>
      <script>
        const jobId = "{job_id}";
        const messages = [
          "Thank you for your patience, this may take up to 1 minute.",
          "Running signal analysis...",
          "Running AI vision check...",
          "Calculating authenticity score..."
        ];
        let msgIdx = 0;
        const statusEl = document.getElementById("status");
        setInterval(() => {{
          msgIdx = (msgIdx + 1) % messages.length;
          statusEl.textContent = messages[msgIdx];
        }}, 8000);

        async function poll() {{
          try {{
            const resp = await fetch("/job-status/" + jobId);
            const data = await resp.json();
            if (data.job_status === "complete") {{ showResult(data); return; }}
            if (data.job_status === "error") {{ showError(data.error || "Analysis failed."); return; }}
            if (data.job_status === "queued") statusEl.textContent = "Position " + data.position + " in queue...";
          }} catch(e) {{}}
          setTimeout(poll, 3000);
        }}

        function showResult(data) {{
          document.getElementById("loading").style.display = "none";
          const colorMap = {{ green:"#22c55e", red:"#ef4444", blue:"#3b82f6" }};
          const color = colorMap[data.color] || "#f59e0b";
          let html = `<h1 style="color:${{color}};font-size:42px">${{data.status}}</h1>
            <h2>Authenticity Score: ${{data.authenticity_score}}/100</h2>
            <p style="color:#aaa">Signal: ${{data.signal_score}}/100 | GPT: ${{data.gpt_score}}/100</p>`;
          if (data.gpt_reasoning) html += `<p style="color:#ccc;max-width:560px;margin:20px auto;font-size:14px">${{data.gpt_reasoning}}</p>`;
          if (data.download_url) html += `<br><a href="${{data.download_url}}" style="background:#f59e0b;color:black;padding:14px 28px;border-radius:8px;text-decoration:none;font-weight:bold">⬇ Download Certified Video</a>`;
          html += `<p style="color:#555;margin-top:40px;font-size:13px">Analyzed by VeriFYD</p>`;
          document.getElementById("result").innerHTML = html;
          document.getElementById("result").style.display = "block";
        }}

        function showError(msg) {{
          document.getElementById("loading").style.display = "none";
          document.getElementById("result").innerHTML =
            `<h2 style="color:#ef4444">Analysis Failed</h2><p style="color:#aaa">${{msg}}</p>`;
          document.getElementById("result").style.display = "block";
        }}

        poll();
      </script>
    </body>
    </html>
    """
    return HTMLResponse(html)


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
        os.environ.get("PAYPAL_PLAN_ID",     ""): "creator",
        os.environ.get("PAYPAL_PLAN_ID_PRO", ""): "pro",
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

        elif event_type in ("BILLING.SUBSCRIPTION.CANCELLED", "BILLING.SUBSCRIPTION.EXPIRED"):
            if email:
                upgrade_user_plan(email, "free")
                log.info("Downgraded %s to free (sub cancelled/expired)", email)

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
    import sqlite3
    from datetime import datetime, timezone
    db_path = "/data/verifyd.db" if os.path.isdir("/data") else "verifyd.db"
    try:
        conn = sqlite3.connect(db_path)
        conn.execute(
            "UPDATE users SET period_uses=0, period_start=? WHERE email_lower=?",
            (datetime.now(timezone.utc).isoformat(), email.lower())
        )
        conn.commit()
        conn.close()
        return {"status": "reset", "email": email}
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








