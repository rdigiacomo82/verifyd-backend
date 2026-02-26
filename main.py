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
from config import (                  # single source of truth for all settings
    BASE_URL,
    UPLOAD_DIR, CERT_DIR, TMP_DIR,
)
from emailer  import send_otp_email
from database import (init_db, insert_certificate, increment_downloads,
                      get_or_create_user, get_user_status, increment_user_uses,
                      is_valid_email, FREE_USES, get_certificate,
                      is_email_verified, create_otp, verify_otp)
from video import clip_first_6_seconds, stamp_video, download_video_ytdlp

log = logging.getLogger("verifyd.main")

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
    Clip → detect → return (authenticity, label, detail, ui_text, color, certify).
    Cleans up the temp clip automatically.
    """
    clip_path = clip_first_6_seconds(source_path)
    try:
        authenticity, label, detail = run_detection(clip_path)
    finally:
        if os.path.exists(clip_path):
            os.remove(clip_path)

    ui_text, color, certify = LABEL_UI.get(label, ("UNKNOWN", "grey", False))
    log.info(
        "Analysis: label=%s authenticity=%d ai_score=%d",
        label, authenticity, detail["ai_score"],
    )
    return authenticity, label, detail, ui_text, color, certify


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

    cid      = str(uuid.uuid4())
    raw_path = f"{UPLOAD_DIR}/{cid}_{file.filename}"

    # Stream directly to disk — no memory limit regardless of file size
    with open(raw_path, "wb") as f:
        while chunk := await file.read(1024 * 1024):  # 1MB chunks
            f.write(chunk)

    sha256 = _sha256(raw_path)

    try:
        authenticity, label, detail, ui_text, color, certify = _run_analysis(raw_path)
    except Exception as e:
        log.exception("Detection failed for %s", raw_path)
        return JSONResponse({"error": str(e)}, status_code=500)

    # ── Count this use ────────────────────────────────────────
    increment_user_uses(email)

    # Persist every result — certified or not
    insert_certificate(
        cert_id       = cid,
        email         = email,
        original_file = file.filename,
        label         = label,
        authenticity  = authenticity,
        ai_score      = detail["ai_score"],
        sha256        = sha256,
    )

    if certify:
        certified_path = f"{CERT_DIR}/{cid}.mp4"
        try:
            stamp_video(raw_path, certified_path, cid)
        except RuntimeError as e:
            log.error("Stamping failed: %s", e)
            return JSONResponse({"error": "Video certification failed"}, status_code=500)

        return {
            "status":             ui_text,
            "authenticity_score": authenticity,
            "certificate_id":     cid,
            "download_url":       f"{BASE_URL}/download/{cid}",
            "color":              color,
            "gpt_reasoning":      detail.get("gpt_reasoning", ""),
            "gpt_flags":          detail.get("gpt_flags", []),
            "signal_score":       detail.get("signal_ai_score", 0),
            "gpt_score":          detail.get("gpt_ai_score", 0),
        }

    return {
        "status":             ui_text,
        "authenticity_score": authenticity,
        "color":              color,
        "gpt_reasoning":      detail.get("gpt_reasoning", ""),
        "gpt_flags":          detail.get("gpt_flags", []),
        "signal_score":       detail.get("signal_ai_score", 0),
        "gpt_score":          detail.get("gpt_ai_score", 0),
    }


@app.get("/download/{cid}")
def download(cid: str):
    path = f"{CERT_DIR}/{cid}.mp4"
    if not os.path.exists(path):
        return JSONResponse({"error": "Certificate not found"}, status_code=404)
    increment_downloads(cid)
    return FileResponse(path, media_type="video/mp4")


@app.get("/certificate/{cid}")
def certificate_lookup(cid: str):
    """Public certificate verification endpoint."""
    cert = get_certificate(cid)
    if not cert:
        return JSONResponse({"error": "Certificate not found"}, status_code=404)
    return {
        "certificate_id":  cert["cert_id"],
        "label":           cert["label"],
        "authenticity":    cert["authenticity"],
        "ai_score":        cert["ai_score"],
        "original_file":   cert["original_file"],
        "upload_time":     cert["upload_time"],
        "download_count":  cert["download_count"],
        "verified_by":     "VeriFYD",
    }


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

        authenticity, label, detail, ui_text, color, _ = _run_analysis(tmp_path)

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
    if key not in ("Honda#6915", "Honda6915", "admin2026"):
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
    if key not in ("Honda#6915", "Honda6915", "admin2026"):
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
    if key not in ("Honda#6915", "Honda6915", "admin2026"):
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
    if key not in ("Honda#6915", "Honda6915", "admin2026"):
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





