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
from database import init_db, insert_certificate, increment_downloads
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
    return {
        "raw_length":        len(raw),
        "masked":            masked,
        "has_at_sign":       "@" in raw,
        "has_plus":          "+" in raw,
        "has_encoded_plus":  "%2B" in raw,
        "starts_with_http":  raw.startswith("http"),
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
#  Routes
# ─────────────────────────────────────────────
@app.post("/upload/")
async def upload(file: UploadFile = File(...), email: str = Form(...)):
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
        }

    return {
        "status":             ui_text,
        "authenticity_score": authenticity,
        "color":              color,
    }


@app.get("/download/{cid}")
def download(cid: str):
    path = f"{CERT_DIR}/{cid}.mp4"
    if not os.path.exists(path):
        return JSONResponse({"error": "Certificate not found"}, status_code=404)
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

PROXY_REQUIRED_DOMAINS = (
    "tiktok.com",
    "instagram.com",
)

YTDLP_DOMAINS = (
    "youtube.com", "youtu.be",
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


@app.get("/analyze-link/", response_class=HTMLResponse)
def analyze_link(video_url: str):
    if not video_url.startswith("http"):
        return HTMLResponse(_error_html("Invalid URL — must start with http."), status_code=400)

    # ── TikTok / Instagram: proxy required ───────────────────────────────────
    # Render's datacenter IP is blocked by TikTok/Instagram CDN.
    # Show a clean holding page until RESIDENTIAL_PROXY_URL is configured.
    if any(d in video_url for d in PROXY_REQUIRED_DOMAINS):
        if not os.environ.get("RESIDENTIAL_PROXY_URL", "").strip():
            return HTMLResponse(_proxy_coming_soon_html(video_url), status_code=200)
        # Proxy configured — fall through to yt-dlp

    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tmp_path = tmp_file.name
    tmp_file.close()

    try:
        use_ytdlp = (
            any(d in video_url for d in PROXY_REQUIRED_DOMAINS) or
            any(d in video_url for d in YTDLP_DOMAINS)
        )

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
