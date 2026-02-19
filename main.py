# ============================================================
#  VeriFYD — main.py
#
#  FastAPI application entry point.
#
#  Detection pipeline:
#    upload / analyze-link
#      → clip_first_10_seconds()
#        → run_detection()          (detection.py)
#          → detect_ai()            (detector.py)
#            → authenticity, label, detail
#              → certify / respond
# ============================================================

from fastapi import FastAPI, UploadFile, File, Form
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
from video import clip_first_10_seconds, stamp_video

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
    clip_path = clip_first_10_seconds(source_path)
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

    with open(raw_path, "wb") as f:
        f.write(await file.read())

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


# Short-link redirects that never resolve to a raw video file.
# These are redirect-only URLs — not the platform domain itself.
SHORTLINK_PATTERNS = ("tiktok.com/t/", "vm.tiktok.com", "vt.tiktok.com")


@app.get("/analyze-link/", response_class=HTMLResponse)
def analyze_link(video_url: str):
    if not video_url.startswith("http"):
        return HTMLResponse(_error_html("Invalid URL — must start with http."), status_code=400)

    # Only block known short-redirect patterns, not entire platforms.
    # Full platform URLs (e.g. tiktok.com/@user/video/123) may still work
    # via yt-dlp in a future update.
    if any(p in video_url for p in SHORTLINK_PATTERNS):
        return HTMLResponse(_error_html(
            "Short share links (e.g. tiktok.com/t/...) cannot be downloaded directly. "
            "Try copying the full video URL from the browser address bar instead."
        ), status_code=400)

    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tmp_path = tmp_file.name
    tmp_file.close()

    try:
        r = requests.get(video_url, stream=True, timeout=30, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        })
        if r.status_code != 200:
            return HTMLResponse(_error_html(
                f"Could not download video (HTTP {r.status_code}). "
                "Make sure the URL points directly to an .mp4 file."
            ))

        # If the server returns HTML it means we got a webpage, not a video
        content_type = r.headers.get("content-type", "")
        if "text/html" in content_type:
            return HTMLResponse(_error_html(
                "The URL returned a webpage, not a video file. "
                "Please provide a direct link to an .mp4 or video file."
            ), status_code=400)

        with open(tmp_path, "wb") as f:
            for chunk in r.iter_content(8192):
                f.write(chunk)

        # Sanity-check file size
        if os.path.getsize(tmp_path) < 1024:
            return HTMLResponse(_error_html(
                "Downloaded file is too small to be a valid video."
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

    except ValueError as e:
        return HTMLResponse(_error_html(str(e)), status_code=400)

    except Exception as e:
        log.exception("analyze-link failed for %s", video_url)
        return HTMLResponse(_error_html(
            "Could not process this video. Please try a direct .mp4 URL."
        ), status_code=500)

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
