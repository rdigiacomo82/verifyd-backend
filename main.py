# ============================================================
#  VeriFYD – main.py  (v2.0.0)
# ============================================================

import os
import uuid
import logging
import hashlib
import subprocess
import json
from datetime import datetime, timezone

import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

# -- Real detectors --------------------------------------------------
from detector import detect_ai
from external_detector import external_ai_score
import database as db

# ============================================================
#  CONFIG (env-driven, with sane defaults)
# ============================================================
BASE_URL    = os.getenv("BASE_URL", "https://verifyd-backend.onrender.com")
UPLOAD_DIR  = os.getenv("UPLOAD_DIR", "videos")
CERT_DIR    = os.getenv("CERT_DIR", "certified")
REVIEW_DIR  = os.getenv("REVIEW_DIR", "review")
TMP_DIR     = os.getenv("TMP_DIR", "tmp")
FFMPEG_BIN  = os.getenv("FFMPEG_BIN", "ffmpeg")
FFPROBE_BIN = os.getenv("FFPROBE_BIN", "ffprobe")

MAX_UPLOAD_MB    = int(os.getenv("MAX_UPLOAD_MB", "500"))
ALLOWED_EXTS     = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
REVIEW_THRESHOLD = (40, 70)  # scores in this range → manual review

for d in (UPLOAD_DIR, CERT_DIR, REVIEW_DIR, TMP_DIR):
    os.makedirs(d, exist_ok=True)

# ============================================================
#  LOGGING
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("verifyd")

log.info("🚀 VeriFYD backend starting")

# Init database on startup
db.init_db()

# ============================================================
#  APP
# ============================================================
app = FastAPI(title="VeriFYD", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
#  HELPERS
# ============================================================

def ffprobe_streams(path: str) -> dict:
    """Return ffprobe JSON for a file. Raises on failure."""
    cmd = [
        FFPROBE_BIN, "-v", "quiet",
        "-print_format", "json",
        "-show_streams",
        path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr[:300]}")
    return json.loads(result.stdout)


def has_audio(probe: dict) -> bool:
    return any(s.get("codec_type") == "audio" for s in probe.get("streams", []))


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ============================================================
#  DETECTION  (combines both detectors)
# ============================================================

def analyze_video(path: str) -> dict:
    """
    Run real detection pipeline.
    Returns dict with individual + combined scores.
    """
    log.info("🔍 Running AI detection on %s", path)

    # Primary detector (ResNet18 features + noise + motion)
    try:
        primary_score = detect_ai(path)
    except Exception as e:
        log.error("Primary detector failed: %s", e)
        primary_score = 50  # uncertain fallback

    # Secondary detector (noise heuristic)
    try:
        secondary_score = external_ai_score(path)
    except Exception as e:
        log.error("Secondary detector failed: %s", e)
        secondary_score = 50

    # Weighted combination: primary is the stronger signal
    # Both return 0-100 where HIGH = likely AI, LOW = likely real
    # We invert to "authenticity" where HIGH = likely real
    combined_ai = (primary_score * 0.7) + (secondary_score * 0.3)
    authenticity = max(0, min(100, 100 - int(combined_ai)))

    result = {
        "authenticity_score": authenticity,
        "primary_ai_score": primary_score,
        "secondary_ai_score": secondary_score,
        "combined_ai_likelihood": int(combined_ai),
    }

    log.info("🔍 Detection result: %s", result)
    return result


# ============================================================
#  STAMPING (audio-safe, with error checking)
# ============================================================

def stamp_video(input_path: str, output_path: str, cert_id: str):
    """
    Two-pass stamp: remux (preserve streams) → watermark + re-encode.
    Raises on FFmpeg failure. Cleans up temp files.
    """
    log.info("🎬 Stamping %s → %s", input_path, output_path)

    temp_mux = os.path.join(TMP_DIR, f"{cert_id}_mux.mp4")

    try:
        # --- Probe input ---
        probe = ffprobe_streams(input_path)
        audio_present = has_audio(probe)
        log.info("   Audio detected: %s", audio_present)

        # --- STEP 1: Remux (copy all streams) ---
        remux_cmd = [
            FFMPEG_BIN, "-y",
            "-i", input_path,
            "-map", "0",
            "-c", "copy",
            "-movflags", "+faststart",
            temp_mux,
        ]
        _run_ffmpeg(remux_cmd, "remux")

        # --- STEP 2: Watermark + re-encode ---
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        watermark = (
            f"drawtext=text='VeriFYD  |  {cert_id[:8]}  |  {timestamp}':"
            "x=w-tw-20:y=h-th-20:"
            "fontsize=16:"
            "fontcolor=white@0.85:"
            "box=1:boxcolor=black@0.4:boxborderw=4"
        )

        final_cmd = [
            FFMPEG_BIN, "-y",
            "-i", temp_mux,
            "-vf", watermark,
            "-map", "0:v",
        ]

        if audio_present:
            final_cmd += [
                "-map", "0:a",
                "-c:a", "aac",
                "-b:a", "192k",
                "-af", "aresample=async=1",
            ]

        final_cmd += [
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "23",
            "-shortest",
            "-movflags", "+faststart",
            output_path,
        ]
        _run_ffmpeg(final_cmd, "watermark")

        log.info("🎬 Stamp complete: %s", output_path)

    finally:
        # Clean up temp file
        if os.path.exists(temp_mux):
            os.remove(temp_mux)
            log.info("   Cleaned temp: %s", temp_mux)


def _run_ffmpeg(cmd: list, label: str):
    """Run an FFmpeg command with timeout and error checking."""
    log.info("   FFmpeg [%s]: %s", label, " ".join(cmd))
    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=300
    )
    if result.returncode != 0:
        log.error("FFmpeg [%s] failed (rc=%d): %s", label, result.returncode, result.stderr[-500:])
        raise RuntimeError(f"FFmpeg {label} failed: {result.stderr[-200:]}")


# ============================================================
#  ROUTES
# ============================================================

@app.get("/")
def home():
    return {"status": "VeriFYD LIVE", "version": "2.0.0"}


@app.get("/health")
def health():
    """Confirm API + FFmpeg are operational."""
    try:
        subprocess.run([FFMPEG_BIN, "-version"], capture_output=True, timeout=5, check=True)
        ffmpeg_ok = True
    except Exception:
        ffmpeg_ok = False
    return {"api": True, "ffmpeg": ffmpeg_ok}


@app.post("/upload/")
async def upload(file: UploadFile = File(...), email: str = Form(...)):
    log.info("📤 Upload from %s — file: %s", email, file.filename)

    # --- Validate extension ---
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED_EXTS:
        raise HTTPException(400, f"Unsupported file type: {ext}. Accepted: {ALLOWED_EXTS}")

    # --- Save upload ---
    cert_id = str(uuid.uuid4())
    raw_path = os.path.join(UPLOAD_DIR, f"{cert_id}_{file.filename}")

    try:
        contents = await file.read()

        # --- Validate size ---
        if len(contents) > MAX_UPLOAD_MB * 1024 * 1024:
            raise HTTPException(413, f"File too large. Max: {MAX_UPLOAD_MB} MB")

        with open(raw_path, "wb") as f:
            f.write(contents)
        log.info("   Saved: %s (%d MB)", raw_path, len(contents) // (1024 * 1024))
    except HTTPException:
        raise
    except Exception as e:
        log.error("Failed to save upload: %s", e)
        raise HTTPException(500, "Failed to save uploaded file")

    # --- Validate it's actually a video ---
    try:
        probe = ffprobe_streams(raw_path)
        video_streams = [s for s in probe.get("streams", []) if s.get("codec_type") == "video"]
        if not video_streams:
            raise HTTPException(400, "Uploaded file contains no video stream")
    except HTTPException:
        raise
    except Exception as e:
        log.error("ffprobe validation failed: %s", e)
        raise HTTPException(400, "Could not read video file — may be corrupt")

    # --- Run real detection ---
    detection = analyze_video(raw_path)
    authenticity = detection["authenticity_score"]

    # --- Route: certify vs review ---
    low, high = REVIEW_THRESHOLD
    if authenticity >= high:
        status = "CERTIFIED"
        certified_path = os.path.join(CERT_DIR, f"{cert_id}_VeriFYD.mp4")
        try:
            stamp_video(raw_path, certified_path, cert_id)
        except Exception as e:
            log.error("Stamping failed: %s", e)
            raise HTTPException(500, "Video certification failed during processing")
        download_url = f"{BASE_URL}/download/{cert_id}"
    elif authenticity >= low:
        status = "UNDER_REVIEW"
        # Copy to review folder, no stamp yet
        review_path = os.path.join(REVIEW_DIR, f"{cert_id}_{file.filename}")
        os.rename(raw_path, review_path)
        certified_path = None
        download_url = None
        log.info("⚠️ Video sent to review (score %d)", authenticity)
    else:
        status = "REJECTED"
        certified_path = None
        download_url = None
        log.warning("🚫 Video rejected (score %d)", authenticity)

    # --- Hash for verification ---
    sha256 = file_sha256(certified_path) if certified_path and os.path.exists(certified_path) else None

    # --- Store in database ---
    try:
        db.insert_certificate(
            cert_id=cert_id,
            email=email,
            original_file=file.filename,
            status=status,
            authenticity=authenticity,
            ai_likelihood=detection["combined_ai_likelihood"],
            primary_score=detection["primary_ai_score"],
            secondary_score=detection["secondary_ai_score"],
            sha256=sha256,
        )
    except Exception as e:
        log.error("DB insert failed (non-fatal): %s", e)

    log.info("📤 Upload complete — cert_id=%s status=%s score=%d", cert_id, status, authenticity)

    return {
        "status": status,
        "certificate_id": cert_id,
        "authenticity_score": authenticity,
        "ai_likelihood": detection["combined_ai_likelihood"],
        "detection_details": {
            "primary": detection["primary_ai_score"],
            "secondary": detection["secondary_ai_score"],
        },
        "sha256": sha256,
        "download_url": download_url,
    }


@app.get("/download/{cid}")
def download(cid: str):
    # Look for certified file (pattern: {cid}_VeriFYD.mp4 or {cid}.mp4)
    for name in (f"{cid}_VeriFYD.mp4", f"{cid}.mp4"):
        path = os.path.join(CERT_DIR, name)
        if os.path.exists(path):
            log.info("📥 Download: %s", path)
            db.increment_downloads(cid)
            return FileResponse(path, media_type="video/mp4", filename=f"VeriFYD_{cid[:8]}.mp4")

    log.warning("📥 Download not found: %s", cid)
    raise HTTPException(404, "Certificate not found")


@app.get("/status/{cid}")
def certificate_status(cid: str):
    """Check certificate status — DB first, then filesystem fallback."""
    # Try database
    record = db.get_certificate(cid)
    if record:
        return {
            "certificate_id": cid,
            "status": record["status"],
            "authenticity_score": record["authenticity"],
            "ai_likelihood": record["ai_likelihood"],
            "sha256": record["sha256"],
            "upload_time": record["upload_time"],
            "download_count": record["download_count"],
        }

    # Filesystem fallback (for pre-DB certificates)
    for name in (f"{cid}_VeriFYD.mp4", f"{cid}.mp4"):
        path = os.path.join(CERT_DIR, name)
        if os.path.exists(path):
            return {
                "certificate_id": cid,
                "status": "CERTIFIED",
                "sha256": file_sha256(path),
            }

    for f in os.listdir(REVIEW_DIR):
        if f.startswith(cid):
            return {"certificate_id": cid, "status": "UNDER_REVIEW"}

    raise HTTPException(404, "Certificate not found")


@app.get("/verify/{cid}")
def verify_certificate(cid: str):
    """Public verification — anyone can check if a cert is legit and hash matches."""
    for name in (f"{cid}_VeriFYD.mp4", f"{cid}.mp4"):
        path = os.path.join(CERT_DIR, name)
        if os.path.exists(path):
            current_hash = file_sha256(path)
            record = db.get_certificate(cid)
            stored_hash = record["sha256"] if record else None

            return {
                "certificate_id": cid,
                "verified": True,
                "hash_match": current_hash == stored_hash if stored_hash else None,
                "sha256": current_hash,
                "stored_sha256": stored_hash,
            }

    raise HTTPException(404, "Certificate not found")
