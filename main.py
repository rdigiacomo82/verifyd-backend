import os
import uuid
import shutil
import subprocess
import requests
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse

from detector import detect_ai

app = FastAPI(title="VeriFYD 5.0")

BASE_URL = "https://verifyd-backend.onrender.com"

UPLOAD_DIR = "videos"
CERT_DIR = "certified"
TMP_DIR = "tmp"
ASSETS_DIR = "assets"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CERT_DIR, exist_ok=True)
os.makedirs(TMP_DIR, exist_ok=True)

LOGO_PATH = os.path.join(ASSETS_DIR, "logo.png")


# =========================================================
# HEALTH
# =========================================================
@app.get("/health")
def health():
    return {"status": "ok"}


# =========================================================
# STAMP VIDEO WITH LOGO + AUDIO PRESERVED
# =========================================================
def stamp_video(input_path, output_path, cert_id):

    temp_norm = os.path.join(TMP_DIR, f"{cert_id}_norm.mp4")

    # Normalize video first
    cmd_norm = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-vf", "scale=720:-2",
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-c:a", "aac",
        "-movflags", "+faststart",
        temp_norm
    ]
    subprocess.run(cmd_norm, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # If logo exists → overlay
    if os.path.exists(LOGO_PATH):
        cmd_stamp = [
            "ffmpeg", "-y",
            "-i", temp_norm,
            "-i", LOGO_PATH,
            "-filter_complex",
            "overlay=10:10",
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "23",
            "-c:a", "copy",
            "-movflags", "+faststart",
            output_path
        ]
    else:
        cmd_stamp = [
            "ffmpeg", "-y",
            "-i", temp_norm,
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "23",
            "-c:a", "copy",
            "-movflags", "+faststart",
            output_path
        ]

    result = subprocess.run(cmd_stamp, capture_output=True)

    if result.returncode != 0:
        raise RuntimeError("Video encode failed")

    os.remove(temp_norm)


# =========================================================
# UPLOAD VIDEO
# =========================================================
@app.post("/upload/")
async def upload(file: UploadFile = File(...), email: str = "guest"):

    cert_id = str(uuid.uuid4())
    raw_path = os.path.join(UPLOAD_DIR, f"{cert_id}_{file.filename}")

    with open(raw_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Run detection
    score = detect_ai(raw_path)

    # ================================
    # REAL VIDEO VERIFIED
    # ================================
    if score >= 70:

        certified_path = os.path.join(CERT_DIR, f"{cert_id}.mp4")

        try:
            stamp_video(raw_path, certified_path, cert_id)
        except Exception:
            return JSONResponse({
                "status": "PROCESSING ERROR",
                "authenticity_score": score,
                "color": "red"
            })

        return {
            "status": "REAL VIDEO VERIFIED",
            "authenticity_score": score,
            "certificate_id": cert_id,
            "download_url": f"{BASE_URL}/download/{cert_id}",
            "color": "green"
        }

    # ================================
    # UNDETERMINED
    # ================================
    elif score >= 50:
        return {
            "status": "UNDETERMINED",
            "authenticity_score": score,
            "color": "blue"
        }

    # ================================
    # AI DETECTED
    # ================================
    else:
        return {
            "status": "AI DETECTED",
            "authenticity_score": score,
            "color": "red"
        }


# =========================================================
# DOWNLOAD CERTIFIED VIDEO
# =========================================================
@app.get("/download/{cid}")
def download(cid: str):

    path = os.path.join(CERT_DIR, f"{cid}.mp4")

    if not os.path.exists(path):
        return JSONResponse({"error": "not found"})

    return FileResponse(
        path,
        media_type="video/mp4",
        filename="verified_video.mp4"
    )


# =========================================================
# ANALYZE VIDEO LINK
# =========================================================
@app.get("/analyze-link/")
def analyze_link(video_url: str, email: str = "guest"):

    if not video_url.startswith("http"):
        return {"error": "Invalid URL"}

    tmp_path = os.path.join(TMP_DIR, f"{uuid.uuid4()}.mp4")

    try:
        # Try direct download
        r = requests.get(video_url, stream=True, timeout=10)

        if r.status_code == 200 and "video" in r.headers.get("content-type", ""):
            with open(tmp_path, "wb") as f:
                for chunk in r.iter_content(8192):
                    f.write(chunk)

        else:
            # Use yt-dlp for social links
            subprocess.run([
                "yt-dlp",
                "-o", tmp_path,
                video_url
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        score = detect_ai(tmp_path)

    except Exception as e:
        return {"error": str(e)}

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    # ================================
    # RESULTS
    # ================================
    if score >= 70:
        return {
            "status": "REAL VIDEO VERIFIED",
            "authenticity_score": score,
            "color": "green"
        }

    elif score >= 50:
        return {
            "status": "UNDETERMINED",
            "authenticity_score": score,
            "color": "blue"
        }

    else:
        return {
            "status": "AI DETECTED",
            "authenticity_score": score,
            "color": "red"
        }








