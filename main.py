import os
import uuid
import shutil
import subprocess
import tempfile
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from detector import detect_ai
from external_detector import external_ai_score

BASE = os.getcwd()
VIDEOS = os.path.join(BASE, "videos")
CERT = os.path.join(BASE, "certified")
ASSETS = os.path.join(BASE, "assets")

os.makedirs(VIDEOS, exist_ok=True)
os.makedirs(CERT, exist_ok=True)

LOGO = os.path.join(ASSETS, "logo.png")

app = FastAPI(title="VeriFYD 4.2")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# UTIL: SCORE → STATUS
# -----------------------------
def score_to_status(score: int):
    if score >= 70:
        return "REAL VIDEO VERIFIED", "green"
    if score >= 50:
        return "FURTHER REVIEW NEEDED", "blue"
    return "AI DETECTED", "red"


# -----------------------------
# STAMP VIDEO WITH LOGO + AUDIO
# -----------------------------
def stamp_video(input_path, output_path, cert_id):
    """
    Adds logo overlay AND keeps audio untouched
    """

    if not os.path.exists(LOGO):
        # fallback if logo missing
        shutil.copy(input_path, output_path)
        return

    cmd = [
        "ffmpeg",
        "-y",
        "-i", input_path,
        "-i", LOGO,
        "-filter_complex",
        "overlay=W-w-20:20",
        "-c:v", "libx264",
        "-crf", "23",
        "-preset", "veryfast",
        "-c:a", "copy",
        output_path
    ]

    r = subprocess.run(cmd, capture_output=True, text=True)

    if r.returncode != 0:
        raise RuntimeError(r.stderr[-400:])


# -----------------------------
# HOME
# -----------------------------
@app.get("/")
def home():
    return {"status": "VeriFYD running"}


@app.get("/health")
def health():
    return {"ok": True}


# -----------------------------
# UPLOAD VIDEO
# -----------------------------
@app.post("/upload/")
async def upload(file: UploadFile = File(...), email: str = Form(...)):
    cid = str(uuid.uuid4())

    raw_path = os.path.join(VIDEOS, f"{cid}_{file.filename}")
    with open(raw_path, "wb") as f:
        f.write(await file.read())

    # --------- AI DETECTION ----------
    primary = detect_ai(raw_path)
    secondary = external_ai_score(raw_path)

    authenticity = int((100 - primary) * 0.6 + (100 - secondary) * 0.4)

    status, color = score_to_status(authenticity)

    # --------- CERTIFY IF REAL ----------
    if authenticity >= 70:
        certified_path = os.path.join(CERT, f"{cid}.mp4")
        stamp_video(raw_path, certified_path, cid)

        return {
            "status": status,
            "authenticity_score": authenticity,
            "certificate_id": cid,
            "download_url": f"https://verifyd-backend.onrender.com/download/{cid}",
            "verify_url": f"https://verifyd-backend.onrender.com/verify/{cid}",
            "color": color
        }

    return {
        "status": status,
        "authenticity_score": authenticity,
        "color": color
    }


# -----------------------------
# DOWNLOAD CERTIFIED VIDEO
# -----------------------------
@app.get("/download/{cid}")
def download(cid: str):
    path = os.path.join(CERT, f"{cid}.mp4")

    if not os.path.exists(path):
        return JSONResponse({"error": "not found"}, status_code=404)

    return FileResponse(path, media_type="video/mp4")


# -----------------------------
# VERIFY PAGE LINK
# -----------------------------
@app.get("/verify/{cid}")
def verify(cid: str):
    path = os.path.join(CERT, f"{cid}.mp4")

    if not os.path.exists(path):
        return {"status": "NOT FOUND"}

    return {
        "status": "VERIFIED BY VERIFYD",
        "certificate_id": cid,
        "verified": True,
        "timestamp": datetime.utcnow().isoformat()
    }


# -----------------------------
# ANALYZE SOCIAL LINK
# -----------------------------
@app.get("/analyze-link/")
def analyze_link(video_url: str):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp_path = tmp.name

        subprocess.run([
            "yt-dlp",
            "-o", tmp_path,
            "-f", "mp4",
            video_url
        ], capture_output=True)

        primary = detect_ai(tmp_path)
        secondary = external_ai_score(tmp_path)

        authenticity = int((100 - primary) * 0.6 + (100 - secondary) * 0.4)
        status, color = score_to_status(authenticity)

        return {
            "status": status,
            "authenticity_score": authenticity,
            "color": color
        }

    except Exception as e:
        return {"error": str(e)}








