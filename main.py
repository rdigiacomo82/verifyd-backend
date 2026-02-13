from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import os, uuid, shutil, subprocess

from detector import detect_ai
from external_detector import external_ai_score

BASE_URL = "https://verifyd-backend.onrender.com"

UPLOAD_DIR = "videos"
CERT_DIR = "certified"
TEMP_DIR = "temp"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CERT_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================
# UTILS
# =============================

def combined_score(local, external):
    try:
        return int((local * 0.4) + ((100 - external) * 0.6))
    except:
        return 50

# =============================
# HOME
# =============================

@app.get("/")
def home():
    return {"status": "VeriFYD API LIVE"}

# =============================
# UPLOAD VIDEO
# =============================

@app.post("/upload/")
async def upload(file: UploadFile = File(...), email: str = Form(...)):

    cert_id = str(uuid.uuid4())
    raw_path = f"{UPLOAD_DIR}/{cert_id}_{file.filename}"

    with open(raw_path, "wb") as buffer:
        buffer.write(await file.read())

    local_score = detect_ai(raw_path)
    external_score = external_ai_score(raw_path)
    final_score = combined_score(local_score, external_score)

    if final_score < 98:
        return {
            "success": True,
            "message": f"AI DETECTED — Authenticity Score: {final_score}"
        }

    certified_path = f"{CERT_DIR}/{cert_id}.mp4"
    shutil.copy(raw_path, certified_path)

    return {
        "success": True,
        "message": f"CERTIFIED REAL VIDEO — Authenticity Score: {final_score}",
        "certificate_id": cert_id,
        "verify_url": f"{BASE_URL}/verify/{cert_id}",
        "download_url": f"{BASE_URL}/download/{cert_id}"
    }

# =============================
# ANALYZE LINK
# =============================

@app.post("/analyze-link/")
async def analyze_link(email: str = Form(...), video_url: str = Form(...)):

    try:
        temp_id = str(uuid.uuid4())
        temp_path = f"{TEMP_DIR}/{temp_id}.mp4"

        # Download using yt-dlp
        cmd = [
            "yt-dlp",
            "-f", "mp4",
            "-o", temp_path,
            "--no-playlist",
            "--quiet",
            video_url
        ]

        subprocess.run(cmd, timeout=90)

        if not os.path.exists(temp_path):
            return {
                "success": True,
                "message": "Could not download video."
            }

        local_score = detect_ai(temp_path)
        external_score = external_ai_score(temp_path)
        final_score = combined_score(local_score, external_score)

        os.remove(temp_path)

        if final_score < 98:
            return {
                "success": True,
                "message": f"AI DETECTED — Authenticity Score: {final_score}"
            }

        return {
            "success": True,
            "message": f"CERTIFIED REAL VIDEO — Authenticity Score: {final_score}"
        }

    except Exception as e:
        return {
            "success": True,
            "message": "Error analyzing video."
        }

# =============================
# VERIFY
# =============================

@app.get("/verify/{cid}")
def verify(cid: str):
    path = f"{CERT_DIR}/{cid}.mp4"
    if not os.path.exists(path):
        return {"status": "not found"}
    return {"status": "valid", "certificate_id": cid}

# =============================
# DOWNLOAD
# =============================

@app.get("/download/{cid}")
def download(cid: str):
    path = f"{CERT_DIR}/{cid}.mp4"
    if not os.path.exists(path):
        return {"error": "not found"}
    return FileResponse(path, media_type="video/mp4")














































