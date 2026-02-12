from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import os, uuid, shutil, subprocess, sqlite3, hashlib

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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def combined_score(local, external):
    return int((local * 0.4) + ((100 - external) * 0.6))

# ================= UPLOAD =================

@app.post("/upload/")
async def upload(file: UploadFile = File(...), email: str = Form(...)):

    cert_id = str(uuid.uuid4())
    raw_path = f"{UPLOAD_DIR}/{cert_id}_{file.filename}"

    with open(raw_path,"wb") as buffer:
        buffer.write(await file.read())

    local_score = detect_ai(raw_path)
    external_score = external_ai_score(raw_path)
    final_score = combined_score(local_score, external_score)

    if final_score < 98:
        return {
            "status": "OK",
            "success": True,
            "data": {
                "result": "AI DETECTED",
                "score": final_score
            }
        }

    certified_path = f"{CERT_DIR}/{cert_id}.mp4"
    shutil.copy(raw_path, certified_path)

    return {
        "status": "OK",
        "success": True,
        "data": {
            "result": "CERTIFIED REAL VIDEO",
            "score": final_score,
            "certificate_id": cert_id,
            "verify_url": f"{BASE_URL}/verify/{cert_id}",
            "download_url": f"{BASE_URL}/download/{cert_id}"
        }
    }

# ================= LINK =================

@app.post("/analyze-link/")
async def analyze_link(email: str = Form(...), video_url: str = Form(...)):

    try:
        temp_id = str(uuid.uuid4())
        temp_path = f"{TEMP_DIR}/{temp_id}.mp4"

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
                "status": "OK",
                "success": True,
                "data": {
                    "result": "ERROR",
                    "score": 0
                }
            }

        local_score = detect_ai(temp_path)
        external_score = external_ai_score(temp_path)
        final_score = combined_score(local_score, external_score)

        os.remove(temp_path)

        if final_score < 98:
            return {
                "status": "OK",
                "success": True,
                "data": {
                    "result": "AI DETECTED",
                    "score": final_score
                }
            }

        return {
            "status": "OK",
            "success": True,
            "data": {
                "result": "CERTIFIED REAL VIDEO",
                "score": final_score
            }
        }

    except Exception as e:
        return {
            "status": "OK",
            "success": True,
            "data": {
                "result": "ERROR",
                "score": 0
            }
        }

@app.get("/")
def home():
    return {"status": "VeriFYD API LIVE"}













































