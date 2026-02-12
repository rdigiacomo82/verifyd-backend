from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import os, uuid, hashlib, sqlite3, shutil, subprocess
from datetime import datetime

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

DB_FILE = "certificates.db"

def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS certificates(
        id TEXT PRIMARY KEY,
        filename TEXT,
        fingerprint TEXT,
        score INTEGER,
        status TEXT,
        created TEXT
    )
    """)
    conn.commit()
    conn.close()

init_db()

def fingerprint(path):
    h = hashlib.sha256()
    with open(path,"rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def combined_score(local_score, external_score):
    return int((local_score * 0.4) + ((100 - external_score) * 0.6))

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
            "success": True,
            "data": {
                "status": "AI DETECTED",
                "authenticity_score": final_score
            }
        }

    certified_path = f"{CERT_DIR}/{cert_id}.mp4"
    shutil.copy(raw_path, certified_path)

    return {
        "success": True,
        "data": {
            "status": "CERTIFIED REAL VIDEO",
            "certificate_id": cert_id,
            "authenticity_score": final_score,
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
        subprocess.run(cmd, timeout=60)

        if not os.path.exists(temp_path):
            return {
                "success": True,
                "data": {
                    "status": "ERROR",
                    "message": "Could not download video"
                }
            }

        local_score = detect_ai(temp_path)
        external_score = external_ai_score(temp_path)
        final_score = combined_score(local_score, external_score)

        os.remove(temp_path)

        if final_score < 98:
            return {
                "success": True,
                "data": {
                    "status": "AI DETECTED",
                    "authenticity_score": final_score
                }
            }

        return {
            "success": True,
            "data": {
                "status": "CERTIFIED REAL VIDEO",
                "authenticity_score": final_score
            }
        }

    except Exception as e:
        return {
            "success": True,
            "data": {
                "status": "ERROR",
                "message": str(e)
            }
        }

@app.get("/")
def home():
    return {"status": "VeriFYD API LIVE"}












































