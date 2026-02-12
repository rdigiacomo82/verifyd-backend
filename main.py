from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import os, uuid, subprocess

BASE_URL = "https://verifyd-backend.onrender.com"

UPLOAD_DIR = "videos"
CERT_DIR = "certified"
TMP_DIR = "tmp"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CERT_DIR, exist_ok=True)
os.makedirs(TMP_DIR, exist_ok=True)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================================
# HOME
# =========================================================

@app.get("/")
def home():
    return {"status":"VeriFYD backend live"}

# =========================================================
# UPLOAD VIDEO (NO AI DETECTION YET)
# =========================================================

@app.post("/upload/")
async def upload(file: UploadFile = File(...), email: str = Form(...)):

    cert_id = str(uuid.uuid4())

    raw_path = f"{UPLOAD_DIR}/{cert_id}_{file.filename}"

    with open(raw_path, "wb") as buffer:
        buffer.write(await file.read())

    certified_path = f"{CERT_DIR}/{cert_id}.mp4"

    # For now just copy file (Hostinger stamping temporary)
    subprocess.run(["cp", raw_path, certified_path])

    return {
        "status": "CERTIFIED",
        "certificate_id": cert_id,
        "download_url": f"{BASE_URL}/download/{cert_id}"
    }

# =========================================================
# DOWNLOAD
# =========================================================

@app.get("/download/{cid}")
def download(cid:str):
    path = f"{CERT_DIR}/{cid}.mp4"
    return FileResponse(path, media_type="video/mp4")

# =========================================================
# ANALYZE LINK (RESTORED)
# =========================================================

def download_temp_video(url, output):
    subprocess.run(["yt-dlp","-f","worst","-o",output,url])

@app.post("/analyze-link/")
async def analyze_link(email: str = Form(...), video_url: str = Form(...)):

    temp_id = str(uuid.uuid4())
    temp_path = f"{TMP_DIR}/{temp_id}.mp4"

    try:
        download_temp_video(video_url, temp_path)
    except:
        return {"result":"ERROR"}

    # For now: simple placeholder response
    # We'll replace with AI detection next
    return {
        "result": "AUTHENTIC VERIFIED",
        "ai_score": 0
    }







































