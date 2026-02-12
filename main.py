from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import os, uuid, subprocess, sqlite3
from datetime import datetime
import cv2
import numpy as np

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
# WATERMARK FUNCTION
# =========================================================

def stamp_video(input_path, output_path):

    watermark_text = "VeriFYD CERTIFIED"

    cmd = [
        "ffmpeg",
        "-y",
        "-i", input_path,
        "-vf",
        f"drawtext=text='{watermark_text}':"
        f"x=6:y=6:"
        f"fontsize=18:"
        f"fontcolor=white@0.9:"
        f"box=1:"
        f"boxcolor=black@0.35:"
        f"boxborderw=6",
        "-c:v","libx264",
        "-preset","fast",
        "-crf","23",
        "-c:a","aac",
        "-b:a","128k",
        output_path
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print("FFmpeg error:")
        print(result.stderr)
        raise Exception("Watermark failed")

# =========================================================
# UPLOAD VIDEO → APPLY WATERMARK
# =========================================================

@app.post("/upload/")
async def upload(file: UploadFile = File(...), email: str = Form(...)):

    cert_id = str(uuid.uuid4())

    raw_path = f"{UPLOAD_DIR}/{cert_id}_{file.filename}"

    with open(raw_path, "wb") as buffer:
        buffer.write(await file.read())

    certified_path = f"{CERT_DIR}/{cert_id}.mp4"

    # 🔒 APPLY WATERMARK
    stamp_video(raw_path, certified_path)

    return {
        "status": "CERTIFIED",
        "certificate_id": cert_id,
        "download_url": f"{BASE_URL}/download/{cert_id}"
    }

# =========================================================
# DOWNLOAD CERTIFIED VIDEO
# =========================================================

@app.get("/download/{cid}")
def download(cid: str):
    path = f"{CERT_DIR}/{cid}.mp4"
    return FileResponse(path, media_type="video/mp4")

# =========================================================
# AI LINK ANALYSIS (kept for your system)
# =========================================================

def analyze_video_ai(path):

    cap = cv2.VideoCapture(path)
    anomaly = 0
    frames_checked = 0

    while frames_checked < 200:
        ret, frame = cap.read()
        if not ret:
            break

        frames_checked += 1

        if frames_checked % 10 != 0:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if np.std(gray) < 8:
            anomaly += 2

        edges = cv2.Canny(gray,100,200)
        if np.mean(edges) < 3:
            anomaly += 1

    cap.release()

    if anomaly > 60:
        return "AI DETECTED", 92
    else:
        return "AUTHENTIC VERIFIED", 98

def download_temp_video(url, output):
    subprocess.run(["yt-dlp","-f","worst","-o",output,url], check=True)

@app.post("/analyze-link/")
async def analyze_link(email: str = Form(...), video_url: str = Form(...)):

    temp_id = str(uuid.uuid4())
    temp_path = f"{TMP_DIR}/{temp_id}.mp4"

    try:
        download_temp_video(video_url, temp_path)
    except:
        return {"result":"ERROR"}

    result, score = analyze_video_ai(temp_path)

    try:
        os.remove(temp_path)
    except:
        pass

    return {
        "result": result,
        "ai_score": score
    }

# =========================================================

@app.get("/")
def home():
    return {"status":"VeriFYD backend live"}






































