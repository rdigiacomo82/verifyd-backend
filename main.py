from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import os, uuid, subprocess, sqlite3
from datetime import datetime
import cv2
import numpy as np
import hashlib

BASE_URL = "https://verifyd-backend.onrender.com"

UPLOAD_DIR = "videos"
CERT_DIR = "certified"
TMP_DIR = "tmp"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CERT_DIR, exist_ok=True)
os.makedirs(TMP_DIR, exist_ok=True)

DB = "certificates.db"

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================= DB =================

def init_db():
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS users(
        email TEXT PRIMARY KEY,
        checks_used INTEGER DEFAULT 0,
        is_paid INTEGER DEFAULT 0,
        created TEXT
    )
    """)
    conn.commit()
    conn.close()

init_db()

def get_user(email):
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE email=?", (email,))
    row = c.fetchone()

    if not row:
        c.execute("INSERT INTO users VALUES (?,?,?,?)",
                  (email,0,0,datetime.utcnow().isoformat()))
        conn.commit()
        conn.close()
        return {"checks_used":0,"is_paid":0}

    conn.close()
    return {"checks_used":row[1],"is_paid":row[2]}

def increment_usage(email):
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute("UPDATE users SET checks_used = checks_used + 1 WHERE email=?", (email,))
    conn.commit()
    conn.close()

# ================= WATERMARK =================

def stamp_video(input_path, output_path):

    watermark = "VeriFYD CERTIFIED"

    command = [
        "ffmpeg","-y",
        "-i", input_path,
        "-vf",
        f"drawtext=text='{watermark}':"
        f"x=5:y=5:"
        f"fontsize=18:"
        f"fontcolor=white@0.85:"
        f"box=1:"
        f"boxcolor=black@0.3:"
        f"boxborderw=5",
        "-c:v","libx264",
        "-preset","fast",
        "-crf","23",
        "-c:a","aac",
        "-b:a","128k",
        output_path
    ]

    subprocess.run(command, check=True)

# ================= AI DETECTION =================

def analyze_video_ai(path, max_seconds):

    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    max_frames = int(fps * max_seconds)

    frame_count = 0
    anomaly = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count > max_frames:
            break

        if frame_count % 15 != 0:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if np.std(gray) < 8:
            anomaly += 2

        edges = cv2.Canny(gray,100,200)
        if np.mean(edges) < 3:
            anomaly += 1

    cap.release()

    score = min(100, anomaly)

    if score > 60:
        return "AI DETECTED", score
    else:
        return "AUTHENTIC VERIFIED", score

def download_video(url, output_path):
    cmd = ["yt-dlp","-f","worst","-o",output_path,url]
    subprocess.run(cmd, check=True)

# ================= LINK ANALYSIS =================

@app.post("/analyze-link/")
async def analyze_link(email: str = Form(...), video_url: str = Form(...)):

    user = get_user(email)

    if user["checks_used"] >= 10:
        return {"result":"LIMIT_REACHED"}

    vid_id = str(uuid.uuid4())
    temp_file = f"{TMP_DIR}/{vid_id}.mp4"

    try:
        download_video(video_url, temp_file)
    except:
        return {"result":"ERROR"}

    result, score = analyze_video_ai(temp_file, 60)

    os.remove(temp_file)
    increment_usage(email)

    return {
        "result": result,
        "ai_score": score
    }

# ================= UPLOAD CERTIFY =================

@app.post("/upload/")
async def upload(file: UploadFile = File(...), email: str = Form(...)):

    cert_id = str(uuid.uuid4())
    raw_path = f"{UPLOAD_DIR}/{cert_id}_{file.filename}"

    with open(raw_path, "wb") as buffer:
        buffer.write(await file.read())

    certified_name = f"{cert_id}.mp4"
    certified_path = f"{CERT_DIR}/{certified_name}"

    # 🔒 APPLY WATERMARK HERE
    stamp_video(raw_path, certified_path)

    increment_usage(email)

    return {
        "status":"CERTIFIED",
        "certificate_id":cert_id,
        "verify_url":f"{BASE_URL}/verify/{cert_id}",
        "download_url":f"{BASE_URL}/download/{cert_id}"
    }

@app.get("/download/{cid}")
def download(cid:str):
    path = f"{CERT_DIR}/{cid}.mp4"
    return FileResponse(path, media_type="video/mp4")

@app.get("/")
def home():
    return {"status":"VeriFYD backend live"}





































