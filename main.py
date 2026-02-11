from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import hashlib, os, uuid, subprocess, sqlite3
from datetime import datetime
import cv2
import numpy as np

BASE_URL = "https://verifyd-backend.onrender.com"
PRICING_URL = "https://vfvid.com/pricing"

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

# ================= DATABASE =================

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

# ================= DOWNLOAD =================

def download_video(url, output_path):
    cmd = ["yt-dlp","-f","worst","-o",output_path,url]
    subprocess.run(cmd, check=True)

# ================= LINK ANALYSIS =================

@app.post("/analyze-link/")
async def analyze_link(email: str = Form(...), video_url: str = Form(...)):

    user = get_user(email)

    if not user["is_paid"] and user["checks_used"] >= 10:
        return {
            "success": True,
            "data": {
                "result": "LIMIT_REACHED",
                "upgrade_url": PRICING_URL
            }
        }

    vid_id = str(uuid.uuid4())
    temp_file = f"{TMP_DIR}/{vid_id}.mp4"

    try:
        download_video(video_url, temp_file)
    except:
        return {"success": True, "data": {"result":"ERROR"}}

    max_seconds = 300 if user["is_paid"] else 60
    result, score = analyze_video_ai(temp_file, max_seconds)

    try:
        os.remove(temp_file)
    except:
        pass

    increment_usage(email)

    return {
        "success": True,
        "data": {
            "result": result,
            "ai_score": score
        }
    }

@app.get("/")
def home():
    return {"status":"live"}






































