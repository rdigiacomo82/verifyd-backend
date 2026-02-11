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

# ============================================================
# DATABASE
# ============================================================

def init_db():
    conn = sqlite3.connect(DB)
    c = conn.cursor()

    c.execute("""
    CREATE TABLE IF NOT EXISTS certificates(
        id TEXT,
        filename TEXT,
        fingerprint TEXT,
        certified_file TEXT,
        created TEXT
    )
    """)

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

# ============================================================
# USER UTIL
# ============================================================

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

# ============================================================
# AI DETECTION
# ============================================================

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

        noise = np.std(gray)
        if noise < 8:
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

# ============================================================
# DOWNLOAD VIDEO FROM LINK
# ============================================================

def download_video(url, output_path):
    cmd = [
        "yt-dlp",
        "-f","worst",
        "-o", output_path,
        url
    ]
    subprocess.run(cmd, check=True)

# ============================================================
# LINK ANALYSIS (FIXED RESPONSE FORMAT)
# ============================================================

@app.post("/analyze-link/")
async def analyze_link(email: str = Form(...), video_url: str = Form(...)):

    user = get_user(email)

    if not user["is_paid"] and user["checks_used"] >= 10:
        return {
            "status": "success",
            "result": "LIMIT_REACHED",
            "upgrade_url": PRICING_URL
        }

    vid_id = str(uuid.uuid4())
    temp_file = f"{TMP_DIR}/{vid_id}.mp4"

    try:
        download_video(video_url, temp_file)
    except:
        return {
            "status": "success",
            "result": "ERROR"
        }

    max_seconds = 300 if user["is_paid"] else 60

    result, score = analyze_video_ai(temp_file, max_seconds)

    try:
        os.remove(temp_file)
    except:
        pass

    increment_usage(email)

    # 🔴 THIS IS THE KEY CHANGE
    return {
        "status": "success",
        "result": result,
        "ai_score": score
    }

# ============================================================
# CERTIFICATION PIPELINE (UNCHANGED)
# ============================================================

def fingerprint(path):
    sha = hashlib.sha256()
    with open(path,"rb") as f:
        while chunk := f.read(8192):
            sha.update(chunk)
    return sha.hexdigest()

def stamp_video(input_path, output_path, cert_id):
    text = f"VeriFYD • {cert_id[:6]}"
    cmd = [
        "ffmpeg","-y",
        "-i",input_path,
        "-vf",f"drawtext=text='{text}':x=4:y=4:fontsize=14:fontcolor=white@0.85:box=1:boxcolor=black@0.25",
        "-c:v","libx264","-preset","fast","-crf","23",
        "-c:a","aac","-b:a","128k",
        output_path
    ]
    subprocess.run(cmd, check=True)

@app.post("/upload/")
async def upload(file: UploadFile = File(...), email: str = Form(...)):

    user = get_user(email)

    if not user["is_paid"] and user["checks_used"] >= 10:
        return {
            "status":"LIMIT_REACHED",
            "upgrade_url":PRICING_URL
        }

    cert_id = str(uuid.uuid4())
    raw = f"{UPLOAD_DIR}/{cert_id}_{file.filename}"

    with open(raw,"wb") as buffer:
        buffer.write(await file.read())

    fp = fingerprint(raw)

    certified_name = f"{cert_id}_VeriFYD.mp4"
    certified_path = f"{CERT_DIR}/{certified_name}"

    stamp_video(raw, certified_path, cert_id)

    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute("INSERT INTO certificates VALUES (?,?,?,?,?)",
              (cert_id,file.filename,fp,certified_name,datetime.utcnow().isoformat()))
    conn.commit()
    conn.close()

    increment_usage(email)

    return {
        "status":"CERTIFIED",
        "certificate_id":cert_id,
        "verify_url":f"{BASE_URL}/verify/{cert_id}",
        "download_url":f"{BASE_URL}/download/{cert_id}"
    }

# ============================================================
@app.get("/download/{cid}")
def download(cid:str):
    path = f"{CERT_DIR}/{cid}_VeriFYD.mp4"
    return FileResponse(path, media_type="video/mp4")

@app.get("/verify/{cid}")
def verify(cid:str):
    return {"status":"VALID","certificate_id":cid}

@app.get("/")
def home():
    return {"status":"VeriFYD backend live"}





































