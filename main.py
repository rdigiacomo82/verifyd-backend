from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
import os, uuid, shutil, subprocess
import cv2
import numpy as np
import requests
import tempfile

BASE_URL = "https://verifyd-backend.onrender.com"

UPLOAD_DIR = "videos"
CERT_DIR = "certified"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CERT_DIR, exist_ok=True)

app = FastAPI()

# --------------------------------------------------
# CORS
# --------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------
# HOME
# --------------------------------------------------
@app.get("/")
def home():
    return {"status": "VeriFYD API LIVE"}

# ==================================================
# 🔬 REAL DETECTION ENGINE
# ==================================================
def analyze_video(file_path):

    cap = cv2.VideoCapture(file_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if frame_count == 0:
        return 0

    samples = []
    step = max(frame_count // 25, 1)

    for i in range(0, frame_count, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        noise = np.std(gray)
        edges = cv2.Laplacian(gray, cv2.CV_64F).var()

        samples.append((noise, edges))

        if len(samples) > 25:
            break

    cap.release()

    if not samples:
        return 0

    noise_avg = np.mean([s[0] for s in samples])
    edge_avg = np.mean([s[1] for s in samples])

    score = 0

    # natural camera noise
    if noise_avg > 20:
        score += 40
    else:
        score -= 20

    # texture detail
    if edge_avg > 30:
        score += 40
    else:
        score -= 20

    # randomness variation
    variance = np.var([s[0] for s in samples])
    if variance > 5:
        score += 20

    score = max(min(score + 50, 100), 0)

    return int(score)

# ==================================================
# 🔊 STAMP VIDEO WITH AUDIO PRESERVED
# ==================================================
def stamp_video(input_path, output_path, cert_id):

    command = [
        "ffmpeg",
        "-y",
        "-i", input_path,

        "-vf",
        f"drawtext=text='VeriFYD CERTIFIED':"
        "x=10:y=10:fontsize=18:fontcolor=white@0.8",

        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",

        # 🔊 KEEP AUDIO
        "-c:a", "aac",
        "-b:a", "192k",
        "-map", "0:v",
        "-map", "0:a?",

        output_path
    ]

    subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

# ==================================================
# 📤 UPLOAD VIDEO
# ==================================================
@app.post("/upload/")
async def upload(file: UploadFile = File(...), email: str = Form(...)):

    cert_id = str(uuid.uuid4())
    raw_path = f"{UPLOAD_DIR}/{cert_id}_{file.filename}"

    with open(raw_path, "wb") as buffer:
        buffer.write(await file.read())

    # 🔬 run detection
    score = analyze_video(raw_path)

    if score < 40:
        return {
            "status": "AI DETECTED",
            "authenticity_score": score
        }

    certified_path = f"{CERT_DIR}/{cert_id}.mp4"

    stamp_video(raw_path, certified_path, cert_id)

    return {
        "status": "CERTIFIED REAL VIDEO",
        "certificate_id": cert_id,
        "authenticity_score": score,
        "download_url": f"{BASE_URL}/download/{cert_id}"
    }

# ==================================================
# 📥 DOWNLOAD
# ==================================================
@app.get("/download/{cid}")
def download(cid: str):
    path = f"{CERT_DIR}/{cid}.mp4"
    return FileResponse(path, media_type="video/mp4")

# ==================================================
# 🔗 ANALYZE LINK
# ==================================================
@app.post("/analyze-link/")
async def analyze_link(email: str = Form(...), video_url: str = Form(...)):

    try:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        r = requests.get(video_url, stream=True, timeout=15)

        for chunk in r.iter_content(1024):
            tmp.write(chunk)

        tmp.close()

        score = analyze_video(tmp.name)

        os.unlink(tmp.name)

        if score < 40:
            result = "AI DETECTED"
        else:
            result = "REAL VIDEO"

        html = f"""
        <html>
        <body style="background:#0b0b0b;color:white;text-align:center;padding-top:120px;font-family:Arial">
        <h1>{result}</h1>
        <h2>Authenticity Score: {score}</h2>
        <p>Analyzed by VeriFYD</p>
        </body>
        </html>
        """

        return HTMLResponse(html)

    except Exception as e:
        return {"error": str(e)}















































