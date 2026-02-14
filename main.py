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
# 🔬 AI DETECTION ENGINE
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
        brightness = np.mean(gray)

        samples.append((noise, edges, brightness))

        if len(samples) > 25:
            break

    cap.release()

    if not samples:
        return 0

    noise_avg = np.mean([s[0] for s in samples])
    edge_avg = np.mean([s[1] for s in samples])
    brightness_var = np.var([s[2] for s in samples])

    score = 50

    # natural sensor noise
    if noise_avg > 18:
        score += 20
    else:
        score -= 20

    # texture detail
    if edge_avg > 25:
        score += 20
    else:
        score -= 20

    # lighting variance
    if brightness_var > 2:
        score += 10
    else:
        score -= 10

    score = max(min(score, 99), 0)
    return int(score)

# ==================================================
# 🔊 WATERMARK + AUDIO PRESERVED
# ==================================================
def stamp_video(input_path, output_path, cert_id):

    drawtext_main = "drawtext=text='VeriFYD':x=10:y=10:fontsize=22:fontcolor=white@0.85"
    drawtext_id = f"drawtext=text='ID:{cert_id}':x=w-tw-20:y=h-th-20:fontsize=16:fontcolor=white@0.7"

    vf = f"{drawtext_main},{drawtext_id}"

    command = [
        "ffmpeg",
        "-y",
        "-i", input_path,

        "-vf", vf,

        "-map", "0:v:0",
        "-map", "0:a?",

        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",

        "-c:a", "aac",
        "-b:a", "192k",

        "-movflags", "+faststart",

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

    score = analyze_video(raw_path)

    if score < 45:
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
        r = requests.get(video_url, stream=True, timeout=20)

        for chunk in r.iter_content(1024):
            tmp.write(chunk)

        tmp.close()

        score = analyze_video(tmp.name)
        os.unlink(tmp.name)

        result = "AI DETECTED" if score < 45 else "REAL VIDEO"

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



















































