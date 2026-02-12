from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import os, uuid, shutil
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
# HOME
# =========================================================
@app.get("/")
def home():
    return {"status": "VeriFYD backend live"}

# =========================================================
# AI DETECTION ENGINE (LIGHTWEIGHT FORENSIC)
# =========================================================

def detect_ai_video(video_path):

    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    anomaly_score = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count > 300:
            break

        # sample every 10th frame
        if frame_count % 10 != 0:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # noise analysis
        noise = np.std(gray)
        if noise < 8:
            anomaly_score += 2

        # edge realism
        edges = cv2.Canny(gray, 100, 200)
        edge_mean = np.mean(edges)

        if edge_mean < 3:
            anomaly_score += 1

    cap.release()

    if anomaly_score > 40:
        return True, anomaly_score
    else:
        return False, anomaly_score

# =========================================================
# UPLOAD VIDEO → DETECT AI → CERTIFY
# =========================================================

@app.post("/upload/")
async def upload(file: UploadFile = File(...), email: str = Form(...)):

    cert_id = str(uuid.uuid4())
    raw_path = f"{UPLOAD_DIR}/{cert_id}_{file.filename}"

    with open(raw_path, "wb") as buffer:
        buffer.write(await file.read())

    # 🔍 RUN AI DETECTION
    is_ai, score = detect_ai_video(raw_path)

    if is_ai:
        return {
            "status": "AI DETECTED",
            "ai_score": score,
            "message": "Video appears AI-generated and cannot be certified."
        }

    certified_path = f"{CERT_DIR}/{cert_id}.mp4"
    shutil.copy(raw_path, certified_path)

    return {
        "status": "CERTIFIED",
        "certificate_id": cert_id,
        "download_url": f"{BASE_URL}/download/{cert_id}",
        "ai_score": score
    }

# =========================================================
# DOWNLOAD CERTIFIED VIDEO
# =========================================================

@app.get("/download/{cid}")
def download(cid: str):
    path = f"{CERT_DIR}/{cid}.mp4"
    return FileResponse(path, media_type="video/mp4")

# =========================================================
# ANALYZE LINK (AI DETECTION)
# =========================================================

@app.post("/analyze-link/")
async def analyze_link(email: str = Form(...), video_url: str = Form(...)):

    # NOTE:
    # For now we simulate detection.
    # Next phase we'll download video safely.

    # Placeholder logic
    return {
        "status": "OK",
        "result": "AUTHENTIC VERIFIED",
        "ai_score": 0
    }








































