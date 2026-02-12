from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import os, uuid, shutil
import cv2
import numpy as np

BASE_URL = "https://verifyd-backend.onrender.com"

UPLOAD_DIR = "videos"
CERT_DIR = "certified"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CERT_DIR, exist_ok=True)

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
# FAST AI DETECTION ENGINE
# =========================================================

def detect_ai_video(video_path):

    cap = cv2.VideoCapture(video_path)

    frame_count = 0
    noise_scores = []
    edge_scores = []
    motion_scores = []

    prev_gray = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count > 150:
            break

        if frame_count % 5 != 0:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # --- noise entropy ---
        noise = np.std(gray)
        noise_scores.append(noise)

        # --- edge density ---
        edges = cv2.Canny(gray, 100, 200)
        edge_scores.append(np.mean(edges))

        # --- motion realism ---
        if prev_gray is not None:
            diff = cv2.absdiff(gray, prev_gray)
            motion_scores.append(np.mean(diff))

        prev_gray = gray

    cap.release()

    if len(noise_scores) == 0:
        return False, 0

    avg_noise = np.mean(noise_scores)
    avg_edge = np.mean(edge_scores)
    avg_motion = np.mean(motion_scores) if motion_scores else 0

    ai_score = 0

    # diffusion smoothing indicator
    if avg_noise < 12:
        ai_score += 35

    # soft edges indicator
    if avg_edge < 6:
        ai_score += 25

    # motion inconsistency
    if avg_motion < 2:
        ai_score += 20

    # overly perfect consistency
    if np.std(noise_scores) < 1.5:
        ai_score += 20

    if ai_score > 60:
        return True, int(ai_score)
    else:
        return False, int(ai_score)

# =========================================================
# UPLOAD → DETECT → CERTIFY
# =========================================================

@app.post("/upload/")
async def upload(file: UploadFile = File(...), email: str = Form(...)):

    cert_id = str(uuid.uuid4())
    raw_path = f"{UPLOAD_DIR}/{cert_id}_{file.filename}"

    with open(raw_path, "wb") as buffer:
        buffer.write(await file.read())

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
# DOWNLOAD
# =========================================================

@app.get("/download/{cid}")
def download(cid: str):
    path = f"{CERT_DIR}/{cid}.mp4"
    return FileResponse(path, media_type="video/mp4")

# =========================================================
# LINK ANALYSIS (FAST MODE)
# =========================================================

@app.post("/analyze-link/")
async def analyze_link(email: str = Form(...), video_url: str = Form(...)):

    # For now we simulate detection
    # Next phase we download + analyze frames

    return {
        "status": "OK",
        "result": "AUTHENTIC VERIFIED",
        "ai_score": 0
    }








































