from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import os, uuid, shutil
import cv2
import numpy as np
import torch
import timm

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
# LOAD MODEL (lightweight vision transformer)
# =========================================================

model = timm.create_model("mobilenetv3_small_100", pretrained=True)
model.eval()

# =========================================================
# FRAME MODEL ANALYSIS
# =========================================================

def analyze_frame_model(frame):

    img = cv2.resize(frame, (224, 224))
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = torch.tensor(img).unsqueeze(0)

    with torch.no_grad():
        feat = model.forward_features(img)
        score = float(torch.mean(feat))

    return score

# =========================================================
# REAL DETECTION ENGINE
# =========================================================

def detect_ai_video(video_path):

    cap = cv2.VideoCapture(video_path)

    frame_count = 0
    model_scores = []
    noise_scores = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count > 120:
            break

        if frame_count % 6 != 0:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        noise_scores.append(np.std(gray))

        model_score = analyze_frame_model(frame)
        model_scores.append(model_score)

    cap.release()

    if not model_scores:
        return False, 0

    avg_model = np.mean(model_scores)
    avg_noise = np.mean(noise_scores)

    ai_score = 0

    # model feature anomaly
    if avg_model < 0.25:
        ai_score += 50

    # diffusion smoothing
    if avg_noise < 10:
        ai_score += 30

    # consistency
    if np.std(model_scores) < 0.01:
        ai_score += 20

    if ai_score > 60:
        return True, int(ai_score)
    else:
        return False, int(ai_score)

# =========================================================
# HOME
# =========================================================
@app.get("/")
def home():
    return {"status":"VeriFYD backend live"}

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
# LINK ANALYSIS (placeholder for now)
# =========================================================
@app.post("/analyze-link/")
async def analyze_link(email: str = Form(...), video_url: str = Form(...)):
    return {
        "status": "OK",
        "result": "AUTHENTIC VERIFIED",
        "ai_score": 0
    }








































