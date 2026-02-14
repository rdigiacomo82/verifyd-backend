from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import os, uuid, subprocess, cv2, numpy as np

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

@app.get("/")
def home():
    return {"status": "VeriFYD FINAL BUILD"}

# --------------------------------------------------
# detection (placeholder)
# --------------------------------------------------
def analyze_video(path):
    cap = cv2.VideoCapture(path)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return 75 if frames > 0 else 50

# --------------------------------------------------
# FINAL AUDIO-SAFE STAMP
# --------------------------------------------------
def stamp_video(input_path, output_path, cert_id):

    print("🔊 stamping with remux pipeline")

    temp_mux = f"{TMP_DIR}/{cert_id}_mux.mp4"

    # STEP 1: REMUX ONLY (no filters)
    remux_cmd = [
        "ffmpeg",
        "-y",
        "-i", input_path,
        "-map", "0",
        "-c", "copy",
        "-movflags", "+faststart",
        temp_mux
    ]

    subprocess.run(remux_cmd)

    # STEP 2: APPLY WATERMARK
    # IMPORTANT: no colons in text
    watermark = (
        f"drawtext=text='VeriFYD ID {cert_id}':"
        "x=w-tw-20:y=h-th-20:"
        "fontsize=16:"
        "fontcolor=white@0.8"
    )

    final_cmd = [
        "ffmpeg",
        "-y",
        "-i", temp_mux,
        "-vf", watermark,
        "-map", "0:v",
        "-map", "0:a?",
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-c:a", "aac",
        "-b:a", "192k",
        "-af", "aresample=async=1",
        "-shortest",
        "-movflags", "+faststart",
        output_path
    ]

    subprocess.run(final_cmd)

# --------------------------------------------------
# upload
# --------------------------------------------------
@app.post("/upload/")
async def upload(file: UploadFile = File(...), email: str = Form(...)):

    cert_id = str(uuid.uuid4())
    raw_path = f"{UPLOAD_DIR}/{cert_id}_{file.filename}"

    with open(raw_path, "wb") as buffer:
        buffer.write(await file.read())

    score = analyze_video(raw_path)

    certified_path = f"{CERT_DIR}/{cert_id}.mp4"
    stamp_video(raw_path, certified_path, cert_id)

    return {
        "status": "CERTIFIED REAL VIDEO",
        "certificate_id": cert_id,
        "authenticity_score": score,
        "download_url": f"{BASE_URL}/download/{cert_id}"
    }

@app.get("/download/{cid}")
def download(cid: str):
    return FileResponse(f"{CERT_DIR}/{cid}.mp4", media_type="video/mp4")
























































