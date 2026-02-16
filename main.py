import os
import uuid
import shutil
import subprocess
import tempfile
import requests
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from detector import detect_ai
from external_detector import external_ai_score

app = FastAPI(title="VeriFYD 3.2")

BASE_URL = "https://verifyd-backend.onrender.com"

UPLOAD_DIR = "videos"
CERT_DIR = "certified"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CERT_DIR, exist_ok=True)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------
# DETECTION
# ------------------------------------
def run_detection(video_path):
    primary = detect_ai(video_path)
    secondary = external_ai_score(video_path)

    avg = (primary + secondary) / 2

    # recalibrated scoring
    score = int(100 - avg * 0.7)

    if score >= 60:
        status = "REAL"
    elif score >= 40:
        status = "REVIEW"
    else:
        status = "AI"

    return score, status, primary, secondary

# ------------------------------------
# STAMP VIDEO (AUDIO SAFE)
# ------------------------------------
def stamp_video(input_path, output_path, cert_id):
    text = f"VeriFYD CERTIFIED {cert_id}"

    cmd = [
        "ffmpeg","-y",
        "-i", input_path,
        "-vf", f"drawtext=text='{text}':x=20:y=h-th-20:fontsize=18:fontcolor=white:box=1:boxcolor=black@0.5",
        "-c:v","libx264",
        "-c:a","copy",
        output_path
    ]

    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-400:])

# ------------------------------------
# UPLOAD
# ------------------------------------
@app.post("/upload/")
async def upload(file: UploadFile = File(...), email: str = Form(...)):
    ext = file.filename.split(".")[-1]
    cert_id = str(uuid.uuid4())
    raw_path = f"{UPLOAD_DIR}/{cert_id}.{ext}"

    with open(raw_path,"wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    score, status, primary, secondary = run_detection(raw_path)

    if status == "AI":
        return {
            "status": "AI DETECTED",
            "authenticity_score": score,
            "color": "red"
        }

    if status == "REVIEW":
        return {
            "status": "Video received. Confidence moderate.",
            "authenticity_score": score,
            "color": "blue"
        }

    # REAL → certify
    certified_path = f"{CERT_DIR}/{cert_id}.mp4"
    stamp_video(raw_path, certified_path, cert_id)

    return {
        "status": "REAL VIDEO VERIFIED",
        "authenticity_score": score,
        "certificate_id": cert_id,
        "download_url": f"{BASE_URL}/download/{cert_id}",
        "color": "green"
    }

# ------------------------------------
# DOWNLOAD
# ------------------------------------
@app.get("/download/{cid}")
def download(cid: str):
    path = f"{CERT_DIR}/{cid}.mp4"
    if not os.path.exists(path):
        return JSONResponse({"error":"not found"},status_code=404)
    return FileResponse(path, media_type="video/mp4", filename=f"{cid}.mp4")

# ------------------------------------
# ANALYZE LINK
# ------------------------------------
@app.get("/analyze-link/")
def analyze_link(video_url: str):

    if not video_url.startswith("http"):
        return {"error":"Invalid URL"}

    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_path = temp.name

    try:
        r = requests.get(video_url, stream=True, timeout=20)
        if r.status_code == 200:
            with open(temp_path,"wb") as f:
                for chunk in r.iter_content(1024):
                    f.write(chunk)
        else:
            return {"error":"Could not download video"}

        score, status, primary, secondary = run_detection(temp_path)

        if status == "AI":
            return {"status":"AI DETECTED","authenticity_score":score,"color":"red"}

        if status == "REVIEW":
            return {"status":"FURTHER REVIEW NEEDED","authenticity_score":score,"color":"blue"}

        return {"status":"REAL VIDEO VERIFIED","authenticity_score":score,"color":"green"}

    except Exception as e:
        return {"error":str(e)}

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)





