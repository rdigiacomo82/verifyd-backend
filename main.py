from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import os, uuid, subprocess

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
# VERIFY FFMPEG EXISTS
# =========================================================

def check_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print("FFmpeg detected.")
    except Exception as e:
        print("FFmpeg NOT detected:", e)

check_ffmpeg()

# =========================================================
# WATERMARK FUNCTION (FORCED)
# =========================================================

def stamp_video(input_path, output_path):

    print("Applying watermark...")

    command = [
        "ffmpeg",
        "-y",
        "-i", input_path,
        "-vf",
        "drawtext=text='VeriFYD CERTIFIED':"
        "x=5:y=5:"
        "fontsize=22:"
        "fontcolor=white:"
        "box=1:"
        "boxcolor=black@0.4:"
        "boxborderw=8",
        "-c:v","libx264",
        "-preset","fast",
        "-crf","23",
        "-c:a","aac",
        "-b:a","128k",
        output_path
    ]

    result = subprocess.run(command, capture_output=True, text=True)

    print(result.stdout)
    print(result.stderr)

    if result.returncode != 0:
        raise Exception("FFmpeg watermark failed")

    print("Watermark complete.")

# =========================================================
# UPLOAD + CERTIFY
# =========================================================

@app.post("/upload/")
async def upload(file: UploadFile = File(...), email: str = Form(...)):

    cert_id = str(uuid.uuid4())

    raw_path = f"{UPLOAD_DIR}/{cert_id}_{file.filename}"

    with open(raw_path, "wb") as buffer:
        buffer.write(await file.read())

    certified_path = f"{CERT_DIR}/{cert_id}.mp4"

    # APPLY WATERMARK
    stamp_video(raw_path, certified_path)

    return {
        "status":"CERTIFIED",
        "certificate_id":cert_id,
        "download_url":f"{BASE_URL}/download/{cert_id}"
    }

# =========================================================
# DOWNLOAD
# =========================================================

@app.get("/download/{cid}")
def download(cid:str):
    path = f"{CERT_DIR}/{cid}.mp4"
    return FileResponse(path, media_type="video/mp4")

# =========================================================

@app.get("/")
def home():
    return {"status":"VeriFYD backend live"}







































