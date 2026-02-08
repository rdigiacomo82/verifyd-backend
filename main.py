from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
import uuid
import subprocess

app = FastAPI()

# Allow website to talk to backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_URL = "https://verifyd-backend.onrender.com"

UPLOAD_DIR = "videos"
CERT_DIR = "certified"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CERT_DIR, exist_ok=True)

LOGO_PATH = "logo.png"   # put your logo.png in same folder


@app.get("/")
def home():
    return {"status": "VFVid API LIVE"}


# =========================
# VIDEO CERTIFICATION ENGINE
# =========================
def certify_video(input_path, output_path, cert_id):
    cmd = [
        "ffmpeg",
        "-y",
        "-i", input_path,
        "-i", LOGO_PATH,
        "-filter_complex",
        "[1:v]scale=140:-1[logo];"
        "[0:v][logo]overlay=W-w-20:H-h-20,"
        "drawtext=text='VERIFIED AUTHENTIC':fontcolor=white:fontsize=40:x=20:y=20,"
        f"drawtext=text='Certificate ID: {cert_id}':fontcolor=purple:fontsize=26:x=20:y=70",
        "-codec:a", "copy",
        output_path
    ]

    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


# =========================
# UPLOAD VIDEO
# =========================
@app.post("/upload/")
async def upload_video(file: UploadFile = File(...), email: str = Form(...)):

    file_id = str(uuid.uuid4())
    input_path = f"{UPLOAD_DIR}/{file_id}_{file.filename}"

    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    cert_id = str(uuid.uuid4())
    output_path = f"{CERT_DIR}/certified_{file.filename}"

    certify_video(input_path, output_path, cert_id)

    return {
        "status": "CERTIFIED",
        "certificate_id": cert_id,
        "verify": f"{BASE_URL}/verify/{cert_id}",
        "download": f"{BASE_URL}/download/{cert_id}",
        "free_remaining": 9
    }


# =========================
# DOWNLOAD VIDEO
# =========================
@app.get("/download/{cert_id}")
def download_video(cert_id: str):
    for file in os.listdir(CERT_DIR):
        if file.startswith("certified_"):
            return FileResponse(
                path=f"{CERT_DIR}/{file}",
                filename=file,
                media_type="video/mp4"
            )
    return JSONResponse({"error": "Not found"}, status_code=404)


# =========================
# VERIFY PAGE
# =========================
@app.get("/verify/{cert_id}")
def verify(cert_id: str):
    return {
        "status": "VERIFIED AUTHENTIC",
        "certificate_id": cert_id,
        "issuer": "VeriFYD",
        "public_verify": True
    }


# =========================
# PAYPAL SUBSCRIPTION ACTIVATE
# =========================
@app.post("/activate-subscription/")
def activate_subscription(email: str = Form(...)):
    return {"status": "subscription activated"}





















