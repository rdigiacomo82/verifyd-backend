import os
import uuid
import subprocess
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI()

# -----------------------------
# FOLDERS
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEO_DIR = os.path.join(BASE_DIR, "videos")
CERT_DIR = os.path.join(BASE_DIR, "certified")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")

os.makedirs(VIDEO_DIR, exist_ok=True)
os.makedirs(CERT_DIR, exist_ok=True)

# -----------------------------
# SERVE STATIC FILES (THIS FIXES LOGO 404)
# -----------------------------
app.mount("/assets", StaticFiles(directory=ASSETS_DIR), name="assets")

# -----------------------------
# STATUS CHECK
# -----------------------------
@app.get("/")
def root():
    return {"status": "VFVid API LIVE"}

# -----------------------------
# UPLOAD VIDEO
# -----------------------------
@app.post("/upload/")
async def upload_video(file: UploadFile = File(...)):
    video_id = str(uuid.uuid4())
    input_path = os.path.join(VIDEO_DIR, f"{video_id}.mp4")
    output_path = os.path.join(CERT_DIR, f"certified_{video_id}.mp4")

    with open(input_path, "wb") as f:
        f.write(await file.read())

    logo_path = os.path.join(ASSETS_DIR, "logo.png")

    # Overlay logo with ffmpeg
    cmd = [
        "ffmpeg",
        "-i", input_path,
        "-i", logo_path,
        "-filter_complex", "overlay=W-w-20:H-h-20",
        "-codec:a", "copy",
        output_path
    ]

    subprocess.run(cmd)

    return {
        "certificate_id": video_id,
        "download": f"/download/{video_id}"
    }

# -----------------------------
# DOWNLOAD CERTIFIED VIDEO
# -----------------------------
@app.get("/download/{video_id}")
def download(video_id: str):
    file_path = os.path.join(CERT_DIR, f"certified_{video_id}.mp4")

    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="video/mp4")

    return {"detail": "Not Found"}
























