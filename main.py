from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import os, uuid, subprocess

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
# UPLOAD VIDEO
# =========================================================

@app.post("/upload/")
async def upload(file: UploadFile = File(...), email: str = Form(...)):

    cert_id = str(uuid.uuid4())
    raw_path = f"{UPLOAD_DIR}/{cert_id}_{file.filename}"

    with open(raw_path, "wb") as buffer:
        buffer.write(await file.read())

    certified_path = f"{CERT_DIR}/{cert_id}.mp4"

    # Temporary copy (Hostinger stamping for now)
    subprocess.run(["cp", raw_path, certified_path])

    return {
        "status": "CERTIFIED",
        "certificate_id": cert_id,
        "download_url": f"{BASE_URL}/download/{cert_id}"
    }

# =========================================================
# DOWNLOAD CERTIFIED VIDEO
# =========================================================

@app.get("/download/{cid}")
def download(cid: str):
    path = f"{CERT_DIR}/{cid}.mp4"
    return FileResponse(path, media_type="video/mp4")

# =========================================================
# SAFE VIDEO DOWNLOAD FUNCTION
# =========================================================

def download_temp_video(url, output_path):

    try:
        result = subprocess.run(
            ["yt-dlp", "-f", "worst", "-o", output_path, url],
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            print("yt-dlp failed:", result.stderr)
            return False

        if not os.path.exists(output_path):
            print("Downloaded file missing")
            return False

        return True

    except Exception as e:
        print("Download exception:", e)
        return False

# =========================================================
# ANALYZE LINK (STABLE VERSION)
# =========================================================

@app.post("/analyze-link/")
async def analyze_link(email: str = Form(...), video_url: str = Form(...)):

    temp_id = str(uuid.uuid4())
    temp_path = f"{TMP_DIR}/{temp_id}.mp4"

    success = download_temp_video(video_url, temp_path)

    if not success:
        return {
            "result": "DOWNLOAD_FAILED",
            "message": "Could not download video from link."
        }

    # Placeholder result until detection added
    return {
        "result": "AUTHENTIC VERIFIED",
        "ai_score": 0
    }








































