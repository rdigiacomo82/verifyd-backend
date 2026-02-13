from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import os, uuid, shutil

app = FastAPI()

BASE_URL = "https://verifyd-backend.onrender.com"

UPLOAD_DIR = "videos"
CERT_DIR = "certified"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CERT_DIR, exist_ok=True)

# ---------------- CORS ----------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- HOME ----------------
@app.get("/")
def home():
    return {"status": "API LIVE"}

# ---------------- UPLOAD ----------------
@app.post("/upload/")
async def upload(file: UploadFile = File(...), email: str = Form(...)):

    cert_id = str(uuid.uuid4())
    raw_path = f"{UPLOAD_DIR}/{cert_id}_{file.filename}"

    with open(raw_path, "wb") as buffer:
        buffer.write(await file.read())

    certified_path = f"{CERT_DIR}/{cert_id}.mp4"
    shutil.copy(raw_path, certified_path)

    return {
        "success": True,
        "data": {
            "status": "CERTIFIED",
            "certificate_id": cert_id,
            "download_url": f"{BASE_URL}/download/{cert_id}"
        }
    }

# ---------------- DOWNLOAD ----------------
@app.get("/download/{cid}")
def download(cid: str):
    path = f"{CERT_DIR}/{cid}.mp4"
    return FileResponse(path, media_type="video/mp4")

# ---------------- ANALYZE LINK (SAFE) ----------------
@app.post("/analyze-link/")
async def analyze_link(email: str = Form(...), video_url: str = Form(...)):

    print("Analyze request received")
    print(email, video_url)

    return {
        "success": True,
        "data": {
            "status": "AI DETECTED",
            "authenticity_score": 78
        }
    }














































