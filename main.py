from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import os, uuid, shutil

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

# --------------------------------------------------
# HOME
# --------------------------------------------------
@app.get("/")
def home():
    return {"status": "VeriFYD API LIVE"}

# --------------------------------------------------
# UPLOAD VIDEO
# --------------------------------------------------
@app.post("/upload/")
async def upload(file: UploadFile = File(...), email: str = Form(...)):
    cert_id = str(uuid.uuid4())
    raw_path = f"{UPLOAD_DIR}/{cert_id}_{file.filename}"

    with open(raw_path, "wb") as buffer:
        buffer.write(await file.read())

    certified_path = f"{CERT_DIR}/{cert_id}.mp4"
    shutil.copy(raw_path, certified_path)

    return {
        "status": "CERTIFIED",
        "certificate_id": cert_id,
        "download_url": f"{BASE_URL}/download/{cert_id}"
    }

# --------------------------------------------------
# DOWNLOAD
# --------------------------------------------------
@app.get("/download/{cid}")
def download(cid: str):
    path = f"{CERT_DIR}/{cid}.mp4"
    return FileResponse(path, media_type="video/mp4")

# --------------------------------------------------
# ANALYZE LINK (ACCEPTS ANY FRONTEND FORMAT)
# --------------------------------------------------
@app.post("/analyze-link/")
async def analyze_link(request: Request):

    try:
        # Try form first
        form = await request.form()
        email = form.get("email")
        video_url = form.get("video_url")

        # If form empty, try JSON
        if not email or not video_url:
            data = await request.json()
            email = data.get("email")
            video_url = data.get("video_url")

        print("Analyze request")
        print("Email:", email)
        print("URL:", video_url)

        # ---- Detection placeholder ----
        score = 78
        status = "AI DETECTED" if score < 85 else "AUTHENTIC VERIFIED"

        return {
            "success": True,
            "data": {
                "status": status,
                "authenticity_score": score
            }
        }

    except Exception as e:
        print("ANALYZE ERROR:", str(e))
        return {
            "success": False,
            "data": {
                "status": "ERROR",
                "authenticity_score": 0
            }
        }













































