import os
import shutil
import uuid
import hashlib
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse

app = FastAPI(title="VeriFYD Video Certification Authority")

UPLOAD_DIR = "videos"
CERT_DIR = "certified"
REVIEW_DIR = "review"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CERT_DIR, exist_ok=True)
os.makedirs(REVIEW_DIR, exist_ok=True)

CERTIFY_THRESHOLD = 80

def fingerprint(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def fake_ai_score():
    return 100  # placeholder until real AI model

@app.get("/")
def home():
    return {"status": "VeriFYD API LIVE"}

@app.post("/upload/")
async def upload_video(file: UploadFile = File(...)):
    cert_id = str(uuid.uuid4())
    temp_path = f"{UPLOAD_DIR}/{cert_id}_{file.filename}"

    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    score = fake_ai_score()

    # PASS → auto certify
    if score >= CERTIFY_THRESHOLD:
        certified_path = f"{CERT_DIR}/{cert_id}_VeriFYD.mp4"
        shutil.copy(temp_path, certified_path)

        return {
            "status": "CERTIFIED",
            "score": score,
            "certificate_id": cert_id,
            "verify": f"https://verifyd-backend.onrender.com/verify/{cert_id}",
            "download": f"https://verifyd-backend.onrender.com/download/{cert_id}"
        }

    # FAIL → private review
    review_path = f"{REVIEW_DIR}/{cert_id}_{file.filename}"
    shutil.move(temp_path, review_path)

    return {
        "status": "UNDER REVIEW",
        "score": score,
        "certificate_id": cert_id
    }

@app.get("/verify/{cert_id}")
def verify(cert_id: str):
    path = f"{CERT_DIR}/{cert_id}_VeriFYD.mp4"
    if os.path.exists(path):
        return {"status": "VALID", "certificate_id": cert_id}
    return {"status": "NOT FOUND"}

@app.get("/download/{cert_id}")
def download(cert_id: str):
    path = f"{CERT_DIR}/{cert_id}_VeriFYD.mp4"
    if os.path.exists(path):
        return FileResponse(path, media_type="video/mp4")
    return JSONResponse({"error": "not found"}, status_code=404)

















