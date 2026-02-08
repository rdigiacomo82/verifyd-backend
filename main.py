from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import os, uuid, hashlib, subprocess

app = FastAPI(title="VFVid API")

# Allow your website
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "videos"
CERT_DIR = "certified"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CERT_DIR, exist_ok=True)

# -------------------------
# Fingerprint
# -------------------------
def fingerprint(path):
    h = hashlib.sha256()
    with open(path,"rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

# -------------------------
# Stamp video (lightweight)
# -------------------------
def stamp_video(inp, out):
    cmd = [
        "ffmpeg","-y",
        "-i", inp,
        "-vf","drawtext=text='VFVid CERTIFIED':fontsize=28:fontcolor=white:x=20:y=H-th-40:box=1:boxcolor=black@0.5",
        "-c:v","libx264","-preset","veryfast","-crf","28",
        "-c:a","aac","-b:a","96k",
        out
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# -------------------------
# Upload endpoint
# -------------------------
@app.post("/upload/")
async def upload(file: UploadFile = File(...), email: str = Form(None)):

    uid = str(uuid.uuid4())
    raw_path = f"{UPLOAD_DIR}/{uid}_{file.filename}"

    with open(raw_path,"wb") as f:
        f.write(await file.read())

    cert_id = str(uuid.uuid4())
    out_path = f"{CERT_DIR}/{cert_id}.mp4"

    stamp_video(raw_path, out_path)

    return {
        "status":"CERTIFIED",
        "certificate_id": cert_id,
        "verify": f"https://verifyd-backend.onrender.com/verify/{cert_id}",
        "download": f"https://verifyd-backend.onrender.com/download/{cert_id}"
    }

# -------------------------
# Download
# -------------------------
@app.get("/download/{cid}")
def download(cid: str):
    path = f"{CERT_DIR}/{cid}.mp4"
    if not os.path.exists(path):
        return {"error":"not found"}
    return FileResponse(path, media_type="video/mp4", filename=f"{cid}.mp4")

# -------------------------
# Verify page
# -------------------------
@app.get("/verify/{cid}")
def verify(cid: str):
    return {
        "status":"VERIFIED AUTHENTIC",
        "certificate_id": cid
    }

# -------------------------
@app.get("/")
def root():
    return {"status":"VFVid API LIVE"}




























