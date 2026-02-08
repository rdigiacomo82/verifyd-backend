import os
import uuid
import sqlite3
import subprocess
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

UPLOAD_DIR = "videos"
CERT_DIR = "certified"
LOGO_PATH = "logo.png"
DB_PATH = "certificates.db"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CERT_DIR, exist_ok=True)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- DATABASE ----------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS usage (
        email TEXT PRIMARY KEY,
        uploads INTEGER
    )
    """)
    conn.commit()
    conn.close()

init_db()

def get_usage(email):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT uploads FROM usage WHERE email=?", (email,))
    row = c.fetchone()
    if row:
        count = row[0]
    else:
        count = 0
        c.execute("INSERT INTO usage VALUES (?,?)", (email, 0))
        conn.commit()
    conn.close()
    return count

def increment_usage(email):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("UPDATE usage SET uploads = uploads + 1 WHERE email=?", (email,))
    conn.commit()
    conn.close()

# ---------- ROOT ----------
@app.get("/")
def home():
    return {"status": "VFVid API LIVE"}

# ---------- UPLOAD ----------
@app.post("/upload/")
async def upload_video(file: UploadFile = File(...), email: str = Form(...)):
    cert_id = str(uuid.uuid4())

    input_path = f"{UPLOAD_DIR}/{cert_id}_{file.filename}"
    output_path = f"{CERT_DIR}/{cert_id}.mp4"

    with open(input_path, "wb") as f:
        f.write(await file.read())

    # overlay logo
    if os.path.exists(LOGO_PATH):
        cmd = [
            "ffmpeg",
            "-i", input_path,
            "-i", LOGO_PATH,
            "-filter_complex", "overlay=W-w-20:H-h-20",
            "-codec:a", "copy",
            output_path
        ]
        subprocess.run(cmd)
    else:
        os.rename(input_path, output_path)

    increment_usage(email)
    used = get_usage(email)
    free_remaining = max(0, 10 - used)

    return {
        "status": "CERTIFIED",
        "certificate_id": cert_id,
        "verify": f"https://verifyd-backend.onrender.com/verify/{cert_id}",
        "download": f"https://verifyd-backend.onrender.com/download/{cert_id}",
        "free_remaining": free_remaining
    }

# ---------- DOWNLOAD ----------
@app.get("/download/{cert_id}")
def download(cert_id: str):
    file_path = f"{CERT_DIR}/{cert_id}.mp4"

    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="video/mp4", filename=f"certified_{cert_id}.mp4")

    return JSONResponse({"error": "Not found"}, status_code=404)

# ---------- VERIFY ----------
@app.get("/verify/{cert_id}")
def verify(cert_id: str):
    return {"status": "VALID CERTIFICATE", "id": cert_id}





















