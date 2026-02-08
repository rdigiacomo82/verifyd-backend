from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import shutil
import uuid
import sqlite3
import os
import subprocess

app = FastAPI()

# CORS (allow your website)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEOS_DIR = os.path.join(BASE_DIR, "videos")
CERT_DIR = os.path.join(BASE_DIR, "certified")
DB_PATH = os.path.join(BASE_DIR, "certificates.db")
LOGO_PATH = os.path.join(BASE_DIR, "logo.png")

os.makedirs(VIDEOS_DIR, exist_ok=True)
os.makedirs(CERT_DIR, exist_ok=True)

# ---------- DATABASE ----------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS certs (
        id TEXT PRIMARY KEY,
        email TEXT,
        filename TEXT
    )
    """)
    conn.commit()
    conn.close()

init_db()

# ---------- HOME ----------
@app.get("/")
def home():
    return {"status": "VFVid API LIVE"}

# ---------- UPLOAD ----------
@app.post("/upload/")
async def upload(file: UploadFile = File(...), email: str = Form(...)):
    cert_id = str(uuid.uuid4())
    input_path = os.path.join(VIDEOS_DIR, file.filename)
    output_name = f"certified_{file.filename}"
    output_path = os.path.join(CERT_DIR, output_name)

    # save original
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # ---------- ADD LOGO OVERLAY ----------
    if os.path.exists(LOGO_PATH):
        try:
            subprocess.run([
                "ffmpeg",
                "-y",
                "-i", input_path,
                "-i", LOGO_PATH,
                "-filter_complex", "overlay=W-w-20:H-h-20",
                output_path
            ], check=True)
        except:
            shutil.copy(input_path, output_path)
    else:
        shutil.copy(input_path, output_path)

    # ---------- SAVE TO DB ----------
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO certs VALUES (?, ?, ?)", (cert_id, email, output_name))
    conn.commit()
    conn.close()

    return {
        "status": "CERTIFIED",
        "certificate_id": cert_id,
        "verify": f"https://verifyd-backend.onrender.com/verify/{cert_id}",
        "download": f"https://verifyd-backend.onrender.com/download/{cert_id}",
        "uploads_used": 1,
        "free_remaining": 9
    }

# ---------- DOWNLOAD ----------
@app.get("/download/{cert_id}")
def download(cert_id: str):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT filename FROM certs WHERE id=?", (cert_id,))
    row = c.fetchone()
    conn.close()

    if not row:
        return {"error": "Not found"}

    file_path = os.path.join(CERT_DIR, row[0])

    if not os.path.exists(file_path):
        return {"error": "File missing"}

    return FileResponse(file_path, media_type="video/mp4", filename=row[0])

# ---------- VERIFY ----------
@app.get("/verify/{cert_id}")
def verify(cert_id: str):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT email, filename FROM certs WHERE id=?", (cert_id,))
    row = c.fetchone()
    conn.close()

    if not row:
        return {"error": "Not found"}

    return {
        "status": "verified",
        "certificate_id": cert_id,
        "owner": row[0],
        "file": row[1]
    }
























