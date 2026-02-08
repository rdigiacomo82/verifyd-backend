from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

import hashlib
import os
import uuid
import subprocess
import sqlite3
from datetime import datetime

app = FastAPI(title="VeriFYD API")

# =========================
# CORS FIX (CRITICAL)
# =========================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# PATHS
# =========================
UPLOAD_DIR = "videos"
CERTIFIED_DIR = "certified"
DB_FILE = "certificates.db"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CERTIFIED_DIR, exist_ok=True)

# =========================
# DATABASE INIT
# =========================
def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()

    c.execute("""
    CREATE TABLE IF NOT EXISTS certificates (
        id TEXT PRIMARY KEY,
        filename TEXT,
        fingerprint TEXT,
        certified_file TEXT,
        email TEXT,
        created_at TEXT
    )
    """)

    conn.commit()
    conn.close()

init_db()

# =========================
# HASH
# =========================
def generate_fingerprint(file_path):
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            sha256.update(chunk)
    return sha256.hexdigest()

# =========================
# STAMP VIDEO (LIGHTWEIGHT)
# =========================
def stamp_video(input_path, output_path, cert_id):

    command = [
        "ffmpeg",
        "-y",
        "-i", input_path,
        "-vf",
        "drawtext=text='VeriFYD CERTIFIED':fontsize=36:fontcolor=white:x=20:y=H-th-40:box=1:boxcolor=black@0.5",
        "-metadata", f"VeriFYD-CertID={cert_id}",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "28",
        "-c:a", "aac",
        "-b:a", "96k",
        output_path
    ]

    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# =========================
# UPLOAD ENDPOINT
# =========================
@app.post("/upload/")
async def upload_video(
    file: UploadFile = File(...),
    email: str = Form(...)
):

    upload_id = str(uuid.uuid4())
    raw_path = os.path.join(UPLOAD_DIR, f"{upload_id}_{file.filename}")

    # SAVE FILE IN CHUNKS (prevents memory crash)
    with open(raw_path, "wb") as buffer:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            buffer.write(chunk)

    fingerprint = generate_fingerprint(raw_path)

    cert_id = str(uuid.uuid4())
    certified_filename = f"{cert_id}.mp4"
    certified_path = os.path.join(CERTIFIED_DIR, certified_filename)

    stamp_video(raw_path, certified_path, cert_id)

    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()

    c.execute("""
        INSERT INTO certificates VALUES (?,?,?,?,?,?)
    """, (
        cert_id,
        file.filename,
        fingerprint,
        certified_filename,
        email,
        datetime.utcnow().isoformat()
    ))

    conn.commit()
    conn.close()

    base = "https://verifyd-backend.onrender.com"

    return {
        "status": "CERTIFIED",
        "certificate_id": cert_id,
        "verify": f"{base}/verify/{cert_id}",
        "download": f"{base}/download/{cert_id}"
    }

# =========================
# VERIFY
# =========================
@app.get("/verify/{cert_id}")
def verify(cert_id: str):

    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT * FROM certificates WHERE id=?", (cert_id,))
    row = c.fetchone()
    conn.close()

    if not row:
        return {"status": "not found"}

    return {
        "status": "verified",
        "certificate_id": row[0],
        "filename": row[1],
        "fingerprint": row[2],
        "email": row[4],
        "created": row[5]
    }

# =========================
# DOWNLOAD
# =========================
@app.get("/download/{cert_id}")
def download(cert_id: str):

    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT certified_file FROM certificates WHERE id=?", (cert_id,))
    row = c.fetchone()
    conn.close()

    if not row:
        return {"error": "not found"}

    path = os.path.join(CERTIFIED_DIR, row[0])

    return FileResponse(
        path,
        media_type="video/mp4",
        filename=row[0]
    )

# =========================
# ROOT
# =========================
@app.get("/")
def root():
    return {"status": "VeriFYD API LIVE"}



























