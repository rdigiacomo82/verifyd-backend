from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse
import hashlib
import os
import uuid
import sqlite3
from datetime import datetime

app = FastAPI(title="VFVid API")

UPLOAD_DIR = "videos"
CERTIFIED_DIR = "certified"
DB_FILE = "certificates.db"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CERTIFIED_DIR, exist_ok=True)

# ==============================
# DATABASE INIT
# ==============================
def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()

    c.execute("""
    CREATE TABLE IF NOT EXISTS certificates (
        id TEXT PRIMARY KEY,
        email TEXT,
        filename TEXT,
        fingerprint TEXT,
        certified_file TEXT,
        created_at TEXT,
        status TEXT
    )
    """)

    conn.commit()
    conn.close()

init_db()

# ==============================
# FINGERPRINT
# ==============================
def fingerprint(path):
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            sha.update(chunk)
    return sha.hexdigest()

# ==============================
# UPLOAD
# ==============================
@app.post("/upload/")
async def upload_video(
    file: UploadFile = File(...),
    email: str = Form(...)
):
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in [".mp4", ".mov", ".avi", ".m4v"]:
        return {"error": "Unsupported file type"}

    upload_id = str(uuid.uuid4())
    raw_path = os.path.join(UPLOAD_DIR, f"{upload_id}_{file.filename}")

    with open(raw_path, "wb") as buffer:
        buffer.write(await file.read())

    fp = fingerprint(raw_path)

    cert_id = str(uuid.uuid4())
    certified_filename = f"{cert_id}.mp4"
    certified_path = os.path.join(CERTIFIED_DIR, certified_filename)

    # STABLE VERSION: just copy video (no ffmpeg)
    with open(raw_path, "rb") as src, open(certified_path, "wb") as dst:
        dst.write(src.read())

    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()

    c.execute("""
        INSERT INTO certificates
        (id, email, filename, fingerprint, certified_file, created_at, status)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        cert_id,
        email,
        file.filename,
        fp,
        certified_filename,
        datetime.utcnow().isoformat(),
        "CERTIFIED"
    ))

    conn.commit()
    conn.close()

    return {
        "status": "CERTIFIED",
        "certificate_id": cert_id,
        "verify": f"https://verifyd-backend.onrender.com/verify/{cert_id}",
        "download": f"https://verifyd-backend.onrender.com/download/{cert_id}"
    }

# ==============================
# VERIFY
# ==============================
@app.get("/verify/{cert_id}")
def verify(cert_id: str):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT * FROM certificates WHERE id=?", (cert_id,))
    row = c.fetchone()
    conn.close()

    if not row:
        return {"error": "Not found"}

    return {
        "status": row[6],
        "filename": row[2],
        "fingerprint": row[3],
        "issued": row[5]
    }

# ==============================
# DOWNLOAD
# ==============================
@app.get("/download/{cert_id}")
def download(cert_id: str):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT certified_file FROM certificates WHERE id=?", (cert_id,))
    row = c.fetchone()
    conn.close()

    if not row:
        return {"error": "Not found"}

    path = os.path.join(CERTIFIED_DIR, row[0])
    return FileResponse(path, media_type="video/mp4", filename=row[0])

# ==============================
# ROOT
# ==============================
@app.get("/")
def root():
    return {"status": "VFVid API LIVE"}



























