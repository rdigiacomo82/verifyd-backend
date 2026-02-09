from fastapi import FastAPI, UploadFile, File
import hashlib
import os
import uuid
import subprocess
import sqlite3
import shutil
from datetime import datetime

# ============================================================
# VeriFYD Video Certification Authority
# ============================================================

app = FastAPI(title="VeriFYD Video Certification Authority")

UPLOAD_DIR = "videos"
CERTIFIED_DIR = "certified"
DB_FILE = "certificates.db"
LOGO_PATH = "assets/logo.png"   # <-- your watermark logo

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CERTIFIED_DIR, exist_ok=True)

# ============================================================
# Database Setup
# ============================================================
def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()

    c.execute("""
    CREATE TABLE IF NOT EXISTS certificates (
        id TEXT PRIMARY KEY,
        filename TEXT,
        fingerprint TEXT,
        certified_file TEXT,
        created_at TEXT
    )
    """)

    conn.commit()
    conn.close()

init_db()

# ============================================================
# Fingerprint Generator
# ============================================================
def generate_fingerprint(file_path: str):
    sha256 = hashlib.sha256()

    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            sha256.update(chunk)

    return sha256.hexdigest()

# ============================================================
# SAFE WATERMARK + CERTIFICATION
# ============================================================
def stamp_video(input_path: str, output_path: str, cert_id: str):
    """
    Adds watermark + metadata.
    If watermark fails, falls back to original certification.
    """

    # If logo missing → just encode without watermark
    if not os.path.exists(LOGO_PATH):
        print("⚠ Logo not found — certifying without watermark")
        return basic_encode(input_path, output_path, cert_id)

    text = f"VeriFYD CERTIFIED {cert_id[:8]}"

    command = [
        "ffmpeg",
        "-y",
        "-i", input_path,
        "-i", LOGO_PATH,
        "-filter_complex",
        f"[0:v][1:v] overlay=20:20, drawtext=text='{text}':x=20:y=110:fontsize=18:fontcolor=white:box=1:boxcolor=black@0.4",

        # metadata
        "-metadata", f"VeriFYD-CertID={cert_id}",
        "-metadata", "VeriFYD-Status=CertifiedAuthentic",

        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-c:a", "aac",
        "-b:a", "128k",

        output_path
    ]

    try:
        subprocess.run(command, check=True)
        print("✅ Watermark applied")
    except Exception as e:
        print("❌ Watermark failed — fallback:", e)
        basic_encode(input_path, output_path, cert_id)

# ============================================================
# FALLBACK ENCODE (NO WATERMARK)
# ============================================================
def basic_encode(input_path: str, output_path: str, cert_id: str):

    command = [
        "ffmpeg",
        "-y",
        "-i", input_path,

        "-metadata", f"VeriFYD-CertID={cert_id}",
        "-metadata", "VeriFYD-Status=CertifiedAuthentic",

        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-c:a", "aac",
        "-b:a", "128k",

        output_path
    ]

    subprocess.run(command, check=True)

# ============================================================
# Upload + Certify Endpoint
# ============================================================
@app.post("/upload/")
async def upload_video(file: UploadFile = File(...)):

    # Save upload
    upload_id = str(uuid.uuid4())
    raw_path = os.path.join(UPLOAD_DIR, f"{upload_id}_{file.filename}")

    with open(raw_path, "wb") as buffer:
        buffer.write(await file.read())

    # Fingerprint
    fingerprint = generate_fingerprint(raw_path)

    # Certificate ID
    cert_id = str(uuid.uuid4())

    certified_filename = f"{cert_id}_VeriFYD.mp4"
    certified_path = os.path.join(CERTIFIED_DIR, certified_filename)

    # Apply watermark safely
    stamp_video(raw_path, certified_path, cert_id)

    # Store in DB
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()

    c.execute("""
        INSERT INTO certificates (id, filename, fingerprint, certified_file, created_at)
        VALUES (?, ?, ?, ?, ?)
    """, (
        cert_id,
        file.filename,
        fingerprint,
        certified_filename,
        datetime.utcnow().isoformat()
    ))

    conn.commit()
    conn.close()

    # Return response (UNCHANGED structure)
    return {
        "message": "✅ Video Certified Successfully by VeriFYD",
        "certificate_id": cert_id,
        "fingerprint": fingerprint,
        "certified_video_file": certified_filename,
        "verification_link": f"https://verifyd-backend.onrender.com/verify/{cert_id}"
    }

# ============================================================
# Verify Endpoint
# ============================================================
@app.get("/verify/{cert_id}")
def verify_video(cert_id: str):

    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()

    c.execute("SELECT * FROM certificates WHERE id=?", (cert_id,))
    row = c.fetchone()

    conn.close()

    if not row:
        return {"status": "❌ Certificate Not Found"}

    return {
        "status": "✅ Certified Authentic Video",
        "certificate_id": row[0],
        "original_filename": row[1],
        "fingerprint": row[2],
        "certified_file": row[3],
        "issued_at": row[4]
    }

# ============================================================
# Home
# ============================================================
@app.get("/")
def home():
    return {
        "service": "VeriFYD Certification Authority",
        "message": "Upload videos at /docs to certify authenticity."
    }


































