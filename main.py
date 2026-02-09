from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import hashlib
import os
import uuid
import subprocess
import sqlite3
from datetime import datetime

app = FastAPI(title="VeriFYD Video Certification Authority")

# --- CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "videos"
CERTIFIED_DIR = "certified"
DB_FILE = "certificates.db"
LOGO_PATH = "assets/logo.png"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CERTIFIED_DIR, exist_ok=True)

# ============================================================
# DATABASE
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
# FINGERPRINT
# ============================================================
def fingerprint(file_path):
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            sha256.update(chunk)
    return sha256.hexdigest()

# ============================================================
# WATERMARK — SMALLER + HIGHER + LEFT
# ============================================================
def stamp_video(input_path, output_path, cert_id):

    if not os.path.exists(LOGO_PATH):
        command = [
            "ffmpeg","-y","-i",input_path,
            "-c:v","libx264","-preset","fast","-crf","23",
            "-c:a","aac","-b:a","128k",
            output_path
        ]
        subprocess.run(command, check=True)
        return

    text = f"VeriFYD {cert_id[:6]}"

    command = [
        "ffmpeg","-y",
        "-i", input_path,
        "-i", LOGO_PATH,
        "-filter_complex",
        # LOGO ALMOST TOUCHING TOP-LEFT EDGE
        f"[0:v][1:v] overlay=0:0,"
        # SMALLER TEXT VERY CLOSE UNDER LOGO
        f"drawtext=text='{text}':x=6:y=32:fontsize=14:fontcolor=white:box=1:boxcolor=black@0.35",
        "-c:v","libx264","-preset","fast","-crf","23",
        "-c:a","aac","-b:a","128k",
        output_path
    ]

    try:
        subprocess.run(command, check=True)
    except:
        command = [
            "ffmpeg","-y","-i",input_path,
            "-c:v","libx264","-preset","fast","-crf","23",
            "-c:a","aac","-b:a","128k",
            output_path
        ]
        subprocess.run(command, check=True)

# ============================================================
# UPLOAD
# ============================================================
@app.post("/upload/")
async def upload(file: UploadFile = File(...)):

    cert_id = str(uuid.uuid4())

    raw_path = f"{UPLOAD_DIR}/{cert_id}_{file.filename}"
    with open(raw_path,"wb") as buffer:
        buffer.write(await file.read())

    fp = fingerprint(raw_path)

    certified_name = f"{cert_id}_VeriFYD.mp4"
    certified_path = f"{CERTIFIED_DIR}/{certified_name}"

    stamp_video(raw_path, certified_path, cert_id)

    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("INSERT INTO certificates VALUES (?,?,?,?,?)",
              (cert_id,file.filename,fp,certified_name,datetime.utcnow().isoformat()))
    conn.commit()
    conn.close()

    base = "https://verifyd-backend.onrender.com"

    return {
        "status": "CERTIFIED",
        "certificate_id": cert_id,
        "verify_url": f"{base}/verify/{cert_id}",
        "download_url": f"{base}/download/{cert_id}",
        "fingerprint": fp
    }

# ============================================================
# VERIFY
# ============================================================
@app.get("/verify/{cid}")
def verify(cid:str):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT * FROM certificates WHERE id=?", (cid,))
    row = c.fetchone()
    conn.close()

    if not row:
        return {"status":"NOT FOUND"}

    return {
        "status":"VALID",
        "certificate_id":cid,
        "file":row[3],
        "issued_at":row[4]
    }

# ============================================================
# DOWNLOAD VIDEO
# ============================================================
@app.get("/download/{cid}")
def download(cid:str):
    path = f"{CERTIFIED_DIR}/{cid}_VeriFYD.mp4"
    if not os.path.exists(path):
        return {"error":"file not found"}
    return FileResponse(path, media_type="video/mp4", filename=f"VeriFYD_{cid}.mp4")

# ============================================================
@app.get("/")
def home():
    return {"status":"VeriFYD backend live"}



































