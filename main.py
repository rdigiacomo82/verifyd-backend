from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import os, uuid, hashlib, sqlite3, shutil
from datetime import datetime

from detector import detect_ai  # <-- connects to detector.py

BASE_URL = "https://verifyd-backend.onrender.com"

UPLOAD_DIR = "videos"
CERT_DIR = "certified"
DB_FILE = "certificates.db"

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

# ================= DATABASE =================

def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS certificates(
        id TEXT PRIMARY KEY,
        filename TEXT,
        fingerprint TEXT,
        score INTEGER,
        status TEXT,
        created TEXT
    )
    """)
    conn.commit()
    conn.close()

init_db()

# ================= FINGERPRINT =================

def generate_fingerprint(path):
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            sha256.update(chunk)
    return sha256.hexdigest()

# ================= HOME =================

@app.get("/")
def home():
    return {"status":"VeriFYD AI detection server live"}

# ================= UPLOAD + DETECT =================

@app.post("/upload/")
async def upload(file: UploadFile = File(...), email: str = Form(...)):

    cert_id = str(uuid.uuid4())
    raw_path = f"{UPLOAD_DIR}/{cert_id}_{file.filename}"

    with open(raw_path, "wb") as buffer:
        buffer.write(await file.read())

    fingerprint = generate_fingerprint(raw_path)

    # 🔍 RUN AI DETECTOR
    score = detect_ai(raw_path)

    # ================= AI BLOCK =================
    if score < 50:
        return {
            "status": "AI GENERATED",
            "authenticity_score": score,
            "message": "Video appears AI generated and cannot be certified."
        }

    # ================= CERTIFY =================
    certified_path = f"{CERT_DIR}/{cert_id}.mp4"
    shutil.copy(raw_path, certified_path)

    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("""
    INSERT INTO certificates VALUES (?,?,?,?,?,?)
    """, (
        cert_id,
        file.filename,
        fingerprint,
        score,
        "CERTIFIED",
        datetime.utcnow().isoformat()
    ))
    conn.commit()
    conn.close()

    return {
        "status": "CERTIFIED REAL VIDEO",
        "certificate_id": cert_id,
        "authenticity_score": score,
        "verify_url": f"{BASE_URL}/verify/{cert_id}",
        "download_url": f"{BASE_URL}/download/{cert_id}"
    }

# ================= VERIFY =================

@app.get("/verify/{cid}")
def verify(cid: str):

    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT * FROM certificates WHERE id=?", (cid,))
    row = c.fetchone()
    conn.close()

    if not row:
        return {"status":"NOT FOUND"}

    return {
        "status":"VALID CERTIFICATE",
        "certificate_id": row[0],
        "filename": row[1],
        "fingerprint": row[2],
        "authenticity_score": row[3],
        "created": row[5]
    }

# ================= DOWNLOAD =================

@app.get("/download/{cid}")
def download(cid: str):
    path = f"{CERT_DIR}/{cid}.mp4"
    return FileResponse(path, media_type="video/mp4")

# ================= LINK ANALYSIS =================

@app.post("/analyze-link/")
async def analyze_link(email: str = Form(...), video_url: str = Form(...)):
    return {
        "status":"UNDER REVIEW",
        "message":"Link detection engine coming next phase"
    }









































