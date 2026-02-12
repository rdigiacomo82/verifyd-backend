from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import os, uuid, hashlib, sqlite3, shutil, requests
from datetime import datetime

from detector import detect_ai
from external_detector import external_ai_score

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

# ================= HASH =================

def fingerprint(path):
    h = hashlib.sha256()
    with open(path,"rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

# ================= SCORE =================

def combined_score(local_score, external_score):
    return int((local_score * 0.4) + ((100 - external_score) * 0.6))

# ================= UPLOAD =================

@app.post("/upload/")
async def upload(file: UploadFile = File(...), email: str = Form(...)):

    cert_id = str(uuid.uuid4())
    raw_path = f"{UPLOAD_DIR}/{cert_id}_{file.filename}"

    with open(raw_path,"wb") as buffer:
        buffer.write(await file.read())

    fp = fingerprint(raw_path)

    local_score = detect_ai(raw_path)
    external_score = external_ai_score(raw_path)
    final_score = combined_score(local_score, external_score)

    print("UPLOAD SCORE:", final_score)

    # 🔴 AGGRESSIVE BLOCK
    if final_score < 98:
        return {
            "status":"AI DETECTED",
            "authenticity_score":final_score,
            "certificate_id":None,
            "verify_url":None,
            "download_url":None
        }

    # 🟢 CERTIFY
    certified_path = f"{CERT_DIR}/{cert_id}.mp4"
    shutil.copy(raw_path, certified_path)

    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("""
    INSERT INTO certificates VALUES (?,?,?,?,?,?)
    """,(cert_id,file.filename,fp,final_score,"CERTIFIED",datetime.utcnow().isoformat()))
    conn.commit()
    conn.close()

    return {
        "status":"CERTIFIED REAL VIDEO",
        "certificate_id":cert_id,
        "authenticity_score":final_score,
        "verify_url":f"{BASE_URL}/verify/{cert_id}",
        "download_url":f"{BASE_URL}/download/{cert_id}"
    }

# ================= VERIFY =================

@app.get("/verify/{cid}")
def verify(cid:str):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT * FROM certificates WHERE id=?",(cid,))
    row = c.fetchone()
    conn.close()

    if not row:
        return {"status":"NOT FOUND"}

    return {
        "status":"VALID CERTIFICATE",
        "certificate_id":row[0],
        "score":row[3],
        "created":row[5]
    }

# ================= DOWNLOAD =================

@app.get("/download/{cid}")
def download(cid:str):
    path=f"{CERT_DIR}/{cid}.mp4"
    return FileResponse(path,media_type="video/mp4")

# ================= LINK ANALYSIS =================

@app.post("/analyze-link/")
async def analyze_link(email: str = Form(...), video_url: str = Form(...)):

    try:
        print("LINK RECEIVED:", video_url)

        temp_path = f"temp_{uuid.uuid4()}.mp4"

        # try downloading
        try:
            r = requests.get(video_url, stream=True, timeout=15)
            with open(temp_path, "wb") as f:
                for chunk in r.iter_content(1024):
                    f.write(chunk)
        except Exception as download_error:
            print("DOWNLOAD FAILED:", download_error)
            return {
                "status":"ERROR",
                "authenticity_score":0,
                "certificate_id":None,
                "verify_url":None,
                "download_url":None,
                "message":"Could not fetch video"
            }

        # run detection
        try:
            local_score = detect_ai(temp_path)
            external_score = external_ai_score(temp_path)
            final_score = combined_score(local_score, external_score)
        except Exception as detect_error:
            print("DETECTION FAILED:", detect_error)
            final_score = 0

        if os.path.exists(temp_path):
            os.remove(temp_path)

        print("LINK SCORE:", final_score)

        if final_score < 98:
            return {
                "status":"AI DETECTED",
                "authenticity_score":final_score,
                "certificate_id":None,
                "verify_url":None,
                "download_url":None
            }

        return {
            "status":"CERTIFIED REAL VIDEO",
            "authenticity_score":final_score,
            "certificate_id":"LINK_ANALYSIS",
            "verify_url":None,
            "download_url":None
        }

    except Exception as e:
        print("LINK CRASH:", str(e))
        return {
            "status":"ERROR",
            "authenticity_score":0,
            "certificate_id":None,
            "verify_url":None,
            "download_url":None,
            "message":str(e)
        }

# ================= HOME =================

@app.get("/")
def home():
    return {"status":"VeriFYD API LIVE"}











































