from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import os, uuid, shutil, sqlite3
from datetime import datetime

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "videos"
CERT_DIR = "certified"
DB = "certificates.db"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CERT_DIR, exist_ok=True)

# DB
def init_db():
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS certs(
        id TEXT,
        filename TEXT,
        created TEXT
    )
    """)
    conn.commit()
    conn.close()

init_db()

@app.get("/")
def home():
    return {"status":"VFVid API LIVE"}

# UPLOAD
@app.post("/upload/")
async def upload(file: UploadFile = File(...), email: str = Form(None)):

    cert_id = str(uuid.uuid4())
    raw_path = f"{UPLOAD_DIR}/{cert_id}_{file.filename}"

    with open(raw_path,"wb") as buffer:
        buffer.write(await file.read())

    # JUST COPY (no ffmpeg)
    out_name = f"{cert_id}.mp4"
    out_path = f"{CERT_DIR}/{out_name}"
    shutil.copy(raw_path, out_path)

    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute("INSERT INTO certs VALUES (?,?,?)",
              (cert_id,file.filename,datetime.utcnow().isoformat()))
    conn.commit()
    conn.close()

    base = "https://verifyd-backend.onrender.com"

    return {
        "status":"CERTIFIED",
        "certificate_id":cert_id,
        "verify":f"{base}/verify/{cert_id}",
        "download":f"{base}/download/{cert_id}"
    }

# VERIFY
@app.get("/verify/{cid}")
def verify(cid:str):
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute("SELECT * FROM certs WHERE id=?",(cid,))
    row = c.fetchone()
    conn.close()

    if not row:
        return {"error":"not found"}

    return {"status":"VALID","certificate_id":cid}

# DOWNLOAD
@app.get("/download/{cid}")
def download(cid:str):
    path = f"{CERT_DIR}/{cid}.mp4"
    if not os.path.exists(path):
        return {"error":"not found"}
    return FileResponse(path, media_type="video/mp4")





























