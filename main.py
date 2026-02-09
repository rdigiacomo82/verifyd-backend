from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import hashlib, os, uuid, subprocess, sqlite3
from datetime import datetime

app = FastAPI(title="VeriFYD API")

# ---------------------------
# CORS
# ---------------------------
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
LOGO = "assets/logo.png"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CERT_DIR, exist_ok=True)

# ---------------------------
# DB
# ---------------------------
def init_db():
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS certs(
        id TEXT,
        filename TEXT,
        fingerprint TEXT,
        created TEXT
    )
    """)
    conn.commit()
    conn.close()

init_db()

# ---------------------------
# FINGERPRINT
# ---------------------------
def fingerprint(path):
    h = hashlib.sha256()
    with open(path,"rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

# ---------------------------
# WATERMARK
# ---------------------------
def watermark(input_path, output_path, cid):

    text = f"VeriFYD CERTIFIED {cid[:8]}"

    cmd = [
        "ffmpeg",
        "-y",
        "-i", input_path,
        "-i", LOGO,
        "-filter_complex",
        f"[0:v][1:v] overlay=20:20, drawtext=text='{text}':x=20:y=110:fontsize=18:fontcolor=white:box=1:boxcolor=black@0.4",
        "-c:v","libx264",
        "-preset","fast",
        "-crf","23",
        "-c:a","aac",
        output_path
    ]

    subprocess.run(cmd, check=True)

# ---------------------------
@app.get("/")
def home():
    return {"status":"VeriFYD LIVE"}

# ---------------------------
# UPLOAD
# ---------------------------
@app.post("/upload/")
async def upload(file: UploadFile = File(...), email: str = Form(None)):

    cid = str(uuid.uuid4())

    raw_path = f"{UPLOAD_DIR}/{cid}_{file.filename}"
    with open(raw_path,"wb") as buffer:
        buffer.write(await file.read())

    # fingerprint
    fp = fingerprint(raw_path)

    # output path
    out_path = f"{CERT_DIR}/{cid}.mp4"

    # watermark
    watermark(raw_path, out_path, cid)

    # store
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute("INSERT INTO certs VALUES (?,?,?,?)",
              (cid,file.filename,fp,datetime.utcnow().isoformat()))
    conn.commit()
    conn.close()

    base = "https://verifyd-backend.onrender.com"

    return {
        "status":"CERTIFIED",
        "certificate_id":cid,
        "verify":f"{base}/verify/{cid}",
        "download":f"{base}/download/{cid}",
        "stream":f"{base}/stream/{cid}"
    }

# ---------------------------
# VERIFY
# ---------------------------
@app.get("/verify/{cid}")
def verify(cid:str):
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute("SELECT * FROM certs WHERE id=?",(cid,))
    row = c.fetchone()
    conn.close()

    if not row:
        return {"status":"NOT FOUND"}

    return {
        "status":"VALID",
        "certificate_id":cid,
        "fingerprint":row[2],
        "issued":row[3]
    }

# ---------------------------
# DOWNLOAD
# ---------------------------
@app.get("/download/{cid}")
def download(cid:str):
    path = f"{CERT_DIR}/{cid}.mp4"
    if not os.path.exists(path):
        return {"error":"not found"}
    return FileResponse(path, media_type="video/mp4")

# ---------------------------
# STREAM
# ---------------------------
@app.get("/stream/{cid}")
def stream(cid:str):
    path = f"{CERT_DIR}/{cid}.mp4"
    if not os.path.exists(path):
        return {"error":"not found"}
    return FileResponse(path, media_type="video/mp4")






























