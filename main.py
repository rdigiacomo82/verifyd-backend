from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import os, uuid, shutil, sqlite3, subprocess
from datetime import datetime

app = FastAPI()

# ---------------- CORS ----------------
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

# ---------------- DB ----------------
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
    return {"status":"API LIVE"}

# ---------------- SAFE WATERMARK ----------------
def try_watermark(input_path, output_path, cid):
    try:
        if not os.path.exists(LOGO):
            shutil.copy(input_path, output_path)
            return

        text = f"VeriFYD {cid[:8]}"

        cmd = [
            "ffmpeg",
            "-y",
            "-i", input_path,
            "-i", LOGO,
            "-filter_complex",
            f"[0:v][1:v] overlay=20:20, drawtext=text='{text}':x=20:y=110:fontsize=18:fontcolor=white:box=1:boxcolor=black@0.4",
            "-c:v","libx264",
            "-preset","ultrafast",
            "-crf","23",
            "-c:a","copy",
            output_path
        ]

        subprocess.run(cmd, check=True)

    except Exception as e:
        print("Watermark failed:", e)
        shutil.copy(input_path, output_path)

# ---------------- UPLOAD ----------------
@app.post("/upload/")
async def upload(file: UploadFile = File(...), email: str = Form(None)):

    try:
        cid = str(uuid.uuid4())
        raw_path = f"{UPLOAD_DIR}/{cid}_{file.filename}"

        with open(raw_path,"wb") as buffer:
            buffer.write(await file.read())

        out_path = f"{CERT_DIR}/{cid}.mp4"

        # NEVER crash here
        try_watermark(raw_path, out_path, cid)

        conn = sqlite3.connect(DB)
        c = conn.cursor()
        c.execute("INSERT INTO certs VALUES (?,?,?)",
                  (cid,file.filename,datetime.utcnow().isoformat()))
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

    except Exception as e:
        print("UPLOAD CRASH:", e)
        return {"status":"ERROR","message":"Upload failed"}

# ---------------- VERIFY ----------------
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

# ---------------- DOWNLOAD ----------------
@app.get("/download/{cid}")
def download(cid:str):
































