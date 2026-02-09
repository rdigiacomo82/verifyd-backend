from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import os, uuid, shutil, sqlite3, subprocess
from datetime import datetime

app = FastAPI()

# ============================================================
# CORS  (for vfvid.com)
# ============================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://vfvid.com",
        "https://www.vfvid.com"
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "videos"
CERT_DIR = "certified"
DB = "certificates.db"
LOGO = "assets/logo.png"
FONT = "assets/arial.ttf"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CERT_DIR, exist_ok=True)

# ============================================================
# DATABASE
# ============================================================
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

# ============================================================
@app.get("/")
def home():
    return {"status": "VFVid API LIVE"}

# ============================================================
# WATERMARK FUNCTION (SAFE)
# ============================================================
def try_watermark(input_path, output_path, cid):

    try:
        if not os.path.exists(LOGO):
            print("Logo missing, copying video")
            shutil.copy(input_path, output_path)
            return

        text = f"VeriFYD CERTIFIED {cid[:8]}"

        filter_string = (
            f"[1:v]scale=140:-1[logo];"
            f"[0:v][logo]overlay=20:20,"
            f"drawtext=fontfile={FONT}:"
            f"text='{text}':"
            f"x=20:y=170:"
            f"fontsize=22:"
            f"fontcolor=white:"
            f"box=1:boxcolor=black@0.4"
        )

        cmd = [
            "ffmpeg",
            "-y",
            "-i", input_path,
            "-i", LOGO,
            "-filter_complex", filter_string,
            "-codec:a", "copy",
            output_path
        ]

        subprocess.run(cmd, check=True)

    except Exception as e:
        print("Watermark failed:", e)
        shutil.copy(input_path, output_path)

# ============================================================
# UPLOAD
# ============================================================
@app.post("/upload/")
async def upload(file: UploadFile = File(...), email: str = Form(None)):

    cid = str(uuid.uuid4())
    raw_path = f"{UPLOAD_DIR}/{cid}_{file.filename}"

    with open(raw_path, "wb") as buffer:
        buffer.write(await file.read())

    out_path = f"{CERT_DIR}/{cid}.mp4"

    # Apply watermark safely
    try_watermark(raw_path, out_path, cid)

    # Store record
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute("INSERT INTO certs VALUES (?,?,?)",
              (cid, file.filename, datetime.utcnow().isoformat()))
    conn.commit()
    conn.close()

    base = "https://verifyd-backend.onrender.com"

    return {
        "status": "CERTIFIED",
        "certificate_id": cid,
        "verify": f"{base}/verify/{cid}",
        "download": f"{base}/download/{cid}",
        "stream": f"{base}/stream/{cid}"
    }

# ============================================================
# VERIFY
# ============================================================
@app.get("/verify/{cid}")
def verify(cid: str):
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute("SELECT * FROM certs WHERE id=?", (cid,))
    row = c.fetchone()
    conn.close()

    if not row:
        return {"error": "not found"}

    return {"status": "VALID", "certificate_id": cid}

# ============================================================
# DOWNLOAD
# ============================================================
@app.get("/download/{cid}")
def download(cid: str):
    path = f"{CERT_DIR}/{cid}.mp4"
    if not os.path.exists(path):
        return {"error": "not found"}
    return FileResponse(path, media_type="video/mp4")

# ============================================================
# STREAM
# ============================================================
@app.get("/stream/{cid}")
def stream(cid: str):
    path = f"{CERT_DIR}/{cid}.mp4"
    if not os.path.exists(path):
        return {"error": "not found"}
    return FileResponse(path, media_type="video/mp4")

































