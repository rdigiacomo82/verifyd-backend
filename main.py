# ============================================================
# VeriFYD – STABLE CORE (Audio + Upload + Analyze Link FIXED)
# ============================================================

import os, uuid, subprocess, tempfile
from fastapi import FastAPI, UploadFile, File, Form, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
import requests

from detector import detect_ai
from external_detector import external_ai_score

BASE_URL = "https://verifyd-backend.onrender.com"

UPLOAD_DIR = "videos"
CERT_DIR   = "certified"
TMP_DIR    = "tmp"

for d in (UPLOAD_DIR, CERT_DIR, TMP_DIR):
    os.makedirs(d, exist_ok=True)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# DETECTION
# ============================================================

def analyze_video(path):

    primary = detect_ai(path)
    secondary = external_ai_score(path)

    ai_likelihood = int((primary * 0.7) + (secondary * 0.3))
    authenticity  = max(0, min(100, 100 - ai_likelihood))

    return {
        "authenticity_score": authenticity,
        "ai_likelihood": ai_likelihood,
        "primary": primary,
        "secondary": secondary
    }

# ============================================================
# SAFE STAMP VIDEO (NO MORE CRASHES)
# ============================================================

def stamp_video(input_path, output_path, cert_id):

    safe_id = cert_id[:8]

    vf = (
        f"drawtext=text='VeriFYD_{safe_id}':"
        "x=w-tw-20:y=h-th-20:"
        "fontsize=18:"
        "fontcolor=white@0.9:"
        "box=1:"
        "boxcolor=black@0.5:"
        "boxborderw=5"
    )

    cmd = [
        "ffmpeg","-y",
        "-i", input_path,
        "-vf", vf,

        # VIDEO
        "-c:v","libx264",
        "-preset","fast",
        "-crf","23",

        # AUDIO (ALWAYS RE-ENCODE)
        "-c:a","aac",
        "-b:a","192k",

        "-movflags","+faststart",
        output_path
    ]

    r = subprocess.run(cmd, capture_output=True, text=True)

    if r.returncode != 0:
        print("FFMPEG ERROR:", r.stderr)
        raise RuntimeError("Stamp failed")

# ============================================================
# UPLOAD
# ============================================================

@app.post("/upload/")
async def upload(file: UploadFile = File(...), email: str = Form(...)):

    cert_id = str(uuid.uuid4())
    raw_path = f"{UPLOAD_DIR}/{cert_id}_{file.filename}"

    with open(raw_path, "wb") as f:
        f.write(await file.read())

    detection = analyze_video(raw_path)
    score = detection["authenticity_score"]

    # TEMP: allow certification so we can test audio
    if score >= 50:
        status = "CERTIFIED"
        out_path = f"{CERT_DIR}/{cert_id}.mp4"
        stamp_video(raw_path, out_path, cert_id)
        download_url = f"{BASE_URL}/download/{cert_id}"
    else:
        status = "UNDER_REVIEW"
        download_url = None

    return {
        "status": status,
        "certificate_id": cert_id,
        "authenticity_score": score,
        "download_url": download_url
    }

# ============================================================
# DOWNLOAD
# ============================================================

@app.get("/download/{cid}")
def download(cid: str):

    path = f"{CERT_DIR}/{cid}.mp4"

    if not os.path.exists(path):
        raise HTTPException(404, "Not found")

    return FileResponse(path, media_type="video/mp4")

# ============================================================
# ANALYZE LINK (DIRECT MP4 ONLY FOR NOW)
# ============================================================

@app.api_route("/analyze-link/", methods=["GET","POST"])
async def analyze_link(request: Request):

    if request.method == "POST":
        form = await request.form()
        video_url = form.get("video_url")
    else:
        video_url = request.query_params.get("video_url")

    if not video_url:
        return {"error":"missing url"}

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")

    try:
        r = requests.get(video_url, stream=True, timeout=20)

        for chunk in r.iter_content(1024):
            tmp.write(chunk)

        tmp.close()

        detection = analyze_video(tmp.name)
        score = detection["authenticity_score"]

        result = "REAL VIDEO" if score >= 50 else "AI DETECTED"

        return HTMLResponse(f"""
        <html>
        <body style="background:black;color:white;text-align:center;padding-top:120px;font-family:Arial">
        <h1>{result}</h1>
        <h2>Authenticity Score: {score}</h2>
        </body>
        </html>
        """)

    except Exception as e:
        return {"error": str(e)}

    finally:
        try:
            os.unlink(tmp.name)
        except:
            pass

# ============================================================

@app.get("/")
def home():
    return {"status":"VeriFYD LIVE"}


