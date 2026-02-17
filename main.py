from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import os, uuid, subprocess, tempfile, requests

from detector import detect_ai
from external_detector import external_ai_score

BASE_URL = "https://verifyd-backend.onrender.com"

UPLOAD_DIR = "videos"
CERT_DIR = "certified"
ASSETS_DIR = "assets"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CERT_DIR, exist_ok=True)

LOGO_PATH = os.path.join(ASSETS_DIR, "logo.png")

app = FastAPI(title="VeriFYD 6.0")

# ---------------------------------------------------
# CORS
# ---------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------
# HEALTH
# ---------------------------------------------------
@app.get("/health")
def health():
    return {"success": True, "data": "ok"}

# ---------------------------------------------------
# AI SCORING
# ---------------------------------------------------
def combined_score(path):
    try:
        primary = detect_ai(path)
        secondary = external_ai_score(path)
        score = int((primary * 0.7) + (secondary * 0.3))
        return score
    except:
        return 50

# ---------------------------------------------------
# VIDEO STAMP WITH LOGO + AUDIO PRESERVED
# ---------------------------------------------------
def stamp_video(input_path, output_path, cert_id):
    if not os.path.exists(LOGO_PATH):
        raise RuntimeError("Logo missing at assets/logo.png")

    vf = (
        f"movie={LOGO_PATH}[logo];"
        "[in][logo]overlay=10:10[out]"
    )

    cmd = [
        "ffmpeg","-y",
        "-i", input_path,
        "-vf", vf,
        "-map","0:v:0",
        "-map","0:a?",
        "-c:v","libx264",
        "-preset","fast",
        "-crf","23",
        "-c:a","copy",
        "-movflags","+faststart",
        output_path
    ]

    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if r.returncode != 0:
        raise RuntimeError(r.stderr.decode()[-300:])

# ---------------------------------------------------
# UPLOAD
# ---------------------------------------------------
@app.post("/upload/")
async def upload(file: UploadFile = File(...), email: str = Form(...)):

    cid = str(uuid.uuid4())
    raw_path = f"{UPLOAD_DIR}/{cid}_{file.filename}"

    with open(raw_path, "wb") as f:
        f.write(await file.read())

    score = combined_score(raw_path)

    # 🔴 AI DETECTED
    if score < 50:
        return {
            "success": True,
            "data": {
                "status": "AI DETECTED",
                "authenticity_score": score,
                "color": "red"
            }
        }

    # 🔵 UNDETERMINED
    if score < 70:
        return {
            "success": True,
            "data": {
                "status": "UNDETERMINED",
                "authenticity_score": score,
                "color": "blue"
            }
        }

    # 🟢 REAL → STAMP
    out_path = f"{CERT_DIR}/{cid}.mp4"
    stamp_video(raw_path, out_path, cid)

    return {
        "success": True,
        "data": {
            "status": "REAL VIDEO VERIFIED",
            "authenticity_score": score,
            "certificate_id": cid,
            "download_url": f"{BASE_URL}/download/{cid}",
            "verify_url": f"{BASE_URL}/verify/{cid}",
            "color": "green"
        }
    }

# ---------------------------------------------------
# DOWNLOAD
# ---------------------------------------------------
@app.get("/download/{cid}")
def download(cid: str):
    path = f"{CERT_DIR}/{cid}.mp4"
    if not os.path.exists(path):
        return {"success": False, "error": "not found"}
    return FileResponse(path, media_type="video/mp4")

# ---------------------------------------------------
# VERIFY PAGE
# ---------------------------------------------------
@app.get("/verify/{cid}")
def verify(cid: str):
    return {
        "success": True,
        "data": {
            "certificate_id": cid,
            "status": "Verified by VeriFYD"
        }
    }

# ---------------------------------------------------
# ANALYZE LINK (POST ONLY)
# ---------------------------------------------------
@app.post("/analyze-link/")
async def analyze_link(email: str = Form(...), video_url: str = Form(...)):

    try:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")

        r = requests.get(video_url, stream=True, timeout=20)
        for chunk in r.iter_content(1024):
            tmp.write(chunk)

        tmp.close()

        score = combined_score(tmp.name)
        os.unlink(tmp.name)

        if score < 50:
            status = "AI DETECTED"
            color = "red"
        elif score < 70:
            status = "UNDETERMINED"
            color = "blue"
        else:
            status = "REAL VIDEO VERIFIED"
            color = "green"

        return {
            "success": True,
            "data": {
                "status": status,
                "authenticity_score": score,
                "color": color
            }
        }

    except Exception as e:
        return {"success": False, "error": str(e)}








