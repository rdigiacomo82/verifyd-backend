from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

import os, uuid, subprocess, tempfile, requests

from detector import detect_ai
from external_detector import external_ai_score

BASE_URL = "https://verifyd-backend.onrender.com"

UPLOAD_DIR = "videos"
CERT_DIR = "certified"
TMP_DIR = "tmp"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CERT_DIR, exist_ok=True)
os.makedirs(TMP_DIR, exist_ok=True)

app = FastAPI(title="VeriFYD", version="3.0")

# --------------------------------------------------
# CORS
# --------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------
# HEALTH
# --------------------------------------------------
@app.get("/")
def home():
    return {"status": "VeriFYD API LIVE"}

@app.get("/health")
def health():
    return {"ok": True}

# ==================================================
# 🎬 STAMP VIDEO (AUDIO SAFE)
# ==================================================
def stamp_video(input_path, output_path, cert_id):

    draw = f"drawtext=text='VeriFYD {cert_id}':x=10:y=10:fontsize=18:fontcolor=white@0.9"

    cmd = [
        "ffmpeg","-y",
        "-i", input_path,

        "-vf", draw,

        "-map","0:v:0",
        "-map","0:a?",

        "-c:v","libx264",
        "-preset","fast",
        "-crf","23",

        "-c:a","aac",
        "-b:a","192k",

        "-movflags","+faststart",

        output_path
    ]

    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    if r.returncode != 0:
        raise RuntimeError(r.stderr.decode()[-500:])

# ==================================================
# 🧠 DETECTION ENGINE
# ==================================================
def run_detection(video_path):

    primary = detect_ai(video_path)
    secondary = external_ai_score(video_path)

    # fusion scoring
    score = int((primary * 0.7) + ((100 - secondary) * 0.3))
    score = max(0, min(score, 100))

    if score >= 70:
        status = "REAL"
    elif score >= 45:
        status = "REVIEW"
    else:
        status = "AI"

    return score, status

# ==================================================
# 📤 UPLOAD VIDEO
# ==================================================
@app.post("/upload/")
async def upload(file: UploadFile = File(...), email: str = Form(...)):

    cert_id = str(uuid.uuid4())
    raw_path = f"{UPLOAD_DIR}/{cert_id}_{file.filename}"

    with open(raw_path, "wb") as buffer:
        buffer.write(await file.read())

    score, status = run_detection(raw_path)

    if status == "AI":
        return {
            "status": "AI DETECTED",
            "authenticity_score": score
        }

    if status == "REVIEW":
        return {
            "status": "UNDER REVIEW",
            "authenticity_score": score
        }

    certified_path = f"{CERT_DIR}/{cert_id}.mp4"
    stamp_video(raw_path, certified_path, cert_id)

    return {
        "status": "CERTIFIED",
        "certificate_id": cert_id,
        "authenticity_score": score,
        "download_url": f"{BASE_URL}/download/{cert_id}"
    }

# ==================================================
# 📥 DOWNLOAD CERTIFIED VIDEO
# ==================================================
@app.get("/download/{cid}")
def download(cid: str):

    path = f"{CERT_DIR}/{cid}.mp4"

    if not os.path.exists(path):
        return JSONResponse({"error": "file not found"}, status_code=404)

    return FileResponse(path, media_type="video/mp4")

# ==================================================
# 📥 SOCIAL VIDEO DOWNLOAD (yt-dlp via python)
# ==================================================
def download_social_video(url):

    out = f"{TMP_DIR}/{uuid.uuid4()}.mp4"

    cmd = [
        "python", "-m", "yt_dlp",
        "-o", out,
        "-f", "mp4",
        url
    ]

    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    return out

# ==================================================
# 🔗 ANALYZE VIDEO LINK
# ==================================================
@app.api_route("/analyze-link/", methods=["GET","POST"])
async def analyze_link(request: Request):

    if request.method == "POST":
        form = await request.form()
        video_url = form.get("video_url")
    else:
        video_url = request.query_params.get("video_url")

    if not video_url:
        return {"error": "Missing video_url"}

    try:

        # SOCIAL LINKS
        if "tiktok" in video_url or "instagram" in video_url or "x.com" in video_url:
            path = download_social_video(video_url)

        # DIRECT VIDEO LINK
        else:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")

            r = requests.get(video_url, stream=True, timeout=20)
            for chunk in r.iter_content(1024):
                tmp.write(chunk)

            tmp.close()
            path = tmp.name

        score, status = run_detection(path)
        os.unlink(path)

        # COLOR OUTPUT
        if status == "REAL":
            color = "#00ff88"
            title = "REAL VIDEO VERIFIED"
        elif status == "REVIEW":
            color = "#00bfff"
            title = "FURTHER REVIEW NEEDED"
        else:
            color = "#ff3b3b"
            title = "AI DETECTED"

        html = f"""
        <html>
        <body style="background:#0b0b0b;color:white;text-align:center;padding-top:120px;font-family:Arial">
        <h1 style="color:{color}">{title}</h1>
        <h2>Authenticity Score: {score}</h2>
        <p>Analyzed by VeriFYD</p>
        </body>
        </html>
        """

        return HTMLResponse(html)

    except Exception as e:
        return {"error": str(e)}



