from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
import os, uuid, subprocess, tempfile
import cv2
import numpy as np
import yt_dlp

BASE_URL = "https://verifyd-backend.onrender.com"

UPLOAD_DIR = "videos"
CERT_DIR = "certified"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CERT_DIR, exist_ok=True)

app = FastAPI()

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

@app.get("/")
def home():
    return {"status": "VeriFYD API LIVE"}

# ==================================================
# DETECTION ENGINE (LIGHT + STABLE)
# ==================================================
def analyze_video(file_path):

    cap = cv2.VideoCapture(file_path)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if frames == 0:
        return 50

    noise_vals = []
    edge_vals = []

    step = max(frames // 25, 1)

    for i in range(0, frames, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        noise_vals.append(np.std(gray))
        edge_vals.append(cv2.Laplacian(gray, cv2.CV_64F).var())

        if len(noise_vals) >= 25:
            break

    cap.release()

    if not noise_vals:
        return 50

    noise = np.mean(noise_vals)
    edges = np.mean(edge_vals)

    score = 60

    if noise > 16:
        score += 15
    else:
        score -= 15

    if edges > 22:
        score += 15
    else:
        score -= 15

    return max(min(score, 95), 5)

# ==================================================
# VIDEO STAMP WITH PERMANENT AUDIO FIX
# ==================================================
def stamp_video(input_path, output_path, cert_id):

    print("🔊 stamping video:", input_path)

    # STATIC watermark text (no dynamic updates)
    watermark = (
        "drawtext=text='VeriFYD':x=10:y=10:fontsize=22:fontcolor=white@0.85,"
        f"drawtext=text='ID\\:{cert_id}':x=w-tw-20:y=h-th-20:fontsize=16:fontcolor=white@0.7"
    )

    # PASS 1 — TRY AUDIO COPY
    cmd_copy = [
        "ffmpeg",
        "-y",
        "-i", input_path,
        "-vf", watermark,
        "-map", "0:v:0",
        "-map", "0:a?",
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-c:a", "copy",
        "-movflags", "+faststart",
        output_path
    ]

    result = subprocess.run(cmd_copy, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    if os.path.exists(output_path) and os.path.getsize(output_path) > 50000:
        print("✅ audio copy succeeded")
        return

    print("⚠️ copy failed → forcing audio re-encode")

    # PASS 2 — FORCE AUDIO ENCODE
    cmd_encode = [
        "ffmpeg",
        "-y",
        "-i", input_path,
        "-vf", watermark,
        "-map", "0:v:0",
        "-map", "0:a?",
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-c:a", "aac",
        "-b:a", "192k",
        "-ac", "2",
        "-ar", "44100",
        "-shortest",
        "-movflags", "+faststart",
        output_path
    ]

    subprocess.run(cmd_encode, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print("✅ audio re-encode complete")

# ==================================================
# UPLOAD
# ==================================================
@app.post("/upload/")
async def upload(file: UploadFile = File(...), email: str = Form(...)):

    cert_id = str(uuid.uuid4())
    raw_path = f"{UPLOAD_DIR}/{cert_id}_{file.filename}"

    with open(raw_path, "wb") as buffer:
        buffer.write(await file.read())

    score = analyze_video(raw_path)

    if score < 50:
        return {
            "status": "AI DETECTED",
            "authenticity_score": score
        }

    certified_path = f"{CERT_DIR}/{cert_id}.mp4"
    stamp_video(raw_path, certified_path, cert_id)

    return {
        "status": "CERTIFIED REAL VIDEO",
        "certificate_id": cert_id,
        "authenticity_score": score,
        "download_url": f"{BASE_URL}/download/{cert_id}"
    }

# ==================================================
# DOWNLOAD
# ==================================================
@app.get("/download/{cid}")
def download(cid: str):
    return FileResponse(f"{CERT_DIR}/{cid}.mp4", media_type="video/mp4")

# ==================================================
# LINK ANALYZER (NO MORE 405)
# ==================================================
@app.api_route("/analyze-link", methods=["GET","POST"])
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
        temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp_video.close()

        ydl_opts = {
            'outtmpl': temp_video.name,
            'format': 'mp4',
            'quiet': True,
            'noplaylist': True
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])

        score = analyze_video(temp_video.name)
        os.unlink(temp_video.name)

        result = "AI DETECTED" if score < 50 else "REAL VIDEO"

        html = f"""
        <html>
        <body style="background:#0b0b0b;color:white;text-align:center;padding-top:120px;font-family:Arial">
        <h1>{result}</h1>
        <h2>Authenticity Score: {score}</h2>
        <p>Analyzed by VeriFYD</p>
        </body>
        </html>
        """

        return HTMLResponse(html)

    except Exception as e:
        return {"error": str(e)}






















































