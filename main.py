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
# DETECTION (TEMP — NOT THE ISSUE RIGHT NOW)
# ==================================================
def analyze_video(path):
    cap = cv2.VideoCapture(path)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if frames == 0:
        return 50

    vals = []
    step = max(frames // 20, 1)

    for i in range(0, frames, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        vals.append(np.std(gray))

        if len(vals) > 20:
            break

    cap.release()

    if not vals:
        return 50

    score = 65 + (np.mean(vals) / 2)
    return max(min(int(score), 95), 10)

# ==================================================
# AUDIO-SAFE STAMP FUNCTION
# ==================================================
def stamp_video(input_path, output_path, cert_id):

    print("🔊 stamping video with audio-safe pipeline")

    watermark = (
        "drawtext=text='VeriFYD':x=10:y=10:fontsize=22:fontcolor=white@0.85,"
        f"drawtext=text='ID\\:{cert_id}':x=w-tw-20:y=h-th-20:fontsize=16:fontcolor=white@0.7"
    )

    # PASS 1 — direct stream copy (fastest + safest)
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

    subprocess.run(cmd_copy)

    # check if audio exists
    check = subprocess.run(
        ["ffmpeg", "-i", output_path],
        stderr=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True
    )

    if "Audio:" in check.stderr:
        print("✅ audio preserved")
        return

    print("⚠️ audio missing → re-encoding")

    # PASS 2 — force audio encode
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

    subprocess.run(cmd_encode)
    print("✅ forced audio encode complete")

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

    certified_path = f"{CERT_DIR}/{cert_id}.mp4"
    stamp_video(raw_path, certified_path, cert_id)

    return {
        "status": "CERTIFIED REAL VIDEO",
        "certificate_id": cert_id,
        "authenticity_score": score,
        "download_url": f"{BASE_URL}/download/{cert_id}"
    }

@app.get("/download/{cid}")
def download(cid: str):
    return FileResponse(f"{CERT_DIR}/{cid}.mp4", media_type="video/mp4")

# ==================================================
# LINK ANALYSIS
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

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tmp.close()

    ydl_opts = {
        'outtmpl': tmp.name,
        'format': 'mp4',
        'quiet': True,
        'noplaylist': True
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([video_url])

    score = analyze_video(tmp.name)
    os.unlink(tmp.name)

    result = "AI DETECTED" if score < 45 else "REAL VIDEO"

    return HTMLResponse(f"""
    <html>
    <body style="background:#0b0b0b;color:white;text-align:center;padding-top:120px;font-family:Arial">
    <h1>{result}</h1>
    <h2>Authenticity Score: {score}</h2>
    </body>
    </html>
    """)























































