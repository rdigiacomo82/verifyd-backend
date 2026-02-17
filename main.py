from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
import os, uuid, subprocess, tempfile

from detection import run_detection

app = FastAPI(title="VeriFYD STABLE")

BASE_URL = "https://verifyd-backend.onrender.com"

UPLOAD_DIR = "videos"
CERT_DIR = "certified"
TMP_DIR = "tmp"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CERT_DIR, exist_ok=True)
os.makedirs(TMP_DIR, exist_ok=True)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"status": "VeriFYD LIVE"}

@app.get("/health")
def health():
    return {"status": "ok"}

# -----------------------------
# VIDEO STAMP
# -----------------------------
def stamp_video(input_path, output_path, cid):

    vf = (
        f"drawtext=text='VeriFYD':x=10:y=10:fontsize=24:fontcolor=white,"
        f"drawtext=text='ID\\:{cid}':x=w-tw-20:y=h-th-20:fontsize=16:fontcolor=white"
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

    return r.returncode == 0

# -----------------------------
# UPLOAD
# -----------------------------
@app.post("/upload/")
async def upload(file: UploadFile = File(...), email: str = Form(...)):

    cid = str(uuid.uuid4())
    raw_path = f"{UPLOAD_DIR}/{cid}_{file.filename}"

    with open(raw_path, "wb") as f:
        f.write(await file.read())

    # analyze first 10 seconds
    clip_path = f"{TMP_DIR}/{cid}_clip.mp4"

    subprocess.run([
        "ffmpeg","-y","-i", raw_path,
        "-t","10","-c","copy",clip_path
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    score, label = run_detection(clip_path)

    if score >= 85:
        color = "green"
        text = "REAL VIDEO VERIFIED"
    elif score >= 60:
        color = "blue"
        text = "VIDEO UNDETERMINED"
    else:
        color = "red"
        text = "AI DETECTED"

    # certify ONLY if real
    if score >= 85:

        certified_path = f"{CERT_DIR}/{cid}.mp4"
        success = stamp_video(raw_path, certified_path, cid)

        if success and os.path.exists(certified_path) and os.path.getsize(certified_path) > 10000:

            return {
                "status": text,
                "authenticity_score": score,
                "certificate_id": cid,
                "download_url": f"{BASE_URL}/download/{cid}",
                "color": color
            }

    return {
        "status": text,
        "authenticity_score": score,
        "color": color
    }

# -----------------------------
# DOWNLOAD
# -----------------------------
@app.get("/download/{cid}")
def download(cid: str):

    path = f"{CERT_DIR}/{cid}.mp4"

    if not os.path.exists(path):
        return JSONResponse({"error":"not found"})

    return FileResponse(path, media_type="video/mp4")

# -----------------------------
# ANALYZE LINK
# -----------------------------
@app.get("/analyze-link/", response_class=HTMLResponse)
def analyze_link(video_url: str):

    if not video_url.startswith("http"):
        return HTMLResponse("<h2>Invalid URL</h2>")

    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    path = temp.name

    try:
        import requests
        r = requests.get(video_url, stream=True, timeout=20)

        with open(path, "wb") as f:
            for chunk in r.iter_content(1024):
                f.write(chunk)

        score, label = run_detection(path)

        if score >= 85:
            color = "green"
            text = "REAL VIDEO VERIFIED"
        elif score >= 60:
            color = "blue"
            text = "VIDEO UNDETERMINED"
        else:
            color = "red"
            text = "AI DETECTED"

        return HTMLResponse(f"""
        <html>
        <body style="background:black;color:white;text-align:center;padding-top:120px;font-family:Arial">
        <h1 style="color:{color};font-size:48px">{text}</h1>
        <h2>Authenticity Score: {score}</h2>
        </body>
        </html>
        """)

    finally:
        if os.path.exists(path):
            os.remove(path)



















