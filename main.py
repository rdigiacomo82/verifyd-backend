from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
import os, uuid, subprocess, requests, tempfile

from detection import run_detection

app = FastAPI(title="VeriFYD AI Engine")

BASE_URL = "https://verifyd-backend.onrender.com"

UPLOAD_DIR = "videos"
CERT_DIR = "certified"
TMP_DIR = "tmp"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CERT_DIR, exist_ok=True)
os.makedirs(TMP_DIR, exist_ok=True)

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
@app.get("/")
def home():
    return {"status": "VeriFYD API LIVE"}

@app.get("/health")
def health():
    return {"status": "ok"}

# ---------------------------------------------------
# TRIM FIRST 10 SECONDS FOR ANALYSIS
# ---------------------------------------------------
def create_analysis_clip(input_path):

    temp_clip = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name

    cmd = [
        "ffmpeg","-y",
        "-i", input_path,
        "-t","10",
        "-c","copy",
        temp_clip
    ]

    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    return temp_clip

# ---------------------------------------------------
# VIDEO STAMP
# ---------------------------------------------------
def stamp_video(input_path, output_path, cert_id):

    vf = (
        "drawtext=text='VeriFYD':x=10:y=10:fontsize=24:"
        "fontcolor=white@0.85:box=1:boxcolor=black@0.4:boxborderw=4,"
        f"drawtext=text='ID\\:{cert_id}':x=w-tw-20:y=h-th-20:fontsize=16:"
        "fontcolor=white@0.85:box=1:boxcolor=black@0.4:boxborderw=4"
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
        raise RuntimeError(r.stderr.decode()[-400:])

# ---------------------------------------------------
# SCORE INTERPRETATION
# ---------------------------------------------------
def interpret_score(score):

    if score >= 85:
        return "REAL VIDEO VERIFIED", "green", "REAL"
    elif score >= 60:
        return "VIDEO UNDETERMINED", "blue", "UNDETERMINED"
    else:
        return "AI DETECTED", "red", "AI"

# ---------------------------------------------------
# UPLOAD VIDEO
# ---------------------------------------------------
@app.post("/upload/")
async def upload(file: UploadFile = File(...), email: str = Form(...)):

    cid = str(uuid.uuid4())
    raw_path = f"{UPLOAD_DIR}/{cid}_{file.filename}"

    with open(raw_path, "wb") as f:
        f.write(await file.read())

    # 🔥 Analyze first 10 seconds only
    clip = create_analysis_clip(raw_path)
    score = run_detection(clip)

    if os.path.exists(clip):
        os.remove(clip)

    text, color, label = interpret_score(score)

    # CERTIFY ONLY STRONG REAL
    if score >= 85:
        certified_path = f"{CERT_DIR}/{cid}.mp4"

        try:
            stamp_video(raw_path, certified_path, cid)
        except Exception as e:
            return {
                "status": "STAMP ERROR",
                "error": str(e),
                "authenticity_score": score,
                "color": "red"
            }

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

# ---------------------------------------------------
# DOWNLOAD
# ---------------------------------------------------
@app.get("/download/{cid}")
def download(cid: str):

    path = f"{CERT_DIR}/{cid}.mp4"

    if not os.path.exists(path):
        return JSONResponse({"error":"not found"})

    return FileResponse(path, media_type="video/mp4")

# ---------------------------------------------------
# ANALYZE LINK
# ---------------------------------------------------
@app.get("/analyze-link/", response_class=HTMLResponse)
def analyze_link(video_url: str):

    if not video_url.startswith("http"):
        return HTMLResponse("<h2>Invalid video URL</h2>")

    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    path = temp.name

    try:
        r = requests.get(video_url, stream=True, timeout=25)

        if r.status_code != 200:
            return HTMLResponse("<h2>Could not download video</h2>")

        with open(path, "wb") as f:
            for chunk in r.iter_content(1024):
                f.write(chunk)

        clip = create_analysis_clip(path)
        score = run_detection(clip)

        if os.path.exists(clip):
            os.remove(clip)

        text, color, label = interpret_score(score)

        html = f"""
        <html>
        <body style="background:black;color:white;text-align:center;padding-top:120px;font-family:Arial">
        <h1 style="color:{color};font-size:48px">{text}</h1>
        <h2>Authenticity Score: {score}</h2>
        <p>Analyzed by VeriFYD</p>
        </body>
        </html>
        """

        return HTMLResponse(html)

    except Exception as e:
        return HTMLResponse(f"<h2>Error: {str(e)}</h2>")

    finally:
        if os.path.exists(path):
            os.remove(path)

















