# ============================================================
#  VeriFYD — main.py
#
#  FastAPI application entry point.
#
#  Detection pipeline:
#    upload / analyze-link
#      → clip_first_10_seconds()
#        → run_detection()          (detection.py)
#          → detect_ai()            (detector.py)
#            → authenticity, label, detail
#              → certify / respond
# ============================================================

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
import os, uuid, subprocess, requests, tempfile, logging

from detection import run_detection   # returns (authenticity, label, detail)

log = logging.getLogger("verifyd.main")

app = FastAPI(title="VeriFYD")

BASE_URL = "https://verifyd-backend.onrender.com"

UPLOAD_DIR = "videos"
CERT_DIR   = "certified"
TMP_DIR    = "tmp"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CERT_DIR,   exist_ok=True)
os.makedirs(TMP_DIR,    exist_ok=True)

# ─────────────────────────────────────────────
#  Label → UI display mapping
#  Keep presentation logic here, detection
#  thresholds stay in detection.py.
# ─────────────────────────────────────────────
LABEL_UI = {
    "REAL":          ("REAL VIDEO VERIFIED", "green",  True),
    "UNDETERMINED":  ("VIDEO UNDETERMINED",  "blue",   False),
    "AI":            ("AI DETECTED",         "red",    False),
}

# ─────────────────────────────────────────────
#  CORS
# ─────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
#  Health
# ─────────────────────────────────────────────
@app.get("/")
def home():
    return {"status": "VeriFYD API LIVE"}

@app.get("/health")
def health():
    return {"status": "ok"}

# ─────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────
def clip_first_10_seconds(input_path: str) -> str:
    """Return path to a 10-second clip of input_path (stream-copy, fast)."""
    clipped = f"{TMP_DIR}/{uuid.uuid4()}.mp4"
    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-t", "10",
        "-c", "copy",
        clipped,
    ]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return clipped


def stamp_video(input_path: str, output_path: str, cert_id: str) -> None:
    """Burn VeriFYD watermark + certificate ID into the video."""
    vf = (
        "drawtext=text='VeriFYD':x=10:y=10:fontsize=24:"
        "fontcolor=white@0.85:box=1:boxcolor=black@0.4:boxborderw=4,"
        f"drawtext=text='ID\\:{cert_id}':x=w-tw-20:y=h-th-20:fontsize=16:"
        "fontcolor=white@0.85:box=1:boxcolor=black@0.4:boxborderw=4"
    )
    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-vf", vf,
        "-map", "0:v:0",
        "-map", "0:a?",
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-c:a", "copy",
        "-movflags", "+faststart",
        output_path,
    ]
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if r.returncode != 0:
        log.error("ffmpeg stamp failed: %s", r.stderr.decode())
        raise RuntimeError("FFMPEG stamp failed")


def _run_analysis(source_path: str) -> tuple:
    """
    Clip → detect → return (authenticity, label, detail, ui_text, color, certify).
    Cleans up the temp clip automatically.
    """
    clip_path = clip_first_10_seconds(source_path)
    try:
        authenticity, label, detail = run_detection(clip_path)
    finally:
        if os.path.exists(clip_path):
            os.remove(clip_path)

    ui_text, color, certify = LABEL_UI.get(label, ("UNKNOWN", "grey", False))
    log.info(
        "Analysis: label=%s authenticity=%d ai_score=%d",
        label, authenticity, detail["ai_score"],
    )
    return authenticity, label, detail, ui_text, color, certify


# ─────────────────────────────────────────────
#  Routes
# ─────────────────────────────────────────────
@app.post("/upload/")
async def upload(file: UploadFile = File(...), email: str = Form(...)):
    cid      = str(uuid.uuid4())
    raw_path = f"{UPLOAD_DIR}/{cid}_{file.filename}"

    with open(raw_path, "wb") as f:
        f.write(await file.read())

    try:
        authenticity, label, detail, ui_text, color, certify = _run_analysis(raw_path)
    except Exception as e:
        log.exception("Detection failed for %s", raw_path)
        return JSONResponse({"error": str(e)}, status_code=500)

    if certify:
        certified_path = f"{CERT_DIR}/{cid}.mp4"
        try:
            stamp_video(raw_path, certified_path, cid)
        except RuntimeError as e:
            log.error("Stamping failed: %s", e)
            return JSONResponse({"error": "Video certification failed"}, status_code=500)

        return {
            "status":             ui_text,
            "authenticity_score": authenticity,
            "certificate_id":     cid,
            "download_url":       f"{BASE_URL}/download/{cid}",
            "color":              color,
        }

    return {
        "status":             ui_text,
        "authenticity_score": authenticity,
        "color":              color,
    }


@app.get("/download/{cid}")
def download(cid: str):
    path = f"{CERT_DIR}/{cid}.mp4"
    if not os.path.exists(path):
        return JSONResponse({"error": "Certificate not found"}, status_code=404)
    return FileResponse(path, media_type="video/mp4")


@app.get("/analyze-link/", response_class=HTMLResponse)
def analyze_link(video_url: str):
    if not video_url.startswith("http"):
        return HTMLResponse("<h2>Invalid URL — must start with http</h2>", status_code=400)

    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tmp_path = tmp_file.name
    tmp_file.close()

    try:
        r = requests.get(video_url, stream=True, timeout=20)
        if r.status_code != 200:
            return HTMLResponse(f"<h2>Could not download video (HTTP {r.status_code})</h2>")

        with open(tmp_path, "wb") as f:
            for chunk in r.iter_content(8192):
                f.write(chunk)

        authenticity, label, detail, ui_text, color, _ = _run_analysis(tmp_path)

        html = f"""
        <html>
        <body style="background:black;color:white;text-align:center;
                     padding-top:120px;font-family:Arial">
          <h1 style="color:{color};font-size:48px">{ui_text}</h1>
          <h2>Authenticity Score: {authenticity}/100</h2>
          <p style="color:#aaa">AI Signal Score: {detail['ai_score']}/100</p>
          <p>Analyzed by VeriFYD</p>
        </body>
        </html>
        """
        return HTMLResponse(html)

    except Exception as e:
        log.exception("analyze-link failed for %s", video_url)
        return HTMLResponse(f"<h2>Error: {str(e)}</h2>", status_code=500)

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
