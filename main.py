from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
import uuid, requests, tempfile, os

from config import BASE_URL, UPLOAD_DIR, CERT_DIR, TMP_DIR
from detection import run_detection
from video import stamp_video

app = FastAPI(title="VeriFYD 4.1")

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
# UPLOAD VIDEO
# ---------------------------------------------------
@app.post("/upload/")
async def upload(file: UploadFile = File(...), email: str = Form(...)):

    cid = str(uuid.uuid4())
    raw_path = f"{UPLOAD_DIR}/{cid}_{file.filename}"

    with open(raw_path, "wb") as f:
        f.write(await file.read())

    score, status = run_detection(raw_path)

    if status == "REAL":
        color = "green"
        text = "REAL VIDEO VERIFIED"
    elif status == "REVIEW":
        color = "blue"
        text = "FURTHER REVIEW NEEDED"
    else:
        color = "red"
        text = "AI DETECTED"

    if score >= 70:
        certified_path = f"{CERT_DIR}/{cid}.mp4"
        stamp_video(raw_path, certified_path, cid)

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
# ANALYZE VIDEO LINK
# ---------------------------------------------------
@app.get("/analyze-link/", response_class=HTMLResponse)
def analyze_link(video_url: str):

    if not video_url.startswith("http"):
        return HTMLResponse("<h2>Invalid video URL</h2>")

    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    path = temp.name

    try:
        r = requests.get(video_url, stream=True, timeout=20)

        if r.status_code != 200:
            return HTMLResponse("<h2>Could not download video</h2>")

        with open(path, "wb") as f:
            for chunk in r.iter_content(1024):
                f.write(chunk)

        score, status = run_detection(path)

        if status == "AI":
            color = "red"
            text = "AI DETECTED"
        elif status == "REVIEW":
            color = "blue"
            text = "FURTHER REVIEW NEEDED"
        else:
            color = "green"
            text = "REAL VIDEO VERIFIED"

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










