from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
import os, uuid, shutil

BASE_URL = "https://verifyd-backend.onrender.com"

UPLOAD_DIR = "videos"
CERT_DIR = "certified"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CERT_DIR, exist_ok=True)

app = FastAPI()

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
# HOME
# ---------------------------------------------------
@app.get("/")
def home():
    return {"status": "VeriFYD API LIVE"}

# ---------------------------------------------------
# UPLOAD VIDEO
# ---------------------------------------------------
@app.post("/upload/")
async def upload(file: UploadFile = File(...), email: str = Form(...)):

    cert_id = str(uuid.uuid4())
    raw_path = f"{UPLOAD_DIR}/{cert_id}_{file.filename}"

    with open(raw_path, "wb") as buffer:
        buffer.write(await file.read())

    certified_path = f"{CERT_DIR}/{cert_id}.mp4"
    shutil.copy(raw_path, certified_path)

    return {
        "status": "CERTIFIED",
        "certificate_id": cert_id,
        "download_url": f"{BASE_URL}/download/{cert_id}"
    }

# ---------------------------------------------------
# DOWNLOAD VIDEO
# ---------------------------------------------------
@app.get("/download/{cid}")
def download(cid: str):
    path = f"{CERT_DIR}/{cid}.mp4"
    return FileResponse(path, media_type="video/mp4")

# ---------------------------------------------------
# ANALYZE LINK (BYPASS HORIZONS — ALWAYS WORKS)
# ---------------------------------------------------
@app.api_route("/analyze-link/", methods=["GET","POST"])
async def analyze_link(request: Request):

    email = None
    video_url = None

    # ---- Get POST form ----
    if request.method == "POST":
        try:
            form = await request.form()
            email = form.get("email")
            video_url = form.get("video_url")
        except:
            pass

    # ---- Get GET params ----
    if request.method == "GET":
        email = request.query_params.get("email")
        video_url = request.query_params.get("video_url")

    print("Analyze request:", email, video_url)

    # ------------------------------------------------
    # TEMP AI DETECTION RESULT (STABLE PLACEHOLDER)
    # ------------------------------------------------
    status = "AI DETECTED"
    score = 78

    # ------------------------------------------------
    # IF request expects JSON → return JSON
    # ------------------------------------------------
    if "application/json" in request.headers.get("accept",""):
        return JSONResponse({
            "status": status,
            "ai_score": score,
            "success": True,
            "data": {
                "status": status,
                "authenticity_score": score
            }
        })

    # ------------------------------------------------
    # OTHERWISE → RETURN RESULT PAGE
    # ------------------------------------------------
    html = f"""
    <html>
    <head>
        <title>VeriFYD Result</title>
        <style>
            body {{
                font-family: Arial;
                background: #0b0b0b;
                color: white;
                text-align: center;
                padding-top: 120px;
            }}
            .box {{
                background: #111;
                padding: 50px;
                border-radius: 12px;
                display: inline-block;
                box-shadow: 0 0 30px rgba(0,255,140,0.2);
            }}
            h1 {{
                color: #00ff9c;
                font-size: 42px;
            }}
            h2 {{
                font-size: 28px;
                margin-top: 20px;
            }}
            p {{
                margin-top: 20px;
                color: #aaa;
            }}
        </style>
    </head>
    <body>
        <div class="box">
            <h1>{status}</h1>
            <h2>Authenticity Score: {score}</h2>
            <p>Analyzed by VeriFYD</p>
        </div>
    </body>
    </html>
    """

    return HTMLResponse(content=html)














































