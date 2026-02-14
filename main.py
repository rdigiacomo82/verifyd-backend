from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import os, uuid, subprocess

print("🚨 VERIFYD BUILD LOADED 🚨")

BASE_URL = "https://verifyd-backend.onrender.com"

UPLOAD_DIR = "videos"
CERT_DIR = "certified"
TMP_DIR = "tmp"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CERT_DIR, exist_ok=True)
os.makedirs(TMP_DIR, exist_ok=True)

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
    return {"status": "VeriFYD LIVE"}

# ==================================================
# 🔊 FINAL AUDIO-SAFE STAMP FUNCTION
# ==================================================
def stamp_video(input_path, output_path, cert_id):

    print("\n===========================")
    print("STAMP START")
    print("INPUT:", input_path)
    print("===========================\n")

    # TEMP REMUX FILE
    temp_mux = f"{TMP_DIR}/{cert_id}_mux.mp4"

    # --------------------------------------------------
    # STEP 1: REMUX (COPY AUDIO EXACTLY)
    # --------------------------------------------------
    remux_cmd = [
        "ffmpeg",
        "-y",
        "-i", input_path,
        "-map", "0",
        "-c", "copy",
        "-movflags", "+faststart",
        temp_mux
    ]

    print("REMUX CMD:", " ".join(remux_cmd))
    subprocess.run(remux_cmd)

    # --------------------------------------------------
    # STEP 2: APPLY WATERMARK (STATIC TEXT)
    # --------------------------------------------------
    watermark = (
        f"drawtext=text='VeriFYD ID {cert_id}':"
        "x=w-tw-20:y=h-th-20:"
        "fontsize=16:"
        "fontcolor=white@0.85"
    )

    final_cmd = [
        "ffmpeg",
        "-y",
        "-i", temp_mux,
        "-vf", watermark,

        "-map", "0:v",
        "-map", "0:a?",

        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",

        "-c:a", "aac",
        "-b:a", "192k",
        "-ac", "2",
        "-ar", "44100",
        "-af", "aresample=async=1",

        "-shortest",
        "-movflags", "+faststart",

        output_path
    ]

    print("FINAL CMD:", " ".join(final_cmd))
    subprocess.run(final_cmd)

    print("STAMP COMPLETE:", output_path)

# ==================================================
# 📤 UPLOAD ROUTE
# ==================================================
@app.post("/upload/")
async def upload(file: UploadFile = File(...), email: str = Form(...)):

    print("\n📤 UPLOAD HIT\n")

    cert_id = str(uuid.uuid4())
    raw_path = f"{UPLOAD_DIR}/{cert_id}_{file.filename}"

    with open(raw_path, "wb") as buffer:
        buffer.write(await file.read())

    print("FILE SAVED:", raw_path)

    certified_path = f"{CERT_DIR}/{cert_id}.mp4"

    stamp_video(raw_path, certified_path, cert_id)

    print("UPLOAD COMPLETE\n")

    return {
        "status": "CERTIFIED REAL VIDEO",
        "certificate_id": cert_id,
        "download_url": f"{BASE_URL}/download/{cert_id}"
    }

# ==================================================
# 📥 DOWNLOAD
# ==================================================
@app.get("/download/{cid}")
def download(cid: str):
    path = f"{CERT_DIR}/{cid}.mp4"
    print("DOWNLOAD:", path)
    return FileResponse(path, media_type="video/mp4")


























































