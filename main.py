from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import os, uuid, subprocess, time

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
# STAMP FUNCTION WITH STRICT ERROR CHECKING
# ==================================================
def stamp_video(input_path, output_path, cert_id):

    print("\n==== STAMP START ====")
    print("INPUT:", input_path)

    temp_mux = f"{TMP_DIR}/{cert_id}_mux.mp4"

    # --- REMUX ---
    remux_cmd = [
        "ffmpeg",
        "-y",
        "-i", input_path,
        "-map", "0",
        "-c", "copy",
        temp_mux
    ]

    remux = subprocess.run(remux_cmd, capture_output=True, text=True)
    print(remux.stderr)

    # --- FINAL ENCODE ---
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
        "-af", "aresample=async=1",
        output_path
    ]

    final = subprocess.run(final_cmd, capture_output=True, text=True)
    print(final.stderr)

    print("==== STAMP END ====")

    if not os.path.exists(output_path):
        print("❌ OUTPUT VIDEO NOT CREATED")
        raise RuntimeError("FFmpeg failed to create output")

# ==================================================
# UPLOAD
# ==================================================
@app.post("/upload/")
async def upload(file: UploadFile = File(...), email: str = Form(...)):

    print("UPLOAD HIT")

    cert_id = str(uuid.uuid4())
    raw_path = f"{UPLOAD_DIR}/{cert_id}_{file.filename}"

    with open(raw_path, "wb") as buffer:
        content = await file.read()
        buffer.write(content)
        buffer.flush()
        os.fsync(buffer.fileno())

    time.sleep(1.5)

    certified_path = f"{CERT_DIR}/{cert_id}.mp4"

    stamp_video(raw_path, certified_path, cert_id)

    return {
        "status": "CERTIFIED REAL VIDEO",
        "certificate_id": cert_id,
        "download_url": f"{BASE_URL}/download/{cert_id}"
    }

@app.get("/download/{cid}")
def download(cid: str):
    path = f"{CERT_DIR}/{cid}.mp4"
    return FileResponse(path, media_type="video/mp4")




























































