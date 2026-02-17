import os
import subprocess
import tempfile
from detector import detect_ai


def clip_first_10_seconds(input_path: str) -> str:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    clip_path = tmp.name

    cmd = [
        "ffmpeg",
        "-y",
        "-i", input_path,
        "-t", "10",
        "-c", "copy",
        clip_path
    ]

    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return clip_path


def run_detection(video_path: str):

    clip_path = clip_first_10_seconds(video_path)

    try:
        ai_score = detect_ai(clip_path)   # 0–100 AI likelihood
    except Exception:
        ai_score = 50
    finally:
        if os.path.exists(clip_path):
            os.remove(clip_path)

    # ---------------------------------------
    # CONVERT AI SCORE → REAL SCORE
    # ---------------------------------------
    real_score = 100 - ai_score

    # ---------------------------------------
    # CLASSIFICATION
    # ---------------------------------------
    if real_score >= 85:
        label = "REAL"
    elif real_score >= 60:
        label = "UNDETERMINED"
    else:
        label = "AI"

    return real_score, label



