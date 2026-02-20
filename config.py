# ============================================================
#  VeriFYD — config.py
# ============================================================

import os

# ── URLs ─────────────────────────────────────────────────────
BASE_URL = os.environ.get(
    "BASE_URL",
    "https://verifyd-backend.onrender.com"
)

# ── Directory layout ─────────────────────────────────────────
# DATA_ROOT points to Render's persistent disk when deployed,
# falls back to local project directory for development.
DATA_ROOT  = os.environ.get("DATA_ROOT", ".")

UPLOAD_DIR = os.path.join(DATA_ROOT, "videos")
CERT_DIR   = os.path.join(DATA_ROOT, "certified")
TMP_DIR    = os.path.join(DATA_ROOT, "tmp")

for _dir in (UPLOAD_DIR, CERT_DIR, TMP_DIR):
    os.makedirs(_dir, exist_ok=True)

# ── ffmpeg binaries ──────────────────────────────────────────
# build.sh installs ffmpeg to /opt/render/project/.render/ffmpeg/
# Fall back to system ffmpeg if that path doesn't exist (local dev)
_RENDER_FFMPEG = "/opt/render/project/.render/ffmpeg/ffmpeg"
_RENDER_FFPROBE = "/opt/render/project/.render/ffmpeg/ffprobe"
FFMPEG_BIN  = os.environ.get("FFMPEG_BIN",  _RENDER_FFMPEG if os.path.exists(_RENDER_FFMPEG) else "ffmpeg")
FFPROBE_BIN = os.environ.get("FFPROBE_BIN", _RENDER_FFPROBE if os.path.exists(_RENDER_FFPROBE) else "ffprobe")

# ── Upload limits ─────────────────────────────────────────────
MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "500"))
