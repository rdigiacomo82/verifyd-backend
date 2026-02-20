#!/usr/bin/env bash
set -e

echo "=== VeriFYD Build ==="

# ── ffmpeg static binary ──────────────────────────────────────
# Using BtbN pinned GitHub release — more reliable than johnvansickle.com
FFMPEG_DIR="/opt/render/project/.render/ffmpeg"

if [ ! -f "$FFMPEG_DIR/ffmpeg" ]; then
    echo "Installing ffmpeg..."
    mkdir -p "$FFMPEG_DIR"
    curl -sL "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-linux64-gpl.tar.xz" \
        -o /tmp/ffmpeg.tar.xz
    tar -xf /tmp/ffmpeg.tar.xz -C /tmp
    cp /tmp/ffmpeg-master-latest-linux64-gpl/bin/ffmpeg  "$FFMPEG_DIR/"
    cp /tmp/ffmpeg-master-latest-linux64-gpl/bin/ffprobe "$FFMPEG_DIR/"
    chmod +x "$FFMPEG_DIR/ffmpeg" "$FFMPEG_DIR/ffprobe"
    rm -rf /tmp/ffmpeg*
    echo "ffmpeg installed"
else
    echo "ffmpeg already present — skipping download"
fi

export PATH="$FFMPEG_DIR:$PATH"

# Verify both binaries are reachable — fail build here, not at runtime
if ! command -v ffmpeg &> /dev/null; then
    echo "ERROR: ffmpeg not found on PATH. Aborting."
    exit 1
fi
echo "ffmpeg OK: $(ffmpeg -version 2>&1 | head -1)"

if ! command -v ffprobe &> /dev/null; then
    echo "ERROR: ffprobe not found on PATH. Aborting."
    exit 1
fi
echo "ffprobe OK: $(ffprobe -version 2>&1 | head -1)"

# ── Node.js (required for YouTube JS challenge solving) ──────
NODE_DIR="/opt/render/project/.render/node"
if [ ! -f "$NODE_DIR/bin/node" ]; then
    echo "Installing Node.js..."
    mkdir -p "$NODE_DIR"
    curl -sL "https://nodejs.org/dist/v20.19.0/node-v20.19.0-linux-x64.tar.xz"         -o /tmp/node.tar.xz
    tar -xf /tmp/node.tar.xz -C /tmp
    cp -r /tmp/node-v20.19.0-linux-x64/* "$NODE_DIR/"
    rm -rf /tmp/node*
    echo "Node.js installed: $($NODE_DIR/bin/node --version)"
else
    echo "Node.js already present — skipping"
fi
export PATH="$NODE_DIR/bin:$PATH"
echo "node OK: $(node --version)"

# ── Python dependencies ───────────────────────────────────────
pip install --upgrade pip
pip install -r requirements.txt
# Install yt-dlp with curl-cffi extra so impersonation handler is registered
pip install --upgrade "yt-dlp[default,curl-cffi]"
# Install yt-dlp-ejs for YouTube JS challenge solving (requires node)
pip install --upgrade yt-dlp-ejs

# Smoke-test key imports
python -c "import cv2, numpy, fastapi, uvicorn, yt_dlp, curl_cffi; print('Python deps OK')"

# Verify yt-dlp CLI is available
yt-dlp --version && echo "yt-dlp OK" || { echo "ERROR: yt-dlp not found"; exit 1; }

echo "=== Build Complete ==="