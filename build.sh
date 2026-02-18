#!/usr/bin/env bash
set -e

echo "=== VeriFYD Build ==="

# Install ffmpeg via static binary (no apt-get needed)
if ! command -v ffmpeg &> /dev/null; then
    echo "Installing ffmpeg..."
    curl -sL https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz -o /tmp/ffmpeg.tar.xz
    tar -xf /tmp/ffmpeg.tar.xz -C /tmp
    mkdir -p /opt/render/project/.render/ffmpeg
    cp /tmp/ffmpeg-*-amd64-static/ffmpeg /opt/render/project/.render/ffmpeg/
    cp /tmp/ffmpeg-*-amd64-static/ffprobe /opt/render/project/.render/ffmpeg/
    rm -rf /tmp/ffmpeg*
    echo "ffmpeg installed"
fi

export PATH="/opt/render/project/.render/ffmpeg:$PATH"

# Verify ffmpeg is reachable — fail the build here rather than at runtime
if ! command -v ffmpeg &> /dev/null; then
    echo "ERROR: ffmpeg not found on PATH after install. Aborting."
    exit 1
fi
echo "ffmpeg OK: $(ffmpeg -version 2>&1 | head -1)"

if ! command -v ffprobe &> /dev/null; then
    echo "ERROR: ffprobe not found on PATH after install. Aborting."
    exit 1
fi
echo "ffprobe OK: $(ffprobe -version 2>&1 | head -1)"

# Python deps
pip install --upgrade pip
pip install -r requirements.txt

# Verify key Python packages imported cleanly
python -c "import cv2, numpy, fastapi, uvicorn; print('Python deps OK')"

echo "=== Build Complete ==="
