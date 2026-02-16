#!/usr/bin/env bash
set -e

echo "=== VeriFYD Build ==="

# -----------------------------
# Install FFmpeg (static build)
# -----------------------------
if ! command -v ffmpeg &> /dev/null; then
    echo "Installing ffmpeg..."
    curl -sL https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz -o /tmp/ffmpeg.tar.xz
    tar -xf /tmp/ffmpeg.tar.xz -C /tmp
    mkdir -p /opt/render/project/.render/ffmpeg
    cp /tmp/ffmpeg-*-amd64-static/ffmpeg /opt/render/project/.render/ffmpeg/
    cp /tmp/ffmpeg-*-amd64-static/ffprobe /opt/render/project/.render/ffmpeg/
    rm -rf /tmp/ffmpeg*
fi

export PATH="/opt/render/project/.render/ffmpeg:$PATH"
ffmpeg -version | head -1

# -----------------------------
# Python deps
# -----------------------------
pip install --upgrade pip
pip install -r requirements.txt

# Install yt-dlp (python version)
pip install yt-dlp

echo "=== Build Complete ==="


