#!/usr/bin/env bash
set -e

echo "=== VeriFYD Build ==="

# ── LibreOffice / Office rendering support ───────────────────
# DOCX/DOC/ODT/XLSX/PPTX original-layout certification requires
# LibreOffice/soffice in the WORKER runtime. Render's apt.txt should
# normally install it before this build script runs. This block:
#   1) detects whether soffice/libreoffice is already available,
#   2) tries a safe apt-get install when the build user has permission,
#   3) creates wrapper commands inside .venv/bin so the worker PATH can find it,
#   4) logs the result clearly without breaking existing production deploys.
#
# Set VERIFYD_REQUIRE_LIBREOFFICE=1 in Render if you want the build to fail
# whenever LibreOffice cannot be installed/found.
install_libreoffice_if_needed() {
    echo "Checking LibreOffice/soffice availability..."

    # Make sure common locations are included for build-time checks.
    export PATH="/usr/lib/libreoffice/program:/opt/libreoffice/program:$PATH"

    local office_bin=""
    if command -v soffice >/dev/null 2>&1; then
        office_bin="$(command -v soffice)"
    elif command -v libreoffice >/dev/null 2>&1; then
        office_bin="$(command -v libreoffice)"
    elif [ -x "/usr/lib/libreoffice/program/soffice" ]; then
        office_bin="/usr/lib/libreoffice/program/soffice"
    elif [ -x "/usr/lib64/libreoffice/program/soffice" ]; then
        office_bin="/usr/lib64/libreoffice/program/soffice"
    elif [ -x "/opt/libreoffice/program/soffice" ]; then
        office_bin="/opt/libreoffice/program/soffice"
    fi

    if [ -z "$office_bin" ]; then
        echo "LibreOffice not found before install attempt."

        if command -v apt-get >/dev/null 2>&1; then
            if [ "$(id -u)" = "0" ]; then
                echo "Attempting apt-get install of LibreOffice packages as root..."
                export DEBIAN_FRONTEND=noninteractive
                apt-get update
                apt-get install -y --no-install-recommends \
                    libreoffice \
                    libreoffice-writer \
                    libreoffice-calc \
                    libreoffice-impress \
                    fontconfig \
                    fonts-dejavu \
                    libxinerama1 \
                    libxrandr2 \
                    libxrender1 \
                    libxtst6 \
                    libxi6 \
                    libcups2 \
                    default-jre-headless || true
                rm -rf /var/lib/apt/lists/* || true
            elif command -v sudo >/dev/null 2>&1; then
                echo "Attempting apt-get install of LibreOffice packages with sudo..."
                export DEBIAN_FRONTEND=noninteractive
                sudo apt-get update
                sudo apt-get install -y --no-install-recommends \
                    libreoffice \
                    libreoffice-writer \
                    libreoffice-calc \
                    libreoffice-impress \
                    fontconfig \
                    fonts-dejavu \
                    libxinerama1 \
                    libxrandr2 \
                    libxrender1 \
                    libxtst6 \
                    libxi6 \
                    libcups2 \
                    default-jre-headless || true
                sudo rm -rf /var/lib/apt/lists/* || true
            else
                echo "WARNING: apt-get exists, but build user is not root and sudo is unavailable."
                echo "WARNING: Render must install LibreOffice via apt.txt or Dockerfile for DOCX layout preservation."
            fi
        else
            echo "WARNING: apt-get not available in this build environment."
            echo "WARNING: Render must install LibreOffice via apt.txt or Dockerfile for DOCX layout preservation."
        fi

        # Re-detect after install attempt.
        if command -v soffice >/dev/null 2>&1; then
            office_bin="$(command -v soffice)"
        elif command -v libreoffice >/dev/null 2>&1; then
            office_bin="$(command -v libreoffice)"
        elif [ -x "/usr/lib/libreoffice/program/soffice" ]; then
            office_bin="/usr/lib/libreoffice/program/soffice"
        elif [ -x "/usr/lib64/libreoffice/program/soffice" ]; then
            office_bin="/usr/lib64/libreoffice/program/soffice"
        elif [ -x "/opt/libreoffice/program/soffice" ]; then
            office_bin="/opt/libreoffice/program/soffice"
        fi
    fi

    if [ -n "$office_bin" ]; then
        echo "LibreOffice detected at: $office_bin"
        "$office_bin" --version || true

        # The Render worker PATH in logs starts with /opt/render/project/src/.venv/bin.
        # Put wrapper scripts there so shutil.which('soffice') can find them at runtime.
        local venv_bin="/opt/render/project/src/.venv/bin"
        mkdir -p "$venv_bin"

        cat > "$venv_bin/soffice" <<EOF
#!/usr/bin/env bash
exec "$office_bin" "\$@"
EOF
        chmod +x "$venv_bin/soffice"

        cat > "$venv_bin/libreoffice" <<EOF
#!/usr/bin/env bash
exec "$office_bin" "\$@"
EOF
        chmod +x "$venv_bin/libreoffice"

        echo "LibreOffice wrappers installed:"
        ls -l "$venv_bin/soffice" "$venv_bin/libreoffice" || true

        export PATH="$venv_bin:/usr/lib/libreoffice/program:/opt/libreoffice/program:$PATH"

        if command -v soffice >/dev/null 2>&1; then
            echo "soffice OK on PATH: $(command -v soffice)"
            soffice --version || true
        else
            echo "WARNING: wrapper creation completed, but soffice is still not on PATH."
        fi
    else
        echo "WARNING: LibreOffice/soffice was not found after install attempt."
        echo "WARNING: DOCX/DOC/ODT/XLSX/PPTX files will fall back to text rendering until the worker image includes LibreOffice."
        if [ "${VERIFYD_REQUIRE_LIBREOFFICE:-0}" = "1" ]; then
            echo "ERROR: VERIFYD_REQUIRE_LIBREOFFICE=1 and LibreOffice is unavailable. Failing build."
            exit 1
        fi
    fi
}

install_libreoffice_if_needed

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

export PATH="/opt/render/project/src/.venv/bin:/usr/lib/libreoffice/program:/opt/libreoffice/program:$FFMPEG_DIR:$PATH"

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
    curl -sL "https://nodejs.org/dist/v20.19.0/node-v20.19.0-linux-x64.tar.xz" \
        -o /tmp/node.tar.xz
    tar -xf /tmp/node.tar.xz -C /tmp
    cp -r /tmp/node-v20.19.0-linux-x64/* "$NODE_DIR/"
    rm -rf /tmp/node*
    echo "Node.js installed: $($NODE_DIR/bin/node --version)"
else
    echo "Node.js already present — skipping"
fi
export PATH="/opt/render/project/src/.venv/bin:/usr/lib/libreoffice/program:/opt/libreoffice/program:$NODE_DIR/bin:$PATH"
echo "node OK: $(node --version)"

# Re-check LibreOffice after PATH changes. This is diagnostic and non-fatal by default.
echo "Final LibreOffice build check:"
echo "  soffice: $(command -v soffice || true)"
echo "  libreoffice: $(command -v libreoffice || true)"
if command -v soffice >/dev/null 2>&1; then
    soffice --version || true
elif command -v libreoffice >/dev/null 2>&1; then
    libreoffice --version || true
elif [ "${VERIFYD_REQUIRE_LIBREOFFICE:-0}" = "1" ]; then
    echo "ERROR: LibreOffice unavailable and VERIFYD_REQUIRE_LIBREOFFICE=1. Failing build."
    exit 1
else
    echo "WARNING: LibreOffice unavailable at build end. DOCX layout rendering will not work until the worker runtime includes it."
fi

# ── MediaPipe face landmarker model (rPPG engine) ────────────
# Downloaded at build time so the worker has no outbound network dependency at runtime.
# face_landmarker.task: 478-point face mesh — forehead + cheek landmark ROI for CHROM rPPG.
# Pinned to float16/v1 — stable versioned URL, not "latest".
MP_DIR="/opt/render/project/.render/mediapipe_models"
MP_MODEL="$MP_DIR/face_landmarker.task"
if [ ! -f "$MP_MODEL" ]; then
    echo "Downloading MediaPipe face landmarker model (~12MB)..."
    mkdir -p "$MP_DIR"
    curl -sL \
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task" \
        -o "$MP_MODEL"
    if [ -f "$MP_MODEL" ] && [ -s "$MP_MODEL" ]; then
        echo "MediaPipe face landmarker downloaded: $(du -sh $MP_MODEL | cut -f1)"
    else
        echo "WARNING: MediaPipe model download failed — rPPG will fall back to Haar cascade"
        rm -f "$MP_MODEL"
    fi
else
    echo "MediaPipe face landmarker already present — skipping ($(du -sh $MP_MODEL | cut -f1))"
fi

# ── Python dependencies ───────────────────────────────────────
pip install --upgrade pip
pip install -r requirements.txt
pip install redis rq
# Install yt-dlp with curl-cffi extra so impersonation handler is registered
pip install --upgrade "yt-dlp[default,curl-cffi]"
# Install yt-dlp-ejs for YouTube JS challenge solving (requires node)
pip install --upgrade yt-dlp-ejs

# Smoke-test key imports (hard fail — these are required)
python -c "import cv2, numpy, fastapi, uvicorn, yt_dlp, curl_cffi, redis, rq, scipy; print('Python deps OK')"

# MediaPipe smoke test — warning only (rPPG falls back to Haar cascade if unavailable)
python -c "import mediapipe; print('mediapipe OK:', mediapipe.__version__)" \
    || echo "WARNING: mediapipe import failed — rPPG engine will use Haar cascade fallback"

# Verify yt-dlp CLI is available
yt-dlp --version && echo "yt-dlp OK" || { echo "ERROR: yt-dlp not found"; exit 1; }

# Final runtime-path hints for Render logs.
echo "=== VeriFYD Build Runtime Checks ==="
echo "PATH=$PATH"
echo "soffice=$(command -v soffice || true)"
echo "libreoffice=$(command -v libreoffice || true)"
if command -v soffice >/dev/null 2>&1; then
    soffice --version || true
elif command -v libreoffice >/dev/null 2>&1; then
    libreoffice --version || true
fi

echo "=== Build Complete ==="