FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /opt/render/project/src

# System/runtime packages
#
# LibreOffice is installed here because Render's native Python worker build
# environment does not allow apt-get as root/sudo, and apt.txt was not making
# soffice available to the running worker. Installing it in Docker ensures the
# same runtime image that starts the worker also contains soffice/libreoffice.
#
# NOTE:
# - LibreOffice improves DOC/DOCX/ODT/XLS/XLSX/PPT/PPTX/RTF-style rendering.
# - LibreOffice does not provide reliable DWG/DXF CAD geometry rendering.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bash \
        curl \
        ca-certificates \
        xz-utils \
        tar \
        git \
        procps \
        ffmpeg \
        libreoffice \
        libreoffice-writer \
        libreoffice-calc \
        libreoffice-impress \
        fontconfig \
        fonts-dejavu \
        fonts-liberation \
        libxinerama1 \
        libxrandr2 \
        libxrender1 \
        libxtst6 \
        libxi6 \
        libcups2 \
        default-jre-headless \
    && rm -rf /var/lib/apt/lists/*

COPY . .

# Match Render's existing virtualenv path so the current worker PATH and
# existing runtime diagnostics continue to behave as expected.
RUN python -m venv /opt/render/project/src/.venv

ENV PATH="/opt/render/project/src/.venv/bin:/usr/lib/libreoffice/program:/usr/lib64/libreoffice/program:/opt/libreoffice/program:/opt/render/project/.render/ffmpeg:/opt/render/project/.render/node/bin:/usr/local/bin:/usr/bin:/bin"

# Install Python dependencies and project build-time assets.
# build.sh already handles ffmpeg/Node/MediaPipe downloads and dependency checks.
# The LibreOffice install attempt inside build.sh is non-fatal; in Docker,
# soffice should already be available from apt above.
RUN chmod +x build.sh \
    && bash build.sh

# Put wrappers in .venv/bin because the existing worker logs show .venv/bin is
# first on PATH at runtime. This makes shutil.which("soffice") resolve cleanly.
RUN set -eux; \
    OFFICE_BIN=""; \
    if command -v soffice >/dev/null 2>&1; then OFFICE_BIN="$(command -v soffice)"; \
    elif command -v libreoffice >/dev/null 2>&1; then OFFICE_BIN="$(command -v libreoffice)"; \
    elif [ -x "/usr/lib/libreoffice/program/soffice" ]; then OFFICE_BIN="/usr/lib/libreoffice/program/soffice"; \
    elif [ -x "/usr/lib64/libreoffice/program/soffice" ]; then OFFICE_BIN="/usr/lib64/libreoffice/program/soffice"; \
    elif [ -x "/opt/libreoffice/program/soffice" ]; then OFFICE_BIN="/opt/libreoffice/program/soffice"; \
    fi; \
    test -n "$OFFICE_BIN"; \
    printf '#!/usr/bin/env bash\nexec "%s" "$@"\n' "$OFFICE_BIN" > /opt/render/project/src/.venv/bin/soffice; \
    printf '#!/usr/bin/env bash\nexec "%s" "$@"\n' "$OFFICE_BIN" > /opt/render/project/src/.venv/bin/libreoffice; \
    chmod +x /opt/render/project/src/.venv/bin/soffice /opt/render/project/src/.venv/bin/libreoffice; \
    echo "LibreOffice selected at: $OFFICE_BIN"; \
    /opt/render/project/src/.venv/bin/soffice --version; \
    command -v ffmpeg; \
    ffmpeg -version | head -1; \
    python --version

# Default command remains the web/API server to avoid breaking any service that
# already uses this Dockerfile for the API. For the Render Background Worker,
# override the Docker command in Render with the worker command shown below:
#
# bash -lc 'echo "=== VeriFYD Worker Runtime Diagnostics ==="; python --version; echo "PATH=$PATH"; echo "soffice=$(command -v soffice || true)"; soffice --version || true; echo "libreoffice=$(command -v libreoffice || true)"; libreoffice --version || true; echo "ffmpeg=$(command -v ffmpeg || true)"; ffmpeg -version | head -1 || true; /opt/render/project/src/.venv/bin/rq worker verifyd --worker-class rq.worker.SimpleWorker --url "$REDIS_URL" & /opt/render/project/src/.venv/bin/rq worker verifyd --worker-class rq.worker.SimpleWorker --url "$REDIS_URL" & wait'
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-10000} --timeout-keep-alive 120"]
