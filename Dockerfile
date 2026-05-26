FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

WORKDIR /opt/render/project/src

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bash \
        curl \
        ca-certificates \
        xz-utils \
        tar \
        git \
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
    && rm -rf /var/lib/apt/lists/*

COPY . .

RUN python -m venv /opt/render/project/src/.venv

ENV PATH="/opt/render/project/src/.venv/bin:/opt/render/project/.render/ffmpeg:/opt/render/project/.render/node/bin:/usr/local/bin:/usr/bin:/bin"

RUN chmod +x build.sh \
    && bash build.sh \
    && (command -v soffice || command -v libreoffice) \
    && (soffice --version || libreoffice --version)

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-10000} --timeout-keep-alive 120"]
