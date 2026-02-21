# ============================================================
#  VeriFYD — video.py
# ============================================================

import os
import uuid
import subprocess
import logging
import requests
import yt_dlp

from config import FFMPEG_BIN, FFPROBE_BIN, TMP_DIR

log = logging.getLogger("verifyd.video")

# ── RapidAPI config ───────────────────────────────────────────
RAPIDAPI_KEY  = os.environ.get("RAPIDAPI_KEY", "")
RAPIDAPI_HOST = "popular-video-downloader.p.rapidapi.com"
RAPIDAPI_PREMIUM_URL = "https://popular-video-downloader.p.rapidapi.com/premium"
RAPIDAPI_GENERAL_URL = "https://popular-video-downloader.p.rapidapi.com/general"

# Platforms RapidAPI handles — everything else falls through to yt-dlp
RAPIDAPI_DOMAINS = (
    "youtube.com", "youtu.be",
    "tiktok.com",
    "instagram.com",
    "twitter.com", "x.com", "t.co",
    "facebook.com", "fb.watch",
)


def is_valid_video(path: str) -> bool:
    """
    Use ffprobe to verify the file is a readable video before processing.
    Returns True if the file contains at least one video stream.
    """
    cmd = [
        FFPROBE_BIN,
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=codec_type",
        "-of", "default=noprint_wrappers=1:nokey=1",
        path,
    ]
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output = r.stdout.decode().strip()
    return r.returncode == 0 and output == "video"


def _download_from_url(direct_url: str, output_path: str) -> None:
    """Download a direct video URL to output_path using requests."""
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(direct_url, stream=True, timeout=60, headers=headers)
    r.raise_for_status()
    with open(output_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)


def _try_rapidapi(url: str, output_path: str) -> bool:
    """
    Try to download via RapidAPI Popular Video Downloader.
    Tries Premium endpoint first, falls back to General.
    Returns True on success, False if RapidAPI couldn't handle it.
    Raises RuntimeError on a clean platform error (private, unavailable etc).
    """
    if not RAPIDAPI_KEY:
        log.warning("RAPIDAPI_KEY not set — skipping RapidAPI")
        return False

    headers = {
        "x-rapidapi-key":  RAPIDAPI_KEY,
        "x-rapidapi-host": RAPIDAPI_HOST,
        "Content-Type":    "application/json",
    }
    payload = {"url": url}

    # Try Premium first, then General
    for endpoint, label in [
        (RAPIDAPI_PREMIUM_URL, "premium"),
        (RAPIDAPI_GENERAL_URL, "general"),
    ]:
        try:
            log.info("RapidAPI %s: requesting %s", label, url)
            resp = requests.post(endpoint, json=payload, headers=headers, timeout=30)

            if resp.status_code == 429:
                log.warning("RapidAPI quota exceeded")
                return False

            if resp.status_code != 200:
                log.warning("RapidAPI %s returned %d", label, resp.status_code)
                continue

            data = resp.json()
            log.info("RapidAPI %s response keys: %s", label, list(data.keys()))

            # Extract download URL from response
            # The API returns different structures — handle all known formats
            video_url = None

            # Format 1: {"url": "https://..."}
            if isinstance(data.get("url"), str) and data["url"].startswith("http"):
                video_url = data["url"]

            # Format 2: {"data": {"url": "..."}} or {"data": {"download_url": "..."}}
            elif isinstance(data.get("data"), dict):
                d = data["data"]
                video_url = (
                    d.get("url") or
                    d.get("download_url") or
                    d.get("video_url") or
                    d.get("mp4")
                )

            # Format 3: {"medias": [{"url": "..."}]}
            elif isinstance(data.get("medias"), list) and data["medias"]:
                for m in data["medias"]:
                    if isinstance(m, dict) and m.get("url", "").startswith("http"):
                        video_url = m["url"]
                        break

            # Format 4: {"links": [{"url": "..."}]}
            elif isinstance(data.get("links"), list) and data["links"]:
                for lnk in data["links"]:
                    if isinstance(lnk, dict) and lnk.get("url", "").startswith("http"):
                        video_url = lnk["url"]
                        break

            if not video_url:
                log.warning("RapidAPI %s: no usable video URL in response: %s", label, str(data)[:300])
                continue

            log.info("RapidAPI %s: downloading from %s", label, video_url[:80])
            _download_from_url(video_url, output_path)

            size = os.path.getsize(output_path)
            if size < 1024:
                log.warning("RapidAPI %s: downloaded file too small (%d bytes)", label, size)
                continue

            log.info("RapidAPI %s: success — %d bytes", label, size)
            return True

        except requests.exceptions.RequestException as e:
            log.warning("RapidAPI %s request failed: %s", label, e)
            continue
        except Exception as e:
            log.warning("RapidAPI %s unexpected error: %s", label, e)
            continue

    return False


def download_video_ytdlp(url: str, output_path: str) -> None:
    """
    Download a video from any supported platform to output_path.

    Strategy:
      1. RapidAPI  — handles YouTube, TikTok, Instagram, Twitter/X, Facebook
                     cleanly without bot detection issues.
      2. yt-dlp    — fallback for everything else (Reddit, Vimeo, Twitch, etc.)
                     and as a backup when RapidAPI fails.

    Raises RuntimeError with a user-friendly message on failure.
    """
    # Delete stale temp file — yt-dlp skips download if file already exists
    if os.path.exists(output_path):
        os.remove(output_path)

    uses_rapidapi = any(d in url for d in RAPIDAPI_DOMAINS)

    # ── Step 1: Try RapidAPI for supported platforms ──────────
    if uses_rapidapi:
        try:
            success = _try_rapidapi(url, output_path)
            if success:
                return
            log.info("RapidAPI failed or unavailable — falling back to yt-dlp")
        except Exception as e:
            log.warning("RapidAPI error: %s — falling back to yt-dlp", e)

    # ── Step 2: yt-dlp fallback ───────────────────────────────
    log.info("Using yt-dlp for: %s", url)

    PROXY_DOMAINS = ("tiktok.com", "instagram.com")
    needs_proxy = any(d in url for d in PROXY_DOMAINS)
    proxy_url = os.environ.get("RESIDENTIAL_PROXY_URL", "").strip() if needs_proxy else None

    ydl_opts = {
        "format":              "bestvideo[ext=mp4][height<=480]+bestaudio[ext=m4a]/best[ext=mp4][height<=480]/best[height<=480]/best",
        "outtmpl":             output_path,
        "quiet":               False,
        "no_warnings":         False,
        "no_cache_dir":        True,
        "verbose":             True,
        "merge_output_format": "mp4",
        "socket_timeout":      30,
        "sleep_interval":      1,
        "max_sleep_interval":  3,
        "writethumbnail":      False,
        "writeinfojson":       False,
        "writesubtitles":      False,
        "ffmpeg_location":     os.path.dirname(FFMPEG_BIN),
        "js_runtimes":         {"node": {"path": "/opt/render/project/.render/node/bin/node"}},
        **({"proxy": proxy_url} if proxy_url else {}),
    }

    # Must delete again — yt-dlp fallback path
    if os.path.exists(output_path):
        os.remove(output_path)

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
    except yt_dlp.utils.DownloadError as e:
        msg = str(e)
        if "Private video" in msg or "private" in msg.lower():
            raise RuntimeError("This video is private and cannot be analyzed.")
        if "login" in msg.lower() or "sign in" in msg.lower():
            raise RuntimeError("This video requires a login and cannot be accessed.")
        if "not available" in msg.lower() or "unavailable" in msg.lower():
            raise RuntimeError("This video is unavailable or has been removed.")
        if "Unsupported URL" in msg:
            raise RuntimeError(
                "This URL is not supported. Please try a direct .mp4 link or "
                "a URL from YouTube, TikTok, Instagram, Twitter/X, or Reddit."
            )
        log.error("yt-dlp download error: %s", msg)
        raise RuntimeError(f"Could not download video: {msg[:200]}")


def clip_first_6_seconds(input_path: str) -> str:
    """
    Stream-copy the first 6 seconds of input_path into a temp file.
    Returns the path to the clipped file.
    Caller is responsible for deleting it after use.
    Raises ValueError if the input file is not valid video.
    """
    if not is_valid_video(input_path):
        raise ValueError(
            "File does not appear to be a valid video. "
            "Please try a direct .mp4 link or a URL from a supported platform."
        )

    clipped = os.path.join(TMP_DIR, f"{uuid.uuid4()}.mp4")
    cmd = [
        FFMPEG_BIN, "-y",
        "-i", input_path,
        "-t", "6",
        "-c", "copy",
        "-movflags", "+faststart",
        clipped,
    ]
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if r.returncode != 0:
        log.error("clip failed: %s", r.stderr.decode()[-300:])
        raise RuntimeError(f"ffmpeg clip failed: {r.stderr.decode()[-300:]}")
    return clipped


def stamp_video(input_path: str, output_path: str, cert_id: str) -> None:
    """
    Burn VeriFYD watermark and certificate ID into the video.
    Top-left:     VeriFYD
    Bottom-right: ID:<cert_id>
    """
    vf = (
        "drawtext=text='VeriFYD':x=10:y=10:fontsize=24:"
        "fontcolor=white@0.85:box=1:boxcolor=black@0.4:boxborderw=4,"
        "drawtext=text='ID\\:" + cert_id + "':x=w-tw-20:y=h-th-20:fontsize=16:"
        "fontcolor=white@0.85:box=1:boxcolor=black@0.4:boxborderw=4"
    )
    cmd = [
        FFMPEG_BIN, "-y",
        "-i", input_path,
        "-vf", vf,
        "-map", "0:v:0",
        "-map", "0:a?",
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-c:a", "copy",
        "-movflags", "+faststart",
        output_path,
    ]
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if r.returncode != 0:
        log.error("stamp failed: %s", r.stderr.decode()[-300:])
        raise RuntimeError(f"ffmpeg stamp failed: {r.stderr.decode()[-300:]}")

