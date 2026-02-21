# ============================================================
#  VeriFYD — video.py
# ============================================================

import os
import uuid
import subprocess
import logging
import yt_dlp

from config import FFMPEG_BIN, FFPROBE_BIN, TMP_DIR

log = logging.getLogger("verifyd.video")

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


def download_video_ytdlp(url: str, output_path: str) -> None:
    """
    Download a video from any yt-dlp supported platform
    (YouTube, Facebook, Twitter/X, Reddit, Vimeo, and 1000+ more).

    Downloads the best available MP4 up to 480p directly to output_path.
    Raises RuntimeError with a clean message on failure.

    TikTok and Instagram block datacenter IPs (like Render's) and require
    a residential proxy — handled separately via RESIDENTIAL_PROXY_URL env var.
    """
    # Proxy: only apply for platforms that actually need it
    PROXY_DOMAINS = ("tiktok.com", "instagram.com")
    needs_proxy = any(d in url for d in PROXY_DOMAINS)
    proxy_url = os.environ.get("RESIDENTIAL_PROXY_URL", "").strip() if needs_proxy else None

    ydl_opts = {
        # Best quality up to 480p — good balance of speed and signal quality.
        # Falls back progressively to ensure something always downloads.
        "format":           "bestvideo[ext=mp4][height<=480]+bestaudio[ext=m4a]/best[ext=mp4][height<=480]/best[height<=480]/best",
        "outtmpl":          output_path,
        "quiet":            False,
        "no_warnings":      False,
        "no_cache_dir":     True,      # Prevent stale 0-byte cached files
        "verbose":          True,      # Keep on for diagnostics — turn off once stable
        "merge_output_format": "mp4",
        # Socket timeout — prevents hung downloads from blocking the server
        "socket_timeout":   30,
        # Respect platform rate limits
        "sleep_interval":   1,
        "max_sleep_interval": 3,
        # Don't write any extra files
        "writethumbnail":   False,
        "writeinfojson":    False,
        "writesubtitles":   False,
        # Use ffmpeg from our configured path for merging separate video+audio streams
        "ffmpeg_location":  os.path.dirname(FFMPEG_BIN),
        # Node.js for YouTube JS challenge solving (installed by build.sh)
        "js_runtimes":      {"node": {"path": "/opt/render/project/.render/node/bin/node"}},
        # Proxy only applied when RESIDENTIAL_PROXY_URL is set AND platform needs it
        **({"proxy": proxy_url} if proxy_url else {}),
    }

    # Delete temp file if it exists from a previous failed request.
    # yt-dlp sees the file, assumes it's already downloaded, and returns 0 bytes.
    if os.path.exists(output_path):
        os.remove(output_path)

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
    except yt_dlp.utils.DownloadError as e:
        msg = str(e)
        # Translate common yt-dlp errors into user-friendly messages
        if "Private video" in msg or "private" in msg.lower():
            raise RuntimeError("This video is private and cannot be analyzed.")
        if "login" in msg.lower() or "sign in" in msg.lower():
            raise RuntimeError("This video requires a login and cannot be accessed.")
        if "not available" in msg.lower() or "unavailable" in msg.lower():
            raise RuntimeError("This video is unavailable or has been removed.")
        if "Unsupported URL" in msg:
            raise RuntimeError(
                "This URL is not supported. Please try a direct .mp4 link or "
                "a URL from YouTube, Twitter/X, Reddit, Vimeo, or similar platforms."
            )
        log.error("yt-dlp download error: %s", msg)
        raise RuntimeError(f"Could not download video: {msg[:200]}")


def clip_first_6_seconds(input_path: str) -> str:
    """
    Stream-copy the first 6 seconds of input_path into a temp file.
    Returns the path to the clipped file.
    Caller is responsible for deleting it after use.
    Raises ValueError (not RuntimeError) if the input file is not valid video.
    """
    if not is_valid_video(input_path):
        raise ValueError(
            "File does not appear to be a valid video. "
            "Direct links to TikTok, Instagram, YouTube and other "
            "social platforms are not supported — please upload a "
            "direct .mp4 file URL instead."
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

