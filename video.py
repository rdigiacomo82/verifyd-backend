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

# ── SMVD RapidAPI config ──────────────────────────────────────
# Social Media Video Downloader by Emmanuel David
RAPIDAPI_KEY  = os.environ.get("RAPIDAPI_KEY", "")
SMVD_HOST     = "social-media-video-downloader.p.rapidapi.com"
SMVD_BASE     = "https://social-media-video-downloader.p.rapidapi.com"

# Platforms SMVD handles
SMVD_DOMAINS = (
    "youtube.com", "youtu.be",
    "tiktok.com",
    "instagram.com",
    "facebook.com", "fb.watch",
)

# yt-dlp handles these without issues
YTDLP_DOMAINS = (
    "twitter.com", "x.com", "t.co",
    "reddit.com", "v.redd.it",
    "twitch.tv",
    "vimeo.com",
    "dailymotion.com",
    "streamable.com",
    "rumble.com",
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
        for chunk in r.iter_content(chunk_size=65536):
            if chunk:
                f.write(chunk)


def _extract_video_id(url: str) -> str:
    """Extract YouTube video ID from various URL formats."""
    import re
    patterns = [
        r'(?:v=|youtu\.be/|shorts/)([a-zA-Z0-9_-]{11})',
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return ""


def _try_smvd_youtube(url: str, output_path: str) -> bool:
    """
    Download YouTube video via SMVD API.
    Returns True on success, False on failure.
    """
    video_id = _extract_video_id(url)
    if not video_id:
        log.warning("SMVD: could not extract YouTube video ID from %s", url)
        return False

    headers = {
        "x-rapidapi-key":  RAPIDAPI_KEY,
        "x-rapidapi-host": SMVD_HOST,
    }

    # Request video details with 480p rendered format
    params = {
        "videoId":          video_id,
        "renderableFormats": "480p",
        "urlAccess":        "proxied",
        "getTranscript":    "false",
    }

    log.info("SMVD YouTube: requesting video ID %s", video_id)
    resp = requests.get(
        f"{SMVD_BASE}/youtube/v3/video/details",
        headers=headers,
        params=params,
        timeout=30,
    )

    if resp.status_code != 200:
        log.warning("SMVD YouTube: API returned %d — %s", resp.status_code, resp.text[:200])
        return False

    data = resp.json()

    if data.get("error"):
        log.warning("SMVD YouTube: API error: %s", data["error"])
        return False

    contents = data.get("contents", [])
    if not contents:
        log.warning("SMVD YouTube: no contents in response")
        return False

    content = contents[0]

    # ── Try renderableVideos first (pre-merged video+audio) ───
    renderable = content.get("renderableVideos", [])
    for rv in renderable:
        render_config = rv.get("renderConfig", {})
        execution_url = render_config.get("executionUrl", "")
        status_url    = render_config.get("statusUrl", "")

        if not execution_url:
            continue

        log.info("SMVD YouTube: triggering render for %s", rv.get("label", "?"))
        try:
            # Trigger render (no auth needed)
            exec_resp = requests.get(execution_url, timeout=30)
            if exec_resp.status_code not in (200, 202):
                log.warning("SMVD render trigger failed: %d", exec_resp.status_code)
                continue

            # Poll status via WebSocket-style HTTP polling
            import time
            import websocket  # websocket-client

            download_url = None
            deadline = time.time() + 120  # 2 min timeout

            def on_message(ws, message):
                nonlocal download_url
                import json
                try:
                    msg = json.loads(message)
                    log.info("SMVD render status: %s", msg.get("status"))
                    if msg.get("status") == "done":
                        download_url = msg.get("output", {}).get("url", "")
                        ws.close()
                    elif msg.get("status") in ("failed", "error"):
                        ws.close()
                except Exception:
                    pass

            def on_error(ws, error):
                log.warning("SMVD render WS error: %s", error)
                ws.close()

            ws = websocket.WebSocketApp(
                status_url,
                on_message=on_message,
                on_error=on_error,
            )
            ws.run_forever(ping_timeout=10)

            if download_url:
                log.info("SMVD YouTube: render complete, downloading")
                _download_from_url(download_url, output_path)
                size = os.path.getsize(output_path)
                if size > 1024:
                    log.info("SMVD YouTube render: success — %d bytes", size)
                    return True

        except Exception as e:
            log.warning("SMVD render error: %s", e)
            continue

    # ── Fallback: try direct video URLs ───────────────────────
    videos = content.get("videos", [])
    audios = content.get("audios", [])

    # Find best video at 480p or lower
    video_url = None
    audio_url = None

    preferred = ["480p", "360p", "240p", "144p"]
    for quality in preferred:
        for v in videos:
            if v.get("label", "").startswith(quality):
                video_url = v.get("url", "")
                break
        if video_url:
            break

    # If no preferred quality found, take the last (lowest) video
    if not video_url and videos:
        video_url = videos[-1].get("url", "")

    # Get audio stream
    if audios:
        audio_url = audios[0].get("url", "")

    if not video_url:
        log.warning("SMVD YouTube: no video URL found in response")
        return False

    # If we have separate video+audio, merge with ffmpeg
    if audio_url and video_url != audio_url:
        log.info("SMVD YouTube: merging video+audio with ffmpeg")
        video_tmp = output_path + ".video.mp4"
        audio_tmp = output_path + ".audio.m4a"
        try:
            _download_from_url(video_url, video_tmp)
            _download_from_url(audio_url, audio_tmp)

            cmd = [
                FFMPEG_BIN, "-y",
                "-i", video_tmp,
                "-i", audio_tmp,
                "-c:v", "copy",
                "-c:a", "copy",
                "-movflags", "+faststart",
                output_path,
            ]
            r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
            if r.returncode != 0:
                log.warning("SMVD ffmpeg merge failed: %s", r.stderr.decode()[-200:])
                return False

            size = os.path.getsize(output_path)
            if size > 1024:
                log.info("SMVD YouTube merged: success — %d bytes", size)
                return True
        finally:
            for f in (video_tmp, audio_tmp):
                if os.path.exists(f):
                    os.remove(f)
    else:
        # Single stream — just download directly
        _download_from_url(video_url, output_path)
        size = os.path.getsize(output_path)
        if size > 1024:
            log.info("SMVD YouTube direct: success — %d bytes", size)
            return True

    return False


def _try_smvd_tiktok(url: str, output_path: str) -> bool:
    """Download TikTok video via SMVD API."""
    headers = {
        "x-rapidapi-key":  RAPIDAPI_KEY,
        "x-rapidapi-host": SMVD_HOST,
    }
    params = {"url": url}

    log.info("SMVD TikTok: requesting %s", url)
    resp = requests.get(
        f"{SMVD_BASE}/tiktok/v3/post/details",
        headers=headers,
        params=params,
        timeout=30,
    )

    if resp.status_code != 200:
        log.warning("SMVD TikTok: API returned %d", resp.status_code)
        return False

    data = resp.json()
    if data.get("error"):
        log.warning("SMVD TikTok: API error: %s", data["error"])
        return False

    contents = data.get("contents", [])
    if not contents:
        return False

    videos = contents[0].get("videos", [])
    if not videos:
        return False

    video_url = videos[0].get("url", "")
    if not video_url:
        return False

    _download_from_url(video_url, output_path)
    size = os.path.getsize(output_path)
    if size > 1024:
        log.info("SMVD TikTok: success — %d bytes", size)
        return True
    return False


def _try_smvd_instagram(url: str, output_path: str) -> bool:
    """Download Instagram video via SMVD API."""
    import re
    # Extract shortcode from Instagram URL
    match = re.search(r'/(?:p|reel|reels)/([A-Za-z0-9_-]+)', url)
    if not match:
        log.warning("SMVD Instagram: could not extract shortcode from %s", url)
        return False

    shortcode = match.group(1)
    headers = {
        "x-rapidapi-key":  RAPIDAPI_KEY,
        "x-rapidapi-host": SMVD_HOST,
    }
    params = {"shortcode": shortcode}

    log.info("SMVD Instagram: requesting shortcode %s", shortcode)
    resp = requests.get(
        f"{SMVD_BASE}/instagram/v3/media/post/details",
        headers=headers,
        params=params,
        timeout=30,
    )

    if resp.status_code != 200:
        log.warning("SMVD Instagram: API returned %d", resp.status_code)
        return False

    data = resp.json()
    if data.get("error"):
        log.warning("SMVD Instagram: API error: %s", data["error"])
        return False

    contents = data.get("contents", [])
    if not contents:
        return False

    videos = contents[0].get("videos", [])
    if not videos:
        return False

    video_url = videos[0].get("url", "")
    if not video_url:
        return False

    _download_from_url(video_url, output_path)
    size = os.path.getsize(output_path)
    if size > 1024:
        log.info("SMVD Instagram: success — %d bytes", size)
        return True
    return False


def _try_smvd(url: str, output_path: str) -> bool:
    """
    Route to the correct SMVD handler based on platform.
    Returns True on success, False on failure.
    """
    if not RAPIDAPI_KEY:
        log.warning("RAPIDAPI_KEY not set — skipping SMVD")
        return False

    try:
        if "youtube.com" in url or "youtu.be" in url:
            return _try_smvd_youtube(url, output_path)
        elif "tiktok.com" in url:
            return _try_smvd_tiktok(url, output_path)
        elif "instagram.com" in url:
            return _try_smvd_instagram(url, output_path)
        else:
            log.warning("SMVD: no handler for %s", url)
            return False
    except Exception as e:
        log.warning("SMVD error: %s", e)
        return False


def download_video_ytdlp(url: str, output_path: str) -> None:
    """
    Download a video from any supported platform to output_path.

    Strategy:
      1. SMVD RapidAPI — YouTube, TikTok, Instagram, Facebook
      2. yt-dlp        — Twitter/X, Reddit, Vimeo, Twitch, and fallback

    Raises RuntimeError with a user-friendly message on failure.
    """
    # Delete stale temp file — yt-dlp skips download if file already exists
    if os.path.exists(output_path):
        os.remove(output_path)

    uses_smvd = any(d in url for d in SMVD_DOMAINS)

    # ── Step 1: Try SMVD for supported platforms ──────────────
    if uses_smvd:
        try:
            success = _try_smvd(url, output_path)
            if success:
                return
            log.info("SMVD failed — falling back to yt-dlp")
        except Exception as e:
            log.warning("SMVD error: %s — falling back to yt-dlp", e)

    # ── Step 2: yt-dlp fallback ───────────────────────────────
    log.info("Using yt-dlp for: %s", url)

    if os.path.exists(output_path):
        os.remove(output_path)

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
    }

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
        "-vf", "scale='min(iw,720)':'min(ih,1280)',scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-crf", "28",
        "-c:a", "aac",
        "-ar", "44100",
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
    Burn VeriFYD logo watermark into the video.
    Top-left: transparent VeriFYD logo at 75% opacity.
    No text overlays, no certificate ID burned in.
    """
    import tempfile
    from PIL import Image
    import numpy as np

    # Absolute path to logo
    logo_src = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "assets", "VeriFYD_Logo.png"
    )
    log.info("stamp_video: logo path = %s, exists = %s", logo_src, os.path.exists(logo_src))

    if not os.path.exists(logo_src):
        log.warning("Logo not found — falling back to text watermark")
        vf = "drawtext=text=\'VeriFYD\':x=10:y=10:fontsize=24:fontcolor=white@0.75"
        cmd = [
            FFMPEG_BIN, "-y", "-i", input_path,
            "-vf", vf,
            "-map", "0:v:0", "-map", "0:a?",
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-c:a", "copy", "-movflags", "+faststart", output_path,
        ]
        r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if r.returncode != 0:
            raise RuntimeError(f"ffmpeg stamp failed: {r.stderr.decode()[-300:]}")
        return

    # Remove black background, resize to 120px wide
    img = Image.open(logo_src).convert("RGBA")
    data = np.array(img)
    r_ch, g_ch, b_ch = data[:,:,0], data[:,:,1], data[:,:,2]
    black_mask = (r_ch < 30) & (g_ch < 30) & (b_ch < 30)
    data[:,:,3][black_mask] = 0
    img = Image.fromarray(data)
    new_width = 160
    new_height = int(img.height * (new_width / img.width))
    img = img.resize((new_width, new_height), Image.LANCZOS)

    tmp_logo = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    img.save(tmp_logo.name)
    tmp_logo.close()

    try:
        cmd = [
            FFMPEG_BIN, "-y",
            "-i", input_path,
            "-i", tmp_logo.name,
            "-filter_complex",
            "[1:v]scale=iw:-1,format=rgba,colorchannelmixer=aa=0.5[logo];"
            "[0:v][logo]overlay=W-w-2:H-h-2",
            "-map", "0:a?",
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-crf", "26",
            "-c:a", "copy",
            "-movflags", "+faststart",
            output_path,
        ]
        r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if r.returncode != 0:
            log.error("stamp failed: %s", r.stderr.decode()[-300:])
            raise RuntimeError(f"ffmpeg stamp failed: {r.stderr.decode()[-300:]}")
    finally:
        if os.path.exists(tmp_logo.name):
            os.remove(tmp_logo.name)





