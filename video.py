# ============================================================
#  VeriFYD — video.py
# ============================================================

import os
import uuid
import subprocess
import logging

from config import FFMPEG_BIN, FFPROBE_BIN, TMP_DIR

log = logging.getLogger("verifyd.video")

# Social/platform URLs that serve redirect pages, not raw video.
# Attempting to download these will always produce an invalid file.
UNSUPPORTED_DOMAINS = (
    "tiktok.com", "instagram.com", "facebook.com",
    "twitter.com", "x.com", "youtube.com", "youtu.be",
)


def is_supported_url(url: str) -> bool:
    """Return False for social platform URLs that block direct download."""
    return not any(domain in url for domain in UNSUPPORTED_DOMAINS)


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


def clip_first_10_seconds(input_path: str) -> str:
    """
    Stream-copy the first 10 seconds of input_path into a temp file.
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
        "-t", "10",
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

