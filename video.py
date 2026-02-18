# ============================================================
#  VeriFYD — video.py
#
#  All video processing helpers live here.
#  main.py imports from this module — nothing video-related
#  should be defined directly in main.py.
#
#  Functions:
#    clip_first_10_seconds(input_path)  → clipped temp path
#    stamp_video(input_path, output_path, cert_id)
# ============================================================

import os
import uuid
import subprocess
import logging

from config import FFMPEG_BIN, TMP_DIR

log = logging.getLogger("verifyd.video")


def clip_first_10_seconds(input_path: str) -> str:
    """
    Stream-copy the first 10 seconds of input_path into a temp file.
    Returns the path to the clipped file.
    Caller is responsible for deleting it after use.
    """
    clipped = os.path.join(TMP_DIR, f"{uuid.uuid4()}.mp4")
    cmd = [
        FFMPEG_BIN, "-y",
        "-i", input_path,
        "-t", "10",
        "-c", "copy",
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

    Note: the colon in 'ID:{cert_id}' is escaped as '\\:' because
    ffmpeg's drawtext filter treats ':' as a key=value separator.
    """
    vf = (
        "drawtext=text='VeriFYD':x=10:y=10:fontsize=24:"
        "fontcolor=white@0.85:box=1:boxcolor=black@0.4:boxborderw=4,"
        f"drawtext=text='ID\\:{cert_id}':x=w-tw-20:y=h-th-20:fontsize=16:"
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
