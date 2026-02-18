# ============================================================
#  VeriFYD — detection.py
#
#  Public-facing detection interface for the VeriFYD system.
#
#  This module is intentionally thin. All signal analysis lives
#  in detector.py. This file owns:
#    - run_detection()  → called by the rest of your system
#    - Label thresholds (tune here without touching the engine)
#    - Optional result packaging / logging
#
#  Usage:
#    from detection import run_detection
#    score, label, detail = run_detection("path/to/video.mp4")
# ============================================================

import logging
from detector import detect_ai          # ← full advanced engine

log = logging.getLogger("verifyd.detection")

# ─────────────────────────────────────────────
#  Label thresholds  (authenticity scale 0–100)
#
#  Authenticity = 100 − AI score, so:
#    100 = definitely real      0 = definitely AI
#
#  Adjust these without touching the engine in detector.py.
# ─────────────────────────────────────────────
THRESHOLD_REAL          = 85   # authenticity ≥ this → REAL       (was 75)
THRESHOLD_UNDETERMINED  = 60   # authenticity ≥ this → UNDETERMINED (was 50)
                               # authenticity <  this → AI


def run_detection(video_path: str) -> tuple:
    """
    Analyze a video file and return an authenticity verdict.

    Parameters
    ----------
    video_path : str
        Absolute or relative path to the video file.

    Returns
    -------
    authenticity_score : int
        0–100.  Higher = more likely real.
    label : str
        One of "REAL", "UNDETERMINED", or "AI".
    detail : dict
        Additional metadata useful for logging or UI display.

    Examples
    --------
    >>> score, label, detail = run_detection("clip.mp4")
    >>> print(f"{label}  ({score}/100)")
    AI  (55/100)
    """
    ai_score = detect_ai(video_path)
    authenticity = 100 - ai_score

    if authenticity >= THRESHOLD_REAL:
        label = "REAL"
    elif authenticity >= THRESHOLD_UNDETERMINED:
        label = "UNDETERMINED"
    else:
        label = "AI"

    detail = {
        "ai_score":         ai_score,
        "authenticity":     authenticity,
        "label":            label,
        "threshold_real":   THRESHOLD_REAL,
        "threshold_undet":  THRESHOLD_UNDETERMINED,
    }

    log.info(
        "Detection complete | file=%s  ai_score=%d  authenticity=%d  label=%s",
        video_path, ai_score, authenticity, label,
    )

    return authenticity, label, detail












