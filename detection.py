# ============================================================
#  VeriFYD — detection.py
#
#  Public-facing detection interface for the VeriFYD system.
#
#  DUAL ENGINE:
#    1. detector.py   — signal-based analysis (noise, edges,
#                       motion, DCT artifacts, saturation, etc.)
#    2. gpt_vision.py — GPT-4o semantic analysis (impossible
#                       elements, AI art artifacts, physics)
#
#  Final score = weighted combination of both engines.
#  If GPT is unavailable, falls back to signal-only mode.
# ============================================================

import logging
from detector    import detect_ai
from gpt_vision  import gpt_vision_score

log = logging.getLogger("verifyd.detection")

# ─────────────────────────────────────────────
#  Label thresholds  (authenticity scale 0–100)
#
#  Authenticity = 100 − combined_ai_score
#    100 = definitely real      0 = definitely AI
# ─────────────────────────────────────────────
THRESHOLD_REAL         = 55
THRESHOLD_UNDETERMINED = 40

# ─────────────────────────────────────────────
#  Engine weights
#  Signal detector: reliable for compression/noise artifacts
#  GPT-4o vision:   reliable for semantic/content anomalies
# ─────────────────────────────────────────────
WEIGHT_SIGNAL = 0.45   # 45% signal detector
WEIGHT_GPT    = 0.55   # 55% GPT-4o vision (semantic is stronger)


def run_detection(video_path: str) -> tuple:
    """
    Analyze a video using dual-engine detection.

    Returns
    -------
    authenticity_score : int   0-100 (higher = more likely real)
    label : str                "REAL", "UNDETERMINED", or "AI"
    detail : dict              Full breakdown of both engine scores
    """

    # ── Engine 1: Signal-based detector ──────────────────────
    signal_ai_score = detect_ai(video_path)
    log.info("Signal detector ai_score: %d", signal_ai_score)

    # ── Engine 2: GPT-4o vision ───────────────────────────────
    gpt_result    = gpt_vision_score(video_path)
    gpt_ai_score  = gpt_result["ai_probability"]
    gpt_available = gpt_result.get("available", False)
    log.info("GPT-4o ai_probability: %d  available: %s", gpt_ai_score, gpt_available)

    # ── Combined score ────────────────────────────────────────
    if gpt_available:
        combined_ai_score = (
            signal_ai_score * WEIGHT_SIGNAL +
            gpt_ai_score    * WEIGHT_GPT
        )
    else:
        combined_ai_score = float(signal_ai_score)
        log.warning("GPT-4o unavailable — using signal detector only")

    combined_ai_score = max(0.0, min(100.0, combined_ai_score))
    authenticity = 100 - int(round(combined_ai_score))

    # ── Label ─────────────────────────────────────────────────
    if authenticity >= THRESHOLD_REAL:
        label = "REAL"
    elif authenticity >= THRESHOLD_UNDETERMINED:
        label = "UNDETERMINED"
    else:
        label = "AI"

    detail = {
        "ai_score":         int(round(combined_ai_score)),
        "authenticity":     authenticity,
        "label":            label,
        "signal_ai_score":  signal_ai_score,
        "gpt_ai_score":     gpt_ai_score,
        "gpt_available":    gpt_available,
        "gpt_reasoning":    gpt_result.get("reasoning", ""),
        "gpt_flags":        gpt_result.get("flags", []),
        "threshold_real":   THRESHOLD_REAL,
        "threshold_undet":  THRESHOLD_UNDETERMINED,
    }

    log.info(
        "Detection complete | signal=%d gpt=%d combined=%d authenticity=%d label=%s",
        signal_ai_score, gpt_ai_score,
        int(round(combined_ai_score)), authenticity, label,
    )

    return authenticity, label, detail












