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
#
#  GPT FALLBACK POLICY (v2):
#    When GPT fails due to rate limiting or API errors, the
#    system no longer trusts the signal detector alone for
#    definitive verdicts. Instead:
#      - Signal score < 90: cap result at UNDETERMINED
#        (protects real videos from false AI verdicts)
#      - Signal score >= 90: allow AI verdict
#        (only extremely obvious AI signals trusted alone)
#    This prevents cases like the BBC video being flagged as
#    AI when GPT was rate-limited and unavailable.
# ============================================================

import logging
from concurrent.futures import ThreadPoolExecutor
from detector    import detect_ai
from gpt_vision  import gpt_vision_score, gpt_vision_score_with_context, extract_key_frames

log = logging.getLogger("verifyd.detection")

THRESHOLD_REAL         = 55
THRESHOLD_UNDETERMINED = 40
WEIGHT_SIGNAL          = 0.40
WEIGHT_GPT             = 0.60
SIGNAL_ONLY_AI_THRESHOLD = 90


def _extract_frames_only(video_path: str) -> list:
    """Extract frames for GPT without running analysis."""
    try:
        return extract_key_frames(video_path)
    except Exception as e:
        log.error("Frame extraction failed: %s", e)
        return []


def _build_physics_context(signal_score: int) -> dict:
    """
    Read the detector's last-run physics signals from a thread-local
    store set by detector.py, and build a context dict for GPT.
    Falls back to signal score alone if detailed signals unavailable.
    """
    try:
        from detector import get_last_physics_signals
        signals = get_last_physics_signals()
    except Exception:
        signals = {}

    return {
        "signal_score":     signal_score,
        "vert_flow":        signals.get("avg_vert_flow", None),
        "upward_ratio":     signals.get("upward_frame_ratio", None),
        "accel_std":        signals.get("accel_std", None),
        "low_corr_count":   signals.get("low_corr_count", None),
        "avg_saturation":   signals.get("avg_saturation", None),
        "avg_sharpness":    signals.get("avg_sharpness", None),
    }


def run_detection(video_path: str) -> tuple:
    # ── Run signal detector first, then pass physics findings to GPT ──
    # Sequential execution allows GPT to use physics engine results as context,
    # significantly improving accuracy on action/physics content.
    from concurrent.futures import ThreadPoolExecutor

    # Step 1: Run signal detector and extract frames in parallel
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_signal = executor.submit(detect_ai, video_path)
        future_frames = executor.submit(_extract_frames_only, video_path)
        signal_ai_score = future_signal.result()
        frames_b64      = future_frames.result()

    log.info("Signal detector ai_score: %d", signal_ai_score)

    # Step 2: Build physics context from signal results to pass to GPT
    physics_context = _build_physics_context(signal_ai_score)

    # Step 3: Run GPT with physics context
    gpt_result = gpt_vision_score_with_context(frames_b64, physics_context)

    gpt_ai_score  = gpt_result["ai_probability"]
    gpt_available = gpt_result.get("available", False)
    gpt_reasoning = gpt_result.get("reasoning", "")
    log.info("GPT-4o ai_probability: %d  available: %s", gpt_ai_score, gpt_available)

    gpt_failed = (
        not gpt_available
        or gpt_reasoning.startswith("GPT analysis error")
        or gpt_reasoning.startswith("GPT vision not configured")
        or gpt_reasoning == "GPT analysis unavailable"
        or gpt_reasoning == "No frames extracted"
        or gpt_reasoning == "Could not extract frames"
        or ("rate" in gpt_reasoning.lower() and "429" in gpt_reasoning)
    )
    # Never treat as failed if GPT returned a real probability score
    if gpt_available and gpt_ai_score > 0:
        gpt_failed = False

    if not gpt_failed:
        signal_gpt_gap = abs(signal_ai_score - gpt_ai_score)

        if signal_gpt_gap > 40 and signal_ai_score > 60:
            # Signal says AI, GPT says real — trust signal more when physics engine fired
            weight_signal = 0.70
            weight_gpt    = 0.30
            log.info("High signal/GPT disagreement (gap=%d) + high signal — boosting signal weight to 70%%",
                     signal_gpt_gap)
        elif signal_gpt_gap > 40 and signal_ai_score < 40:
            # Signal says real, GPT says AI — trust signal more for low-signal-score videos
            weight_signal = 0.60
            weight_gpt    = 0.40
            log.info("High signal/GPT disagreement (gap=%d) + low signal — boosting signal weight to 60%%",
                     signal_gpt_gap)
        else:
            weight_signal = WEIGHT_SIGNAL
            weight_gpt    = WEIGHT_GPT

        combined_ai_score = (
            signal_ai_score * weight_signal +
            gpt_ai_score    * weight_gpt
        )
    else:
        combined_ai_score = float(signal_ai_score)
        log.warning("GPT-4o failed — using signal detector only (signal=%d)", signal_ai_score)

    combined_ai_score = max(0.0, min(100.0, combined_ai_score))
    authenticity = 100 - int(round(combined_ai_score))

    if authenticity >= THRESHOLD_REAL:
        label = "REAL"
    elif authenticity >= THRESHOLD_UNDETERMINED:
        label = "UNDETERMINED"
    else:
        label = "AI"

    # GPT fallback safety caps
    if gpt_failed and label == "AI" and signal_ai_score < SIGNAL_ONLY_AI_THRESHOLD:
        label = "UNDETERMINED"
        authenticity = THRESHOLD_UNDETERMINED + 5
        log.warning("GPT unavailable + signal=%d < %d — capping at UNDETERMINED",
                    signal_ai_score, SIGNAL_ONLY_AI_THRESHOLD)

    if gpt_failed and label == "REAL":
        label = "UNDETERMINED"
        authenticity = THRESHOLD_REAL - 1
        log.warning("GPT unavailable — capping REAL at UNDETERMINED (signal=%d)",
                    signal_ai_score)

    detail = {
        "ai_score":         int(round(combined_ai_score)),
        "authenticity":     authenticity,
        "label":            label,
        "signal_ai_score":  signal_ai_score,
        "gpt_ai_score":     gpt_ai_score,
        "gpt_available":    gpt_available,
        "gpt_failed":       gpt_failed,
        "gpt_reasoning":    gpt_reasoning,
        "gpt_flags":        gpt_result.get("flags", []),
        "threshold_real":   THRESHOLD_REAL,
        "threshold_undet":  THRESHOLD_UNDETERMINED,
    }

    log.info(
        "Detection complete | signal=%d gpt=%d(%s) combined=%d authenticity=%d label=%s",
        signal_ai_score, gpt_ai_score,
        "ok" if not gpt_failed else "FAILED",
        int(round(combined_ai_score)), authenticity, label,
    )

    return authenticity, label, detail











