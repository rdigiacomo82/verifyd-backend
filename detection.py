# ============================================================
#  VeriFYD — detection.py  v4
#
#  N-ENGINE ARCHITECTURE:
#    Engine 1: detector.py        — pixel-level signal analysis
#    Engine 2: gpt_vision.py      — GPT-4o 12-dimension rubric
#    Engine 3: metadata_detector.py — container/encoder forensics
#    Engine 4: audio_detector.py  — spectral/voice synthesis
#
#  Each engine runs independently and returns a 0–100 AI score
#  plus a confidence level. The final score is a weighted blend
#  where weights depend on content type and per-engine confidence.
#
#  BLEND PHILOSOPHY:
#    - Signal engine is the anchor — has direct pixel measurement.
#      When it's highly confident (score <35 or >75), it gets
#      elevated weight and can override low-confidence engines.
#    - GPT engine provides semantic reasoning. After v5 rubric
#      rewrite, it's now more consistent and debuggable.
#    - Metadata engine is low-weight but high-precision: a strong
#      metadata AI signal (e.g. known AI encoder) can boost score;
#      a strong real signal (camera make/model + GPS) reduces it.
#    - Audio engine catches a dimension no visual engine can: voice
#      synthesis artifacts and missing environmental audio.
#    - When engines clash, signal wins over GPT (validated in testing).
#      Metadata and audio are treated as auxiliary boosters.
#
#  CONTENT-TYPE WEIGHTS:
#    talking_head / selfie  → GPT weight elevated (texture analysis)
#    action / cinematic     → Signal weight elevated (physics)
#    all types              → Metadata/audio contribute as auxiliary
#
#  DETAIL DICT:
#    Every engine's score, confidence, evidence, and GPT dimension
#    scores are stored in the detail dict for logging and feedback.
# ============================================================

import logging
from detector          import detect_ai
from gpt_vision        import gpt_vision_score_with_context, extract_key_frames
from metadata_detector import analyze_metadata
from audio_detector    import analyze_audio

log = logging.getLogger("verifyd.detection")

THRESHOLD_REAL         = 55
THRESHOLD_UNDETERMINED = 40

# ─────────────────────────────────────────────────────────────
#  Per-content-type base weights for signal + GPT engines
#  (metadata and audio are always auxiliary — see _aux_adjustment)
# ─────────────────────────────────────────────────────────────
_ENGINE_WEIGHTS = {
    #                       signal  gpt
    "talking_head":         (0.35,  0.65),
    "selfie":               (0.35,  0.65),
    "single_subject":       (0.45,  0.55),
    "action":               (0.60,  0.40),
    "cinematic":            (0.55,  0.45),
    "static":               (0.50,  0.50),
    "default":              (0.45,  0.55),
}


def run_detection(video_path: str) -> tuple:
    """
    Run all available detection engines and blend into final verdict.

    Returns:
        authenticity : int 0–100 (100 = definitely real, 0 = definitely AI)
        label        : "REAL" | "UNDETERMINED" | "AI"
        detail       : dict with all engine scores and metadata
    """

    # ══════════════════════════════════════════════════════════
    # ENGINE 1: Signal detector (pixel-level)
    # ══════════════════════════════════════════════════════════
    signal_ai_score, signal_context = detect_ai(video_path)
    content_type = signal_context.get("content_type", "default")
    log.info("Engine 1 — Signal: ai_score=%d  content_type=%s",
             signal_ai_score, content_type)

    # ══════════════════════════════════════════════════════════
    # ENGINE 2: GPT-4o 12-dimension rubric
    # ══════════════════════════════════════════════════════════
    frames_b64 = extract_key_frames(video_path)
    gpt_result = gpt_vision_score_with_context(frames_b64, signal_context)
    gpt_ai_score  = gpt_result["ai_probability"]
    gpt_available = gpt_result.get("available", False)
    gpt_scores    = gpt_result.get("scores", {})
    log.info("Engine 2 — GPT: ai_score=%d  available=%s  generator=%s",
             gpt_ai_score, gpt_available, gpt_result.get("generator_guess", "?"))

    # ══════════════════════════════════════════════════════════
    # ENGINE 3: Metadata forensics
    # ══════════════════════════════════════════════════════════
    meta_result     = analyze_metadata(video_path)
    meta_ai_score   = meta_result["metadata_ai_score"]
    meta_confidence = meta_result["confidence"]
    log.info("Engine 3 — Metadata: ai_score=%d  confidence=%s",
             meta_ai_score, meta_confidence)

    # ══════════════════════════════════════════════════════════
    # ENGINE 4: Audio analysis
    # ══════════════════════════════════════════════════════════
    audio_result     = analyze_audio(video_path)
    audio_ai_score   = audio_result["audio_ai_score"]
    audio_confidence = audio_result["confidence"]
    audio_available  = audio_result.get("available", False)
    log.info("Engine 4 — Audio: ai_score=%d  confidence=%s",
             audio_ai_score, audio_confidence)

    # ══════════════════════════════════════════════════════════
    # BLEND — signal + GPT (primary)
    # ══════════════════════════════════════════════════════════
    gpt_failed = (
        not gpt_available or
        gpt_result.get("reasoning", "").startswith("GPT analysis error")
    )

    if gpt_failed:
        combined_primary = float(signal_ai_score)
        w_sig, w_gpt     = 1.0, 0.0
        blend_mode       = "signal-only (GPT failed)"
    else:
        w_sig, w_gpt = _ENGINE_WEIGHTS.get(content_type, _ENGINE_WEIGHTS["default"])

        # Override weights based on confidence / clash patterns
        clash_real = signal_ai_score < 50 and gpt_ai_score > 50
        clash_ai   = signal_ai_score > 65 and gpt_ai_score < 40
        both_real  = signal_ai_score < 45 and gpt_ai_score < 45
        both_ai    = signal_ai_score > 65 and gpt_ai_score > 65
        sig_confident = signal_ai_score < 35 or signal_ai_score > 75

        if clash_real:
            # Signal has pixel evidence of real — GPT over-calling AI
            w_sig, w_gpt = 0.90, 0.10
            blend_mode   = "clash→real (signal wins)"
        elif clash_ai:
            # Signal caught AI artifacts — GPT under-calling
            w_sig, w_gpt = 0.70, 0.30
            blend_mode   = "clash→AI (signal leads)"
        elif both_real:
            w_sig, w_gpt = 0.40, 0.60
            blend_mode   = "both-real"
        elif both_ai:
            w_sig, w_gpt = 0.40, 0.60
            blend_mode   = "both-AI"
        elif sig_confident:
            # Keep content-type weights but floor signal at 60%
            w_sig = max(w_sig, 0.60)
            w_gpt = 1.0 - w_sig
            blend_mode = "signal-confident"
        else:
            blend_mode = f"default ({content_type})"

        combined_primary = signal_ai_score * w_sig + gpt_ai_score * w_gpt

        # Both-agree bonus/penalty
        if both_real:
            combined_primary -= 5
        elif both_ai:
            combined_primary += 5

    log.info("Primary blend: mode=%s  signal=%d  gpt=%d  w_sig=%.0f%%  w_gpt=%.0f%%  combined=%.1f",
             blend_mode, signal_ai_score, gpt_ai_score, w_sig*100, w_gpt*100, combined_primary)

    # ══════════════════════════════════════════════════════════
    # AUXILIARY ADJUSTMENTS — metadata + audio
    # These engines are capped to ±12 adjustment each so they
    # can't flip a verdict alone, but they push ambiguous cases.
    # ══════════════════════════════════════════════════════════
    meta_adjustment  = _aux_adjustment(meta_ai_score,  meta_confidence,  cap=12)
    audio_adjustment = _aux_adjustment(audio_ai_score, audio_confidence, cap=10)

    # Suppress audio adjustment for no_audio confidence (it's already baked in)
    if audio_confidence == "no_audio":
        audio_adjustment = _aux_adjustment(audio_ai_score, "low", cap=6)

    combined_final = combined_primary + meta_adjustment + audio_adjustment
    combined_final = max(0.0, min(100.0, combined_final))

    log.info("Aux adjustments: meta=%+.1f  audio=%+.1f  final=%.1f",
             meta_adjustment, audio_adjustment, combined_final)

    # ══════════════════════════════════════════════════════════
    # VERDICT
    # ══════════════════════════════════════════════════════════
    authenticity = 100 - int(round(combined_final))

    if authenticity >= THRESHOLD_REAL:
        label = "REAL"
    elif authenticity >= THRESHOLD_UNDETERMINED:
        label = "UNDETERMINED"
    else:
        label = "AI"

    # ── Detail dict — full evidence trail ────────────────────
    detail = {
        # Final verdict
        "ai_score":        int(round(combined_final)),
        "authenticity":    authenticity,
        "label":           label,
        "blend_mode":      blend_mode,

        # Engine 1 — Signal
        "signal_ai_score": signal_ai_score,
        "content_type":    content_type,

        # Engine 2 — GPT
        "gpt_ai_score":    gpt_ai_score,
        "gpt_available":   gpt_available,
        "gpt_reasoning":   gpt_result.get("reasoning", ""),
        "gpt_flags":       gpt_result.get("flags", []),
        "gpt_scores":      gpt_scores,               # NEW: per-dimension breakdown
        "generator_guess": gpt_result.get("generator_guess", "Unknown"),  # NEW

        # Engine 3 — Metadata
        "metadata_ai_score":  meta_ai_score,
        "metadata_confidence":meta_confidence,
        "metadata_evidence":  meta_result.get("evidence", []),
        "metadata_adjustment":meta_adjustment,

        # Engine 4 — Audio
        "audio_ai_score":    audio_ai_score,
        "audio_confidence":  audio_confidence,
        "audio_evidence":    audio_result.get("evidence", []),
        "audio_available":   audio_available,
        "audio_adjustment":  audio_adjustment,

        # Blend weights
        "weight_signal":   w_sig,
        "weight_gpt":      w_gpt,

        # Thresholds (for frontend display)
        "threshold_real":  THRESHOLD_REAL,
        "threshold_undet": THRESHOLD_UNDETERMINED,
    }

    log.info(
        "Detection complete | signal=%d gpt=%d meta=%d audio=%d "
        "combined=%.1f auth=%d label=%s",
        signal_ai_score, gpt_ai_score, meta_ai_score, audio_ai_score,
        combined_final, authenticity, label,
    )

    return authenticity, label, detail


# ─────────────────────────────────────────────────────────────
#  Auxiliary engine adjustment helper
# ─────────────────────────────────────────────────────────────
def _aux_adjustment(ai_score: int, confidence: str, cap: int = 12) -> float:
    """
    Convert an auxiliary engine score into a ±cap adjustment
    for the primary blend. Neutral (50) = 0 adjustment.
    High confidence amplifies; low confidence attenuates.
    """
    if confidence == "low":
        multiplier = 0.25
    elif confidence == "medium":
        multiplier = 0.55
    elif confidence == "high":
        multiplier = 0.85
    elif confidence == "no_audio":
        multiplier = 0.30
    else:
        multiplier = 0.30

    # Raw deviation from neutral (50) → ±cap
    deviation = (ai_score - 50) / 50.0 * cap * multiplier
    return round(deviation, 2)










