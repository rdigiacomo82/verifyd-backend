# ============================================================
#  VeriFYD — detection.py  v3
#
#  DUAL ENGINE:
#    1. detector.py   — signal-based (pixel-level evidence)
#    2. gpt_vision.py — GPT-4o semantic (visual reasoning)
#
#  WEIGHTING RULES (in priority order):
#    1. GPT failed            → signal 100%
#    2. Both agree strongly   → combined ± 5pt bonus
#    3. CLASH: signal<50, GPT>60 (GPT says AI, signal says real)
#              → signal 90% (GPT is wrong in every tested case)
#    4. CLASH: signal>65, GPT<40 (signal says AI, GPT misses it)
#              → signal 70% / GPT 30%
#    5. Signal confident (<35 or >75) → signal 60% / GPT 40%
#    6. Default               → signal 40% / GPT 60%
# ============================================================

import logging
from detector    import detect_ai
from gpt_vision  import gpt_vision_score_with_context, extract_key_frames

log = logging.getLogger("verifyd.detection")

THRESHOLD_REAL         = 55
THRESHOLD_UNDETERMINED = 40


def run_detection(video_path: str) -> tuple:
    # ── Engine 1: Signal detector ─────────────────────────────
    signal_ai_score = detect_ai(video_path)
    log.info("Signal detector ai_score: %d", signal_ai_score)

    # ── Engine 2: GPT-4o ──────────────────────────────────────
    frames_b64      = extract_key_frames(video_path)
    physics_context = {"signal_score": signal_ai_score}
    gpt_result      = gpt_vision_score_with_context(frames_b64, physics_context)
    gpt_ai_score    = gpt_result["ai_probability"]
    gpt_available   = gpt_result.get("available", False)
    log.info("GPT-4o ai_probability: %d  available: %s", gpt_ai_score, gpt_available)

    # ── Combine ───────────────────────────────────────────────
    gpt_failed = (
        not gpt_available or
        gpt_result.get("reasoning", "").startswith("GPT analysis error")
    )

    if gpt_failed:
        combined      = float(signal_ai_score)
        w_sig, w_gpt  = 1.0, 0.0
        mode          = "signal-only (GPT failed)"

    else:
        # Determine blend mode
        clash_real  = signal_ai_score < 50 and gpt_ai_score > 60   # GPT says AI, signal says real
        clash_ai    = signal_ai_score > 65 and gpt_ai_score < 40   # signal says AI, GPT misses it
        both_real   = signal_ai_score < 45 and gpt_ai_score < 45
        both_ai     = signal_ai_score > 65 and gpt_ai_score > 65
        confident   = signal_ai_score < 35 or signal_ai_score > 75

        if clash_real:
            # Signal has pixel evidence of real — GPT is over-calling AI
            combined, w_sig, w_gpt = (
                signal_ai_score * 0.90 + gpt_ai_score * 0.10,
                0.90, 0.10
            )
            mode = "clash→real (signal wins)"
        elif clash_ai:
            # Signal caught AI artifacts — GPT is under-calling
            combined, w_sig, w_gpt = (
                signal_ai_score * 0.70 + gpt_ai_score * 0.30,
                0.70, 0.30
            )
            mode = "clash→AI (signal leads)"
        elif both_real:
            combined  = signal_ai_score * 0.40 + gpt_ai_score * 0.60 - 5
            w_sig, w_gpt = 0.40, 0.60
            mode = "both-real bonus"
        elif both_ai:
            combined  = signal_ai_score * 0.40 + gpt_ai_score * 0.60 + 5
            w_sig, w_gpt = 0.40, 0.60
            mode = "both-AI bonus"
        elif confident:
            combined, w_sig, w_gpt = (
                signal_ai_score * 0.60 + gpt_ai_score * 0.40,
                0.60, 0.40
            )
            mode = "signal-confident"
        else:
            combined, w_sig, w_gpt = (
                signal_ai_score * 0.40 + gpt_ai_score * 0.60,
                0.40, 0.60
            )
            mode = "default"

        log.info("Blend mode: %s | signal=%d gpt=%d w_sig=%.0f%% w_gpt=%.0f%%",
                 mode, signal_ai_score, gpt_ai_score, w_sig*100, w_gpt*100)

    combined_ai_score = max(0.0, min(100.0, combined))
    authenticity      = 100 - int(round(combined_ai_score))

    if authenticity >= THRESHOLD_REAL:
        label = "REAL"
    elif authenticity >= THRESHOLD_UNDETERMINED:
        label = "UNDETERMINED"
    else:
        label = "AI"

    detail = {
        "ai_score":        int(round(combined_ai_score)),
        "authenticity":    authenticity,
        "label":           label,
        "signal_ai_score": signal_ai_score,
        "gpt_ai_score":    gpt_ai_score,
        "gpt_available":   gpt_available,
        "gpt_reasoning":   gpt_result.get("reasoning", ""),
        "gpt_flags":       gpt_result.get("flags", []),
        "weight_signal":   w_sig,
        "weight_gpt":      w_gpt,
        "blend_mode":      mode if not gpt_failed else "signal-only",
        "threshold_real":  THRESHOLD_REAL,
        "threshold_undet": THRESHOLD_UNDETERMINED,
    }

    log.info(
        "Detection complete | signal=%d gpt=%d combined=%d authenticity=%d label=%s",
        signal_ai_score, gpt_ai_score,
        int(round(combined_ai_score)), authenticity, label,
    )
    return authenticity, label, detail











