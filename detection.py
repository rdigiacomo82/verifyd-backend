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


def _check_metadata_override(video_path: str) -> tuple:
    """
    Check for definitive metadata signals before running engines.
    Returns (override: bool, ai_score: int, label: str, reason: str)
    """
    import subprocess, json as _json, os

    # 1. Check sidecar file written by SMVD TikTok downloader
    sidecar = video_path.replace(".mp4", ".meta.json")
    if os.path.exists(sidecar):
        try:
            with open(sidecar) as sf:
                meta = _json.load(sf)
            aigc = int(meta.get("aigc_label_type", 0))
            log.info("METADATA: sidecar aigc_label_type=%d source=%s", aigc, meta.get("source", "?"))
            if aigc == 2:
                reason = "TikTok/Douyin AIGC label: officially marked as AI-generated content"
                log.info("METADATA_OVERRIDE: aigc_label_type=2 → FULLY AI GENERATED")
                return True, 97, "AI", reason
            elif aigc == 1:
                reason = "TikTok/Douyin AIGC label: AI-assisted content"
                log.info("METADATA: aigc_label_type=1 → AI assisted (weak signal, not override)")
        except Exception as e:
            log.warning("METADATA: sidecar read error: %s", e)

    # 2. Check MP4 format tags via ffprobe (uploaded files)
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_format", "-show_streams", video_path],
            capture_output=True, text=True, timeout=15
        )
        data = _json.loads(result.stdout)
        fmt_tags = data.get("format", {}).get("tags", {})
        encoder  = fmt_tags.get("encoder", fmt_tags.get("ENCODER", "")).lower()
        vendor   = ""
        for s in data.get("streams", []):
            vendor = s.get("tags", {}).get("vendor_id", "")
            if vendor:
                break

        # TikTok/Douyin AIGC tag in format tags
        aigc_info = fmt_tags.get("aigc_info", "")
        if aigc_info:
            try:
                aigc_data  = _json.loads(aigc_info)
                aigc_label = int(aigc_data.get("aigc_label_type", 0))
                if aigc_label == 2:
                    reason = "TikTok/Douyin AIGC label: officially marked as AI-generated content"
                    log.info("METADATA_OVERRIDE: aigc_label_type=2 → FULLY AI GENERATED")
                    return True, 97, "AI", reason
                elif aigc_label == 1:
                    log.info("METADATA: aigc_label_type=1 → AI assisted (weak signal, not override)")
            except Exception:
                pass

        # Lavf encoder + null vendor = AI pipeline fingerprint (weak, not override)
        if "lavf" in encoder and vendor in ("[0][0][0][0]", "", "0000"):
            log.info("METADATA: Lavf encoder + null vendor_id detected (AI pipeline fingerprint — weak signal, not override)")

    except Exception as e:
        log.warning("METADATA: ffprobe error: %s", e)

    return False, 0, "", ""


def run_detection(video_path: str) -> tuple:
    # ── Pre-check: Metadata override ─────────────────────────
    override, ov_ai_score, ov_label, ov_reason = _check_metadata_override(video_path)
    if override:
        authenticity = 100 - ov_ai_score
        log.info("Metadata override → ai_score=%d authenticity=%d label=%s | %s",
                 ov_ai_score, authenticity, ov_label, ov_reason)
        return authenticity, ov_label, {
            "ai_score":        ov_ai_score,
            "signal_ai_score": ov_ai_score,
            "gpt_ai_score":    ov_ai_score,
            "gpt_reasoning":   ov_reason,
            "gpt_flags":       ["metadata_override"],
            "content_type":    "unknown",
            "blend_mode":      "metadata-override",
            "weight_signal":   1.0,
            "weight_gpt":      0.0,
        }

    # ── Engine 1: Signal detector ─────────────────────────────
    signal_ai_score, signal_context = detect_ai(video_path)
    log.info("Signal detector ai_score: %d  content_type: %s",
             signal_ai_score, signal_context.get("content_type", "unknown"))

    # ── Engine 2: GPT-4o ──────────────────────────────────────
    frames_b64      = extract_key_frames(video_path)
    gpt_result      = gpt_vision_score_with_context(frames_b64, signal_context)
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
        # clash_real: signal says real BUT only override GPT if GPT is not highly confident
        # If GPT >= 75, it's seeing strong AI artifacts — don't let signal dismiss it
        clash_real  = signal_ai_score < 50 and gpt_ai_score > 50 and gpt_ai_score < 75
        clash_ai    = signal_ai_score > 65 and gpt_ai_score < 40   # signal says AI, GPT misses it
        gpt_dominant = gpt_ai_score >= 75 and signal_ai_score < 60  # GPT highly confident AI, signal unsure
        both_real   = signal_ai_score < 45 and gpt_ai_score < 45
        both_ai     = signal_ai_score > 65 and gpt_ai_score > 65
        confident   = signal_ai_score < 35 or signal_ai_score > 75

        if gpt_dominant:
            # GPT is highly confident AI but signal is borderline — trust GPT heavily
            combined, w_sig, w_gpt = (
                signal_ai_score * 0.25 + gpt_ai_score * 0.75,
                0.25, 0.75
            )
            mode = "gpt-dominant (GPT highly confident AI)"
        elif clash_real:
            # Signal has pixel evidence of real — GPT is moderately calling AI
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









