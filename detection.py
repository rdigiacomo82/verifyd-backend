# ============================================================
#  VeriFYD — detection.py  v4
#
#  DUAL ENGINE:
#    1. detector.py   — signal-based (pixel-level evidence)
#    2. gpt_vision.py — GPT-4o semantic (visual reasoning)
#
#  HARD OVERRIDE (checked BEFORE engines run):
#    0. Metadata pre-check — AIGC tag, TikTok AI label, known
#       generator encoder signatures → instant AI result
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

import json
import logging
import os
import subprocess

from detector    import detect_ai
from gpt_vision  import gpt_vision_score_with_context, extract_key_frames

log = logging.getLogger("verifyd.detection")

THRESHOLD_REAL         = 55
THRESHOLD_UNDETERMINED = 40

# ─────────────────────────────────────────────────────────────
#  Metadata pre-check — hard override before signal/GPT engines
# ─────────────────────────────────────────────────────────────

def _check_metadata(video_path: str) -> dict | None:
    """
    Extract MP4/MOV format-level metadata via ffprobe and check for
    known AI-generation signatures embedded by generators or platforms.

    Returns a result dict (same shape as run_detection return) if a
    definitive AI signal is found, otherwise returns None so normal
    detection proceeds.

    Known signals checked:
      1. aigc_info tag — TikTok/Douyin official AI content label
         {"aigc_label_type": 2} = fully AI generated
         {"aigc_label_type": 1} = AI assisted
      2. Encoder fingerprint — known AI video generator encoders
         Lavf + no camera vendor = re-encoded AI output
      3. vendor_id "[0][0][0][0]" — stripped/generic vendor (AI pipeline)
    """
    try:
        ffprobe_bin = os.environ.get("FFPROBE_BIN", "ffprobe")
        result = subprocess.run(
            [ffprobe_bin, "-v", "quiet", "-print_format", "json",
             "-show_format", "-show_streams", video_path],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode != 0:
            return None

        data       = json.loads(result.stdout)
        fmt_tags   = data.get("format", {}).get("tags", {})
        streams    = data.get("streams", [])
        video_tags = next(
            (s.get("tags", {}) for s in streams if s.get("codec_type") == "video"),
            {}
        )

        # ── Signal 1: TikTok/Douyin AIGC label ───────────────
        aigc_raw = fmt_tags.get("aigc_info", "")
        if aigc_raw:
            try:
                aigc = json.loads(aigc_raw)
                aigc_type = int(aigc.get("aigc_label_type", -1))
                if aigc_type == 2:
                    log.info("METADATA_OVERRIDE: aigc_label_type=2 → FULLY AI GENERATED")
                    return _metadata_ai_result(
                        "TikTok/Douyin AIGC label: officially marked as AI-generated content",
                        ai_score=97
                    )
                elif aigc_type == 1:
                    log.info("METADATA_OVERRIDE: aigc_label_type=1 → AI ASSISTED")
                    return _metadata_ai_result(
                        "TikTok/Douyin AIGC label: officially marked as AI-assisted content",
                        ai_score=85
                    )
            except (json.JSONDecodeError, ValueError):
                pass

        # ── Signal 2: Known AI generator comment/tag patterns ─
        comment = fmt_tags.get("comment", "").lower()
        title   = fmt_tags.get("title",   "").lower()
        for field in (comment, title):
            for marker in ("aigc", "ai generated", "ai-generated", "sora", "kling",
                           "runway", "pika", "hailuo", "luma", "vidu", "cogvideo"):
                if marker in field:
                    log.info("METADATA_OVERRIDE: AI marker '%s' in tags", marker)
                    return _metadata_ai_result(
                        f"Metadata tag contains AI generation marker: '{marker}'",
                        ai_score=90
                    )

        # ── Signal 3: Encoder + vendor fingerprint ────────────
        encoder   = fmt_tags.get("encoder", "")
        vendor_id = video_tags.get("vendor_id", "")
        # Lavf (FFmpeg libavformat) with stripped vendor = re-encoded AI output
        # Real camera videos have camera vendor IDs: "appl", "FFMP", manufacturer codes
        # "[0][0][0][0]" is the null vendor — AI pipelines strip camera provenance
        is_lavf_reencoded = encoder.startswith("Lavf") and vendor_id == "[0][0][0][0]"
        # Only use encoder alone as a weak supporting signal (not a hard override)
        # — it's common in legitimate re-uploads too.
        if is_lavf_reencoded:
            log.info(
                "METADATA: Lavf encoder + null vendor_id detected "
                "(AI pipeline fingerprint — weak signal, not override)"
            )
            # Return None — let signal+GPT engines handle it, but log the evidence
            # This is available in the logs for debugging but doesn't hard-override
            # because many re-uploaded real videos also go through FFmpeg pipelines.

        return None

    except Exception as e:
        log.warning("Metadata pre-check failed: %s", e)
        return None


def _metadata_ai_result(reason: str, ai_score: int = 95) -> tuple:
    """Build a complete run_detection-compatible return tuple for metadata overrides."""
    authenticity = 100 - ai_score
    label = "AI" if authenticity < THRESHOLD_UNDETERMINED else "UNDETERMINED"
    detail = {
        "ai_score":        ai_score,
        "authenticity":    authenticity,
        "label":           label,
        "signal_ai_score": ai_score,
        "gpt_ai_score":    ai_score,
        "gpt_available":   False,
        "gpt_reasoning":   reason,
        "gpt_flags":       [reason],
        "weight_signal":   1.0,
        "weight_gpt":      0.0,
        "blend_mode":      "metadata-override",
        "threshold_real":  THRESHOLD_REAL,
        "threshold_undet": THRESHOLD_UNDETERMINED,
        "metadata_override": True,
        "metadata_reason": reason,
    }
    log.info("Metadata override → ai_score=%d authenticity=%d label=%s | %s",
             ai_score, authenticity, label, reason)
    return authenticity, label, detail


def run_detection(video_path: str) -> tuple:
    # ── Engine 0: Metadata pre-check (hard override) ──────────
    metadata_result = _check_metadata(video_path)
    if metadata_result is not None:
        return metadata_result

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
        clash_real  = signal_ai_score < 50 and gpt_ai_score > 50   # GPT uncertain/wrong, signal says real
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









