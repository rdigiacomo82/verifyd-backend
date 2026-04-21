# ============================================================
#  VeriFYD — photo_detection.py  v1
#
#  DUAL ENGINE for photos (parallel to detection.py for video):
#    1. photo_detector.py — signal-based (ELA, PRNU, DCT, etc.)
#    2. gpt_vision.py     — GPT-4o semantic analysis
#
#  Key differences from video detection:
#    - No multi-clip, no temporal signals, no LAVF/CHAN_CORR boost
#    - No hybrid flag (single image, no variance across clips)
#    - ELA is a strong differentiator not available in video
#    - EXIF metadata is more diagnostic for photos
#    - Real device ceiling applies (same as video)
#    - Simpler blend logic — fewer edge cases
#
#  Returns: (authenticity: int, label: str, detail: dict)
#  Same signature as run_detection_multiclip() — drop-in compatible.
# ============================================================
import os
import logging
import base64
import cv2

log = logging.getLogger("verifyd.photo_detection")

THRESHOLD_REAL         = 55
THRESHOLD_UNDETERMINED = 40


# ─────────────────────────────────────────────────────────────
#  Metadata check (reuse logic from detection.py)
# ─────────────────────────────────────────────────────────────
def _check_photo_metadata(image_path: str) -> tuple:
    """
    Check image metadata for definitive signals.
    Returns (override: bool, ai_score: int, label: str, reason: str)
    """
    import subprocess, json as _j
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_format", image_path],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode != 0:
            return False, 0, "", ""

        data  = _j.loads(result.stdout)
        tags  = data.get("format", {}).get("tags", {})
        sw    = tags.get("software", tags.get("Software", "")).lower()

        # Explicit AI generator tag
        ai_tools = ["midjourney", "dall-e", "dall·e", "stable diffusion",
                    "firefly", "kling", "sora", "runway", "pika", "ideogram",
                    "leonardo", "adobe firefly", "nightcafe", "image creator"]
        for tool in ai_tools:
            if tool in sw:
                reason = f"Image software tag identifies AI generator: {sw}"
                log.info("PHOTO_META_OVERRIDE: AI tool in software tag → %s", sw)
                return True, 97, "AI", reason

        # Real device metadata
        make  = tags.get("make",  tags.get("Make",  ""))
        model = tags.get("model", tags.get("Model", ""))
        if make or model:
            reason = f"Real camera EXIF: {make} {model}".strip()
            log.info("PHOTO_META: Real camera EXIF detected → %s", reason)
            return True, 10, "REAL", reason

    except Exception as e:
        log.debug("PHOTO_META: error: %s", e)

    return False, 0, "", ""


# ─────────────────────────────────────────────────────────────
#  Extract frames for GPT (single image)
# ─────────────────────────────────────────────────────────────
def _extract_photo_frames(image_path: str, n_crops: int = 4) -> list:
    """
    For photos, extract the full image plus regional crops.
    This gives GPT both the overall view and close-up texture detail.
    Returns list of base64-encoded JPEG strings.
    """
    import tempfile
    frames = []
    try:
        img = cv2.imread(image_path)
        if img is None:
            return []

        h, w = img.shape[:2]

        # Full image (resized to max 1024px on longest side)
        max_dim = 1024
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            img_full = cv2.resize(img, (int(w * scale), int(h * scale)))
        else:
            img_full = img.copy()

        _, buf = cv2.imencode(".jpg", img_full, [cv2.IMWRITE_JPEG_QUALITY, 90])
        frames.append(base64.b64encode(buf.tobytes()).decode("utf-8"))

        # Regional crops for texture/detail analysis
        # Divide into quadrants + center crop
        crops = [
            (0,     0,     w//2,   h//2),   # top-left
            (w//2,  0,     w,      h//2),   # top-right
            (w//4,  h//4,  3*w//4, 3*h//4), # center
            (0,     h//2,  w//2,   h),      # bottom-left
        ]
        for x1, y1, x2, y2 in crops[:n_crops - 1]:
            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            # Resize crop to 512px
            ch, cw = crop.shape[:2]
            if max(ch, cw) > 512:
                cs = 512 / max(ch, cw)
                crop = cv2.resize(crop, (int(cw * cs), int(ch * cs)))
            _, cbuf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
            frames.append(base64.b64encode(cbuf.tobytes()).decode("utf-8"))

        log.info("photo_detection: extracted %d frames/crops from %s", len(frames), image_path)
        return frames

    except Exception as e:
        log.error("photo_detection: frame extraction error: %s", e)
        return []


# ─────────────────────────────────────────────────────────────
#  Build GPT context for photos
# ─────────────────────────────────────────────────────────────
def _build_photo_gpt_context(signal_score: int, signal_context: dict) -> dict:
    """
    Build the context dict passed to gpt_vision_score_with_context.
    Maps photo signal names to the names gpt_vision.py expects.
    """
    ctx = dict(signal_context)
    ctx["content_type"]  = signal_context.get("content_type", "portrait")
    ctx["signal_score"]  = signal_score
    ctx["is_photo"]      = True

    # Map photo-specific signals to GPT-readable names
    ctx["avg_noise"]     = signal_context.get("avg_noise", 500)
    ctx["avg_saturation"] = signal_context.get("avg_saturation", 80)
    ctx["flat_noise"]    = signal_context.get("flat_noise", 1.0)
    ctx["chan_corr"]     = signal_context.get("chan_corr", 0.7)

    # Photo-specific hints
    ela = signal_context.get("ela_score", 50)
    dct = signal_context.get("dct_uniformity", 50)
    tex = signal_context.get("texture_var", 500)
    meta = signal_context.get("meta_adjustment", 0)

    photo_notes = []
    if ela > 25:
        photo_notes.append(
            f"ELA (Error Level Analysis) score={ela:.0f}: "
            "JPEG re-compression inconsistencies detected. "
            "Composited or AI-generated regions show boundary artifacts. "
            "Check for edges where different parts of the image have "
            "different compression levels — this is a hallmark of AI compositing."
        )
    if signal_context.get("flat_noise", 1.0) < 0.25:
        photo_notes.append(
            "Very low sensor noise in flat regions (no PRNU detected). "
            "Real camera sensors leave a unique noise fingerprint. "
            "AI-rendered images have near-zero noise in flat areas."
        )
    if signal_context.get("chan_corr", 0.7) > 0.93:
        photo_notes.append(
            f"High RGB channel correlation ({signal_context['chan_corr']:.3f}). "
            "AI renders have near-perfect inter-channel lock. "
            "Real cameras have independent channel noise."
        )
    if dct > 65:
        photo_notes.append(
            "Unusually uniform DCT compression blocks. "
            "AI-generated images have suspiciously consistent block energy. "
            "Real photos have high variance in compression block structure."
        )
    if tex < 80:
        photo_notes.append(
            "Very low local texture variance. "
            "AI skin and surfaces are unnaturally smooth across patches. "
            "Score skin_texture high if skin looks airbrushed or porcelain."
        )
    if meta < -10:
        photo_notes.append(
            f"Real camera EXIF detected (metadata_adjustment={meta}). "
            "Camera make/model/GPS present. This is a strong real camera indicator."
        )
    if meta > 30:
        photo_notes.append(
            "AI generator software tag found in EXIF metadata. "
            "Score generator_artifacts 9-10 immediately."
        )

    if photo_notes:
        ctx["photo_signal_notes"] = "\n".join(
            [f"PHOTO SIGNAL ANALYSIS — guide your inspection:"] + 
            [f"  • {n}" for n in photo_notes]
        )

    return ctx


# ─────────────────────────────────────────────────────────────
#  Main detection function
# ─────────────────────────────────────────────────────────────
def run_photo_detection(image_path: str) -> tuple:
    """
    Full dual-engine photo detection.

    Args:
        image_path: path to image file (jpg, png, webp, heic)

    Returns:
        (authenticity: int, label: str, detail: dict)
        Same signature as run_detection_multiclip().
    """
    from photo_detector import detect_ai_photo
    from gpt_vision import gpt_vision_score_with_context

    # ── Metadata fast-path ───────────────────────────────────
    override, ov_score, ov_label, ov_reason = _check_photo_metadata(image_path)

    if override and ov_label == "AI":
        authenticity = 100 - ov_score
        log.info("Photo metadata override → AI (score=%d) | %s", ov_score, ov_reason)
        return authenticity, "AI", {
            "ai_score":        ov_score,
            "signal_ai_score": ov_score,
            "gpt_ai_score":    ov_score,
            "gpt_reasoning":   ov_reason,
            "gpt_flags":       ["metadata_override"],
            "content_type":    "photo",
            "blend_mode":      "metadata-override",
            "weight_signal":   1.0,
            "weight_gpt":      0.0,
        }

    # ── Engine 1: Signal detector ────────────────────────────
    signal_score, signal_context = detect_ai_photo(image_path)
    log.info("Photo signal score: %d  content_type: %s",
             signal_score, signal_context.get("content_type", "photo"))

    # ── Engine 2: GPT-4o vision ──────────────────────────────
    frames = _extract_photo_frames(image_path)
    if not frames:
        log.warning("Photo detection: no frames extracted — signal only")
        gpt_result = {
            "ai_probability": 50,
            "reasoning": "Image could not be processed for visual analysis.",
            "flags": [], "available": False, "scores": {}, "generator_guess": "Unknown"
        }
    else:
        gpt_context = _build_photo_gpt_context(signal_score, signal_context)
        # Inject photo signal notes into context for _build_physics_summary
        if "photo_signal_notes" in gpt_context:
            gpt_context["_extra_context"] = gpt_context.pop("photo_signal_notes")
        gpt_result = gpt_vision_score_with_context(frames, gpt_context)

    gpt_score     = gpt_result.get("ai_probability", 50)
    gpt_available = gpt_result.get("available", False)
    gpt_reasoning = gpt_result.get("reasoning", "")
    gpt_flags     = gpt_result.get("flags", [])
    log.info("Photo GPT score: %d  available: %s", gpt_score, gpt_available)

    # ── Blend ────────────────────────────────────────────────
    gpt_failed = (
        not gpt_available or
        gpt_reasoning.startswith("GPT analysis error")
    )

    if gpt_failed:
        combined = float(signal_score)
        w_sig, w_gpt = 1.0, 0.0
        mode = "signal-only (GPT failed)"
    else:
        gpt_dominant  = gpt_score >= 75 and signal_score < 60
        clash_ai      = signal_score > 65 and gpt_score < 40
        clash_real    = signal_score < 45 and gpt_score > 55 and gpt_score < 75
        both_real     = signal_score < 45 and gpt_score < 45
        both_ai       = signal_score > 65 and gpt_score > 65

        if gpt_dominant:
            combined, w_sig, w_gpt = (
                signal_score * 0.25 + gpt_score * 0.75, 0.25, 0.75
            )
            mode = "gpt-dominant"
        elif clash_ai:
            combined, w_sig, w_gpt = (
                signal_score * 0.70 + gpt_score * 0.30, 0.70, 0.30
            )
            mode = "clash→AI (signal leads)"
        elif clash_real:
            combined, w_sig, w_gpt = (
                signal_score * 0.85 + gpt_score * 0.15, 0.85, 0.15
            )
            mode = "clash→real (signal wins)"
        elif both_real:
            combined  = signal_score * 0.40 + gpt_score * 0.60 - 5
            w_sig, w_gpt = 0.40, 0.60
            mode = "both-real bonus"
        elif both_ai:
            combined  = signal_score * 0.40 + gpt_score * 0.60 + 5
            w_sig, w_gpt = 0.40, 0.60
            mode = "both-AI bonus"
        else:
            combined, w_sig, w_gpt = (
                signal_score * 0.40 + gpt_score * 0.60, 0.40, 0.60
            )
            mode = "default"

        log.info("Photo blend: %s | signal=%d gpt=%d w_sig=%.0f%% w_gpt=%.0f%%",
                 mode, signal_score, gpt_score, w_sig * 100, w_gpt * 100)

    # ── Real device metadata ceiling ─────────────────────────
    # Same logic as video: if EXIF confirms real camera but signals
    # score high (unusual lighting, extreme macro, etc.), cap at UNDETERMINED
    if override and ov_label == "REAL" and combined >= 70:
        old = combined
        combined = min(combined, 58.0)
        log.info(
            "Photo real device ceiling: combined %.1f→%.1f "
            "(EXIF confirmed real camera, preventing AI verdict) | %s",
            old, combined, ov_reason
        )

    # ── Generator artifact hard floor ────────────────────────
    gen_score = gpt_result.get("scores", {}).get("generator_artifacts", 0)
    if gen_score >= 8:
        combined = max(combined, 90.0)
        log.info("Photo: generator_artifacts=%d → floor 90", gen_score)

    combined_ai_score = max(0.0, min(100.0, combined))
    authenticity      = 100 - int(round(combined_ai_score))
    authenticity      = max(0, min(100, authenticity))

    label = (
        "REAL"          if authenticity >= THRESHOLD_REAL        else
        "UNDETERMINED"  if authenticity >= THRESHOLD_UNDETERMINED else
        "AI"
    )

    # ── Reasoning ────────────────────────────────────────────
    content_type   = signal_context.get("content_type", "photo")
    content_labels = {
        "portrait": "photo",
        "action":   "photo",
        "scene":    "photo",
        "photo":    "photo",
    }
    content_friendly = content_labels.get(content_type, "photo")

    _gpt_specific = (
        gpt_reasoning and len(gpt_reasoning) > 60 and
        "inconclusive" not in gpt_reasoning.lower() and
        "error" not in gpt_reasoning.lower()
    )

    if label == "REAL":
        prefix = (
            f"This {content_friendly} shows authentic camera characteristics. "
            if authenticity >= 75
            else f"This {content_friendly} appears to be genuine camera footage. "
        )
    elif label == "UNDETERMINED":
        prefix = (
            f"This {content_friendly} could not be conclusively verified "
            f"({authenticity}% authenticity). "
        )
    else:
        prefix = f"This {content_friendly} shows AI generation signatures. "

    user_reasoning = (prefix + gpt_reasoning) if _gpt_specific else prefix

    log.info(
        "Photo detection complete | signal=%d gpt=%d combined=%d "
        "authenticity=%d label=%s mode=%s",
        signal_score, gpt_score, int(round(combined_ai_score)),
        authenticity, label, mode
    )

    detail = {
        "ai_score":           int(round(combined_ai_score)),
        "signal_ai_score":    signal_score,
        "gpt_ai_score":       gpt_score,
        "gpt_available":      gpt_available,
        "gpt_reasoning":      user_reasoning,
        "gpt_flags":          gpt_flags,
        "gpt_scores":         gpt_result.get("scores", {}),
        "generator_guess":    gpt_result.get("generator_guess", "Unknown"),
        "weight_signal":      w_sig,
        "weight_gpt":         w_gpt,
        "blend_mode":         mode,
        "content_type":       content_type,
        "ela_score":          signal_context.get("ela_score", 0),
        "flat_noise":         signal_context.get("flat_noise", 0),
        "chan_corr":           signal_context.get("chan_corr", 0),
        "metadata":           signal_context.get("metadata", {}),
        "threshold_real":     THRESHOLD_REAL,
        "threshold_undet":    THRESHOLD_UNDETERMINED,
    }

    return authenticity, label, detail
