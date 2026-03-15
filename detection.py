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

        # Android/iOS device metadata = definitive real camera recording
        # These tags are written by the device OS and cannot be faked by AI generators
        android_ver = fmt_tags.get("com.android.version", "")
        apple_make  = fmt_tags.get("com.apple.quicktime.make", "")
        apple_model = fmt_tags.get("com.apple.quicktime.model", "")
        major_brand = fmt_tags.get("major_brand", "")
        creation_time = fmt_tags.get("creation_time", "")

        is_android = bool(android_ver)
        is_apple   = bool(apple_make or apple_model)
        # mp42/isom with creation_time and no Lavf encoder = real device recording
        is_real_container = (
            major_brand in ("mp42", "isom", "M4V ", "qt  ") and
            bool(creation_time) and
            "lavf" not in encoder.lower()
        )

        if is_android:
            log.info("METADATA: Android device recording detected (version=%s) → real camera", android_ver)
            return True, 3, "REAL", f"Android device recording (Android {android_ver})"
        elif is_apple:
            log.info("METADATA: Apple device recording detected → real camera")
            return True, 3, "REAL", f"Apple device recording ({apple_make} {apple_model})"
        elif is_real_container:
            log.info("METADATA: Real device container (brand=%s, creation_time=%s) → likely real camera", major_brand, creation_time[:10])
            return True, 15, "REAL", f"Real device container (brand={major_brand}, recorded {creation_time[:10]})"

    except Exception as e:
        log.warning("METADATA: ffprobe error: %s", e)

    return False, 0, "", ""


def run_detection(video_path: str) -> tuple:
    # ── Pre-check: Metadata override ─────────────────────────
    override, ov_ai_score, ov_label, ov_reason = _check_metadata_override(video_path)
    if override and ov_label == "AI":
        authenticity = 100 - ov_ai_score
        log.info("Metadata override → ai_score=%d authenticity=%d label=AI | %s",
                 ov_ai_score, authenticity, ov_reason)
        return authenticity, "AI", {
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
    elif override and ov_label == "REAL":
        # Device metadata confirms real camera — still run full detection
        # but pass the device confirmation as a strong prior to both engines
        log.info("Metadata real-device signal → ai_score=%d | %s (running full detection)", 
                 ov_ai_score, ov_reason)
        # ov_ai_score is very low (3-15) — will be incorporated via signal_context

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

    # Build content-aware user-facing reasoning for single-clip path
    user_reasoning = _build_content_aware_reasoning(
        label=label,
        authenticity=authenticity,
        content_type=signal_context.get("content_type", "cinematic"),
        signal_scores=[signal_ai_score],
        gpt_score=gpt_ai_score,
        hybrid_flag=False,
        gpt_reasoning=gpt_result.get("reasoning", ""),
        gpt_flags=gpt_result.get("flags", []),
        n_clips=1,
    )
    detail["gpt_reasoning"] = user_reasoning

    log.info(
        "Detection complete | signal=%d gpt=%d combined=%d authenticity=%d label=%s",
        signal_ai_score, gpt_ai_score,
        int(round(combined_ai_score)), authenticity, label,
    )
    return authenticity, label, detail













# ─────────────────────────────────────────────────────────────
#  Multi-clip detection with content-aware reasoning
# ─────────────────────────────────────────────────────────────

def _build_content_aware_reasoning(
    label: str,
    authenticity: int,
    content_type: str,
    signal_scores: list,
    gpt_score: int,
    hybrid_flag: bool,
    gpt_reasoning: str,
    gpt_flags: list,
    n_clips: int,
) -> str:
    """
    Build the user-facing explanation shown in the result box.
    GPT now writes a specific 2-3 sentence explanation based on what it
    actually observed in the frames + the signal data passed to it.
    We use that directly, with a brief structural prefix for context.
    Falls back to templated text only if GPT reasoning is missing/generic.
    """
    score_variance = max(signal_scores) - min(signal_scores) if len(signal_scores) > 1 else 0

    content_labels = {
        "talking_head":   "talking head video",
        "selfie":         "selfie video",
        "action":         "action footage",
        "cinematic":      "cinematic footage",
        "static":         "static scene",
        "single_subject": "subject footage",
    }
    content_friendly = content_labels.get(content_type, "video")

    # Check if GPT produced a specific explanation (>60 chars, not a placeholder)
    _gpt_specific = (
        gpt_reasoning
        and len(gpt_reasoning) > 60
        and "inconclusive" not in gpt_reasoning.lower()
        and "pending" not in gpt_reasoning.lower()
        and "error" not in gpt_reasoning.lower()
    )

    # ── Use GPT's specific explanation as the primary text ────
    if _gpt_specific:
        # Add a brief structural prefix so the user knows what the score means
        if label == "REAL":
            if hybrid_flag:
                prefix = (
                    f"This {content_friendly} is predominantly real camera footage "
                    f"with some AI-generated elements detected across {n_clips} samples. "
                )
            elif authenticity >= 80:
                prefix = f"This {content_friendly} shows strong authentic camera indicators. "
            else:
                prefix = f"This {content_friendly} appears to be genuine footage. "
        elif label == "UNDETERMINED":
            if hybrid_flag:
                prefix = (
                    f"This {content_friendly} contains a mix of real and AI-generated content "
                    f"across {n_clips} sampled segments. "
                )
            else:
                prefix = (
                    f"This {content_friendly} could not be conclusively verified "
                    f"({authenticity}% authenticity). "
                )
        else:  # AI
            if hybrid_flag:
                prefix = (
                    f"This {content_friendly} contains AI-generated content across "
                    f"multiple segments. "
                )
            else:
                prefix = f"This {content_friendly} shows AI generation signatures. "

        return prefix + gpt_reasoning

    # ── Fallback templates if GPT gave a generic/failed response ──
    if label == "REAL":
        if hybrid_flag:
            return (
                f"This {content_friendly} is predominantly real camera footage. "
                f"Analysis across {n_clips} samples detected some AI-generated elements — "
                f"likely added graphics or still images — but the underlying footage shows "
                f"genuine sensor noise and organic motion physics."
            )
        return (
            f"This {content_friendly} shows authentic camera characteristics. "
            f"Sensor noise patterns, motion physics, and lighting coherence across "
            f"{n_clips} sampled segment{'s' if n_clips > 1 else ''} are consistent "
            f"with real camera footage. Authenticity score: {authenticity}%."
        )

    elif label == "UNDETERMINED":
        if hybrid_flag:
            return (
                f"This {content_friendly} contains a mix of real and AI-generated content. "
                f"Analysis across {n_clips} time samples found variation — some segments show "
                f"authentic camera characteristics while others show AI synthesis signatures. "
                f"Consistent with documentary or explainer content that mixes real and AI visuals."
            )
        return (
            f"This {content_friendly} could not be conclusively verified. "
            f"Signal analysis returned {authenticity}% authenticity — borderline between "
            f"real and AI-generated. Some elements appear authentic while others are ambiguous."
        )

    else:  # AI
        if hybrid_flag:
            flags_text = (
                f"including {gpt_flags[0]}" if gpt_flags
                else "including rendering artifacts and unnatural motion"
            )
            return (
                f"This {content_friendly} contains AI-generated content across "
                f"{n_clips} time samples, {flags_text}. While some segments may contain "
                f"real footage, AI-generated portions are significant. "
                f"Authenticity score: {authenticity}%."
            )
        flags_text = f"Key indicators: {gpt_flags[0]}. " if gpt_flags else ""
        return (
            f"This {content_friendly} shows clear AI generation signatures. "
            f"{flags_text}Signal analysis detected rendering artifacts and motion patterns "
            f"inconsistent with real camera footage. Authenticity score: {authenticity}%."
        )


def run_detection_multiclip(video_path: str) -> tuple:
    """
    Multi-clip detection entry point.
    Extracts 1-3 clips from different positions, runs signal detection
    on each in parallel, runs GPT once with frames from all clips combined,
    then merges results with content-aware reasoning.

    Returns: (authenticity, label, detail)
    Same signature as run_detection() — drop-in replacement.
    """
    import concurrent.futures
    from video import extract_clips_for_detection

    # ── Metadata override — fast path ────────────────────────
    override, ov_ai_score, ov_label, ov_reason = _check_metadata_override(video_path)
    if override and ov_label == "AI":
        authenticity = 100 - ov_ai_score
        log.info("Metadata override → ai_score=%d auth=%d label=AI", ov_ai_score, authenticity)
        return authenticity, "AI", {
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

    # ── Extract clips from multiple positions ─────────────────
    try:
        clips = extract_clips_for_detection(video_path)
    except Exception as e:
        log.warning("Multi-clip extraction failed (%s) — falling back to single clip", e)
        return run_detection(video_path)

    if not clips:
        log.warning("No clips extracted — falling back to single clip detection")
        return run_detection(video_path)

    n_clips = len(clips)
    log.info("Multi-clip detection: %d clips extracted", n_clips)

    # ── Run signal detection on all clips in parallel ─────────
    def _detect_one(clip_path_offset):
        clip_path, offset_pct = clip_path_offset
        try:
            score, context = detect_ai(clip_path)
            log.info("Clip @%.0f%%: signal_score=%d content=%s",
                     offset_pct * 100, score, context.get("content_type", "?"))
            return score, context, offset_pct
        except Exception as e:
            log.warning("Signal detection failed for clip @%.0f%%: %s", offset_pct * 100, e)
            return None, {}, offset_pct

    clip_results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
        futures = [ex.submit(_detect_one, c) for c in clips]
        for f in concurrent.futures.as_completed(futures):
            clip_results.append(f.result())

    # Sort by offset for consistency
    clip_results.sort(key=lambda x: x[2])

    # Filter valid results
    valid = [(score, ctx, pct) for score, ctx, pct in clip_results if score is not None]
    if not valid:
        log.warning("All clip detections failed — falling back to single clip")
        return run_detection(video_path)

    signal_scores = [score for score, _, _ in valid]

    # Weighted average — weight by noise level (more noise = more real data)
    noise_weights = []
    for _, ctx, _ in valid:
        noise = ctx.get("noise_laplacian", ctx.get("noise", 500))
        noise_weights.append(float(noise))
    total_w = sum(noise_weights) or 1.0
    norm_weights = [w / total_w for w in noise_weights]

    signal_ai_score = int(round(sum(
        s * w for s, w in zip(signal_scores, norm_weights)
    )))

    # Use context from highest-noise clip (most data-rich)
    best_ctx_idx = noise_weights.index(max(noise_weights))
    signal_context = valid[best_ctx_idx][1]
    content_type = signal_context.get("content_type", "cinematic")

    # ── Hybrid detection ──────────────────────────────────────
    score_variance = max(signal_scores) - min(signal_scores) if len(signal_scores) > 1 else 0
    signal_avg_pre = sum(signal_scores) / len(signal_scores)

    # Primary hybrid trigger: variance > 15 between clips
    hybrid_flag = score_variance > 15

    # Secondary hybrid trigger: all clips score high AI (>75) but GPT
    # sees mostly neutral scores (all 5s) — means GPT frames landed on
    # real content while signal clips landed on AI segments.
    # This is the "postal truck" pattern: AI stills scattered in real footage.
    # We check this after GPT runs below, so set a pre-flag here.
    _all_clips_high_ai = len(signal_scores) >= 2 and min(signal_scores) > 75
    _pre_hybrid_candidate = _all_clips_high_ai  # refined after GPT

    if hybrid_flag:
        log.info("HYBRID FLAG (variance): signal variance=%d across clips %s",
                 score_variance, signal_scores)
        signal_context["hybrid_detected"] = True
        signal_context["clip_signal_scores"] = signal_scores

    # ── GPT vision — single call with frames from all clips ───
    all_frames = []
    frames_per_clip = max(2, 8 // n_clips)
    for clip_path, _ in clips:
        try:
            frames = extract_key_frames(clip_path, n_frames=frames_per_clip)
            all_frames.extend(frames)
        except Exception as e:
            log.warning("Frame extraction failed for clip: %s", e)

    if not all_frames:
        log.warning("No frames extracted — using signal only")
        gpt_result = {"ai_probability": 50, "reasoning": "Frame extraction failed",
                      "flags": [], "available": False}
    else:
        # Add hybrid context to GPT prompt
        if hybrid_flag:
            signal_context["_extra_context"] = (
                f"NOTE: This video was sampled at {n_clips} points in time. "
                f"Signal scores varied significantly ({min(signal_scores)}-{max(signal_scores)}), "
                f"suggesting mixed content — some segments real, some AI-generated."
            )
        gpt_result = gpt_vision_score_with_context(all_frames, signal_context)

    gpt_ai_score  = gpt_result.get("ai_probability", 50)
    gpt_available = gpt_result.get("available", False)
    gpt_reasoning = gpt_result.get("reasoning", "")
    gpt_flags     = gpt_result.get("flags", [])

    # Secondary hybrid check: all clips scored high AI but GPT is neutral (<50)
    # This means signal clips hit AI segments but GPT frames hit real content
    # Classic mixed/hybrid documentary pattern
    if _pre_hybrid_candidate and not hybrid_flag and gpt_ai_score < 55:
        hybrid_flag = True
        log.info(
            "HYBRID FLAG (gpt-neutral + high-signal): "
            "all clips AI (min=%d) but gpt=%d → mixed content",
            min(signal_scores), gpt_ai_score
        )
        signal_context["hybrid_detected"] = True
        signal_context["clip_signal_scores"] = signal_scores

    log.info("Multi-clip: signal_avg=%d (clips=%s var=%d) gpt=%d hybrid=%s",
             signal_ai_score, signal_scores, score_variance, gpt_ai_score, hybrid_flag)

    # ── Blend signal + GPT (same logic as run_detection) ─────
    gpt_refused = gpt_result.get("gpt_refused", False)
    gpt_failed = (not gpt_available or gpt_reasoning.startswith("GPT analysis error")) and not gpt_refused

    if gpt_failed:
        combined_ai_score = float(signal_ai_score)
        w_sig, w_gpt = 1.0, 0.0
        mode = "signal-only (GPT failed)"
    else:
        clash_real   = signal_ai_score < 50 and gpt_ai_score > 50 and gpt_ai_score < 75
        clash_ai     = signal_ai_score > 65 and gpt_ai_score < 40
        gpt_dominant = gpt_ai_score >= 75 and signal_ai_score < 60
        both_real    = signal_ai_score < 45 and gpt_ai_score < 45
        both_ai      = signal_ai_score > 65 and gpt_ai_score > 65

        if gpt_dominant:
            combined_ai_score = signal_ai_score * 0.25 + gpt_ai_score * 0.75
            w_sig, w_gpt = 0.25, 0.75
            mode = "gpt-dominant"
        elif clash_real:
            combined_ai_score = signal_ai_score * 0.90 + gpt_ai_score * 0.10
            w_sig, w_gpt = 0.90, 0.10
            mode = "clash→real"
        elif clash_ai:
            combined_ai_score = signal_ai_score * 0.70 + gpt_ai_score * 0.30
            w_sig, w_gpt = 0.70, 0.30
            mode = "clash→AI"
        elif both_real:
            combined_ai_score = signal_ai_score * 0.40 + gpt_ai_score * 0.60 - 5
            w_sig, w_gpt = 0.40, 0.60
            mode = "both-real bonus"
        elif both_ai:
            combined_ai_score = signal_ai_score * 0.40 + gpt_ai_score * 0.60 + 5
            w_sig, w_gpt = 0.40, 0.60
            mode = "both-AI bonus"
        else:
            combined_ai_score = signal_ai_score * 0.40 + gpt_ai_score * 0.60
            w_sig, w_gpt = 0.40, 0.60
            mode = "default"

    # Hybrid adjustment — only clamp when clips GENUINELY disagree
    # True hybrid = some clips real (<45) AND some clips AI (>65)
    # NOT hybrid = all clips agree on AI with varying confidence
    _has_real_clips   = any(s < 45 for s in signal_scores)
    _has_ai_clips     = any(s > 65 for s in signal_scores)
    _true_hybrid      = hybrid_flag and _has_real_clips and _has_ai_clips
    _both_engines_ai  = signal_ai_score > 70 and gpt_ai_score > 65

    if _true_hybrid and not gpt_failed and not _both_engines_ai:
        # Genuine mixed content — pull toward UNDETERMINED
        if combined_ai_score > 65:
            combined_ai_score = min(combined_ai_score, 58)
            log.info("Hybrid adjustment: high-AI clamped to %.1f (genuine mixed content)", combined_ai_score)
        elif combined_ai_score < 30:
            combined_ai_score = max(combined_ai_score, 35)
            log.info("Hybrid adjustment: high-REAL clamped to %.1f (genuine mixed content)", combined_ai_score)
        else:
            log.info("Hybrid adjustment: combined=%.1f already in mixed range", combined_ai_score)
    elif hybrid_flag and not _true_hybrid:
        log.info("Hybrid flag set but all clips agree (%s) — no clamping applied", signal_scores)

    combined_ai_score = max(0.0, min(100.0, combined_ai_score))
    authenticity = 100 - int(round(combined_ai_score))
    authenticity = max(0, min(100, authenticity))

    label = (
        "REAL"          if authenticity >= THRESHOLD_REAL        else
        "UNDETERMINED"  if authenticity >= THRESHOLD_UNDETERMINED else
        "AI"
    )

    # ── Content-aware reasoning ───────────────────────────────
    user_reasoning = _build_content_aware_reasoning(
        label=label,
        authenticity=authenticity,
        content_type=content_type,
        signal_scores=signal_scores,
        gpt_score=gpt_ai_score,
        hybrid_flag=hybrid_flag,
        gpt_reasoning=gpt_reasoning,
        gpt_flags=gpt_flags,
        n_clips=n_clips,
    )

    log.info(
        "Multi-clip detection complete | clips=%d signal_avg=%d gpt=%d "
        "combined=%d authenticity=%d label=%s hybrid=%s mode=%s",
        n_clips, signal_ai_score, gpt_ai_score,
        int(round(combined_ai_score)), authenticity, label, hybrid_flag, mode,
    )

    # Cleanup clips
    for clip_path, _ in clips:
        try:
            os.remove(clip_path)
        except Exception:
            pass

    detail = {
        "ai_score":           int(round(combined_ai_score)),
        "signal_ai_score":    signal_ai_score,
        "gpt_ai_score":       gpt_ai_score,
        "gpt_available":      gpt_available,
        "gpt_reasoning":      user_reasoning,
        "gpt_flags":          gpt_flags,
        "weight_signal":      w_sig,
        "weight_gpt":         w_gpt,
        "blend_mode":         mode,
        "content_type":       content_type,
        "n_clips":            n_clips,
        "clip_signal_scores": signal_scores,
        "hybrid_detected":    hybrid_flag,
        "threshold_real":     THRESHOLD_REAL,
        "threshold_undet":    THRESHOLD_UNDETERMINED,
    }
    return authenticity, label, detail









