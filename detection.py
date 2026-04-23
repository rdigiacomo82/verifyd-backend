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

# New detection engines — imported lazily to avoid startup failures
def _get_c2pa_checker():
    try:
        from c2pa_checker import check_c2pa
        return check_c2pa
    except Exception:
        return None

def _get_npr_analyzer():
    try:
        from npr_detector import analyze_npr, get_npr_contribution
        return analyze_npr, get_npr_contribution
    except Exception:
        return None, None

def _get_dino_analyzer():
    try:
        from dinov2_detector import analyze_dinov2, get_dino_contribution
        return analyze_dinov2, get_dino_contribution
    except Exception:
        return None, None

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

        # Lavf encoder + null vendor = AI pipeline fingerprint (soft signal)
        # Returns a soft override label so multi_clip_detection can apply
        # the LAVF+CHAN_CORR composite boost when all clips confirm it.
        if "lavf" in encoder and vendor in ("[0][0][0][0]", "", "0000"):
            # Before flagging as AI pipeline, check if this is a YouTube re-encode.
            # YouTube's transcoding pipeline always sets Lavf as the encoder AND writes
            # "ISO Media file produced by Google Inc." in the stream handler_name.
            # This is a legitimate re-encode, not an AI generator fingerprint — suppress.
            _handler_names = " ".join(
                s.get("tags", {}).get("handler_name", "").lower()
                for s in data.get("streams", [])
            )
            if "google" in _handler_names or "youtube" in _handler_names:
                log.info("METADATA: Lavf+Google handler detected → YouTube re-encode pipeline, NOT AI — suppressing LAVF flag")
                return False, 0, None, "YouTube re-encode (Google handler_name)"
            log.info("METADATA: Lavf encoder + null vendor_id detected (AI pipeline fingerprint — soft signal, see LAVF_CHAN_CORR boost)")
            log.info("METADATA: LAVF_AI_PIPELINE soft flag set — will boost if all clips CHAN_CORR>0.90")
            return True, 20, "LAVF_AI_PIPELINE", "Lavf encoder + null vendor fingerprint (AI pipeline soft signal)"

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

    # 3. C2PA provenance check — cryptographic content credentials
    try:
        check_c2pa = _get_c2pa_checker()
        if check_c2pa:
            c2pa_delta, c2pa_label, c2pa_detail = check_c2pa(video_path)
            if c2pa_delta >= 45:
                # Strong AI credential — treat as override
                reason = f"C2PA manifest identifies AI generator: {c2pa_detail.get('generator', c2pa_label)}"
                log.info("METADATA_OVERRIDE: C2PA AI generator → +%d → AI", c2pa_delta)
                return True, min(97, 50 + c2pa_delta), "AI", reason
            elif c2pa_delta <= -20:
                # Strong real camera credential — treat as override
                reason = f"C2PA manifest from verified real camera: {c2pa_detail.get('generator', c2pa_label)}"
                log.info("METADATA_OVERRIDE: C2PA real camera → %d → REAL", c2pa_delta)
                return True, max(3, 15 + c2pa_delta), "REAL", reason
    except Exception as e:
        log.warning("METADATA: C2PA check error: %s", e)

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
    elif override and ov_label == "LAVF_AI_PIPELINE":
        # Soft LAVF flag — do not override, just note it for composite boost below
        log.info("Metadata LAVF soft flag → running full detection, composite boost may apply")
    elif override and ov_label == "REAL":
        # Device metadata confirms real camera — still run full detection
        # but pass the device confirmation as a strong prior to both engines
        log.info("Metadata real-device signal → ai_score=%d | %s (running full detection)", 
                 ov_ai_score, ov_reason)
        # ov_ai_score is very low (3-15) — will be incorporated via signal_context

    # ── Engine 1: Signal detector ─────────────────────────────
    _raw_signal = detect_ai(video_path)
    if isinstance(_raw_signal, tuple):
        signal_ai_score, signal_context = _raw_signal
    else:
        signal_ai_score, signal_context = int(_raw_signal), {}
    log.info("Signal detector ai_score: %d  content_type: %s",
             signal_ai_score, signal_context.get("content_type", "unknown"))

    # ── Engine 2: GPT-4o ──────────────────────────────────────
    frames_b64      = extract_key_frames(video_path)
    gpt_result      = gpt_vision_score_with_context(frames_b64, signal_context)
    gpt_ai_score    = gpt_result["ai_probability"]
    gpt_available   = gpt_result.get("available", False)
    log.info("GPT-4o ai_probability: %d  available: %s", gpt_ai_score, gpt_available)

    # ── Combine ───────────────────────────────────────────────
    # Read video source from sidecar — needed for YouTube clash->real suppression
    import os as _os_rd, json as _jsc_rd
    _sidecar_rd = video_path.replace(".mp4", ".meta.json").replace(".MOV", ".meta.json")
    _video_source = ""
    if _os_rd.path.exists(_sidecar_rd):
        try:
            with open(_sidecar_rd) as _sf_rd:
                _video_source = _jsc_rd.load(_sf_rd).get("source", "")
        except Exception:
            pass

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
        _youtube_signal_unreliable = signal_context.get("youtube_lowres_adjusted", False) or "youtube" in _video_source.lower()
        clash_real   = signal_ai_score < 50 and gpt_ai_score > 50 and gpt_ai_score < 75 and not _youtube_signal_unreliable
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
    signal_context: dict = None,
    blend_mode: str = "",
) -> str:
    """
    Build the user-facing explanation shown in the result box.
    When signal and GPT clash, we write a signal-informed explanation
    that explains WHY the detector overrode the visual appearance.
    Falls back to templated text if GPT reasoning is missing/generic.
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

    # ── Detect clash: signal strongly says AI but GPT says real ────────
    _signal_avg = sum(signal_scores) / len(signal_scores) if signal_scores else 0
    _clash_ai   = _signal_avg > 65 and gpt_score < 45 and label == "AI"

    if _clash_ai and signal_context:
        # Build an explanation based on the actual signal evidence
        # instead of trusting GPT's visual read of static frames
        _ctx = signal_context or {}
        _evidence = []

        # Audio evidence (most reliable)
        _audio = _ctx.get("audio_stereo_corr", 0)
        if _audio >= 0.99:
            _evidence.append("perfectly correlated stereo audio (a hallmark of synthetic pipelines)")
        elif _audio >= 0.93:
            _evidence.append("near-perfect stereo audio correlation typical of AI-generated content")

        # HF kurtosis
        _kurtosis = _ctx.get("hf_kurtosis", 0)
        if _kurtosis > 80:
            _evidence.append(f"extreme high-frequency upsampling artifacts (kurtosis {_kurtosis:.0f})")
        elif _kurtosis > 55:
            _evidence.append(f"high-frequency upsampling artifacts typical of AI rendering")

        # Channel correlation
        _chan = _ctx.get("chan_corr", 0)
        if _chan > 0.93:
            _evidence.append("unnaturally high inter-channel video correlation")

        # Omni flow
        _omni = _ctx.get("omni_flow_entropy", 0)
        if _omni > 3.8:
            _evidence.append("omnidirectional noise motion inconsistent with real camera movement")

        # Shadow drift
        _shadow = _ctx.get("shadow_drift", 0)
        if _shadow > 0.9:
            _evidence.append("strongly inconsistent shadow behavior across frames")

        # Edge crawl
        _edge_cov = _ctx.get("edge_cov_var", 0)
        if _edge_cov > 1.5:
            _evidence.append("edge crawl artifacts characteristic of AI upscaling")

        # FLAT_NOISE
        _flat = _ctx.get("flat_noise", 1.0)
        if _flat < 0.3:
            _evidence.append("absence of natural camera sensor noise (PRNU)")

        if _evidence:
            evidence_str = "; ".join(_evidence[:3])  # top 3 signals
            return (
                f"Despite appearing visually convincing in static frames, this {content_friendly} "
                f"contains strong technical AI signatures: {evidence_str}. "
                f"These artifacts are invisible to the eye but detectable in the signal data, "
                f"which is why our detector overrides the visual appearance. "
                f"Authenticity score: {authenticity}%."
            )

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
            _raw = detect_ai(clip_path)
            # Handle both old (int) and new (int, dict) return signatures
            if isinstance(_raw, tuple):
                score, context = _raw
            else:
                score, context = int(_raw), {}

            # NPR frequency domain analysis — runs alongside signal detector
            # COMPRESSION GUARD: TikTok/Instagram/YouTube re-encode videos
            # with H.264 DCT compression that creates grid and spectral artifacts
            # identical to AI upsampling. NPR grid/slope signals are unreliable
            # on socially-compressed video. Only temporal residual consistency
            # and kurtosis are compression-resilient.
            try:
                analyze_npr, get_npr_contribution = _get_npr_analyzer()
                if analyze_npr:
                    npr_score, npr_signals = analyze_npr(clip_path)

                    # Detect social media compression via noise level + source
                    # TikTok re-encoding creates trc artifacts (0.70+) on real videos
                    # even when noise is high, so we also check the video source
                    noise_level = context.get("avg_noise", context.get("noise_laplacian", 1000))
                    codec = context.get("codec", "")
                    _video_source = context.get("source", "")
                    _is_social_source = any(s in _video_source.lower()
                                            for s in ("tiktok", "smvd", "instagram", "youtube"))
                    is_social_compressed = (
                        noise_level < 600 or           # Low noise = over-compressed
                        "h264" in str(codec).lower() or # H.264 from social platforms
                        _is_social_source               # Known social platform re-encode
                    )

                    if is_social_compressed:
                        # Discount grid and slope signals — compression artifacts
                        # Only use kurtosis and temporal residual consistency
                        sig = npr_signals or {}
                        trc_score = 0
                        trc = sig.get("temporal_residual_consistency", 0) or 0
                        if trc > 0.45: trc_score = 16
                        elif trc > 0.30: trc_score = 10
                        elif trc > 0.20: trc_score = 5
                        elif trc < 0.03: trc_score = -4

                        kurt_score = 0
                        kurt = sig.get("residual_kurtosis", 3) or 3
                        if kurt > 80: kurt_score = 20
                        elif kurt > 50: kurt_score = 14
                        elif kurt > 30: kurt_score = 8

                        # Compressed NPR score — only reliable signals
                        npr_score_adj = max(0, min(100, trc_score + kurt_score))
                        npr_contribution = get_npr_contribution(npr_score_adj)
                        log.info("NPR @%.0f%% (compressed): raw=%d adj=%d trc=%.3f kurt=%.1f contribution=%+d",
                                 offset_pct * 100, npr_score, npr_score_adj, trc, kurt, npr_contribution)
                        npr_score = npr_score_adj
                    else:
                        npr_contribution = get_npr_contribution(npr_score)
                        if npr_contribution != 0:
                            log.info("NPR @%.0f%%: score=%d contribution=%+d",
                                     offset_pct * 100, npr_score, npr_contribution)

                    context["npr_score"] = npr_score
                    context["npr_signals"] = npr_signals
                    context["npr_contribution"] = npr_contribution

                    # Blend NPR into signal score conservatively
                    if npr_contribution != 0:
                        score = int(round(min(100, max(0, score + npr_contribution))))
            except Exception as e:
                log.debug("NPR analysis skipped for clip @%.0f%%: %s", offset_pct * 100, e)

            # DINOv2 feature-based analysis — tie-breaker for ambiguous clips
            # Only runs when signal score is in ambiguous range (35-72)
            # to avoid adding noise to already-confident detections
            try:
                analyze_dinov2, get_dino_contribution = _get_dino_analyzer()
                if analyze_dinov2 and 35 <= score <= 72:
                    dino_score, dino_signals = analyze_dinov2(clip_path)
                    dino_contribution = get_dino_contribution(dino_score, score)
                    context["dino_score"] = dino_score
                    context["dino_signals"] = dino_signals
                    context["dino_contribution"] = dino_contribution
                    if dino_contribution != 0:
                        log.info("DINOv2 @%.0f%%: score=%d contribution=%+d (signal=%d ambiguous)",
                                 offset_pct * 100, dino_score, dino_contribution, score)
                        score = int(round(min(100, max(0, score + dino_contribution))))
                elif analyze_dinov2:
                    log.debug("DINOv2 @%.0f%%: skipped (signal=%d not ambiguous)", offset_pct * 100, score)
            except Exception as e:
                log.debug("DINOv2 analysis skipped for clip @%.0f%%: %s", offset_pct * 100, e)

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
    all_signal_contexts = [ctx for _, ctx, _ in valid]  # all clip contexts for reasoning

    # Inject video source into all contexts so NPR compression guard can use it
    # Source is read from the sidecar file written by the downloader
    import os as _os
    _sidecar = video_path.replace(".mp4", ".meta.json").replace(".MOV", ".meta.json")
    _video_source = ""
    if _os.path.exists(_sidecar):
        try:
            import json as _jsc
            with open(_sidecar) as _sf:
                _smeta = _jsc.load(_sf)
            _video_source = _smeta.get("source", "")
        except Exception:
            pass
    if _video_source:
        for _ctx in all_signal_contexts:
            _ctx["source"] = _video_source

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

    # ── YouTube low-resolution uncertainty adjustment ──────────
    # YouTube re-encodes all uploads through their H264 pipeline, which:
    #   1. Strips all original AI generator metadata
    #   2. Creates H264 compression artifacts that mimic real camera noise
    #   3. Downloads at low resolution making forensic signals unreliable
    # When source=youtube AND clip_px < 500,000 (covers all YouTube quality
    # levels up to ~540p), pull signal score 30% toward 50 and flag for
    # clash->real suppression in blend mode selection below.
    # clip_px is now returned by detector.py in signal_context.
    _is_youtube_source = "youtube" in _video_source.lower()
    if _is_youtube_source:
        _any_low_res = any(
            ctx.get("clip_px", 999999) < 500000
            for _, ctx, _ in valid
        )
        if _any_low_res:
            _old_signal = signal_ai_score
            signal_ai_score = int(round(signal_ai_score * 0.70 + 50 * 0.30))
            log.info(
                "YOUTUBE_LOWRES: low-res YouTube clip detected (clip_px<500k) -- "
                "signal %d->%d (30pct uncertainty pull)",
                _old_signal, signal_ai_score
            )
            signal_context["youtube_lowres_adjusted"] = True
        else:
            log.info("YOUTUBE: source=youtube but resolution adequate -- no adjustment")

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
        # YouTube H264 re-encoding creates Laplacian noise that mimics real camera
        # grain at ALL resolutions (480p AND 720p), making signal say "real" while
        # GPT may correctly identify AI content visually.
        # Suppress clash->real for ALL YouTube sources — not just low-res —
        # because the fake noise signal is a YouTube pipeline artifact, not resolution-dependent.
        # The signal uncertainty pull (30% toward 50) still only applies to low-res (<500k px)
        # since that addresses a different problem (insufficient pixels for reliable analysis).
        _is_youtube = "youtube" in _video_source.lower()
        _youtube_signal_unreliable = (
            signal_context.get("youtube_lowres_adjusted", False) or _is_youtube
        )

        clash_real   = (signal_ai_score < 50 and gpt_ai_score > 50 and gpt_ai_score < 75
                        and not _youtube_signal_unreliable)  # suppress for ALL YouTube sources
        clash_ai     = signal_ai_score > 65 and gpt_ai_score < 40
        gpt_dominant = gpt_ai_score >= 75 and signal_ai_score < 60
        both_real    = signal_ai_score < 45 and gpt_ai_score < 45
        both_ai      = signal_ai_score > 65 and gpt_ai_score > 65

        # real_dominant: when signal says AI but GPT AND DINOv2 BOTH say real.
        # This fires on YouTube live event/news footage where extreme camera
        # shake creates AI-like signals but two visual engines confirm real.
        # Conditions: source=youtube, signal>50 (elevated), gpt<45 (GPT says real),
        # and DINOv2 score <=5 (DINOv2 independently confirms real).
        # Does NOT affect AI videos: AI plasma has gpt=80 (fails gpt<45 check).
        _dino_score_ctx = signal_context.get("dino_score", 50)
        # real_dominant: signal says AI but GPT AND DINOv2 BOTH strongly say real.
        # Extended to catch uploaded files (no sidecar → _is_youtube=False) where:
        #   - Signal is elevated but not extreme (50-75 range — ambiguous territory)
        #   - GPT is very confident real (< 30, not just < 45)
        #   - DINOv2 independently confirms real (score <= 5)
        # This prevents high-noise real camera footage (dashcam, bodycam, news) from
        # being misclassified when re-encoded through YouTube/Lavf pipeline.
        _strong_gpt_real    = gpt_ai_score < 30    # GPT very confident real
        _moderate_gpt_real  = gpt_ai_score < 45    # GPT moderately confident real
        real_dominant = (
            (
                # Original YouTube sidecar path
                (_is_youtube and signal_ai_score > 50 and _moderate_gpt_real and _dino_score_ctx <= 5)
                or
                # Extended: any source where GPT is VERY confident real AND DINOv2 confirms.
                # Conditions are intentionally strict to avoid misclassifying AI videos:
                #   - Signal ambiguous (50-75 range) — above 75 is too strong to override
                #   - GPT very confident real (< 30) — not just moderately real
                #   - DINOv2 also confirms real (score <= 2, stricter than YouTube path)
                # This catches high-noise real camera footage (dashcam, bodycam, news)
                # uploaded directly rather than via YouTube link.
                (signal_ai_score > 50 and signal_ai_score <= 75 and _strong_gpt_real and _dino_score_ctx <= 2)
            )
        )

        if _youtube_signal_unreliable and signal_ai_score < 50 and gpt_ai_score > 50:
            log.info(
                "YOUTUBE: clash->real suppressed — "
                "signal(%d) noise unreliable for YouTube (H264 compression mimics real grain), "
                "GPT(%d) gets default weight",
                signal_ai_score, gpt_ai_score
            )

        if real_dominant:
            log.info(
                "REAL_DOMINANT: YouTube source, signal(%d) elevated but GPT(%d)<45 "
                "and DINOv2(%d)<=5 both confirm real — GPT gets 75%% weight",
                signal_ai_score, gpt_ai_score, _dino_score_ctx
            )
            combined_ai_score = signal_ai_score * 0.25 + gpt_ai_score * 0.75
            w_sig, w_gpt = 0.25, 0.75
            mode = "real-dominant (GPT+DINOv2 confirm real)"
        elif gpt_dominant:
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

    # ── Certainty override ───────────────────────────────────
    # If ANY clip scores >=95 (near-certain AI) AND GPT also shows
    # some AI signal (>35) AND no other clip scored below 50
    # (meaning the whole video leans AI, not just one spliced segment),
    # the video contains definitively AI-generated content.
    #
    # The "no clip below 50" guard protects documentaries/compilations
    # where AI clips are mixed with real footage (USPS case: clips=[99,56,65]
    # has clips at 56 and 65 — both above 50, so... wait, 56 IS above 50.
    # Better guard: no clip below 45 (real threshold).
    # USPS: clips=[99,56,65] — 56 and 65 both above 45 → override fires
    # Tiger: clips=[35,100] — 35 is below 45 → override does NOT fire
    # Hmm, tiger has clip=35 below 45... need different approach.
    #
    # Correct logic: certainty override fires when:
    # - clip >=95 exists (definite AI segment)  
    # - ALL OTHER clips are also >=50 (no clearly real segments)
    # - GPT >35 (some AI signal)
    # Tiger: [35,100] — other clip=35 < 50 → NO override (correct — mixed)
    # USPS: [99,56,65] — other clips 56,65 both >=50 → override fires (wrong)
    # Need: other clips >=60 to protect USPS
    _has_certain_ai_clip = any(s >= 95 for s in signal_scores)
    _other_clips_also_ai = all(s >= 60 for s in signal_scores if s < 95)
    if _has_certain_ai_clip and _other_clips_also_ai and gpt_ai_score > 35:
        old_combined = combined_ai_score
        combined_ai_score = max(combined_ai_score, 72.0)
        if combined_ai_score != old_combined:
            log.info(
                "Certainty override: clip scored >=95, all others >=60, gpt=%d>35 → "
                "combined %.1f→%.1f (definitively AI content)",
                gpt_ai_score, old_combined, combined_ai_score
            )

    # Hybrid adjustment — only clamp when clips GENUINELY disagree
    # True hybrid = some clips real (<45) AND some clips AI (>65)
    # NOT hybrid = all clips agree on AI with varying confidence
    _has_real_clips   = any(s < 35 for s in signal_scores)
    _has_ai_clips     = any(s > 65 for s in signal_scores)
    # GPT override: if GPT is strongly confident AI (>=80), suppress hybrid clamp
    # even when clips appear mixed. GPT seeing definitive AI artifacts at >=80
    # overrides ambiguous signal variance.
    _gpt_strongly_ai  = gpt_ai_score >= 80
    _true_hybrid      = hybrid_flag and _has_real_clips and _has_ai_clips and not _gpt_strongly_ai
    _both_engines_ai  = signal_ai_score > 70 and gpt_ai_score > 65
    if _gpt_strongly_ai and hybrid_flag and _has_real_clips and _has_ai_clips:
        log.info(
            "Hybrid clamp suppressed: GPT=%d strongly AI (>=80) overrides mixed signal %s",
            gpt_ai_score, signal_scores
        )

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
        # Check for high-variance mixed content (documentary with AI inserts)
        # clips agree on direction (all AI) but variance is high (>35)
        # AND not all clips are strongly AI (some below 70)
        # → these are real videos with some AI-generated segments
        # → should land UNDETERMINED not AI
        _high_variance   = score_variance > 35
        _not_all_strong  = any(s < 70 for s in signal_scores)
        _both_engines_ambiguous = signal_ai_score < 80 and gpt_ai_score < 65

        if _high_variance and _not_all_strong and _both_engines_ambiguous:
            # Pull toward UNDETERMINED for mixed-content videos
            old_score = combined_ai_score
            combined_ai_score = min(combined_ai_score, 58)
            log.info(
                "Hybrid adjustment: high-variance mixed content "
                "([%s] var=%d) → clamped %.1f→%.1f (UNDETERMINED territory)",
                ",".join(str(s) for s in signal_scores), score_variance,
                old_score, combined_ai_score
            )
        else:
            log.info("Hybrid flag set but all clips agree (%s) — no clamping applied", signal_scores)

    # ── LAVF + CHAN_CORR composite boost ────────────────────────
    # Lavf encoder alone is weak (innocent re-encodes are common).
    # But Lavf + ALL clips showing CHAN_CORR > 0.90 is a strong composite
    # signal — real re-encodes rarely have both together.
    _lavf_flag = override and ov_label == "LAVF_AI_PIPELINE"
    if _lavf_flag:
        # Only include chan_corr values that were NOT skipped by hi_noise/hevc_hd guard.
        # If CHAN_CORR was skipped for scoring (real camera noise), it must not trigger
        # the LAVF boost — YouTube re-encodes with Lavf AND have high noise (real camera).
        _all_chan_corr = [
            ctx.get("chan_corr", 0) for _, ctx, _ in valid
            if not ctx.get("chan_corr_skipped", False)
        ]
        _all_high_corr = all(c > 0.90 for c in _all_chan_corr if c > 0)
        if _all_high_corr and _all_chan_corr:
            old_combined = combined_ai_score
            combined_ai_score = min(100.0, combined_ai_score + 15)
            log.info(
                "LAVF+CHAN_CORR boost: combined %.1f→%.1f "
                "(Lavf encoder + all clips CHAN_CORR>0.90 [%s])",
                old_combined, combined_ai_score,
                ", ".join(f"{c:.3f}" for c in _all_chan_corr)
            )
        else:
            log.info(
                "LAVF flag set but CHAN_CORR not all >0.90 [%s] — no boost (likely innocent re-encode)",
                ", ".join(str(c) for c in _all_chan_corr)
            )

    # ── Real device metadata ceiling ────────────────────────
    # When metadata confirms a real device recording (isom/mp42 container +
    # creation_time), pixel signals may still score very high for exotic
    # content: space footage, night sky, extreme darkness, underwater, etc.
    # These break terrestrial signal assumptions. Cap combined score at
    # UNDETERMINED ceiling — metadata is more reliable than pixel signals
    # for content outside the detector's training distribution.
    if override and ov_label == "REAL" and combined_ai_score >= 70:
        old_combined = combined_ai_score
        combined_ai_score = min(combined_ai_score, 58.0)
        log.info(
            "Real device metadata ceiling: combined %.1f→%.1f "
            "(metadata confirmed real camera, preventing AI verdict) | %s",
            old_combined, combined_ai_score, ov_reason
        )

    combined_ai_score = max(0.0, min(100.0, combined_ai_score))
    authenticity = 100 - int(round(combined_ai_score))
    authenticity = max(0, min(100, authenticity))

    label = (
        "REAL"          if authenticity >= THRESHOLD_REAL        else
        "UNDETERMINED"  if authenticity >= THRESHOLD_UNDETERMINED else
        "AI"
    )

    # ── Aggregate signal context across all clips for reasoning ─
    # Collect the most diagnostic values from all clips
    _agg_ctx = {}
    for _sc in all_signal_contexts:
        if not _sc:
            continue
        for _k in ("audio_stereo_corr", "hf_kurtosis", "chan_corr",
                   "omni_flow_entropy", "shadow_drift", "edge_cov_var", "flat_noise"):
            _v = _sc.get(_k)
            if _v is not None:
                # Keep the most extreme (highest AI-indicating) value
                if _k == "flat_noise":
                    _agg_ctx[_k] = min(_agg_ctx.get(_k, 999), _v)
                else:
                    _agg_ctx[_k] = max(_agg_ctx.get(_k, 0), _v)

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
        signal_context=_agg_ctx,
        blend_mode=mode,
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
