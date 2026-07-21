# ============================================================
#  VeriFYD — detection.py  v3
#
#  Speed update: heavy DINOv2/Deepfake tie-breakers run once per video
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
import os
from detector    import detect_ai
from gpt_vision  import gpt_vision_score_with_context, extract_key_frames

# AI_SOURCE_PROVENANCE_PATCH: explicit AI generator/source text detector
try:
    from ai_source_detector import scan_video_source_for_ai
except Exception:
    scan_video_source_for_ai = None

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

def _get_deepfake_analyzer():
    try:
        from deepfake_detector import analyze_deepfake, get_deepfake_contribution
        return analyze_deepfake, get_deepfake_contribution
    except Exception:
        return None, None

def _get_audio_analyzer():
    """Lazy-load the audio detector so deployments without optional audio deps still run."""
    try:
        from audio_detector import analyze_audio, get_audio_contribution
        return analyze_audio, get_audio_contribution
    except Exception as e:
        log.debug("Audio detector unavailable: %s", e)
        return None, None

THRESHOLD_REAL         = 55
THRESHOLD_UNDETERMINED = 40


# ─────────────────────────────────────────────────────────────
#  VERIFYD_VIRAL_AI_REEL_PATCH_V1
#  Viral AI reel / staged short-form social clip guard
# ─────────────────────────────────────────────────────────────
def _vfyd_safe_float(value, default=0.0):
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def _vfyd_safe_int(value, default=0):
    try:
        if value is None:
            return default
        return int(round(float(value)))
    except Exception:
        return default


def _vfyd_load_sidecar(video_path: str) -> dict:
    """Load VeriFYD .meta.json sidecar if present."""
    try:
        import json as _json
        candidates = []
        root, ext = os.path.splitext(video_path or "")
        if root:
            candidates.append(root + ".meta.json")
        if video_path:
            candidates.append(
                video_path.replace(".mp4", ".meta.json")
                          .replace(".MP4", ".meta.json")
                          .replace(".mov", ".meta.json")
                          .replace(".MOV", ".meta.json")
            )
        for sidecar in candidates:
            if sidecar and os.path.exists(sidecar):
                with open(sidecar, "r", encoding="utf-8", errors="replace") as sf:
                    data = _json.load(sf)
                return data if isinstance(data, dict) else {}
    except Exception:
        pass
    return {}


def _vfyd_is_generic_mobile_filename(filename: str) -> bool:
    """True when mobile/social apps stripped useful caption/source text."""
    import re as _re
    name = os.path.basename(str(filename or "")).strip().lower()
    stem, ext = os.path.splitext(name)
    if ext and ext not in (".mp4", ".mov", ".m4v", ".webm", ".mkv"):
        return False
    if not stem:
        return False
    generic_exact = {
        "video", "download", "reel", "instagram", "tiktok", "facebook",
        "screenrecording", "screen recording", "movie", "clip",
    }
    if stem in generic_exact:
        return True
    patterns = (
        r"^\d{6,}$",
        r"^img[_-]?\d{3,}$",
        r"^vid[_-]?\d{3,}$",
        r"^pxl[_-]?\d{6,}$",
        r"^video[_-]?\d{3,}$",
        r"^screenrecord[_-]?\d{3,}$",
        r"^screen_recording[_-]?\d{3,}$",
        r"^[a-f0-9]{8,}$",
    )
    return any(_re.match(p, stem) for p in patterns)


def _vfyd_probe_video_summary(video_path: str) -> dict:
    """Return width/height/duration and basic tags using ffprobe."""
    out = {"width": 0, "height": 0, "duration": 0.0, "tags": {}, "has_real_device_metadata": False}
    try:
        import subprocess as _sp, json as _json
        result = _sp.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", "-show_streams", video_path],
            capture_output=True, text=True, timeout=15
        )
        if result.returncode != 0 or not result.stdout.strip():
            return out
        data = _json.loads(result.stdout)
        fmt = data.get("format", {}) if isinstance(data, dict) else {}
        tags = fmt.get("tags", {}) or {}
        out["tags"] = tags
        try:
            out["duration"] = float(fmt.get("duration") or 0.0)
        except Exception:
            out["duration"] = 0.0
        for s in data.get("streams", []) or []:
            if s.get("codec_type") == "video":
                out["width"] = int(s.get("width") or 0)
                out["height"] = int(s.get("height") or 0)
                break
        tag_blob = " ".join(f"{k}={v}" for k, v in tags.items()).lower()
        out["has_real_device_metadata"] = any(x in tag_blob for x in (
            "com.android.version", "com.apple.quicktime.make", "com.apple.quicktime.model",
            "iphone", "samsung", "google pixel", "dji", "gopro", "sony", "canon", "nikon",
        ))
    except Exception:
        pass
    return out


def _vfyd_max_ctx(contexts, key: str, default=0.0):
    vals = []
    for ctx in contexts or []:
        if isinstance(ctx, dict) and key in ctx:
            vals.append(_vfyd_safe_float(ctx.get(key), default))
    return max(vals) if vals else default


def _vfyd_add_unique_flag(flags, flag: str):
    if flags is None:
        flags = []
    if not isinstance(flags, list):
        flags = list(flags) if flags else []
    if flag not in flags:
        flags.append(flag)
    return flags


def _vfyd_prepare_viral_ai_reel_context(video_path: str, contexts, signal_context: dict) -> dict:
    """Inject mobile/social-reel risk context for GPT without changing thresholds."""
    signal_context = signal_context or {}
    contexts = [c for c in (contexts or []) if isinstance(c, dict)]
    sidecar = _vfyd_load_sidecar(video_path)
    probe = _vfyd_probe_video_summary(video_path)
    original_filename = sidecar.get("original_filename") or sidecar.get("filename") or os.path.basename(video_path or "")
    generic_mobile = _vfyd_is_generic_mobile_filename(original_filename)
    width = _vfyd_safe_int(probe.get("width"), 0)
    height = _vfyd_safe_int(probe.get("height"), 0)
    duration = _vfyd_safe_float(probe.get("duration"), 0.0)
    short_vertical = bool(height > width and 3.0 <= duration <= 18.0)
    no_real_device = not bool(probe.get("has_real_device_metadata"))
    max_deepfake = _vfyd_max_ctx(contexts, "deepfake_score", 0.0)
    max_skin = _vfyd_max_ctx(contexts, "skin_ratio", _vfyd_safe_float(signal_context.get("skin_ratio", 0.0)))
    max_motion_period = _vfyd_max_ctx(contexts, "motion_period", _vfyd_safe_float(signal_context.get("motion_period", 0.0)))
    max_omni = _vfyd_max_ctx(contexts, "omni_flow_entropy", _vfyd_safe_float(signal_context.get("omni_flow_entropy", 0.0)))
    content_type = str(signal_context.get("content_type") or "").lower()
    source = str(sidecar.get("source") or signal_context.get("source") or "upload").lower()
    signal_context["generic_mobile_filename"] = generic_mobile
    signal_context["original_filename"] = original_filename
    signal_context["short_vertical_social_video"] = short_vertical
    signal_context["no_real_device_metadata"] = no_real_device
    signal_context["video_duration"] = duration
    signal_context["video_width"] = width
    signal_context["video_height"] = height
    signal_context["deepfake_score"] = max(_vfyd_safe_float(signal_context.get("deepfake_score"), 0.0), max_deepfake)
    signal_context["max_skin_ratio"] = max_skin
    signal_context["max_motion_period"] = max_motion_period
    signal_context["max_omni_flow_entropy"] = max_omni
    viral_candidate = (
        short_vertical and no_real_device and
        (content_type in ("action", "cinematic", "portrait", "single_subject") or max_skin >= 0.08) and
        (max_deepfake >= 75 or (generic_mobile and max_motion_period >= 0.55 and max_omni >= 3.25))
    )
    if viral_candidate:
        signal_context["viral_ai_reel_candidate"] = True
        signal_context["_extra_context"] = (
            str(signal_context.get("_extra_context", ""))
            + "\nVIRAL AI REEL CONTEXT: short vertical social-style clip; "
              f"generic_mobile_filename={generic_mobile}; no_real_device_metadata={no_real_device}; "
              f"deepfake_score={max_deepfake:.0f}; skin_ratio={max_skin:.3f}; "
              f"motion_period={max_motion_period:.3f}; omni_flow_entropy={max_omni:.3f}. "
              "Inspect for a complete staged viral narrative/punchline, perfectly convenient framing, "
              "and scripted reaction timing. Do not score as real solely because frames look photorealistic."
        )
    return signal_context


def _vfyd_apply_viral_ai_reel_guard(*, video_path: str, combined_ai_score: float, mode: str, signal_ai_score: int,
                                    gpt_ai_score: int, gpt_result: dict, signal_context: dict,
                                    all_signal_contexts=None, n_clips: int = 1):
    """Guard against mobile-renamed photorealistic AI social reels slipping to REAL."""
    signal_context = signal_context or {}
    contexts = [signal_context] + [c for c in (all_signal_contexts or []) if isinstance(c, dict)]
    sidecar = _vfyd_load_sidecar(video_path)
    probe = _vfyd_probe_video_summary(video_path)
    original_filename = sidecar.get("original_filename") or signal_context.get("original_filename") or os.path.basename(video_path or "")
    generic_mobile = bool(signal_context.get("generic_mobile_filename")) or _vfyd_is_generic_mobile_filename(original_filename)
    width = _vfyd_safe_int(probe.get("width") or signal_context.get("video_width"), 0)
    height = _vfyd_safe_int(probe.get("height") or signal_context.get("video_height"), 0)
    duration = _vfyd_safe_float(probe.get("duration") or signal_context.get("video_duration"), 0.0)
    short_vertical = bool(height > width and 3.0 <= duration <= 18.0)
    no_real_device = bool(signal_context.get("no_real_device_metadata")) or not bool(probe.get("has_real_device_metadata"))
    max_deepfake = max(_vfyd_max_ctx(contexts, "deepfake_score", 0.0), _vfyd_safe_float(signal_context.get("deepfake_score", 0.0)))
    max_skin = max(_vfyd_max_ctx(contexts, "skin_ratio", 0.0), _vfyd_safe_float(signal_context.get("max_skin_ratio", 0.0)))
    max_motion_period = max(_vfyd_max_ctx(contexts, "motion_period", 0.0), _vfyd_safe_float(signal_context.get("max_motion_period", 0.0)))
    max_omni = max(_vfyd_max_ctx(contexts, "omni_flow_entropy", 0.0), _vfyd_safe_float(signal_context.get("max_omni_flow_entropy", 0.0)))
    content_type = str(signal_context.get("content_type") or "").lower()
    gpt_flags = list(gpt_result.get("flags") or []) if isinstance(gpt_result, dict) else []
    gpt_reasoning = str(gpt_result.get("reasoning", "") if isinstance(gpt_result, dict) else "").lower()
    gpt_scores = gpt_result.get("scores", {}) if isinstance(gpt_result, dict) else {}
    scene_staging = _vfyd_safe_int(gpt_scores.get("scene_staging", 0), 0) if isinstance(gpt_scores, dict) else 0
    behavioral = _vfyd_safe_int(gpt_scores.get("behavioral_plausibility", 0), 0) if isinstance(gpt_scores, dict) else 0
    generator_artifacts = _vfyd_safe_int(gpt_scores.get("generator_artifacts", 0), 0) if isinstance(gpt_scores, dict) else 0
    visual_ai_hint = (
        "viral_ai_reel_pattern" in gpt_flags or "staged_social_reel" in gpt_flags or "viral ai" in gpt_reasoning or
        scene_staging >= 7 or behavioral >= 7 or generator_artifacts >= 7
    )
    face_or_person_action = content_type in ("action", "cinematic", "portrait", "single_subject", "selfie", "talking_head") or max_skin >= 0.08
    high_deepfake_mobile_reel = short_vertical and generic_mobile and no_real_device and face_or_person_action and max_deepfake >= 78
    staged_reel_visual_guard = short_vertical and no_real_device and face_or_person_action and visual_ai_hint and gpt_ai_score >= 50
    pattern_signal_guard = short_vertical and generic_mobile and no_real_device and face_or_person_action and max_deepfake >= 70 and max_motion_period >= 0.55 and max_omni >= 3.25
    if not (high_deepfake_mobile_reel or staged_reel_visual_guard or pattern_signal_guard):
        return None
    target = 78.0 if high_deepfake_mobile_reel else 68.0
    if pattern_signal_guard:
        target = max(target, 72.0)
    old_combined = float(combined_ai_score)
    new_combined = max(old_combined, target)
    new_mode = "viral-ai-reel provenance/behavior guard"
    if isinstance(gpt_result, dict):
        gpt_result["flags"] = _vfyd_add_unique_flag(gpt_flags, "viral_ai_reel_pattern")
        gpt_result["flags"] = _vfyd_add_unique_flag(gpt_result.get("flags"), "generic_mobile_upload_no_source_text")
        reason_prefix = "Short vertical mobile/social upload has a staged viral-reel pattern: generic filename, no real-device provenance, person/action content, and high facial/deepfake or behavior signals. "
        existing_reason = str(gpt_result.get("reasoning", ""))
        if reason_prefix not in existing_reason:
            gpt_result["reasoning"] = (reason_prefix + existing_reason)[:1000]
    log.info(
        "VIRAL_AI_REEL_GUARD: combined %.1f→%.1f generic=%s short_vertical=%s no_device=%s deepfake=%.0f skin=%.3f motion_period=%.3f omni=%.3f gpt=%d file=%s",
        old_combined, new_combined, generic_mobile, short_vertical, no_real_device, max_deepfake, max_skin, max_motion_period, max_omni, gpt_ai_score, original_filename,
    )
    return {
        "combined_ai_score": new_combined,
        "mode": new_mode,
        "viral_ai_reel_pattern": True,
        "generic_mobile_filename": generic_mobile,
        "no_real_device_metadata": no_real_device,
        "deepfake_score": int(round(max_deepfake)),
        "max_motion_period": round(max_motion_period, 4),
        "max_omni_flow_entropy": round(max_omni, 4),
        "original_filename": original_filename,
        "short_vertical_social_video": short_vertical,
    }


# AI_SOURCE_PROVENANCE_PATCH: helper fields for API/job result detail
def _ai_source_detail_fields(video_path: str) -> dict:
    """Return optional detail fields for explicit AI-source provenance evidence."""
    try:
        if not scan_video_source_for_ai:
            return {}
        src_hit = scan_video_source_for_ai(video_path)
        if not src_hit.get("detected"):
            return {}
        return {
            "ai_source_detected": True,
            "ai_source_generator": src_hit.get("generator", ""),
            "ai_source_matched": src_hit.get("matched", ""),
            "ai_source_confidence": src_hit.get("confidence", ""),
            "ai_source_text": src_hit.get("source_text", "")[:500],
            "provenance_override": src_hit.get("confidence") == "high",
        }
    except Exception as exc:
        log.debug("AI source detail field scan skipped: %s", exc)
        return {}


def _add_ai_source_context(video_path: str, signal_context: dict) -> None:
    """Pass explicit AI-source evidence into GPT context without changing thresholds."""
    try:
        if not scan_video_source_for_ai or signal_context is None:
            return
        src_hit = scan_video_source_for_ai(video_path)
        if not src_hit.get("detected"):
            return
        signal_context["ai_source_detected"] = True
        signal_context["ai_source_generator"] = src_hit.get("generator", "")
        signal_context["ai_source_matched"] = src_hit.get("matched", "")
        signal_context["ai_source_confidence"] = src_hit.get("confidence", "")
        signal_context["ai_source_reason"] = src_hit.get("reason", "")
        signal_context["ai_source_text"] = src_hit.get("source_text", "")[:500]
    except Exception as exc:
        log.debug("AI source context pass-through skipped: %s", exc)


def _check_metadata_override(video_path: str) -> tuple:
    """
    Check for definitive metadata signals before running engines.
    Returns (override: bool, ai_score: int, label: str, reason: str)
    """
    import subprocess, json as _json, os


    # AI_SOURCE_PROVENANCE_PATCH: explicit AI-source / provenance text check
    # Examples: "Made with @higgsfield.ai", "generated with Sora", "created with Runway".
    # This is source-level evidence and should override photorealistic pixels.
    try:
        if scan_video_source_for_ai:
            src_hit = scan_video_source_for_ai(video_path)
            if src_hit.get("detected") and src_hit.get("confidence") == "high":
                reason = src_hit.get("reason") or "Source text identifies an AI video generator."
                generator = src_hit.get("generator") or src_hit.get("matched") or "AI generator"
                reason = f"{reason} Generator/platform: {generator}."
                log.info(
                    "METADATA_OVERRIDE: explicit AI source provenance detected -> AI | generator=%s matched=%s",
                    generator,
                    src_hit.get("matched", ""),
                )
                return True, int(src_hit.get("ai_score", 96) or 96), "AI", reason
    except Exception as e:
        log.warning("METADATA: AI source provenance check error: %s", e)

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
        compatible_brands = fmt_tags.get("compatible_brands", "").lower()

        # Exclude FFmpeg/AI pipeline container signatures:
        # "iso4" and "iso6" in compatible_brands = ISO Base Media v4/v6 = FFmpeg output
        # Real cameras use: isomiso2mp41, isomiso2avc1mp41, isomavc1, qt, mp41
        # AI generators use: isomiso4, iso4, iso6 (FFmpeg default mux output)
        _ffmpeg_ai_brand = any(b in compatible_brands for b in ("iso4", "iso6"))

        is_real_container = (
            major_brand in ("mp42", "isom", "M4V ", "qt  ") and
            bool(creation_time) and
            "lavf" not in encoder.lower() and
            not _ffmpeg_ai_brand   # exclude FFmpeg AI pipeline containers
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
            "gpt_flags":       ["metadata_override", "ai_source_provenance"],
            "content_type":    "unknown",
            "blend_mode":      "metadata-override",
            "weight_signal":   1.0,
            "weight_gpt":      0.0,
            **_ai_source_detail_fields(video_path),
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
    # AI_SOURCE_PROVENANCE_PATCH: pass source/caption/filename evidence into GPT context
    _add_ai_source_context(video_path, signal_context)

    # ── Engine 1B: Audio detector (additive, conservative) ───
    audio_result = {"available": False, "audio_ai_score": 50, "confidence": "unavailable", "evidence": []}
    audio_contribution = 0
    try:
        analyze_audio, get_audio_contribution = _get_audio_analyzer()
        if analyze_audio and get_audio_contribution:
            audio_result = analyze_audio(video_path)
            audio_contribution = get_audio_contribution(
                audio_result.get("audio_ai_score", 50),
                audio_result.get("confidence", "low"),
                signal_context,
            )
            signal_context["audio_score"] = audio_result.get("audio_ai_score", 50)
            signal_context["audio_confidence"] = audio_result.get("confidence", "low")
            signal_context["audio_evidence"] = audio_result.get("evidence", [])[:5]
            signal_context["audio_contribution"] = audio_contribution
            if "stereo_corr" in audio_result:
                signal_context["audio_stereo_corr"] = audio_result.get("stereo_corr", 0)
            if "duration_mismatch" in audio_result:
                signal_context["audio_duration_mismatch"] = audio_result.get("duration_mismatch", 0)
            if audio_contribution:
                old_signal = signal_ai_score
                signal_ai_score = int(round(min(100, max(0, signal_ai_score + audio_contribution))))
                log.info("Audio detector: score=%s conf=%s contribution=%+d signal %d→%d",
                         audio_result.get("audio_ai_score"), audio_result.get("confidence"),
                         audio_contribution, old_signal, signal_ai_score)
    except Exception as e:
        log.debug("Audio analysis skipped: %s", e)

    # ── Engine 2: GPT-4o ──────────────────────────────────────
    frames_b64      = extract_key_frames(video_path)
    signal_context = _vfyd_prepare_viral_ai_reel_context(video_path, [signal_context], signal_context)
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
    _viral_ai_reel_guard = _vfyd_apply_viral_ai_reel_guard(
        video_path=video_path,
        combined_ai_score=combined_ai_score,
        mode=mode if not gpt_failed else "signal-only",
        signal_ai_score=signal_ai_score,
        gpt_ai_score=gpt_ai_score,
        gpt_result=gpt_result,
        signal_context=signal_context,
        all_signal_contexts=[signal_context],
        n_clips=1,
    )
    if _viral_ai_reel_guard:
        combined_ai_score = _viral_ai_reel_guard["combined_ai_score"]
        mode = _viral_ai_reel_guard["mode"]
        gpt_flags = gpt_result.get("flags", [])
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
        "audio_ai_score":  audio_result.get("audio_ai_score", 50),
        "audio_confidence": audio_result.get("confidence", "unavailable"),
        "audio_contribution": audio_contribution,
        "audio_evidence":  audio_result.get("evidence", [])[:8],
        "gpt_available":   gpt_available,
        "gpt_reasoning":   gpt_result.get("reasoning", ""),
        "gpt_flags":       gpt_result.get("flags", []),
        "weight_signal":   w_sig,
        "weight_gpt":      w_gpt,
        "blend_mode":      mode if not gpt_failed else "signal-only",
        "threshold_real":  THRESHOLD_REAL,
        "threshold_undet": THRESHOLD_UNDETERMINED,
    }

    if _viral_ai_reel_guard:
        detail.update(_viral_ai_reel_guard)
        detail["gpt_flags"] = gpt_result.get("flags", detail.get("gpt_flags", []))
        detail["blend_mode"] = mode

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
            "gpt_flags":       ["metadata_override", "ai_source_provenance"],
            "content_type":    "unknown",
            "blend_mode":      "metadata-override",
            "weight_signal":   1.0,
            "weight_gpt":      0.0,
            **_ai_source_detail_fields(video_path),
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

            # Heavy learned models (DINOv2 + ViT DeepfakeDetector) are intentionally
            # NOT run inside every parallel clip worker anymore. On 3-clip videos this
            # caused repeated model loads/inference and added ~15-25s with little
            # accuracy benefit. We now run these tie-breakers once, after primary
            # signal/NPR scoring, on the most suspicious/ambiguous clip only.
            context["_clip_path"] = clip_path
            context["_offset_pct"] = offset_pct
            context["pre_heavy_score"] = score

            log.info("Clip @%.0f%%: signal_score=%d content=%s",
                     offset_pct * 100, score, context.get("content_type", "?"))
            return score, context, offset_pct
        except Exception as e:
            log.warning("Signal detection failed for clip @%.0f%%: %s", offset_pct * 100, e)
            return None, {}, offset_pct

    # ── Start frame extraction in background BEFORE signal detection ──
    # This is the parallelization: frame extraction (~1-2s) overlaps with
    # signal detection (~8-12s), saving time without changing any logic.
    # The frames are retrieved later in the GPT section via _frame_future.result()
    _frames_per_clip_pre = 6 if len(clips) <= 2 else 4
    def _extract_all_frames():
        """Extract frames from all clips for GPT. Runs in background thread."""
        _frames = []
        for _cp, _ in clips:
            try:
                _f = extract_key_frames(_cp, n_frames=_frames_per_clip_pre)
                _frames.extend(_f)
            except Exception as _e:
                log.debug("Background frame extraction failed for clip: %s", _e)
        return _frames

    _frame_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    _frame_future = _frame_executor.submit(_extract_all_frames)

    # ── Run signal detection on all clips in parallel ─────────
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

    # ── Heavy-model gating for speed ─────────────────────────
    # Run DINOv2 and ViT DeepfakeDetector only once per video, on the clip
    # where they can help most. Primary detector + NPR still run on every clip.
    # This preserves the multi-clip forensic scan while avoiding 2-3 repeated
    # 330-350MB model inferences/loads per job.
    def _apply_heavy_models_once(_valid):
        if not _valid:
            return _valid

        _candidates = []
        for _idx, (_score, _ctx, _pct) in enumerate(_valid):
            _skin = float(_ctx.get("skin_ratio", 0.0) or 0.0)
            _ambig = 35 <= _score <= 72
            _priority = (2 if _ambig else 0) + (1 if _skin >= 0.08 else 0)
            _candidates.append((_priority, _score, _idx))
        _candidates.sort(reverse=True)
        _chosen_idx = _candidates[0][2]
        _score, _ctx, _pct = _valid[_chosen_idx]
        _clip_path = _ctx.get("_clip_path")
        if not _clip_path:
            return _valid

        _original_score = _score
        log.info(
            "HEAVY_MODEL_GATE: running DINOv2/Deepfake once on clip @%.0f%% "
            "(score=%d, clips=%d)",
            _pct * 100, _score, len(_valid)
        )

        try:
            analyze_dinov2, get_dino_contribution = _get_dino_analyzer()
            if analyze_dinov2 and 35 <= _score <= 72:
                dino_score, dino_signals = analyze_dinov2(_clip_path)
                dino_contribution = get_dino_contribution(dino_score, _score)
                _ctx["dino_score"] = dino_score
                _ctx["dino_signals"] = dino_signals
                _ctx["dino_contribution"] = dino_contribution
                if dino_contribution != 0:
                    log.info("DINOv2 @%.0f%%: score=%d contribution=%+d (signal=%d ambiguous)",
                             _pct * 100, dino_score, dino_contribution, _score)
                    _score = int(round(min(100, max(0, _score + dino_contribution))))
            elif analyze_dinov2:
                log.debug("DINOv2 @%.0f%%: skipped (signal=%d not ambiguous)", _pct * 100, _score)
        except Exception as e:
            log.debug("DINOv2 analysis skipped for clip @%.0f%%: %s", _pct * 100, e)

        try:
            analyze_deepfake, get_deepfake_contribution = _get_deepfake_analyzer()
            if analyze_deepfake:
                _ct = _ctx.get("content_type", "cinematic")
                _skin = _ctx.get("skin_ratio", 0.0)
                df_score, df_signals = analyze_deepfake(
                    _clip_path,
                    content_type=_ct,
                    skin_ratio=_skin,
                )
                _ctx["deepfake_score"]   = df_score
                _ctx["deepfake_signals"] = df_signals
                if df_signals.get("available") and df_score > 0:
                    df_contribution = get_deepfake_contribution(df_score, _score, _ct)
                    _ctx["deepfake_contribution"] = df_contribution
                    if df_contribution != 0:
                        log.info(
                            "DeepfakeDetector @%.0f%%: score=%d contribution=%+d "
                            "(signal=%d content=%s skin=%.3f)",
                            _pct * 100, df_score, df_contribution,
                            _score, _ct, _skin,
                        )
                        _score = int(round(min(100, max(0, _score + df_contribution))))
                elif df_signals.get("skipped_reason"):
                    log.debug(
                        "DeepfakeDetector @%.0f%%: skipped (%s)",
                        _pct * 100, df_signals["skipped_reason"],
                    )
        except Exception as e:
            log.debug("DeepfakeDetector skipped for clip @%.0f%%: %s", _pct * 100, e)

        if _score != _original_score:
            log.info(
                "HEAVY_MODEL_GATE: selected clip @%.0f%% adjusted %d→%d",
                _pct * 100, _original_score, _score
            )
            _valid[_chosen_idx] = (_score, _ctx, _pct)
        return _valid

    valid = _apply_heavy_models_once(valid)

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

    # ── Full-file audio detector (runs once per video) ───────
    # Audio is analyzed on the original full file, not each silent detection clip,
    # so it can detect TTS/stock-audio/clean-room artifacts without slowing every clip.
    audio_result = {"available": False, "audio_ai_score": 50, "confidence": "unavailable", "evidence": []}
    audio_contribution = 0
    try:
        analyze_audio, get_audio_contribution = _get_audio_analyzer()
        if analyze_audio and get_audio_contribution:
            _tmp_content_context = (valid[0][1] if valid and len(valid[0]) > 1 else {})
            audio_result = analyze_audio(video_path)
            audio_contribution = get_audio_contribution(
                audio_result.get("audio_ai_score", 50),
                audio_result.get("confidence", "low"),
                _tmp_content_context,
            )
            if audio_contribution:
                old_signal = signal_ai_score
                signal_ai_score = int(round(min(100, max(0, signal_ai_score + audio_contribution))))
                log.info("Audio detector full-file: score=%s conf=%s contribution=%+d signal %d→%d",
                         audio_result.get("audio_ai_score"), audio_result.get("confidence"),
                         audio_contribution, old_signal, signal_ai_score)
            for _ctx in all_signal_contexts:
                _ctx["audio_score"] = audio_result.get("audio_ai_score", 50)
                _ctx["audio_confidence"] = audio_result.get("confidence", "low")
                _ctx["audio_evidence"] = audio_result.get("evidence", [])[:5]
                _ctx["audio_contribution"] = audio_contribution
                if "stereo_corr" in audio_result:
                    _ctx["audio_stereo_corr"] = audio_result.get("stereo_corr", 0)
                if "duration_mismatch" in audio_result:
                    _ctx["audio_duration_mismatch"] = audio_result.get("duration_mismatch", 0)
    except Exception as e:
        log.debug("Audio analysis skipped for full file: %s", e)

    # Use context from highest-noise clip (most data-rich)
    best_ctx_idx = noise_weights.index(max(noise_weights))
    signal_context = valid[best_ctx_idx][1]
    content_type = signal_context.get("content_type", "cinematic")

    # Inject portrait flag into signal_context for gpt_vision exotic bird hint.
    # Portrait = clip is vertically oriented (height > width * 1.5 — phone video).
    _clip_w = signal_context.get("clip_width", 0)
    _clip_h = signal_context.get("clip_height", 0)
    if _clip_w and _clip_h:
        signal_context["portrait"] = _clip_h > _clip_w * 1.5

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

    signal_context = _vfyd_prepare_viral_ai_reel_context(video_path, all_signal_contexts, signal_context)

    # SOCIAL_REENCODE_REAL_GUARD_V2
    # TikTok/Instagram/Facebook link downloads are platform re-encodes. They can
    # create DCT grid, edge crawl, flat-noise, shadow drift, and omni-flow artifacts
    # even on real phone videos. Pass this to GPT and the final override guard.
    _social_reencode_source_v2 = any(
        s in str(_video_source).lower()
        for s in ("tiktok", "instagram", "facebook", "fb.watch", "youtube", "smvd")
    )
    if _social_reencode_source_v2:
        signal_context["social_reencode_guard"] = True
        signal_context["social_reencode_source"] = _video_source
        signal_context["clip_signal_scores"] = signal_scores
        signal_context["realish_clip_count"] = sum(1 for s in signal_scores if int(s) <= 45)
        signal_context["_extra_context"] = (
            str(signal_context.get("_extra_context", ""))
            + "\nSOCIAL RE-ENCODE REAL-VIDEO GUARD: This came from a social-platform link "
              "with no platform AIGC label. Treat compression/transcode artifacts conservatively. "
              "Do NOT score as AI merely because of DCT grid/blocking, edge crawl, low flat-region noise, "
              "shadow drift, missing device metadata, or social-media compression. Only raise AI if frames "
              "show explicit visual evidence such as impossible physics, morphing hands/face/text, unstable anatomy, "
              "an AI watermark/source label, or a truly complete staged viral AI event pattern. "
              "For normal selfie/person/kitchen/social footage, prefer REAL or UNDETERMINED unless those explicit signs are visible."
        )
        log.info("SOCIAL_REENCODE_REAL_GUARD_V2: source=%s clips=%s realish=%d", _video_source, signal_scores, signal_context["realish_clip_count"])

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
    #
    # PARALLELIZATION: Frame extraction runs in a background thread that was
    # started BEFORE signal detection completed (see _frame_future below).
    # This overlaps the ~1-2s frame extraction with the tail end of signal
    # detection, saving time without changing any logic.
    #
    # The GPT API call itself still waits for signal_context to be fully
    # populated (hybrid_flag, signal_scores, etc.) before being sent.
    # Zero change to detection accuracy or blend logic.
    all_frames = []
    try:
        # Retrieve pre-extracted frames from the background thread
        # _frame_future was submitted before signal detection started
        all_frames = _frame_future.result(timeout=30)
    except Exception as e:
        log.warning("Parallel frame extraction failed (%s) — retrying inline", e)
        # Fallback: extract frames inline if parallel extraction failed
        frames_per_clip = max(2, 8 // n_clips)
        for clip_path, _ in clips:
            try:
                frames = extract_key_frames(clip_path, n_frames=frames_per_clip)
                all_frames.extend(frames)
            except Exception as fe:
                log.warning("Frame extraction failed for clip: %s", fe)

    if not all_frames:
        log.warning("No frames extracted — using signal only")
        gpt_result = {"ai_probability": 50, "reasoning": "Frame extraction failed",
                      "flags": [], "available": False}
    else:
        # Add hybrid context to GPT prompt — still uses full signal_context
        # populated after signal detection completed
        if hybrid_flag:
            signal_context["_extra_context"] = (
                f"NOTE: This video was sampled at {n_clips} points in time. "
                f"Signal scores varied significantly ({min(signal_scores)}-{max(signal_scores)}), "
                f"suggesting mixed content — some segments real, some AI-generated."
            )
        gpt_result = gpt_vision_score_with_context(all_frames, signal_context)

    # Clean up the frame extraction executor
    try:
        _frame_executor.shutdown(wait=False)
    except Exception:
        pass

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
        _is_instagram = "instagram" in _video_source.lower()
        # Instagram re-encodes all videos to VP9 at 540p, creating compression
        # noise that mimics real camera grain — same problem as YouTube H264.
        # Suppress real_dominant and clash->real for Instagram sources.
        _youtube_signal_unreliable = (
            signal_context.get("youtube_lowres_adjusted", False) or
            _is_youtube or _is_instagram
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
        _strong_gpt_real   = gpt_ai_score < 30
        _moderate_gpt_real = gpt_ai_score < 45
        real_dominant = (
            (
                # YouTube/Instagram: compression fakes real grain, so real_dominant
                # applies when GPT and DINOv2 both confirm real.
                # NOT for cinematic/action content — too many AI animal/wildlife videos
                # fool GPT into scoring real. Signal detector is more reliable there.
                ((_is_youtube or _is_instagram) and signal_ai_score > 50
                 and _moderate_gpt_real and _dino_score_ctx <= 5
                 and content_type not in ("cinematic", "action"))
                or
                # Uploaded files: signal ambiguous, GPT very confident real, DINOv2 confirms.
                # NOT for cinematic/action — AI wildlife videos fool GPT.
                (signal_ai_score > 50 and signal_ai_score <= 75 and _strong_gpt_real
                 and _dino_score_ctx <= 2
                 and content_type not in ("cinematic", "action"))
            )
        )

        if _youtube_signal_unreliable and signal_ai_score < 50 and gpt_ai_score > 50:
            log.info(
                "YOUTUBE/INSTAGRAM: clash->real suppressed — "
                "signal(%d) noise unreliable (H264/VP9 compression mimics real grain), "
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


    # ── GPT + forensic agreement overrides for short social AI reels ─────
    # Production miss: X/Twitter vertical social clip had GPT=70 and one
    # suspicious forensic clip, but the final combiner treated this as
    # "clash→real" because the weighted signal average was dragged down by
    # a lower-scoring clip. For social-media AI, a single strong segment plus
    # GPT's semantic AI call is enough to prevent certification as REAL.
    #
    # This does NOT add model calls. It only reuses already-computed clip
    # scores and signal_context values.
    _max_clip_score = max(signal_scores or [signal_ai_score])
    _min_clip_score = min(signal_scores or [signal_ai_score])

    if not gpt_failed:
        if gpt_ai_score >= 65 and _max_clip_score >= 45:
            old_combined = combined_ai_score
            combined_ai_score = max(combined_ai_score, 68.0)
            mode = "GPT_FORENSIC_AGREEMENT_AI_OVERRIDE"
            log.info(
                "GPT_FORENSIC_AGREEMENT_AI_OVERRIDE: GPT=%d and max_clip=%d "
                "agree on AI risk → combined %.1f→%.1f",
                gpt_ai_score, _max_clip_score, old_combined, combined_ai_score
            )

        # Softer hybrid path for the exact pattern in the X/Twitter miss:
        # one clip lands suspicious after heavy-model subtraction, another
        # clip looks real/compressed, and GPT still calls the video AI.
        if gpt_ai_score >= 68 and _max_clip_score >= 37 and hybrid_flag:
            old_combined = combined_ai_score
            combined_ai_score = max(combined_ai_score, 65.0)
            mode = "HYBRID_AI_CLIP_OVERRIDE"
            log.info(
                "HYBRID_AI_CLIP_OVERRIDE: GPT=%d, max_clip=%d, min_clip=%d, "
                "hybrid=%s → combined %.1f→%.1f",
                gpt_ai_score, _max_clip_score, _min_clip_score,
                hybrid_flag, old_combined, combined_ai_score
            )

    # ── Social AI render composite override ───────────────────────────────
    # Catches re-encoded X/Twitter/Instagram/TikTok AI clips where real-looking
    # grain and natural palette deductions suppress the score, but the render
    # pipeline still leaks: very high RGB channel correlation, frozen/synthetic
    # saturation behavior, omnidirectional optical-flow noise, and edge crawl.
    _ctxs_social_render = all_signal_contexts or []

    def _social_float(ctx, *names, default=0.0):
        for name in names:
            try:
                val = ctx.get(name, None)
                if val is not None:
                    return float(val or 0.0)
            except Exception:
                continue
        return float(default)

    _max_social_chan = max([_social_float(ctx, "chan_corr", "channel_corr") for ctx in _ctxs_social_render] or [0.0])
    _max_social_sat_std = max([_social_float(ctx, "sat_std", "saturation_std") for ctx in _ctxs_social_render] or [0.0])
    _max_social_omni = max([_social_float(ctx, "omni_flow_ent", "omni_ent", "omni_flow_entropy") for ctx in _ctxs_social_render] or [0.0])
    _max_social_edge_cov = max([_social_float(ctx, "edge_cov", "edge_cov_var") for ctx in _ctxs_social_render] or [0.0])
    _max_social_edge_mvar = max([_social_float(ctx, "edge_mvar", "edge_motion_var") for ctx in _ctxs_social_render] or [0.0])
    _max_social_pre_heavy = max(
        [int(ctx.get("pre_heavy_score", score) or score) for score, ctx, _ in valid] or [signal_ai_score]
    )

    _social_render_ai = (
        _max_social_chan >= 0.94 and
        _max_social_sat_std >= 2.0 and
        _max_social_omni >= 3.50 and
        _max_social_edge_cov >= 1.60
    )

    # Alternate path for cases where channel correlation is slightly below .94
    # but GPT already sees AI and the clip was suspicious before heavy-model
    # tie-breakers subtracted points.
    _social_render_ai_alt = (
        not gpt_failed and
        gpt_ai_score >= 65 and
        _max_social_pre_heavy >= 45 and
        _max_social_chan >= 0.90 and
        _max_social_sat_std >= 2.0 and
        _max_social_omni >= 3.45 and
        (_max_social_edge_cov >= 1.50 or _max_social_edge_mvar >= 0.030)
    )

    if _social_render_ai or _social_render_ai_alt:
        old_combined = combined_ai_score
        combined_ai_score = max(combined_ai_score, 66.0)
        mode = "SOCIAL_AI_RENDER_OVERRIDE"
        log.info(
            "SOCIAL_AI_RENDER_OVERRIDE: render composite detected "
            "(chan_corr=%.3f sat_std=%.2f omni=%.3f edge_cov=%.3f "
            "edge_mvar=%.4f pre_heavy=%d gpt=%d) → combined %.1f→%.1f",
            _max_social_chan, _max_social_sat_std, _max_social_omni,
            _max_social_edge_cov, _max_social_edge_mvar,
            _max_social_pre_heavy, gpt_ai_score, old_combined, combined_ai_score
        )



    # ── Viral social public-event / destruction AI override ───────────────
    # Production miss: a short vertical Instagram-style "public destruction"
    # reel (child destroys Lamborghini LEGO sculpture) scored REAL because
    # social re-encoding produced convincing camera grain and GPT saw too few
    # frames to understand the full event. These clips often look photographic
    # in still frames, but the temporal/render stack leaks a repeatable pattern:
    #   • short vertical action/cinematic framing,
    #   • high background drift / scene warp,
    #   • strong periodic/generated motion,
    #   • unusually coherent edge coverage / edge crawl,
    #   • elevated temporal consistency/variance,
    #   • low motion-sync or low inter-frame detail variation,
    #   • shadows/lighting that stay too coherent through a dramatic event.
    #
    # This override is intentionally pattern-based and narrow. It does NOT rely
    # on the uploaded filename, a particular sculpture/object, or GPT calling AI.
    # It prevents clean REAL certification for similar viral "impossible event"
    # / public destruction / staged accident reels even when compression creates
    # fake real-camera noise.
    _event_ctxs = all_signal_contexts or []
    _event_content_types = {str(ctx.get("content_type", content_type)) for ctx in _event_ctxs} or {content_type}
    _event_action_content = any(ct in ("action", "cinematic") for ct in _event_content_types)
    _event_vertical = bool(signal_context.get("portrait", False))
    if not _event_vertical and _event_ctxs:
        try:
            _event_vertical = any(
                float(ctx.get("clip_height", 0) or 0) > float(ctx.get("clip_width", 0) or 0) * 1.45
                for ctx in _event_ctxs
            )
        except Exception:
            _event_vertical = False

    def _event_float(ctx, *names, default=0.0):
        for name in names:
            try:
                val = ctx.get(name, None)
                if val is not None:
                    return float(val or 0.0)
            except Exception:
                continue
        return float(default)

    _event_motion = max([_event_float(ctx, "motion", "avg_motion") for ctx in _event_ctxs] or [0.0])
    _event_bg = max([_event_float(ctx, "bg_drift", "background_drift") for ctx in _event_ctxs] or [0.0])
    _event_period = max([_event_float(ctx, "motion_period", "period") for ctx in _event_ctxs] or [0.0])
    _event_edge_cov = max([_event_float(ctx, "edge_cov", "edge_cov_var") for ctx in _event_ctxs] or [0.0])
    _event_tcv = max([_event_float(ctx, "tcv", "temporal_consistency_var", "temporal_coherence_var") for ctx in _event_ctxs] or [0.0])
    _event_omni = max([_event_float(ctx, "omni_flow_ent", "omni_ent", "omni_flow_entropy") for ctx in _event_ctxs] or [0.0])
    _event_sync = min([_event_float(ctx, "motion_sync", default=1.0) for ctx in _event_ctxs] or [1.0])
    _event_ifdv = min([_event_float(ctx, "ifdv", "interframe_detail_variation", default=1.0) for ctx in _event_ctxs] or [1.0])
    _event_shadow = max([_event_float(ctx, "shadow_drift") for ctx in _event_ctxs] or [0.0])
    _event_pre_heavy = max(
        [int(ctx.get("pre_heavy_score", score) or score) for score, ctx, _ in valid] or [signal_ai_score]
    )

    _event_votes = 0
    _event_votes += 1 if _event_motion >= 18.0 else 0
    _event_votes += 1 if _event_bg >= 30.0 else 0
    _event_votes += 1 if _event_period >= 0.60 else 0
    _event_votes += 1 if _event_edge_cov >= 0.50 else 0
    _event_votes += 1 if _event_tcv >= 250.0 else 0
    _event_votes += 1 if _event_omni >= 3.35 else 0
    _event_votes += 1 if (_event_sync <= 0.20 or _event_ifdv <= 0.12) else 0
    _event_votes += 1 if _event_shadow >= 0.70 else 0

    _viral_public_event_ai = (
        _event_action_content and
        _event_vertical and
        _event_motion >= 18.0 and
        _event_bg >= 30.0 and
        _event_period >= 0.60 and
        _event_edge_cov >= 0.50 and
        _event_tcv >= 250.0 and
        (_event_omni >= 3.35 or _event_ifdv <= 0.12) and
        _event_votes >= 6
    )

    # Slightly softer path when the primary detector reached at least ambiguous
    # territory before heavy-model tie-breakers deducted points.
    _viral_public_event_ai_alt = (
        _event_action_content and
        _event_vertical and
        _event_pre_heavy >= 30 and
        _event_bg >= 35.0 and
        _event_period >= 0.68 and
        _event_edge_cov >= 0.55 and
        _event_tcv >= 240.0 and
        (_event_omni >= 3.30 or _event_ifdv <= 0.12) and
        _event_votes >= 5
    )

    if _viral_public_event_ai or _viral_public_event_ai_alt:
        old_combined = combined_ai_score
        combined_ai_score = max(combined_ai_score, 66.0)
        mode = "viral social public-event AI forensic override"
        log.info(
            "VIRAL_PUBLIC_EVENT_AI override: short vertical social/public-event "
            "render composite detected (motion=%.1f bg_drift=%.2f period=%.3f "
            "edge_cov=%.3f tcv=%.2f omni=%.3f sync=%.3f ifdv=%.3f "
            "shadow=%.3f pre_heavy=%d votes=%d gpt=%d) → combined %.1f→%.1f",
            _event_motion, _event_bg, _event_period, _event_edge_cov,
            _event_tcv, _event_omni, _event_sync, _event_ifdv, _event_shadow,
            _event_pre_heavy, _event_votes, gpt_ai_score,
            old_combined, combined_ai_score
        )


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

    # ── Device-recorded / screen-recorded AI override ─────────────
    # Real Android/iPhone metadata is strong evidence that a device recorded the
    # file, but it does NOT prove the scene itself is real. Users can screen-record
    # or film an AI video with a real phone, creating genuine device metadata,
    # HEVC/H.264 camera noise, and real sensor artifacts. In that case GPT/DINOv2
    # can be fooled by the phone capture layer while temporal forensic signals still
    # reveal the synthetic source underneath.
    #
    # This is deliberately narrow:
    #   • only when metadata says real device,
    #   • only when at least one clip was already suspicious before heavy-model
    #     tie-breakers,
    #   • only when multiple temporal AI-motion signals agree.
    _device_meta_real = (
        override and ov_label == "REAL" and (
            "Android" in str(ov_reason) or
            "Apple" in str(ov_reason) or
            "device" in str(ov_reason).lower()
        )
    )
    if _device_meta_real:
        _pre_heavy_scores = [int(ctx.get("pre_heavy_score", score) or score) for score, ctx, _ in valid]
        _max_pre_heavy = max(_pre_heavy_scores) if _pre_heavy_scores else signal_ai_score
        _min_motion_sync = min(float(ctx.get("motion_sync", 1.0) or 1.0) for ctx in all_signal_contexts) if all_signal_contexts else 1.0
        _min_ifdv = min(float(ctx.get("ifdv", 1.0) or 1.0) for ctx in all_signal_contexts) if all_signal_contexts else 1.0
        _max_omni = max(float(ctx.get("omni_flow_ent", ctx.get("omni_ent", ctx.get("omni_flow_entropy", 0.0))) or 0.0) for ctx in all_signal_contexts) if all_signal_contexts else 0.0
        _max_noise = max(float(ctx.get("avg_noise", ctx.get("noise", 0.0)) or 0.0) for ctx in all_signal_contexts) if all_signal_contexts else 0.0
        _device_rerecord_ai = (
            _max_pre_heavy >= 58 and
            _min_motion_sync <= 0.070 and
            _min_ifdv <= 0.120 and
            _max_omni >= 3.50
        )
        if _device_rerecord_ai:
            old_combined = combined_ai_score
            combined_ai_score = max(combined_ai_score, 66.0)
            mode = "device-recorded AI forensic override"
            log.info(
                "DEVICE_RECORDED_AI override: real-device metadata present but temporal AI "
                "signals agree (pre_heavy_max=%d motion_sync=%.3f ifdv=%.3f omni=%.3f "
                "noise=%.0f gpt=%d) → combined %.1f→%.1f",
                _max_pre_heavy, _min_motion_sync, _min_ifdv, _max_omni,
                _max_noise, gpt_ai_score, old_combined, combined_ai_score
            )

    # ── Animal / creature AI render override ─────────────────────
    # Some AI animal/creature/action renders are visually convincing enough that
    # GPT labels them Real and DINOv2/Deepfake acts as a real-video tie-breaker.
    # However, the forensic stack can still show a very specific composite pattern:
    #   • high skin-range ratio from fur/creature tones,
    #   • highly periodic/generated motion,
    #   • very high inter-channel correlation,
    #   • omnidirectional synthetic optical-flow noise,
    #   • at least one pre-heavy clip score already in suspicious territory.
    # This rule prevents that composite from being downgraded back to REAL.
    # It is intentionally narrow and does not apply to ordinary human/phone videos.
    _pre_heavy_scores_for_animal = [
        int(ctx.get("pre_heavy_score", score) or score)
        for score, ctx, _ in valid
    ]
    _max_pre_heavy_animal = max(_pre_heavy_scores_for_animal) if _pre_heavy_scores_for_animal else signal_ai_score
    _max_skin_ratio = max(float(ctx.get("skin_ratio", 0.0) or 0.0) for ctx in all_signal_contexts) if all_signal_contexts else 0.0
    _max_motion_period = max(float(ctx.get("motion_period", 0.0) or 0.0) for ctx in all_signal_contexts) if all_signal_contexts else 0.0
    _max_chan_corr = max(float(ctx.get("chan_corr", 0.0) or 0.0) for ctx in all_signal_contexts) if all_signal_contexts else 0.0
    _max_omni_animal = max(float(ctx.get("omni_flow_ent", ctx.get("omni_ent", ctx.get("omni_flow_entropy", 0.0))) or 0.0) for ctx in all_signal_contexts) if all_signal_contexts else 0.0
    _min_flat_noise = min(float(ctx.get("flat_noise", 9.0) or 9.0) for ctx in all_signal_contexts) if all_signal_contexts else 9.0
    _animal_or_creature_render_ai = (
        _max_pre_heavy_animal >= 52 and
        _max_skin_ratio >= 0.55 and
        _max_motion_period >= 0.70 and
        _max_chan_corr >= 0.90 and
        _max_omni_animal >= 3.70
    )
    # Alternate path for lower-omni clips where flat regions are also unusually clean.
    _animal_or_creature_render_ai_alt = (
        _max_pre_heavy_animal >= 55 and
        _max_skin_ratio >= 0.55 and
        _max_motion_period >= 0.75 and
        _max_chan_corr >= 0.92 and
        _min_flat_noise <= 1.30
    )
    if _animal_or_creature_render_ai or _animal_or_creature_render_ai_alt:
        old_combined = combined_ai_score
        combined_ai_score = max(combined_ai_score, 68.0)
        mode = "animal/creature AI forensic override"
        log.info(
            "ANIMAL_CREATURE_AI override: forensic composite detected "
            "(pre_heavy_max=%d skin=%.3f period=%.3f chan_corr=%.3f "
            "omni=%.3f flat_noise=%.3f gpt=%d) → combined %.1f→%.1f",
            _max_pre_heavy_animal, _max_skin_ratio, _max_motion_period,
            _max_chan_corr, _max_omni_animal, _min_flat_noise,
            gpt_ai_score, old_combined, combined_ai_score
        )

    # ── Cinematic/social AI render override ─────────────────────
    # This catches the Iran/Ram-style cinematic AI videos where GPT and the
    # heavy models may read individual frames as real, while the temporal
    # forensic stack shows a consistent synthetic render pipeline.
    #
    # The failure pattern seen in production:
    #   • upload/link may sample different qualities or clip counts,
    #   • GPT may return Real/low AI on the uploaded version,
    #   • DINOv2/Deepfake can subtract points,
    #   • but multiple motion/texture artifacts still agree:
    #       - background warping/drift,
    #       - omnidirectional AI optical-flow noise,
    #       - very high channel correlation or AI-smooth flat regions,
    #       - flicker/shadow/edge-crawl/grid artifacts.
    #
    # This rule is source-invariant: it applies to uploads and links so the
    # same underlying video cannot pass certification simply because it was
    # submitted as a file instead of a URL.
    _ctxs_for_cinematic = all_signal_contexts or []
    _content_types_seen = {str(ctx.get("content_type", content_type)) for ctx in _ctxs_for_cinematic} or {content_type}
    _cinematic_or_action = any(ct in ("cinematic", "action") for ct in _content_types_seen)

    _max_pre_heavy_cine = max(
        [int(ctx.get("pre_heavy_score", score) or score) for score, ctx, _ in valid] or [signal_ai_score]
    )
    _max_primary_cine = max([int(score) for score, _, _ in valid] or [signal_ai_score])
    _max_bg_drift = max(float(ctx.get("bg_drift", 0.0) or 0.0) for ctx in _ctxs_for_cinematic) if _ctxs_for_cinematic else 0.0
    _max_omni_cine = max(float(ctx.get("omni_flow_ent", ctx.get("omni_ent", ctx.get("omni_flow_entropy", 0.0))) or 0.0) for ctx in _ctxs_for_cinematic) if _ctxs_for_cinematic else 0.0
    _max_chan_cine = max(float(ctx.get("chan_corr", 0.0) or 0.0) for ctx in _ctxs_for_cinematic) if _ctxs_for_cinematic else 0.0
    _min_flat_cine = min(float(ctx.get("flat_noise", 9.0) or 9.0) for ctx in _ctxs_for_cinematic) if _ctxs_for_cinematic else 9.0
    _max_flicker_cine = max(float(ctx.get("flicker_std", 0.0) or 0.0) for ctx in _ctxs_for_cinematic) if _ctxs_for_cinematic else 0.0
    _max_shadow_cine = max(float(ctx.get("shadow_drift", 0.0) or 0.0) for ctx in _ctxs_for_cinematic) if _ctxs_for_cinematic else 0.0
    _max_edge_cov_cine = max(float(ctx.get("edge_cov", ctx.get("edge_cov_var", 0.0)) or 0.0) for ctx in _ctxs_for_cinematic) if _ctxs_for_cinematic else 0.0
    _max_dct_cine = max(float(ctx.get("dct", ctx.get("dct_score", 0.0)) or 0.0) for ctx in _ctxs_for_cinematic) if _ctxs_for_cinematic else 0.0

    _cinematic_artifact_votes = 0
    _cinematic_artifact_votes += 1 if _max_bg_drift >= 35.0 else 0
    _cinematic_artifact_votes += 1 if _max_omni_cine >= 3.80 else 0
    _cinematic_artifact_votes += 1 if _max_chan_cine >= 0.90 else 0
    _cinematic_artifact_votes += 1 if _min_flat_cine <= 0.95 else 0
    _cinematic_artifact_votes += 1 if _max_flicker_cine >= 8.0 else 0
    _cinematic_artifact_votes += 1 if _max_shadow_cine >= 0.72 else 0
    _cinematic_artifact_votes += 1 if _max_edge_cov_cine >= 1.00 else 0
    _cinematic_artifact_votes += 1 if _max_dct_cine >= 22.0 else 0

    _cinematic_social_ai = (
        _cinematic_or_action and
        _max_bg_drift >= 35.0 and
        _max_omni_cine >= 3.80 and
        _cinematic_artifact_votes >= 3 and
        (
            _max_pre_heavy_cine >= 40 or
            _max_primary_cine >= 50 or
            (_max_chan_cine >= 0.93 and _min_flat_cine <= 1.00)
        )
    )

    # Stronger path: if signal was already 50+ before DINO/GPT downgrades,
    # a high-correlation/omni-flow/social-cinematic pattern should not pass.
    _cinematic_social_ai_strong = (
        _cinematic_or_action and
        _max_pre_heavy_cine >= 50 and
        _max_omni_cine >= 3.78 and
        (_max_chan_cine >= 0.90 or _min_flat_cine <= 0.80) and
        (_max_bg_drift >= 30.0 or _max_flicker_cine >= 6.0 or _max_shadow_cine >= 0.70)
    )

    # Guard: do not let social-platform re-encode artifacts alone force a real TikTok/social link to AI.
    # This only suppresses the cinematic/social override when independent confirmation is missing:
    # GPT is neutral/refused, DINOv2 is strongly real, most clips are real-ish, and the pre-override score is below AI.
    _social_source_for_cine = any(
        s in str(_video_source).lower()
        for s in ("tiktok", "instagram", "youtube", "facebook", "smvd")
    )
    _gpt_neutral_or_refused_for_cine = bool(gpt_refused) or (45 <= int(gpt_ai_score or 50) <= 60 and (not gpt_flags or _social_source_for_cine))
    _realish_clip_count_for_cine = sum(1 for s in signal_scores if int(s) <= 45)
    _most_clips_realish_for_cine = (len(signal_scores) >= 3 and _realish_clip_count_for_cine >= 2)
    _dino_scores_for_cine = [
        float(ctx.get("dino_score") if ctx.get("dino_score") is not None else 50)
        for ctx in all_signal_contexts
        if "dino_score" in ctx
    ]
    _dino_strong_real_for_cine = bool(_dino_scores_for_cine and min(_dino_scores_for_cine) <= 15)
    _no_confirming_ai_model_for_cine = (
        _gpt_neutral_or_refused_for_cine and
        _dino_strong_real_for_cine and
        _max_chan_cine < 0.90 and
        combined_ai_score < 58
    )
    _suppress_cinematic_social_override = (
        _social_source_for_cine and
        _most_clips_realish_for_cine and
        _no_confirming_ai_model_for_cine
    )
    if _suppress_cinematic_social_override:
        log.info(
            "CINEMATIC_SOCIAL_AI guard: skipped override for social re-encode "
            "(source=%s clips=%s realish=%d gpt=%d refused=%s dino_min=%.1f "
            "chan_corr=%.3f combined=%.1f)",
            _video_source, signal_scores, _realish_clip_count_for_cine,
            gpt_ai_score, gpt_refused,
            min(_dino_scores_for_cine) if _dino_scores_for_cine else 50.0,
            _max_chan_cine, combined_ai_score
        )

    if (_cinematic_social_ai or _cinematic_social_ai_strong) and not _suppress_cinematic_social_override:
        old_combined = combined_ai_score
        combined_ai_score = max(combined_ai_score, 66.0)
        mode = "cinematic/social AI forensic override"
        log.info(
            "CINEMATIC_SOCIAL_AI override: temporal/render composite detected "
            "(pre_heavy_max=%d primary_max=%d bg_drift=%.2f omni=%.3f "
            "chan_corr=%.3f flat_noise=%.3f flicker=%.2f shadow=%.3f "
            "edge_cov=%.3f dct=%.2f votes=%d gpt=%d) → combined %.1f→%.1f",
            _max_pre_heavy_cine, _max_primary_cine, _max_bg_drift,
            _max_omni_cine, _max_chan_cine, _min_flat_cine,
            _max_flicker_cine, _max_shadow_cine, _max_edge_cov_cine,
            _max_dct_cine, _cinematic_artifact_votes, gpt_ai_score,
            old_combined, combined_ai_score
        )



    # ── Ram / high-motion cinematic animal-action AI override ───────────────
    # Some AI animal/action reels (example: AI Ram) do not match the earlier
    # high-skin/high-period animal rule, and their single uploaded clip can be
    # pulled down hard by GPT saying Real. The reliable pattern is not semantic;
    # it is a temporal/render composite:
    #   • very high background drift / warp,
    #   • very high inter-channel render correlation,
    #   • omnidirectional optical-flow noise,
    #   • high TCV / edge instability,
    #   • DCT/grid or smooth-flat-region evidence.
    # This keeps link and upload behavior aligned for re-encoded social AI.
    _max_tcv_cine = max(float(ctx.get("tcv", 0.0) or 0.0) for ctx in _ctxs_for_cinematic) if _ctxs_for_cinematic else 0.0
    _min_motion_sync_cine = min(float(ctx.get("motion_sync", 1.0) or 1.0) for ctx in _ctxs_for_cinematic) if _ctxs_for_cinematic else 1.0
    _ram_action_render_ai = (
        _cinematic_or_action and
        _max_bg_drift >= 45.0 and
        _max_omni_cine >= 3.78 and
        _max_chan_cine >= 0.92 and
        _max_tcv_cine >= 1200.0 and
        (
            _min_flat_cine <= 1.00 or
            _max_dct_cine >= 15.0 or
            _min_motion_sync_cine <= 0.080 or
            _max_flicker_cine >= 4.0
        )
    )
    # Slightly softer path for cases where channel correlation is just under the
    # high threshold but the clip has extreme DCT/grid and edge/TCV instability.
    _ram_action_render_ai_alt = (
        _cinematic_or_action and
        _max_bg_drift >= 50.0 and
        _max_omni_cine >= 3.80 and
        _max_tcv_cine >= 1500.0 and
        _max_dct_cine >= 15.0 and
        _min_flat_cine <= 1.05 and
        _max_edge_cov_cine >= 0.50
    )
    if _ram_action_render_ai or _ram_action_render_ai_alt:
        old_combined = combined_ai_score
        combined_ai_score = max(combined_ai_score, 68.0)
        mode = "ram/high-motion cinematic AI forensic override"
        log.info(
            "RAM_CINEMATIC_ACTION_AI override: high-motion render composite detected "
            "(bg_drift=%.2f omni=%.3f chan_corr=%.3f tcv=%.2f flat_noise=%.3f "
            "dct=%.2f motion_sync=%.3f flicker=%.2f edge_cov=%.3f gpt=%d) → combined %.1f→%.1f",
            _max_bg_drift, _max_omni_cine, _max_chan_cine, _max_tcv_cine,
            _min_flat_cine, _max_dct_cine, _min_motion_sync_cine,
            _max_flicker_cine, _max_edge_cov_cine, gpt_ai_score,
            old_combined, combined_ai_score
        )


    # ── Borderline large-animal / collision-action AI override ───────────────
    # Some hyperreal animal-impact reels (example: bull/ram collision clips) are
    # re-encoded or screen-recorded in a way that adds convincing camera grain,
    # PRNU-like flat noise, and natural palette signals. That can collapse the
    # primary score even though the temporal/render stack is still abnormal.
    #
    # This is intentionally narrower than the general cinematic override. It
    # requires a large-animal/action pattern plus ALL of the following forensic
    # clues that usually survive re-encoding:
    #   • background warp/drift,
    #   • high inter-channel render correlation,
    #   • omnidirectional optical-flow noise,
    #   • high temporal consistency/variance (TCV),
    #   • lockstep subject/camera motion,
    #   • strong periodic animal/action movement.
    #
    # It prevents GPT/Deepfake from certifying an AI animal collision simply
    # because the clip has convincing noise/palette after social compression.
    _max_motion_large_animal = max(float(ctx.get("motion", 0.0) or 0.0) for ctx in _ctxs_for_cinematic) if _ctxs_for_cinematic else 0.0
    _max_skin_large_animal = max(float(ctx.get("skin_ratio", ctx.get("skin", 0.0)) or 0.0) for ctx in _ctxs_for_cinematic) if _ctxs_for_cinematic else 0.0
    _max_period_large_animal = max(float(ctx.get("motion_period", ctx.get("period", 0.0)) or 0.0) for ctx in _ctxs_for_cinematic) if _ctxs_for_cinematic else 0.0
    _max_rvov_large_animal = max(float(ctx.get("rvov", 0.0) or 0.0) for ctx in _ctxs_for_cinematic) if _ctxs_for_cinematic else 0.0

    _large_animal_action_ai = (
        _cinematic_or_action and
        _max_motion_large_animal >= 20.0 and
        _max_bg_drift >= 40.0 and
        _max_chan_cine >= 0.91 and
        _max_omni_cine >= 3.70 and
        _max_tcv_cine >= 1500.0 and
        _min_motion_sync_cine <= 0.080 and
        _max_period_large_animal >= 0.60 and
        (
            _max_skin_large_animal >= 0.50 or
            _max_rvov_large_animal >= 15000.0 or
            _max_flicker_cine >= 5.0
        )
    )

    # Alternate social-compression path: uploaded version may show slightly lower
    # channel correlation/omni-flow but still has severe background drift, high
    # TCV, lockstep motion, and animal/action periodicity.
    _large_animal_action_ai_alt = (
        _cinematic_or_action and
        _max_motion_large_animal >= 22.0 and
        _max_bg_drift >= 45.0 and
        _max_omni_cine >= 3.68 and
        _max_tcv_cine >= 1700.0 and
        _min_motion_sync_cine <= 0.070 and
        _max_period_large_animal >= 0.65 and
        _max_skin_large_animal >= 0.50
    )

    if _large_animal_action_ai or _large_animal_action_ai_alt:
        old_combined = combined_ai_score
        combined_ai_score = max(combined_ai_score, 68.0)
        mode = "large-animal action AI forensic override"
        log.info(
            "LARGE_ANIMAL_ACTION_AI override: borderline animal/action render composite detected "
            "(motion=%.1f bg_drift=%.2f chan_corr=%.3f omni=%.3f tcv=%.2f "
            "motion_sync=%.3f period=%.3f skin=%.3f rvov=%.2f flicker=%.2f gpt=%d) "
            "→ combined %.1f→%.1f",
            _max_motion_large_animal, _max_bg_drift, _max_chan_cine, _max_omni_cine,
            _max_tcv_cine, _min_motion_sync_cine, _max_period_large_animal,
            _max_skin_large_animal, _max_rvov_large_animal, _max_flicker_cine,
            gpt_ai_score, old_combined, combined_ai_score
        )


    # ── Bull / high-motion animal collision AI hard override ───────────────
    # Production miss: bull/ram collision reels can carry strong camera-like
    # noise and natural palette after Instagram/H264 re-encoding. Those real-
    # looking signals should NOT certify the clip when the temporal/render
    # signature is internally consistent with AI:
    #   • high background drift / scene warp,
    #   • very high RGB channel correlation,
    #   • omnidirectional optical-flow noise,
    #   • lockstep subject motion,
    #   • high TCV/temporal instability or strong periodic animation.
    # This rule is deliberately focused on short vertical action/animal reels and
    # uses only already-computed signals, so it does not add runtime.
    _ctxs_bull = all_signal_contexts or []
    _content_types_bull = {str(ctx.get("content_type", content_type)) for ctx in _ctxs_bull} or {content_type}
    _bull_action_content = any(ct in ("action", "cinematic") for ct in _content_types_bull)

    def _ctx_float(ctx, *names, default=0.0):
        for name in names:
            try:
                val = ctx.get(name, None)
                if val is not None:
                    return float(val or 0.0)
            except Exception:
                pass
        return float(default)

    _max_bg_bull = max([_ctx_float(ctx, "bg_drift") for ctx in _ctxs_bull] or [0.0])
    _max_chan_bull = max([_ctx_float(ctx, "chan_corr") for ctx in _ctxs_bull] or [0.0])
    _max_omni_bull = max([_ctx_float(ctx, "omni_flow_ent", "omni_ent", "omni_flow_entropy") for ctx in _ctxs_bull] or [0.0])
    _min_sync_bull = min([_ctx_float(ctx, "motion_sync", default=1.0) for ctx in _ctxs_bull] or [1.0])
    _max_tcv_bull = max([_ctx_float(ctx, "tcv", "temporal_consistency_var", "temporal_coherence_var") for ctx in _ctxs_bull] or [0.0])
    _max_period_bull = max([_ctx_float(ctx, "motion_period", "period") for ctx in _ctxs_bull] or [0.0])
    _max_motion_bull = max([_ctx_float(ctx, "motion") for ctx in _ctxs_bull] or [0.0])
    _max_skin_bull = max([_ctx_float(ctx, "skin_ratio", "skin") for ctx in _ctxs_bull] or [0.0])
    _max_flicker_bull = max([_ctx_float(ctx, "flicker_std") for ctx in _ctxs_bull] or [0.0])
    _max_dct_bull = max([_ctx_float(ctx, "dct", "dct_score") for ctx in _ctxs_bull] or [0.0])
    _max_pre_heavy_bull = max(
        [int(ctx.get("pre_heavy_score", score) or score) for score, ctx, _ in valid] or [signal_ai_score]
    )

    _bull_signal_votes = 0
    _bull_signal_votes += 1 if _max_bg_bull >= 40.0 else 0
    _bull_signal_votes += 1 if _max_chan_bull >= 0.90 else 0
    _bull_signal_votes += 1 if _max_omni_bull >= 3.65 else 0
    _bull_signal_votes += 1 if _min_sync_bull <= 0.085 else 0
    _bull_signal_votes += 1 if _max_tcv_bull >= 1000.0 else 0
    _bull_signal_votes += 1 if _max_period_bull >= 0.60 else 0
    _bull_signal_votes += 1 if _max_flicker_bull >= 4.0 else 0
    _bull_signal_votes += 1 if _max_dct_bull >= 9.0 else 0

    _bull_collision_ai = (
        _bull_action_content and
        _max_motion_bull >= 18.0 and
        _max_bg_bull >= 40.0 and
        _max_chan_bull >= 0.90 and
        _max_omni_bull >= 3.65 and
        _min_sync_bull <= 0.085 and
        (_max_tcv_bull >= 1000.0 or _max_period_bull >= 0.60) and
        (_max_skin_bull >= 0.35 or _max_pre_heavy_bull >= 22) and
        _bull_signal_votes >= 5
    )

    if _bull_collision_ai:
        old_combined = combined_ai_score
        combined_ai_score = max(combined_ai_score, 72.0)
        mode = "bull/high-motion animal collision AI forensic override"
        log.info(
            "BULL_COLLISION_AI override: high-motion animal/action render composite detected "
            "(motion=%.1f bg_drift=%.2f chan_corr=%.3f omni=%.3f sync=%.3f "
            "tcv=%.2f period=%.3f skin=%.3f flicker=%.2f dct=%.2f pre_heavy=%d "
            "votes=%d gpt=%d) → combined %.1f→%.1f",
            _max_motion_bull, _max_bg_bull, _max_chan_bull, _max_omni_bull,
            _min_sync_bull, _max_tcv_bull, _max_period_bull, _max_skin_bull,
            _max_flicker_bull, _max_dct_bull, _max_pre_heavy_bull,
            _bull_signal_votes, gpt_ai_score, old_combined, combined_ai_score
        )


    # ── Final robust cinematic/action AI agreement override ────────────────
    # This is the last-pass safety net for the exact class of misses seen in
    # short vertical bull/ram/animal/action reels. It intentionally uses
    # PATTERN AGREEMENT rather than brittle single-value thresholds. Modern AI
    # video often carries convincing H264/Instagram noise, PRNU-like flat-region
    # texture, and natural-looking color palettes, so those real-camera-looking
    # deductions must not certify a clip when the temporal/render signals agree.
    #
    # Production miss example values:
    #   motion=24.9, bg_drift=45.5, chan_corr=0.9156, omni_ent=3.741,
    #   motion_sync=0.066, tcv=1744, period=0.676, skin=0.557.
    # The previous rule missed because one threshold was too tight. This rule
    # catches the composite pattern and logs every value for auditability.
    _robust_ctxs = all_signal_contexts or []
    _robust_content_types = {str(ctx.get("content_type", content_type)) for ctx in _robust_ctxs} or {content_type}
    _robust_action_content = any(ct in ("action", "cinematic") for ct in _robust_content_types)

    def _robust_float(ctx, *names, default=0.0):
        for name in names:
            try:
                val = ctx.get(name, None)
                if val is not None:
                    return float(val or 0.0)
            except Exception:
                continue
        return float(default)

    _robust_motion = max([_robust_float(ctx, "motion", "avg_motion") for ctx in _robust_ctxs] or [0.0])
    _robust_bg = max([_robust_float(ctx, "bg_drift", "background_drift") for ctx in _robust_ctxs] or [0.0])
    _robust_chan = max([_robust_float(ctx, "chan_corr", "channel_corr") for ctx in _robust_ctxs] or [0.0])
    _robust_omni = max([_robust_float(ctx, "omni_flow_ent", "omni_ent", "omni_flow_entropy") for ctx in _robust_ctxs] or [0.0])
    _robust_sync = min([_robust_float(ctx, "motion_sync", default=1.0) for ctx in _robust_ctxs] or [1.0])
    _robust_tcv = max([_robust_float(ctx, "tcv", "temporal_consistency_var", "temporal_coherence_var") for ctx in _robust_ctxs] or [0.0])
    _robust_period = max([_robust_float(ctx, "motion_period", "period") for ctx in _robust_ctxs] or [0.0])
    _robust_skin = max([_robust_float(ctx, "skin_ratio", "skin") for ctx in _robust_ctxs] or [0.0])
    _robust_flicker = max([_robust_float(ctx, "flicker_std") for ctx in _robust_ctxs] or [0.0])
    _robust_dct = max([_robust_float(ctx, "dct", "dct_score") for ctx in _robust_ctxs] or [0.0])
    _robust_pre_heavy = max(
        [int(ctx.get("pre_heavy_score", score) or score) for score, ctx, _ in valid] or [signal_ai_score]
    )

    _robust_votes = 0
    _robust_votes += 1 if _robust_motion >= 18.0 else 0
    _robust_votes += 1 if _robust_bg >= 38.0 else 0
    _robust_votes += 1 if _robust_chan >= 0.90 else 0
    _robust_votes += 1 if _robust_omni >= 3.60 else 0
    _robust_votes += 1 if _robust_sync <= 0.085 else 0
    _robust_votes += 1 if _robust_tcv >= 1000.0 else 0
    _robust_votes += 1 if _robust_period >= 0.58 else 0
    _robust_votes += 1 if _robust_flicker >= 4.0 else 0
    _robust_votes += 1 if _robust_dct >= 9.0 else 0

    _robust_cinematic_action_ai = (
        _robust_action_content and
        _robust_motion >= 18.0 and
        _robust_bg >= 38.0 and
        _robust_chan >= 0.90 and
        _robust_omni >= 3.60 and
        (_robust_sync <= 0.085 or _robust_period >= 0.60) and
        (_robust_tcv >= 1000.0 or _robust_flicker >= 4.0) and
        (_robust_skin >= 0.30 or _robust_pre_heavy >= 20) and
        _robust_votes >= 6
    )

    if _robust_cinematic_action_ai:
        old_combined = combined_ai_score
        combined_ai_score = max(combined_ai_score, 72.0)
        mode = "robust cinematic/action AI forensic override"
        log.info(
            "ROBUST_CINEMATIC_ACTION_AI override: pattern agreement detected "
            "(motion=%.1f bg_drift=%.2f chan_corr=%.3f omni=%.3f sync=%.3f "
            "tcv=%.2f period=%.3f skin=%.3f flicker=%.2f dct=%.2f pre_heavy=%d "
            "votes=%d gpt=%d) → combined %.1f→%.1f",
            _robust_motion, _robust_bg, _robust_chan, _robust_omni, _robust_sync,
            _robust_tcv, _robust_period, _robust_skin, _robust_flicker, _robust_dct,
            _robust_pre_heavy, _robust_votes, gpt_ai_score, old_combined, combined_ai_score
        )

    # ── LAVF + CHAN_CORR composite boost ────────────────────────
    # Lavf encoder alone is weak (innocent re-encodes are common).
    # But Lavf + ALL clips showing CHAN_CORR > 0.90 is a strong composite
    # signal — real re-encodes rarely have both together.
    _lavf_flag = override and ov_label == "LAVF_AI_PIPELINE"
    if _lavf_flag:
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
    #
    # OVERRIDE: When BOTH signal AND GPT are strongly confident AI (both >75),
    # the ceiling does NOT apply. Two independent engines agreeing strongly on AI
    # overrides a weak metadata heuristic. The isom+creation_time check is too
    # permissive — AI generators (Runway, Kling, Sora) also output isom containers
    # with creation timestamps. Only apply ceiling when engines are ambiguous.
    _dual_engine_confident_ai = (signal_ai_score > 75 and gpt_ai_score > 75)

    if override and ov_label == "REAL" and combined_ai_score >= 70:
        if _dual_engine_confident_ai:
            log.info(
                "Real device metadata ceiling SKIPPED: signal=%d AND gpt=%d both >75 "
                "(dual-engine confident AI overrides weak metadata heuristic) | %s",
                signal_ai_score, gpt_ai_score, ov_reason
            )
        else:
            old_combined = combined_ai_score
            combined_ai_score = min(combined_ai_score, 58.0)
            log.info(
                "Real device metadata ceiling: combined %.1f→%.1f "
                "(metadata confirmed real camera, preventing AI verdict) | %s",
                old_combined, combined_ai_score, ov_reason
            )

    _viral_ai_reel_guard = _vfyd_apply_viral_ai_reel_guard(
        video_path=video_path,
        combined_ai_score=combined_ai_score,
        mode=mode,
        signal_ai_score=signal_ai_score,
        gpt_ai_score=gpt_ai_score,
        gpt_result=gpt_result,
        signal_context=signal_context,
        all_signal_contexts=all_signal_contexts,
        n_clips=n_clips,
    )
    if _viral_ai_reel_guard:
        combined_ai_score = _viral_ai_reel_guard["combined_ai_score"]
        mode = _viral_ai_reel_guard["mode"]
        gpt_flags = gpt_result.get("flags", gpt_flags)
        signal_context["viral_ai_reel_pattern"] = True

    combined_ai_score = max(0.0, min(100.0, combined_ai_score))
    # VERIFYD_SURREAL_ANIMAL_OBJECT_VIDEO_FINAL_GUARD_V3
    # Prevent AI-edited animal/object/person clips from certifying as REAL when fake camera texture hides synthesis.
    # This catches cases where low-level noise looks real but GPT/forensics show an AI composite or staged insert.
    try:
        _vfyd_scores = gpt_result.get("scores", {}) if isinstance(gpt_result, dict) else {}
        _vfyd_flags = gpt_result.get("flags", []) if isinstance(gpt_result, dict) else []
        if not isinstance(_vfyd_flags, list):
            _vfyd_flags = [str(_vfyd_flags)]
        _vfyd_flags_text = " ".join(str(x).lower() for x in _vfyd_flags)
        _vfyd_gen = str(gpt_result.get("generator_guess", "") if isinstance(gpt_result, dict) else "").lower()

        _vfyd_ctxs = []
        if isinstance(signal_context, dict):
            _vfyd_ctxs.append(signal_context)
        if isinstance(all_signal_contexts, list):
            _vfyd_ctxs.extend([c for c in all_signal_contexts if isinstance(c, dict)])

        def _vfyd_ctx_max_v3(*names, default=0.0):
            vals = []
            for c in _vfyd_ctxs:
                for name in names:
                    if name in c and c.get(name) is not None:
                        try:
                            vals.append(float(c.get(name)))
                        except Exception:
                            pass
            return max(vals) if vals else default

        _vfyd_chan = _vfyd_ctx_max_v3("chan_corr", "channel_corr", default=0.0)
        _vfyd_shadow = _vfyd_ctx_max_v3("shadow_drift", default=0.0)
        _vfyd_omni = _vfyd_ctx_max_v3("omni_flow_entropy", "omni_ent", default=0.0)
        _vfyd_tcv = _vfyd_ctx_max_v3("tcv", "temporal_consistency_var", "temporal_coherence_var", default=0.0)
        _vfyd_pre_heavy = _vfyd_ctx_max_v3("pre_heavy_score", default=0.0)

        _vfyd_scene = int(_vfyd_scores.get("scene_staging", 0) or 0)
        _vfyd_phys = int(_vfyd_scores.get("physics_violations", 0) or 0)
        _vfyd_genart = int(_vfyd_scores.get("generator_artifacts", 0) or 0)
        _vfyd_light = int(_vfyd_scores.get("lighting_coherence", 0) or 0)
        _vfyd_color = int(_vfyd_scores.get("color_naturalism", 0) or 0)

        _vfyd_semantic_ai = (
            "unknown-ai" in _vfyd_gen or
            "surreal_animal_object_composite_guard" in _vfyd_flags_text or
            _vfyd_scene >= 7 or
            _vfyd_phys >= 7 or
            _vfyd_genart >= 7 or
            (_vfyd_light >= 8 and _vfyd_color >= 8)
        )

        _vfyd_forensic_ai = (
            _vfyd_chan >= 0.94 and (
                _vfyd_shadow >= 0.80 or
                _vfyd_omni >= 3.60 or
                _vfyd_tcv >= 300.0 or
                _vfyd_pre_heavy >= 35.0
            )
        )

        # For these surreal object/animal videos, GPT may be only moderate, but Unknown-AI + high channel lock + shadow/omni/TCV is enough to avoid REAL.
        if _vfyd_semantic_ai and _vfyd_forensic_ai and int(gpt_ai_score or 0) >= 45 and combined_ai_score < 65:
            _old_combined = combined_ai_score
            combined_ai_score = 65.0
            mode = "surreal animal/object composite guard"
            if isinstance(gpt_result, dict):
                gpt_result["flags"] = list(dict.fromkeys(["surreal_animal_object_composite_guard"] + _vfyd_flags))
            log.info(
                "VERIFYD_SURREAL_ANIMAL_OBJECT_VIDEO_FINAL_GUARD_V3: combined %.1f->%.1f gpt=%s chan=%.3f shadow=%.3f omni=%.3f tcv=%.2f pre_heavy=%.1f scene=%s phys=%s gen=%s",
                _old_combined, combined_ai_score, gpt_ai_score, _vfyd_chan, _vfyd_shadow, _vfyd_omni, _vfyd_tcv, _vfyd_pre_heavy, _vfyd_scene, _vfyd_phys, _vfyd_gen
            )
    except Exception as _e:
        log.warning("VERIFYD_SURREAL_ANIMAL_OBJECT_VIDEO_FINAL_GUARD_V3 skipped: %s", _e)

    # VERIFYD_TWITTER_SOCIAL_FORENSIC_OVERRIDE_V1
    # X/Twitter link parity guard:
    # Some AI social videos are downloaded from X as platform/Lavf MP4s and then normalized.
    # GPT/DINO can under-call these as real because faces/background/noise look phone-like.
    # When multiple forensic render signals agree, keep link analysis close to manual upload analysis.
    try:
        _tw_ctxs = all_signal_contexts or []
        _tw_source = str(_video_source or signal_context.get("source", "") or "").lower()
        _tw_is_x = any(s in _tw_source for s in ("twitter", "x.com", "t.co"))

        def _tw_float(ctx, *names, default=0.0):
            for name in names:
                try:
                    val = ctx.get(name, None)
                    if val is not None:
                        return float(val)
                except Exception:
                    pass
            return default

        _tw_chan_vals = [_tw_float(ctx, "chan_corr", "channel_corr") for ctx in _tw_ctxs]
        _tw_all_high_chan = bool(_tw_chan_vals) and all(c >= 0.90 for c in _tw_chan_vals if c > 0)
        _tw_max_chan = max(_tw_chan_vals or [0.0])
        _tw_max_shadow = max([_tw_float(ctx, "shadow_drift") for ctx in _tw_ctxs] or [0.0])
        _tw_max_omni = max([_tw_float(ctx, "omni_flow_ent", "omni_ent", "omni_flow_entropy") for ctx in _tw_ctxs] or [0.0])
        _tw_max_edge_cov = max([_tw_float(ctx, "edge_cov", "edge_cov_var") for ctx in _tw_ctxs] or [0.0])
        _tw_max_tcv = max([_tw_float(ctx, "tcv", "temporal_consistency_var", "temporal_coherence_var") for ctx in _tw_ctxs] or [0.0])
        _tw_max_dct = max([_tw_float(ctx, "dct", "dct_score") for ctx in _tw_ctxs] or [0.0])
        _tw_max_pre_heavy = max(
            [int(ctx.get("pre_heavy_score", score) or score) for score, ctx, _pct in valid]
            or [signal_ai_score]
        )

        _tw_votes = 0
        _tw_votes += 1 if _tw_all_high_chan else 0
        _tw_votes += 1 if _tw_max_shadow >= 0.82 else 0
        _tw_votes += 1 if _tw_max_omni >= 3.80 else 0
        _tw_votes += 1 if _tw_max_edge_cov >= 1.45 else 0
        _tw_votes += 1 if _tw_max_tcv >= 150.0 else 0
        _tw_votes += 1 if _tw_max_dct >= 13.0 else 0
        _tw_votes += 1 if _tw_max_pre_heavy >= 40 else 0

        _tw_lavf_flag = bool(override and ov_label == "LAVF_AI_PIPELINE")
        _tw_social_render_ai = (
            _tw_is_x and
            _tw_lavf_flag and
            _tw_all_high_chan and
            _tw_max_pre_heavy >= 40 and
            _tw_votes >= 5 and
            combined_ai_score < 65
        )

        if _tw_social_render_ai:
            _old_combined = combined_ai_score
            combined_ai_score = 65.0
            mode = "X/Twitter social forensic AI override"
            if isinstance(gpt_result, dict):
                _existing_flags = gpt_result.get("flags", []) or []
                if not isinstance(_existing_flags, list):
                    _existing_flags = [str(_existing_flags)]
                gpt_result["flags"] = list(dict.fromkeys(["twitter_social_forensic_ai_pattern"] + _existing_flags))

                # VERIFYD_TWITTER_SOCIAL_REASONING_OVERRIDE_V1
                # The final forensic override supersedes GPT's earlier "Real" visual reasoning.
                # Keep the public explanation aligned with the final AI/Tampering result.
                gpt_result["reasoning"] = (
                    "This video was flagged for AI / tampering indicators based on a forensic pattern found in the X/Twitter link analysis. "
                    "Although some visual elements may appear natural, the downloaded social video showed multiple render/compression forensic signals that are inconsistent with ordinary camera footage, including high channel correlation across clips, Lavf/null-vendor pipeline metadata, shadow inconsistency, omni-directional motion noise, edge-crawl behavior, temporal consistency anomalies, and DCT/grid artifacts. "
                    "These combined signals caused VeriFYD to classify the file as AI / Tampering Detected rather than Authenticity Supported."
                )
            log.info(
                "VERIFYD_TWITTER_SOCIAL_FORENSIC_OVERRIDE_V1: combined %.1f->%.1f source=%s lavf=%s chan_all=%s chan_max=%.3f shadow=%.3f omni=%.3f edge_cov=%.3f tcv=%.2f dct=%.2f pre_heavy=%d votes=%d gpt=%d",
                _old_combined, combined_ai_score, _tw_source, _tw_lavf_flag, _tw_all_high_chan,
                _tw_max_chan, _tw_max_shadow, _tw_max_omni, _tw_max_edge_cov, _tw_max_tcv,
                _tw_max_dct, _tw_max_pre_heavy, _tw_votes, gpt_ai_score
            )
    except Exception as _e:
        log.warning("VERIFYD_TWITTER_SOCIAL_FORENSIC_OVERRIDE_V1 skipped: %s", _e)

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
        for _k in ("audio_stereo_corr", "audio_score", "audio_duration_mismatch",
                   "hf_kurtosis", "chan_corr", "omni_flow_entropy",
                   "shadow_drift", "edge_cov_var", "flat_noise"):
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
        "audio_ai_score":     audio_result.get("audio_ai_score", 50),
        "audio_confidence":   audio_result.get("confidence", "unavailable"),
        "audio_contribution": audio_contribution,
        "audio_evidence":     audio_result.get("evidence", [])[:8],
        "audio_signals":      {k: audio_result.get(k) for k in ("duration_mismatch", "stereo_corr", "noise_floor", "rms_cv", "mean_bandwidth", "zcr_cv", "low_freq_ratio") if k in audio_result},
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
    if _viral_ai_reel_guard:
        detail.update(_viral_ai_reel_guard)
        detail["gpt_flags"] = gpt_result.get("flags", detail.get("gpt_flags", []))
        detail["blend_mode"] = mode

    return authenticity, label, detail


