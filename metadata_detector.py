# ============================================================
#  VeriFYD — metadata_detector.py
#
#  Forensic metadata analysis for AI video detection.
#  Uses ffprobe (already in stack) to extract container-level
#  signals that AI generators consistently fail to fake.
#
#  WHY THIS WORKS:
#  Real cameras write rich, consistent metadata: camera make/model,
#  GPS, lens info, creation timestamps, standard encoder strings.
#  AI video generators typically output bare container files with
#  minimal or suspicious metadata — or they copy metadata templates
#  that have telltale inconsistencies (e.g. creation date in 1970,
#  encoder string = "Lavf" with no camera make, suspicious codec
#  params that no camera ships with).
#
#  SIGNALS:
#    - Camera make/model presence
#    - Encoder string vs known AI encoder fingerprints
#    - Creation timestamp plausibility
#    - Container format vs claimed camera source
#    - Codec parameter consistency with real hardware
#    - Known AI output container signatures
#
#  Returns:
#    metadata_ai_score  : 0–100 (higher = more likely AI)
#    evidence           : list of string findings
#    metadata_dict      : raw extracted fields for logging
# ============================================================

import os
import json
import subprocess
import logging
from typing import Optional

log = logging.getLogger("verifyd.metadata")

FFPROBE_BIN = os.environ.get("FFPROBE_BIN", "ffprobe")

# ─────────────────────────────────────────────────────────────
#  Known AI encoder / writer strings
#  These appear in the 'encoder', 'handler_name', or 'comment'
#  tags of AI-generated video files.
# ─────────────────────────────────────────────────────────────
_AI_ENCODER_STRINGS = [
    "lavf",           # FFmpeg Lavformat — common in AI pipeline outputs
    "lavc",           # FFmpeg Lavcodec
    "ffmpeg",         # Explicit FFmpeg tag (AI tools almost always use FFmpeg)
    "libx264",        # Raw libx264 without camera wrapper
    "libx265",        # Raw libx265 without camera wrapper
    "openh264",       # Browser/web encoder — not a camera
    "sora",           # OpenAI Sora
    "runway",         # Runway ML
    "kling",          # Kuaishou Kling
    "pika",           # Pika Labs
    "midjourney",     # Midjourney video
    "stability",      # Stability AI
    "lumiere",        # Google Lumiere
    "gen-2",          # Runway Gen-2
    "animatediff",    # AnimateDiff
    "zeroscope",      # ZeroScope
    "modelscope",     # ModelScope
    "cogvideo",       # CogVideo
    "videocrafter",   # VideoCrafter
]

# Real camera encoder strings / handler patterns
_REAL_CAMERA_STRINGS = [
    "apple",          # iPhone QuickTime
    "gopro",          # GoPro
    "sony",           # Sony cameras
    "canon",          # Canon cameras
    "nikon",          # Nikon cameras
    "panasonic",      # Panasonic
    "olympus",        # Olympus
    "fujifilm",       # Fujifilm
    "dji",            # DJI drones
    "samsung",        # Samsung
    "android",        # Android camera
    "qualcomm",       # Snapdragon camera pipeline
    "mediatek",       # MediaTek camera pipeline
    "hevc main",      # Standard HEVC profile from camera
    "avc coding",     # Standard AVC from camera
    "video handler",  # Standard QuickTime video handler
    "sound handler",  # Standard QuickTime audio handler
    "core media",     # Apple Core Media (iPhone)
    "atomos",         # Atomos recorder
    "blackmagic",     # Blackmagic Design
]

# Container format → expected camera source mapping
# AI tools almost always output MP4 or WebM; never output MOV from a camera brand
_AI_SUSPICIOUS_FORMATS = {
    "matroska,webm": "WebM/MKV — AI tools commonly output this format",
}

# Codec profiles that no real camera ships with
_AI_CODEC_TELLS = [
    ("codec_name", "vp8"),     # VP8 — browser/AI only, no cameras
    ("codec_name", "vp9"),     # VP9 — browser/AI only
    ("codec_name", "av1"),     # AV1 — AI tools, not cameras yet
    ("profile", "baseline"),   # H.264 Baseline — streaming/AI, not camera
    ("profile", "constrained baseline"),
]


# ─────────────────────────────────────────────────────────────
#  Core analysis
# ─────────────────────────────────────────────────────────────
def analyze_metadata(video_path: str) -> dict:
    """
    Run ffprobe on video_path and score the metadata forensically.

    Returns:
        metadata_ai_score  : int 0–100
        evidence           : list of str findings (for logging/display)
        confidence         : "high" | "medium" | "low" (how reliable this score is)
        metadata_dict      : raw fields for storage
        available          : bool
    """
    probe = _run_ffprobe(video_path)
    if probe is None:
        return {
            "metadata_ai_score": 50,
            "evidence": ["ffprobe failed — metadata unavailable"],
            "confidence": "low",
            "metadata_dict": {},
            "available": False,
        }

    score    = 0
    evidence = []
    raw      = _extract_fields(probe)

    # ── 1. Encoder / writer string ──────────────────────────
    encoder_str = " ".join(filter(None, [
        raw.get("encoder", ""),
        raw.get("handler_name_video", ""),
        raw.get("handler_name_audio", ""),
        raw.get("comment", ""),
        raw.get("software", ""),
        raw.get("major_brand", ""),
        raw.get("compatible_brands", ""),
    ])).lower()

    found_ai_encoder    = False
    found_real_encoder  = False

    for s in _AI_ENCODER_STRINGS:
        if s in encoder_str:
            found_ai_encoder = True
            evidence.append(f"AI encoder detected: '{s}' in metadata strings")
            score += 20
            break  # don't double-count

    for s in _REAL_CAMERA_STRINGS:
        if s in encoder_str:
            found_real_encoder = True
            evidence.append(f"Real camera/device string detected: '{s}'")
            score -= 15
            break

    if not found_ai_encoder and not found_real_encoder and encoder_str.strip():
        evidence.append(f"Encoder string present but unrecognized: '{encoder_str[:80]}'")
    elif not encoder_str.strip():
        evidence.append("No encoder/handler metadata present (unusual for real cameras)")
        score += 10

    # ── 2. Camera make/model ─────────────────────────────────
    make  = raw.get("com.apple.quicktime.make", "") or raw.get("make", "")
    model = raw.get("com.apple.quicktime.model", "") or raw.get("model", "")

    if make and model:
        evidence.append(f"Camera identified: {make} {model}")
        score -= 20
    elif make or model:
        evidence.append(f"Partial camera metadata: make='{make}' model='{model}'")
        score -= 10
    else:
        evidence.append("No camera make/model in metadata (AI generators omit this)")
        score += 15

    # ── 3. Creation timestamp plausibility ───────────────────
    creation_time = raw.get("creation_time", "")
    if creation_time:
        try:
            from datetime import datetime, timezone
            # Parse ISO8601 (ffprobe outputs e.g. "2024-03-15T10:22:00.000000Z")
            ts_str = creation_time.replace("Z", "+00:00")
            ts = datetime.fromisoformat(ts_str)
            year = ts.year

            if year < 2000:
                evidence.append(f"Suspicious creation timestamp: {creation_time} (year {year} — AI pipeline default)")
                score += 20
            elif year > 2030:
                evidence.append(f"Impossible future timestamp: {creation_time}")
                score += 15
            else:
                evidence.append(f"Plausible creation timestamp: {creation_time[:10]}")
                score -= 5
        except Exception:
            evidence.append(f"Unparseable creation timestamp: {creation_time[:30]}")
            score += 5
    else:
        evidence.append("No creation timestamp in metadata")
        score += 8

    # ── 4. Container format ──────────────────────────────────
    fmt = raw.get("format_name", "")
    for suspicious_fmt, reason in _AI_SUSPICIOUS_FORMATS.items():
        if suspicious_fmt in fmt.lower():
            evidence.append(f"Container format: {reason}")
            score += 10
            break

    # MOV with no Apple metadata = suspicious
    if "mov" in fmt.lower() and not make and not model:
        evidence.append("MOV container but no Apple camera metadata (possible AI output)")
        score += 8

    # ── 5. Codec profile tells ───────────────────────────────
    for field, val in _AI_CODEC_TELLS:
        if raw.get(field, "").lower() == val.lower():
            evidence.append(f"AI-associated codec: {field}={val}")
            score += 15
            break

    # ── 6. Audio track presence ──────────────────────────────
    # Real phone/camera videos almost always have audio.
    # AI generators often output video-only.
    has_audio = raw.get("has_audio", False)
    if not has_audio:
        evidence.append("No audio track (AI generators often output video-only)")
        score += 12
    else:
        evidence.append("Audio track present")
        score -= 3

    # ── 7. GPS / location data ────────────────────────────────
    # Phones often embed GPS. AI generators never do.
    location = raw.get("location", "") or raw.get("com.apple.quicktime.location.ISO6709", "")
    if location:
        evidence.append(f"GPS/location data present: {location[:30]}")
        score -= 10

    # ── Clamp final score ─────────────────────────────────────
    score = max(0, min(100, 50 + score))  # start from 50, adjust up/down

    # Confidence is higher when we found strong signals
    strong_signals = sum([
        found_ai_encoder, found_real_encoder,
        bool(make and model),
        bool(location),
    ])
    confidence = "high" if strong_signals >= 2 else "medium" if strong_signals == 1 else "low"

    log.info("metadata_detector: score=%d  confidence=%s  evidence_count=%d",
             score, confidence, len(evidence))

    return {
        "metadata_ai_score": score,
        "evidence":          evidence,
        "confidence":        confidence,
        "metadata_dict":     raw,
        "available":         True,
    }


# ─────────────────────────────────────────────────────────────
#  ffprobe helpers
# ─────────────────────────────────────────────────────────────
def _run_ffprobe(video_path: str) -> Optional[dict]:
    """Run ffprobe and return parsed JSON, or None on failure."""
    cmd = [
        FFPROBE_BIN,
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        video_path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=30)
        if result.returncode != 0:
            log.warning("ffprobe returned %d for %s", result.returncode, video_path)
            return None
        return json.loads(result.stdout.decode("utf-8", errors="replace"))
    except Exception as e:
        log.error("ffprobe error: %s", e)
        return None


def _extract_fields(probe: dict) -> dict:
    """Flatten ffprobe JSON into a single-level dict of relevant fields."""
    raw = {}

    fmt = probe.get("format", {})
    raw["format_name"]  = fmt.get("format_name", "")
    raw["duration"]     = fmt.get("duration", "")
    raw["size"]         = fmt.get("size", "")

    tags = fmt.get("tags", {})
    # Normalize tag keys to lowercase
    tags_lower = {k.lower(): v for k, v in tags.items()}
    for key in [
        "encoder", "software", "comment", "major_brand", "compatible_brands",
        "creation_time", "location", "make", "model",
        "com.apple.quicktime.make", "com.apple.quicktime.model",
        "com.apple.quicktime.location.iso6709",
    ]:
        raw[key] = tags_lower.get(key, "")

    has_audio = False
    for stream in probe.get("streams", []):
        codec_type = stream.get("codec_type", "")
        stags = {k.lower(): v for k, v in stream.get("tags", {}).items()}

        if codec_type == "video":
            raw["codec_name"]    = stream.get("codec_name", "")
            raw["profile"]       = stream.get("profile", "")
            raw["width"]         = stream.get("width", 0)
            raw["height"]        = stream.get("height", 0)
            raw["r_frame_rate"]  = stream.get("r_frame_rate", "")
            raw["pix_fmt"]       = stream.get("pix_fmt", "")
            raw["handler_name_video"] = stags.get("handler_name", "")
            # Some cameras write creation_time per-stream too
            if not raw.get("creation_time"):
                raw["creation_time"] = stags.get("creation_time", "")

        elif codec_type == "audio":
            has_audio = True
            raw["audio_codec"]       = stream.get("codec_name", "")
            raw["handler_name_audio"] = stags.get("handler_name", "")

    raw["has_audio"] = has_audio
    return raw
