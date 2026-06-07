# ============================================================
#  VeriFYD — audio_detector.py  v2
#
#  Audio-domain AI / synthetic-media detector for VeriFYD video and standalone audio files.
#
#  Design goals:
#    - Additive only: does not replace the existing pixel/GPT engines.
#    - Conservative contribution: audio can lift or lower the visual score,
#      but cannot independently certify or condemn normal silent clips.
#    - Full-file analysis: run once per video, not once per extracted clip.
#    - Graceful degradation: if ffmpeg/librosa/scipy are unavailable, return
#      a neutral unavailable result without breaking the worker.
#
#  Signals:
#    1. Missing/very short audio track
#    2. Audio/video duration mismatch
#    3. Stereo channel correlation / mono-panned stock audio
#    4. Spectral noise floor / missing microphone room tone
#    5. RMS volume variation / over-consistent TTS dynamics
#    6. Silence distribution
#    7. Spectral bandwidth
#    8. Zero-crossing/prosody variation
#    9. Low-frequency ambience / room tone
#
#  Returns:
#    analyze_audio(media_path) -> dict
#    get_audio_contribution(score, confidence, context) -> signed int delta
# ============================================================

from __future__ import annotations

import base64
import json
import logging
import os
import subprocess
import tempfile
from typing import Optional, Tuple

import numpy as np

log = logging.getLogger("verifyd.audio")

FFMPEG_BIN = os.environ.get("FFMPEG_BIN", "ffmpeg")
FFPROBE_BIN = os.environ.get("FFPROBE_BIN", "ffprobe")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
GPT_MODEL = os.environ.get("VERIFYD_AUDIO_GPT_MODEL", os.environ.get("VERIFYD_GPT_MODEL", "gpt-4o"))

SAMPLE_RATE = int(os.environ.get("VERIFYD_AUDIO_SAMPLE_RATE", "22050"))
ANALYSIS_SECONDS = float(os.environ.get("VERIFYD_AUDIO_ANALYSIS_SECONDS", "30"))
MIN_AUDIO_SECONDS = 1.0


def analyze_audio(video_path: str) -> dict:
    """
    Analyze the audio track of a video or a standalone audio file for synthetic-audio indicators.

    The returned audio_ai_score is 0-100, where higher means more AI-like.
    This score should be blended through get_audio_contribution(), not used
    directly as the final result.
    """
    media = _get_media_info(video_path)
    if not media.get("available"):
        return {
            "audio_ai_score": 50,
            "evidence": ["Audio metadata unavailable"],
            "confidence": "low",
            "available": False,
        }

    audio_duration = float(media.get("audio_duration") or 0.0)
    video_duration = float(media.get("video_duration") or 0.0)
    has_video = bool(media.get("has_video"))
    has_audio = bool(media.get("has_audio")) and audio_duration >= MIN_AUDIO_SECONDS

    if not has_audio:
        # Missing audio is only a weak signal. Many legitimate clips are muted,
        # screen-recorded, exported without audio, or intentionally silent.
        return {
            "audio_ai_score": 58,
            "evidence": ["No usable audio track detected; treated as a weak signal only"],
            "confidence": "no_audio",
            "available": True,
            "has_audio": False,
            "has_video": has_video,
            "video_duration": round(video_duration, 3),
            "audio_duration": round(audio_duration, 3),
            "duration_mismatch": 0.0,
        }

    wav_mono = _extract_audio_wav(video_path, channels=1)
    if wav_mono is None:
        return {
            "audio_ai_score": 50,
            "evidence": ["Audio extraction failed"],
            "confidence": "low",
            "available": False,
            "has_audio": True,
            "has_video": has_video,
            "video_duration": round(video_duration, 3),
            "audio_duration": round(audio_duration, 3),
        }

    try:
        result = _analyze_wav(wav_mono, audio_duration, video_duration)
        # Add stereo correlation from a separate short stereo extraction.
        stereo_corr = _stereo_correlation(video_path)
        result["stereo_corr"] = stereo_corr
        if stereo_corr is not None:
            if stereo_corr >= 0.985:
                result["audio_ai_score"] = min(100, result["audio_ai_score"] + 10)
                result["evidence"].append(
                    f"Near-perfect stereo channel correlation ({stereo_corr:.3f}); common in mono-panned stock/TTS audio"
                )
            elif stereo_corr >= 0.94:
                result["audio_ai_score"] = min(100, result["audio_ai_score"] + 5)
                result["evidence"].append(
                    f"Very high stereo channel correlation ({stereo_corr:.3f})"
                )
            elif stereo_corr <= 0.80:
                result["audio_ai_score"] = max(0, result["audio_ai_score"] - 4)
                result["evidence"].append(
                    f"Natural stereo variation detected ({stereo_corr:.3f})"
                )

        duration_mismatch = abs(audio_duration - video_duration) if (has_video and video_duration > 0) else 0.0
        result["duration_mismatch"] = round(duration_mismatch, 4)
        if has_video and video_duration >= 2.0 and duration_mismatch > 0.20:
            result["audio_ai_score"] = min(100, result["audio_ai_score"] + 8)
            result["evidence"].append(
                f"Audio/video duration mismatch ({duration_mismatch:.3f}s); suggests separately assembled media"
            )
        elif has_video and video_duration >= 2.0 and duration_mismatch > 0.08:
            result["audio_ai_score"] = min(100, result["audio_ai_score"] + 4)
            result["evidence"].append(
                f"Small audio/video duration mismatch ({duration_mismatch:.3f}s)"
            )

        result["audio_ai_score"] = int(max(0, min(100, round(result["audio_ai_score"]))))
        result["video_duration"] = round(video_duration, 3)
        result["audio_duration"] = round(audio_duration, 3)
        result["has_audio"] = True
        result["has_video"] = has_video
        result["available"] = True
        return result
    finally:
        try:
            os.remove(wav_mono)
        except Exception:
            pass


def get_audio_contribution(audio_ai_score: int, confidence: str = "low", context: Optional[dict] = None) -> int:
    """
    Convert audio_ai_score into a conservative signed contribution for the
    existing signal score.

    Positive values increase AI likelihood; negative values support real capture.
    This cap prevents audio from overpowering the visual detector.
    """
    context = context or {}
    confidence = str(confidence or "low").lower()
    try:
        score = int(round(float(audio_ai_score)))
    except Exception:
        score = 50

    if confidence in ("unavailable", "low"):
        cap = 4
    elif confidence == "no_audio":
        # Missing audio is common and should not break existing real-video behavior.
        cap = 2
    elif confidence == "medium":
        cap = 8
    else:
        cap = 12

    if score >= 82:
        delta = cap
    elif score >= 72:
        delta = max(2, int(round(cap * 0.70)))
    elif score >= 62:
        delta = max(1, int(round(cap * 0.40)))
    elif score <= 25:
        delta = -min(cap, 8)
    elif score <= 35:
        delta = -max(2, int(round(cap * 0.60)))
    elif score <= 42:
        delta = -max(1, int(round(cap * 0.30)))
    else:
        delta = 0

    # For action/cinematic/social clips, no-audio is especially non-diagnostic.
    content_type = str(context.get("content_type", "") or "").lower()
    if confidence == "no_audio" and content_type in ("action", "cinematic", "static"):
        delta = min(delta, 1)

    return int(max(-cap, min(cap, delta)))


def _analyze_wav(wav_path: str, audio_duration: float, video_duration: float) -> dict:
    try:
        import librosa
    except Exception:
        log.warning("audio_detector: librosa not installed — skipping")
        return {
            "audio_ai_score": 50,
            "evidence": ["librosa unavailable; audio analysis skipped"],
            "confidence": "low",
            "available": False,
        }

    evidence = []
    score_delta = 0

    try:
        y, sr = librosa.load(
            wav_path,
            sr=SAMPLE_RATE,
            duration=min(float(audio_duration or ANALYSIS_SECONDS), ANALYSIS_SECONDS),
            mono=True,
        )
    except Exception as e:
        log.warning("audio_detector: librosa load failed: %s", e)
        return {
            "audio_ai_score": 50,
            "evidence": [f"Audio load error: {str(e)[:100]}"],
            "confidence": "low",
            "available": False,
        }

    if len(y) < int(sr * 0.5):
        return {
            "audio_ai_score": 50,
            "evidence": ["Audio too short for reliable analysis"],
            "confidence": "low",
            "available": False,
        }

    stft = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
    noise_floor = float(np.percentile(stft, 5))

    if noise_floor < 0.0005:
        score_delta += 18
        evidence.append(f"Unnaturally clean spectral floor ({noise_floor:.6f}); little microphone/room noise")
    elif noise_floor < 0.002:
        score_delta += 7
        evidence.append(f"Very low spectral noise floor ({noise_floor:.6f})")
    else:
        score_delta -= 8
        evidence.append(f"Natural microphone/room noise floor detected ({noise_floor:.4f})")

    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
    rms_nonzero = rms[rms > 1e-6]
    rms_cv = 0.0
    if len(rms_nonzero) > 10:
        rms_cv = float(np.std(rms_nonzero) / (np.mean(rms_nonzero) + 1e-10))
        if rms_cv < 0.10:
            score_delta += 16
            evidence.append(f"Unnaturally consistent volume (RMS CV={rms_cv:.3f}); TTS/stock-audio-like dynamics")
        elif rms_cv < 0.20:
            score_delta += 6
            evidence.append(f"Low volume variation (RMS CV={rms_cv:.3f})")
        else:
            score_delta -= 6
            evidence.append(f"Natural volume variation detected (RMS CV={rms_cv:.3f})")

    silence_threshold = max(0.006, float(np.percentile(rms, 20)) * 0.70) if len(rms) else 0.01
    silence_ratio = float(np.mean(rms < silence_threshold)) if len(rms) else 0.0
    if silence_ratio > 0.75:
        score_delta += 10
        evidence.append(f"Mostly silent/placeholder audio ({silence_ratio*100:.0f}% low-energy frames)")
    elif 0.08 <= silence_ratio <= 0.65:
        score_delta -= 2
        evidence.append(f"Organic silence/energy distribution ({silence_ratio*100:.0f}% low-energy frames)")
    else:
        evidence.append(f"Silence distribution measured ({silence_ratio*100:.0f}% low-energy frames)")

    spectral_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    mean_bw = float(np.mean(spectral_bw))
    if mean_bw < 550:
        score_delta += 12
        evidence.append(f"Narrow spectral bandwidth ({mean_bw:.0f} Hz); filtered/TTS-like audio")
    elif mean_bw < 950:
        score_delta += 4
        evidence.append(f"Below-average spectral bandwidth ({mean_bw:.0f} Hz)")
    else:
        score_delta -= 4
        evidence.append(f"Normal spectral bandwidth ({mean_bw:.0f} Hz)")

    zcr = librosa.feature.zero_crossing_rate(y, frame_length=2048, hop_length=512)[0]
    zcr_cv = float(np.std(zcr) / (np.mean(zcr) + 1e-9)) if len(zcr) else 0.0
    if zcr_cv < 0.18:
        score_delta += 8
        evidence.append(f"Low zero-crossing/prosody variation (ZCR CV={zcr_cv:.3f})")
    elif zcr_cv > 0.35:
        score_delta -= 4
        evidence.append(f"Natural prosody/noise variation (ZCR CV={zcr_cv:.3f})")
    else:
        evidence.append(f"Moderate zero-crossing variation (ZCR CV={zcr_cv:.3f})")

    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    low_bins = np.where(freqs < 500)[0]
    low_ratio = 0.0
    if len(low_bins) > 0 and stft.shape[0] > low_bins[-1]:
        low_energy = float(np.mean(stft[low_bins, :]))
        total_energy = float(np.mean(stft))
        low_ratio = low_energy / (total_energy + 1e-9)
        if low_ratio < 0.05:
            score_delta += 10
            evidence.append(f"Low-frequency room tone/ambience is weak ({low_ratio:.3f})")
        else:
            score_delta -= 5
            evidence.append(f"Natural low-frequency ambience present ({low_ratio:.3f})")

    base_audio_ai_score = int(max(0, min(100, round(50 + score_delta))))

    strong_hits = 0
    strong_hits += 1 if noise_floor < 0.0005 else 0
    strong_hits += 1 if (len(rms_nonzero) > 10 and rms_cv < 0.10) else 0
    strong_hits += 1 if mean_bw < 550 else 0
    strong_hits += 1 if low_ratio and low_ratio < 0.05 else 0
    confidence = "high" if strong_hits >= 3 else "medium" if strong_hits >= 1 else "low"

    gpt_review = _gpt_spectrogram_review(
        wav_path,
        {
            "base_audio_ai_score": base_audio_ai_score,
            "audio_duration": round(audio_duration, 3),
            "noise_floor": round(noise_floor, 8),
            "rms_cv": round(rms_cv, 4),
            "silence_ratio": round(silence_ratio, 4),
            "mean_bandwidth": round(mean_bw, 2),
            "zcr_cv": round(zcr_cv, 4),
            "low_freq_ratio": round(low_ratio, 4),
        },
    )
    audio_ai_score, gpt_adjustment = _blend_audio_gpt(base_audio_ai_score, gpt_review)

    if gpt_review.get("available"):
        evidence.append(
            f"GPT spectrogram review score={gpt_review.get('gpt_audio_score', 50)} "
            f"adjusted score {gpt_adjustment:+d}; {gpt_review.get('reasoning', '')[:140]}"
        )

    log.info(
        "audio_detector: score=%d base=%d gpt=%s adj=%+d conf=%s duration=%.1fs noise_floor=%.6f rms_cv=%.3f bw=%.0f zcr_cv=%.3f low=%.3f",
        audio_ai_score, base_audio_ai_score, gpt_review.get("gpt_audio_score", "na"), gpt_adjustment,
        confidence, audio_duration, noise_floor, rms_cv, mean_bw, zcr_cv, low_ratio,
    )

    return {
        "audio_ai_score": audio_ai_score,
        "base_audio_ai_score": base_audio_ai_score,
        "evidence": evidence,
        "confidence": confidence,
        "available": True,
        "noise_floor": round(noise_floor, 8),
        "rms_cv": round(rms_cv, 4),
        "silence_ratio": round(silence_ratio, 4),
        "mean_bandwidth": round(mean_bw, 2),
        "zcr_cv": round(zcr_cv, 4),
        "low_freq_ratio": round(low_ratio, 4),
        "gpt_audio_score": gpt_review.get("gpt_audio_score", 0),
        "gpt_audio_available": bool(gpt_review.get("available", False)),
        "gpt_audio_adjustment": int(gpt_adjustment),
        "gpt_audio_reasoning": gpt_review.get("reasoning", ""),
        "gpt_audio_flags": gpt_review.get("flags", []),
    }


def _blend_audio_gpt(base_score: int, gpt_review: dict) -> tuple[int, int]:
    """Blend GPT spectrogram review as a weak nudge only (max +/-10)."""
    if not gpt_review or not gpt_review.get("available"):
        return int(base_score), 0
    try:
        gpt_score = int(round(float(gpt_review.get("gpt_audio_score", 50))))
    except Exception:
        return int(base_score), 0

    # GPT can only nudge the detector, not overrule it.
    raw_delta = gpt_score - int(base_score)
    adjustment = int(round(raw_delta * 0.25))
    adjustment = max(-10, min(10, adjustment))

    # Keep very strong signal results from being pulled too far by visualized charts.
    if base_score >= 82 and adjustment < 0:
        adjustment = max(adjustment, -5)
    if base_score <= 18 and adjustment > 0:
        adjustment = min(adjustment, 5)

    final_score = int(max(0, min(100, int(base_score) + adjustment)))
    return final_score, adjustment


def _gpt_spectrogram_review(wav_path: str, feature_summary: dict) -> dict:
    """Ask GPT-4o to review a visual audio forensic panel as a weak second opinion."""
    if not OPENAI_API_KEY:
        return {"available": False, "gpt_audio_score": 0, "reasoning": "GPT audio-spectrogram review unavailable; no API key.", "flags": []}

    png_path = None
    try:
        png_path = _build_audio_diagnostic_png(wav_path)
        if not png_path or not os.path.exists(png_path):
            return {"available": False, "gpt_audio_score": 0, "reasoning": "Audio diagnostic image unavailable.", "flags": []}

        with open(png_path, "rb") as fh:
            b64 = base64.b64encode(fh.read()).decode("ascii")

        prompt = (
            "You are reviewing an audio forensic visualization for synthetic-audio detection. "
            "The image contains a waveform, RMS/loudness envelope, and mel-spectrogram generated from an uploaded audio file. "
            "You are NOT listening to the audio. Use only visible chart structure plus the provided numeric features. "
            "Return a weak second-opinion AI-likelihood score from 0 to 100, where 0=strongly natural/real recording, "
            "50=inconclusive, 100=strong synthetic/AI indicators. Do not overstate confidence. "
            "Natural mastered music can be clean and compressed; do not label music AI merely because it is polished. "
            f"Numeric features: {json.dumps(feature_summary, sort_keys=True)}. "
            "Respond only as JSON with keys: gpt_audio_score, reasoning, flags."
        )

        import urllib.request
        payload = {
            "model": GPT_MODEL,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "low"}},
                ],
            }],
            "temperature": 0.1,
            "max_tokens": 260,
        }
        req = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {OPENAI_API_KEY}"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=45) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        raw = data["choices"][0]["message"]["content"].strip()
        if "```" in raw:
            raw = raw.split("```")[1].replace("json", "", 1).strip()
        parsed = json.loads(raw)
        score = int(max(0, min(100, round(float(parsed.get("gpt_audio_score", 50))))))
        flags = [str(x)[:120] for x in parsed.get("flags", [])][:5] if isinstance(parsed.get("flags", []), list) else []
        return {
            "available": True,
            "gpt_audio_score": score,
            "reasoning": str(parsed.get("reasoning", ""))[:400],
            "flags": flags,
        }
    except Exception as e:
        log.warning("audio_detector: GPT spectrogram review skipped: %s", e)
        return {"available": False, "gpt_audio_score": 0, "reasoning": f"GPT spectrogram review unavailable: {str(e)[:100]}", "flags": []}
    finally:
        try:
            if png_path and os.path.exists(png_path):
                os.remove(png_path)
        except Exception:
            pass


def _build_audio_diagnostic_png(wav_path: str) -> Optional[str]:
    """Create a compact PNG with waveform, RMS envelope, and mel spectrogram."""
    try:
        import librosa
        from PIL import Image, ImageDraw

        y, sr = librosa.load(wav_path, sr=SAMPLE_RATE, duration=min(ANALYSIS_SECONDS, 30), mono=True)
        if len(y) < int(sr * 0.5):
            return None

        width = 720
        wave_h = 120
        rms_h = 80
        spec_h = 280
        pad = 28
        img = Image.new("RGB", (width, wave_h + rms_h + spec_h + pad * 4), (12, 12, 12))
        draw = ImageDraw.Draw(img)

        # Waveform panel
        y_norm = y / (np.max(np.abs(y)) + 1e-9)
        xs = np.linspace(0, len(y_norm) - 1, width).astype(int)
        wave = y_norm[xs]
        y0 = pad
        mid = y0 + wave_h // 2
        draw.text((8, y0 - 20), "Waveform", fill=(220, 220, 220))
        prev = (0, mid)
        for x, val in enumerate(wave):
            pt = (x, int(mid - val * (wave_h // 2 - 4)))
            draw.line([prev, pt], fill=(80, 190, 255))
            prev = pt

        # RMS panel
        rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
        rms = rms / (np.max(rms) + 1e-9) if len(rms) else np.zeros(1)
        xs2 = np.linspace(0, len(rms) - 1, width).astype(int)
        rms_view = rms[xs2]
        y1 = y0 + wave_h + pad
        draw.text((8, y1 - 20), "RMS / Loudness Envelope", fill=(220, 220, 220))
        prev = (0, y1 + rms_h)
        for x, val in enumerate(rms_view):
            pt = (x, int(y1 + rms_h - val * (rms_h - 4)))
            draw.line([prev, pt], fill=(110, 230, 120))
            prev = pt

        # Mel spectrogram panel
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512, n_mels=96)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_db = np.flipud(mel_db)
        m = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-9)
        arr = (m * 255).astype(np.uint8)
        spec = Image.fromarray(arr, mode="L").resize((width, spec_h)).convert("RGB")
        # simple dark-blue to orange palette approximation
        spec_arr = np.array(spec)[:, :, 0]
        color = np.zeros((spec_h, width, 3), dtype=np.uint8)
        color[:, :, 0] = np.clip(spec_arr * 1.6, 0, 255).astype(np.uint8)
        color[:, :, 1] = np.clip((spec_arr - 40) * 1.2, 0, 255).astype(np.uint8)
        color[:, :, 2] = np.clip(180 - spec_arr * 0.4, 0, 180).astype(np.uint8)
        spec_img = Image.fromarray(color, mode="RGB")
        y2 = y1 + rms_h + pad
        draw.text((8, y2 - 20), "Mel Spectrogram", fill=(220, 220, 220))
        img.paste(spec_img, (0, y2))

        fd, out = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        img.save(out, "PNG")
        return out
    except Exception as e:
        log.debug("audio_detector: diagnostic image failed: %s", e)
        return None


def _get_media_info(video_path: str) -> dict:
    cmd = [
        FFPROBE_BIN,
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        video_path,
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=25)
        if proc.returncode != 0:
            return {"available": False}
        data = json.loads(proc.stdout.decode("utf-8", errors="replace") or "{}")
        fmt = data.get("format", {}) or {}
        format_duration = _safe_float(fmt.get("duration"), 0.0)
        video_duration = 0.0
        audio_duration = 0.0
        has_video = False
        has_audio = False
        for stream in data.get("streams", []) or []:
            if stream.get("codec_type") == "video":
                has_video = True
                video_duration = max(video_duration, _safe_float(stream.get("duration"), 0.0))
            elif stream.get("codec_type") == "audio":
                has_audio = True
                audio_duration = max(audio_duration, _safe_float(stream.get("duration"), 0.0))

        # Many MP3/M4A/OGG files expose duration only at the container level.
        # For audio-only uploads, treat the container duration as audio duration.
        if has_audio and not audio_duration and format_duration:
            audio_duration = format_duration
        if has_video and not video_duration and format_duration:
            video_duration = format_duration

        return {
            "available": True,
            "has_video": has_video,
            "has_audio": has_audio,
            "video_duration": video_duration,
            "audio_duration": audio_duration,
            "format_name": fmt.get("format_name", ""),
        }
    except Exception as e:
        log.debug("audio_detector: ffprobe failed: %s", e)
        return {"available": False}


def _extract_audio_wav(video_path: str, channels: int = 1, seconds: float = ANALYSIS_SECONDS) -> Optional[str]:
    try:
        fd, wav_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        cmd = [
            FFMPEG_BIN, "-y", "-i", video_path,
            "-vn",
            "-ac", str(channels),
            "-ar", str(SAMPLE_RATE),
            "-t", str(seconds),
            "-f", "wav",
            wav_path,
        ]
        proc = subprocess.run(cmd, capture_output=True, timeout=70)
        if proc.returncode == 0 and os.path.exists(wav_path) and os.path.getsize(wav_path) > 256:
            return wav_path
        try:
            os.remove(wav_path)
        except Exception:
            pass
        return None
    except Exception as e:
        log.debug("audio_detector: wav extraction failed: %s", e)
        return None


def _stereo_correlation(video_path: str) -> Optional[float]:
    wav_path = _extract_audio_wav(video_path, channels=2, seconds=min(ANALYSIS_SECONDS, 20))
    if not wav_path:
        return None
    try:
        import soundfile as sf
        data, _sr = sf.read(wav_path, always_2d=True)
        if data.shape[1] < 2 or data.shape[0] < 1000:
            return None
        left = data[:, 0].astype(float)
        right = data[:, 1].astype(float)
        if np.std(left) < 1e-8 or np.std(right) < 1e-8:
            return None
        return float(np.corrcoef(left, right)[0, 1])
    except Exception:
        return None
    finally:
        try:
            os.remove(wav_path)
        except Exception:
            pass


def _safe_float(value, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except Exception:
        return default
