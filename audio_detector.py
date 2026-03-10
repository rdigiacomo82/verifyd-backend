# ============================================================
#  VeriFYD — audio_detector.py
#
#  Audio-domain AI detection engine.
#  Analyzes the audio track of a video for synthesis artifacts,
#  unnatural spectral characteristics, and voice synthesis tells.
#
#  WHY AUDIO MATTERS:
#  AI video generators either (a) have no audio, (b) use a
#  separate TTS/voice synthesis system, or (c) use generic
#  stock audio. Real camera audio has predictable properties:
#  environmental background noise, microphone self-noise,
#  natural speech prosody, and continuous ambient sound.
#
#  AI audio signatures:
#    - Silence or missing audio (video-only)
#    - Spectral floor too clean (no microphone noise)
#    - Speech that starts/stops perfectly (TTS timing)
#    - Missing or unnatural environmental background
#    - Audio/video duration mismatch
#    - Abnormal frequency distribution (TTS has characteristic peaks)
#    - Perfectly consistent volume (no natural variation)
#
#  This module requires: librosa, numpy, scipy (optional)
#  Gracefully degrades if librosa not available.
#
#  Returns:
#    audio_ai_score  : 0–100
#    evidence        : list of str findings
#    confidence      : "high" | "medium" | "low" | "no_audio"
#    available       : bool
# ============================================================

import os
import json
import subprocess
import tempfile
import logging
import numpy as np
from typing import Optional

log = logging.getLogger("verifyd.audio")

FFMPEG_BIN  = os.environ.get("FFMPEG_BIN", "ffmpeg")
FFPROBE_BIN = os.environ.get("FFPROBE_BIN", "ffprobe")

# ─────────────────────────────────────────────────────────────
#  Config
# ─────────────────────────────────────────────────────────────
SAMPLE_RATE      = 22050   # librosa default, good for speech analysis
ANALYSIS_SECONDS = 30      # max audio to analyze (first 30s)
MIN_AUDIO_SECONDS = 1.0    # below this → treat as no audio


def analyze_audio(video_path: str) -> dict:
    """
    Extract audio from video and run spectral AI detection analysis.

    Returns:
        audio_ai_score  : int 0–100
        evidence        : list of str findings
        confidence      : "high" | "medium" | "low" | "no_audio"
        available       : bool
    """
    # ── Step 1: Check for audio track ────────────────────────
    duration = _get_audio_duration(video_path)
    if duration is None or duration < MIN_AUDIO_SECONDS:
        log.info("audio_detector: no audio track or too short (%.1fs)", duration or 0)
        return {
            "audio_ai_score": 62,   # no audio is a mild AI signal
            "evidence": ["No audio track — AI generators often produce video-only output"],
            "confidence": "no_audio",
            "available": True,
        }

    # ── Step 2: Extract audio to WAV ─────────────────────────
    wav_path = _extract_audio_wav(video_path)
    if wav_path is None:
        return {
            "audio_ai_score": 50,
            "evidence": ["Audio extraction failed"],
            "confidence": "low",
            "available": False,
        }

    try:
        return _analyze_wav(wav_path, duration)
    finally:
        try:
            os.remove(wav_path)
        except Exception:
            pass


def _analyze_wav(wav_path: str, duration: float) -> dict:
    """Load WAV and run spectral analysis."""
    try:
        import librosa
    except ImportError:
        log.warning("audio_detector: librosa not installed — skipping analysis")
        return {
            "audio_ai_score": 50,
            "evidence": ["librosa not installed — audio analysis skipped"],
            "confidence": "low",
            "available": False,
        }

    score    = 0
    evidence = []

    try:
        # Load up to ANALYSIS_SECONDS
        y, sr = librosa.load(wav_path, sr=SAMPLE_RATE,
                             duration=min(duration, ANALYSIS_SECONDS))
    except Exception as e:
        log.error("audio_detector: librosa load error: %s", e)
        return {
            "audio_ai_score": 50,
            "evidence": [f"Audio load error: {str(e)[:80]}"],
            "confidence": "low",
            "available": False,
        }

    if len(y) < sr * 0.5:
        return {
            "audio_ai_score": 50,
            "evidence": ["Audio too short for analysis"],
            "confidence": "low",
            "available": False,
        }

    # ── Signal 1: Spectral floor / noise floor ────────────────
    # Real microphones always have a noise floor (mic self-noise + room).
    # TTS / AI audio often has a perfectly clean spectral floor.
    stft        = np.abs(librosa.stft(y))
    noise_floor = np.percentile(stft, 5)   # bottom 5% of spectral energy

    if noise_floor < 0.0005:
        evidence.append(f"Unnaturally clean spectral floor ({noise_floor:.6f}) — "
                        "no microphone noise detected. Consistent with synthesized audio.")
        score += 20
    elif noise_floor < 0.002:
        evidence.append(f"Very low spectral noise floor ({noise_floor:.6f}) — slightly clean")
        score += 8
    else:
        evidence.append(f"Natural microphone noise floor detected ({noise_floor:.4f})")
        score -= 10

    # ── Signal 2: Volume consistency ─────────────────────────
    # Real speech and environmental audio has natural volume variation.
    # TTS audio is often unnaturally consistent in volume.
    rms = librosa.feature.rms(y=y)[0]
    rms_nonzero = rms[rms > 1e-6]
    if len(rms_nonzero) > 10:
        rms_cv = float(np.std(rms_nonzero) / np.mean(rms_nonzero))  # coefficient of variation
        if rms_cv < 0.10:
            evidence.append(f"Unnaturally consistent volume (CV={rms_cv:.3f}) — "
                            "real audio always varies. Consistent with TTS.")
            score += 18
        elif rms_cv < 0.20:
            evidence.append(f"Low volume variation (CV={rms_cv:.3f}) — slightly suspicious")
            score += 7
        else:
            evidence.append(f"Natural volume variation detected (CV={rms_cv:.3f})")
            score -= 8

    # ── Signal 3: Silence distribution ───────────────────────
    # Real speech has organic silence patterns — pauses between words vary.
    # TTS has very uniform inter-word silences.
    silence_threshold = 0.01
    silent_frames     = np.sum(rms < silence_threshold)
    silence_ratio     = silent_frames / len(rms) if len(rms) > 0 else 0

    if silence_ratio > 0.70:
        evidence.append(f"Mostly silent audio ({silence_ratio*100:.0f}% silence) — "
                        "audio may be stock/placeholder")
        score += 12
    elif silence_ratio < 0.05 and duration > 3:
        # Real speech has pauses; continuous audio with no silence = music or consistent noise
        evidence.append(f"No silence gaps in speech ({silence_ratio*100:.0f}%) — natural for music/ambience")
        # neutral — don't penalize, could be music or ambient sound
    else:
        evidence.append(f"Normal silence distribution ({silence_ratio*100:.0f}% silence)")

    # ── Signal 4: Spectral bandwidth / TTS frequency tells ───
    # TTS systems (especially older ones) have characteristic frequency peaks
    # and often lack the full spectral spread of real microphone audio.
    spectral_bw  = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    mean_bw      = float(np.mean(spectral_bw))
    # Real speech: typically 800–2500 Hz bandwidth
    # TTS: often narrower or shows peaks at specific frequencies
    if mean_bw < 600:
        evidence.append(f"Narrow spectral bandwidth ({mean_bw:.0f} Hz) — "
                        "consistent with low-quality TTS or filtered audio")
        score += 15
    elif mean_bw < 1000:
        evidence.append(f"Below-average spectral bandwidth ({mean_bw:.0f} Hz)")
        score += 5
    else:
        evidence.append(f"Normal spectral bandwidth ({mean_bw:.0f} Hz)")
        score -= 5

    # ── Signal 5: Zero-crossing rate consistency ──────────────
    # Real speech has variable ZCR following natural prosody.
    # TTS tends toward more uniform ZCR.
    zcr    = librosa.feature.zero_crossing_rate(y)[0]
    zcr_cv = float(np.std(zcr) / (np.mean(zcr) + 1e-9))
    if zcr_cv < 0.20:
        evidence.append(f"Low prosody variation (ZCR_CV={zcr_cv:.3f}) — consistent with TTS")
        score += 10
    else:
        evidence.append(f"Natural prosody variation (ZCR_CV={zcr_cv:.3f})")
        score -= 5

    # ── Signal 6: Background ambient presence ─────────────────
    # Real outdoor/indoor recordings always have some background hum:
    # HVAC, traffic, room tone, wind. We check for sub-500Hz energy.
    freqs   = librosa.fft_frequencies(sr=sr)
    low_bin = np.where(freqs < 500)[0]
    if len(low_bin) > 0 and stft.shape[0] > low_bin[-1]:
        low_energy  = float(np.mean(stft[low_bin, :]))
        total_energy = float(np.mean(stft))
        low_ratio = low_energy / (total_energy + 1e-9)
        if low_ratio < 0.05:
            evidence.append(f"No ambient low-frequency background ({low_ratio:.3f}) — "
                            "real environments always have low-freq hum")
            score += 12
        else:
            evidence.append(f"Natural ambient low-frequency content present ({low_ratio:.3f})")
            score -= 7

    # ── Clamp and return ──────────────────────────────────────
    score = max(0, min(100, 50 + score))

    strong_signals = sum([
        noise_floor < 0.0005,
        len(rms_nonzero) > 10 and rms_cv < 0.10,
        mean_bw < 600,
    ])
    confidence = "high" if strong_signals >= 2 else "medium" if strong_signals == 1 else "low"

    log.info("audio_detector: score=%d  confidence=%s  duration=%.1fs",
             score, confidence, duration)

    return {
        "audio_ai_score": score,
        "evidence":       evidence,
        "confidence":     confidence,
        "available":      True,
    }


# ─────────────────────────────────────────────────────────────
#  ffmpeg / ffprobe helpers
# ─────────────────────────────────────────────────────────────
def _get_audio_duration(video_path: str) -> Optional[float]:
    """Return audio track duration in seconds, or None if no audio."""
    cmd = [
        FFPROBE_BIN, "-v", "quiet", "-print_format", "json",
        "-show_streams", "-select_streams", "a:0", video_path
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=20)
        data = json.loads(result.stdout.decode("utf-8", errors="replace"))
        streams = data.get("streams", [])
        if not streams:
            return None
        dur = streams[0].get("duration")
        return float(dur) if dur else None
    except Exception as e:
        log.warning("audio_detector: ffprobe duration error: %s", e)
        return None


def _extract_audio_wav(video_path: str) -> Optional[str]:
    """Extract audio to a temporary WAV file. Returns path or None."""
    try:
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        wav_path = tmp.name
        tmp.close()

        cmd = [
            FFMPEG_BIN, "-y", "-i", video_path,
            "-vn",                        # no video
            "-ac", "1",                   # mono
            "-ar", str(SAMPLE_RATE),      # resample
            "-t", str(ANALYSIS_SECONDS),  # max duration
            "-f", "wav",
            wav_path
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=60)
        if result.returncode == 0 and os.path.exists(wav_path):
            return wav_path
        log.warning("audio_detector: ffmpeg audio extract failed (code %d)", result.returncode)
        if os.path.exists(wav_path):
            os.remove(wav_path)
        return None
    except Exception as e:
        log.error("audio_detector: extract error: %s", e)
        return None
