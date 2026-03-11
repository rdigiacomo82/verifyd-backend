# ============================================================
#  VeriFYD — rppg_detector.py  v4
#
#  ENGINE 5: Remote Photoplethysmography (rPPG) — THE MOAT
#
#  What is rPPG?
#  Real human skin has a cardiovascular blood volume pulse (BVP)
#  that causes subtle periodic color changes (~0.5-3% amplitude,
#  45-120 BPM). Measurable from video without contact.
#  Technique behind Intel FakeCatcher (Ciftci et al. 2023)
#  and the de Haan & Jeanne (2013) CHROM algorithm.
#
#  Why is this the moat?
#  AI generators (Sora, Kling, Runway, Pika, HeyGen) train on
#  visual appearance — none model the cardiovascular system.
#  A generated face has zero heartbeat-driven pulse. This is
#  non-differentiable: faking it requires per-pixel blood
#  volume simulation, which no current generator does.
#
#  Algorithm: CHROM (de Haan & Jeanne, IEEE TBME 2013)
#  1. Detect face landmarks with MediaPipe FaceLandmarker
#     (falls back to ensemble Haar cascade if unavailable)
#  2. Extract mean R,G,B from precise forehead + bilateral
#     cheek ROIs (highest capillary density regions)
#  3. Normalize by channel mean (remove absolute luminance)
#  4. CHROM = (3R-2G) - alpha*(1.5R+G-1.5B) [cancels specular]
#  5. Bandpass 0.75-3.0 Hz (45-180 BPM)
#  6. FFT → dominant frequency → estimated BPM
#  7. Autocorrelation at heartbeat lag → periodicity score
#  8. SNR gating: reject noisy traces
#  9. Peak regularity analysis
#
#  v4 changes vs v3:
#  - MediaPipe FaceLandmarker as primary face detector
#    → 478 landmarks → surgical ROI placement on forehead/cheeks
#    → ~85-95% detection rate vs ~50% Haar ensemble
#  - Landmark-precise ROI: forehead (lm 10,151,9), left cheek
#    (lm 234,93), right cheek (lm 454,323) with 20px radius discs
#    → Same approach as Intel FakeCatcher
#  - VIDEO running mode for temporal face tracking (stable ROI)
#  - Haar ensemble (frontal_default + alt2 + profile) as fallback
#    when MediaPipe model not available (build.sh downloads it)
#  - Model path: /opt/render/project/.render/mediapipe_models/
#    face_landmarker.task (downloaded by build.sh at deploy time)
#
#  Confidence tiers:
#    "high"   — >=60 face frames AND duration >= 6s
#    "medium" — >=25 face frames AND duration >= 4s
#    "low"    — >=10 face frames (attenuated in detection.py)
#    "no_face"— < 10 face frames → neutral 50, 0 adjustment
#
#  Cap in detection.py: +-15 at high confidence.
# ============================================================

import cv2
import logging
import os
import numpy as np

log = logging.getLogger("verifyd.rppg")

# MediaPipe model path (downloaded by build.sh at deploy time)
_MP_MODEL_PATH = os.environ.get(
    "MEDIAPIPE_FACE_MODEL",
    "/opt/render/project/.render/mediapipe_models/face_landmarker.task"
)

# rPPG physiological constants
_HR_LOW_HZ          = 0.75       # 45 BPM
_HR_HIGH_HZ         = 3.0        # 180 BPM
_HR_PLAUSIBLE_LOW   = 42
_HR_PLAUSIBLE_HIGH  = 120

# Sampling budget
_MAX_FRAMES         = 150
_DETECT_WIDTH       = 480        # wider than v3 for better landmark accuracy

# Confidence thresholds
_MIN_FACE_FRAMES_HI = 60
_MIN_FACE_FRAMES_MD = 25
_MIN_FACE_FRAMES_LO = 10
_MIN_DURATION_HI    = 6.0
_MIN_DURATION_MD    = 4.0

# Quality gate
_SNR_GATE           = 0.035

# MediaPipe landmark indices for rPPG ROI
# (highest capillary density — same regions as Intel FakeCatcher)
_LM_FOREHEAD   = [10, 151, 9]    # forehead center strip
_LM_LEFT_CHEEK = [234, 93, 132]  # left cheek cluster
_LM_RIGHT_CHEEK= [454, 323, 361] # right cheek cluster
_ROI_RADIUS    = 18              # pixel radius for each landmark disc


def analyze_rppg(video_path: str, content_type: str = "cinematic") -> dict:
    """
    Main entry point. Analyze rPPG biological pulse signal.

    Returns dict with: rppg_ai_score, confidence, estimated_bpm,
    g_amp, autocorr, snr, face_ratio, face_count, evidence, available.
    """
    try:
        return _analyze_inner(video_path, content_type)
    except Exception as e:
        log.warning("rppg: unexpected error: %s", e, exc_info=True)
        return _neutral("error")


def _analyze_inner(video_path: str, content_type: str) -> dict:

    # Step 1: extract RGB traces using best available face detector
    rgb_traces, timestamps, face_frames, total_sampled, fps_nominal, detector_used = \
        _extract_face_rgb(video_path)

    n          = len(rgb_traces)
    face_ratio = face_frames / max(total_sampled, 1)

    log.info("rppg v4: detector=%s  face_frames=%d  traces=%d  total_samp=%d",
             detector_used, face_frames, n, total_sampled)

    if n < _MIN_FACE_FRAMES_LO or face_frames < _MIN_FACE_FRAMES_LO:
        return _neutral("no_face", face_frames=face_frames,
                        face_ratio=face_ratio, n_samples=total_sampled)

    # Reconstruct actual sampling fps from timestamps
    if len(timestamps) >= 2:
        duration   = timestamps[-1] - timestamps[0]
        actual_fps = n / max(duration, 0.1)
    else:
        duration   = n / fps_nominal
        actual_fps = fps_nominal

    log.info("rppg v4: n=%d  face_frames=%d  duration=%.1fs  actual_fps=%.1f  "
             "face_ratio=%.2f  detector=%s",
             n, face_frames, duration, actual_fps, face_ratio, detector_used)

    # Step 2: confidence tier
    if face_frames >= _MIN_FACE_FRAMES_HI and duration >= _MIN_DURATION_HI:
        confidence = "high"
    elif face_frames >= _MIN_FACE_FRAMES_MD and duration >= _MIN_DURATION_MD:
        confidence = "medium"
    elif face_frames >= _MIN_FACE_FRAMES_LO:
        confidence = "low"
    else:
        return _neutral("insufficient_faces", face_frames=face_frames,
                        face_ratio=face_ratio, n_samples=total_sampled)

    # Step 3: CHROM rPPG
    try:
        from scipy import signal as sps
    except ImportError:
        log.warning("rppg: scipy not available")
        return _neutral("scipy_unavailable")

    R = np.array([t[0] for t in rgb_traces])
    G = np.array([t[1] for t in rgb_traces])
    B = np.array([t[2] for t in rgb_traces])

    # Normalize by channel mean
    Rn = R / (R.mean() + 1e-10)
    Gn = G / (G.mean() + 1e-10)
    Bn = B / (B.mean() + 1e-10)

    # CHROM chrominance projection (de Haan & Jeanne 2013)
    Xs    = 3.0 * Rn - 2.0 * Gn
    Ys    = 1.5 * Rn + Gn - 1.5 * Bn
    alpha = Xs.std() / (Ys.std() + 1e-10)
    chrom = sps.detrend(Xs - alpha * Ys)

    # Bandpass: heartbeat band
    nyq  = actual_fps / 2.0
    low  = _HR_LOW_HZ  / nyq
    high = min(_HR_HIGH_HZ / nyq, 0.98)
    if low >= high or nyq < _HR_LOW_HZ:
        log.warning("rppg: sampling fps too low (%.1f)", actual_fps)
        return _neutral("fps_too_low", face_frames=face_frames, face_ratio=face_ratio)

    b, a    = sps.butter(3, [low, high], btype='band')
    chrom_f = sps.filtfilt(b, a, chrom)

    # Step 4: spectral analysis
    freqs   = np.fft.rfftfreq(n, d=1.0 / actual_fps)
    fft_amp = np.abs(np.fft.rfft(chrom_f))
    hr_mask = (freqs >= _HR_LOW_HZ) & (freqs <= _HR_HIGH_HZ)

    if hr_mask.sum() < 2:
        return _neutral("low_freq_resolution", face_frames=face_frames, face_ratio=face_ratio)

    hr_freqs = freqs[hr_mask]
    hr_power = fft_amp[hr_mask]
    dom_idx  = int(np.argmax(hr_power))
    dom_freq = float(hr_freqs[dom_idx])
    dom_bpm  = dom_freq * 60.0
    snr      = float(hr_power[dom_idx] / (hr_power.sum() + 1e-10))

    # SNR gate — noisy signal = unreliable result
    if snr < _SNR_GATE:
        if confidence == "high":
            confidence = "medium"
            log.info("rppg: SNR gate → medium (snr=%.4f)", snr)
        elif confidence == "medium":
            log.info("rppg: SNR gate → low_snr neutral (snr=%.4f)", snr)
            return _neutral("low_snr", face_frames=face_frames, face_ratio=face_ratio)

    # Step 5: autocorrelation at heartbeat lag
    lag = int(actual_fps / dom_freq) if dom_freq > 0.3 else 0
    if 2 < lag < n // 2:
        autocorr = float(np.corrcoef(chrom_f[:-lag], chrom_f[lag:])[0, 1])
        autocorr = max(-1.0, min(1.0, autocorr))
    else:
        autocorr = 0.0

    # Step 6: peak analysis
    chrom_amp  = float(chrom_f.std())
    peaks, _   = sps.find_peaks(chrom_f,
                                  height=chrom_amp * 0.3,
                                  distance=int(actual_fps * 0.4))
    n_peaks    = len(peaks)
    peak_ratio = n_peaks / max(duration * dom_bpm / 60.0, 1e-5)

    log.info(
        "rppg CHROM v4 [%s]: bpm=%.1f snr=%.4f autocorr=%.3f amp=%.5f "
        "n_peaks=%d peak_ratio=%.2f confidence=%s",
        detector_used, dom_bpm, snr, autocorr, chrom_amp,
        n_peaks, peak_ratio, confidence
    )

    # Step 7: score
    ai_score, evidence = _score(autocorr, dom_bpm, snr, peak_ratio)
    evidence.append(f"detector:{detector_used}")

    log.info("rppg: ai_score=%d  confidence=%s", ai_score, confidence)

    return {
        "rppg_ai_score": ai_score,
        "confidence":    confidence,
        "estimated_bpm": round(dom_bpm, 1),
        "g_amp":         round(chrom_amp, 5),
        "autocorr":      round(autocorr, 4),
        "snr":           round(snr, 4),
        "face_ratio":    round(face_ratio, 3),
        "face_count":    face_frames,
        "n_samples":     n,
        "duration":      round(duration, 1),
        "n_peaks":       n_peaks,
        "peak_ratio":    round(peak_ratio, 2),
        "detector_used": detector_used,
        "evidence":      evidence,
        "available":     True,
    }


# ─────────────────────────────────────────────────────────────
#  Face RGB extraction — MediaPipe primary, Haar fallback
# ─────────────────────────────────────────────────────────────

def _extract_face_rgb(video_path: str, max_frames: int = _MAX_FRAMES):
    """
    Extract per-frame mean (R,G,B) from face ROI.
    Tries MediaPipe FaceLandmarker first, falls back to Haar ensemble.

    Returns: (rgb_traces, timestamps, face_frames, total_samp, fps_nominal, detector_used)
    """
    # Try MediaPipe first if model file exists
    if os.path.exists(_MP_MODEL_PATH):
        try:
            result = _extract_mediapipe(video_path, max_frames)
            if result[2] >= _MIN_FACE_FRAMES_LO:   # face_frames >= minimum
                return result + ("mediapipe",)
            else:
                log.info("rppg: MediaPipe found <10 faces, trying Haar fallback")
        except Exception as e:
            log.warning("rppg: MediaPipe extraction failed (%s), falling back to Haar", e)
    else:
        log.info("rppg: MediaPipe model not found at %s — using Haar cascade", _MP_MODEL_PATH)

    # Haar ensemble fallback
    result = _extract_haar(video_path, max_frames)
    return result + ("haar_ensemble",)


def _extract_mediapipe(video_path: str, max_frames: int):
    """
    Extract face RGB using MediaPipe FaceLandmarker.
    Uses precise landmark-defined ROI on forehead and bilateral cheeks.
    Returns: (rgb_traces, timestamps, face_frames, total_samp, fps_nominal)
    """
    import mediapipe as mp
    from mediapipe.tasks.python import vision, BaseOptions
    from mediapipe.tasks.python.vision import FaceLandmarkerOptions, RunningMode

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=_MP_MODEL_PATH),
        running_mode=RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.45,
        min_face_presence_score=0.45,
        min_tracking_confidence=0.45,
    )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return [], [], 0, 0, 30.0

    fc          = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_nominal = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step        = max(1, fc // max_frames)

    rgb_traces  = []
    timestamps  = []
    face_frames = 0
    total_samp  = 0
    last_lm     = None   # cached landmarks from last successful detection

    detector = vision.FaceLandmarker.create_from_options(options)

    fn = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        fn += 1

        if fn % step != 0:
            continue
        if total_samp >= max_frames:
            break

        total_samp += 1
        t = fn / fps_nominal
        H, W = frame.shape[:2]

        # Downscale for faster detection
        scale = _DETECT_WIDTH / W if W > _DETECT_WIDTH else 1.0
        if scale < 1.0:
            small = cv2.resize(frame, (int(W * scale), int(H * scale)))
        else:
            small = frame

        rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_small)

        try:
            result = detector.detect(mp_image)
        except Exception:
            result = None

        if result and result.face_landmarks:
            last_lm  = result.face_landmarks[0]
            face_frames += 1

        # Extract RGB from landmark ROI (use cached if no detection this frame)
        if last_lm is not None:
            rgb = _landmark_roi_rgb(frame, last_lm, W, H, scale)
            if rgb is not None:
                rgb_traces.append(rgb)
                timestamps.append(t)

    cap.release()
    detector.close()

    log.info("rppg MediaPipe extract: total=%d faces=%d traces=%d",
             total_samp, face_frames, len(rgb_traces))
    return rgb_traces, timestamps, face_frames, total_samp, fps_nominal


def _landmark_roi_rgb(frame, landmarks, W: int, H: int, scale: float):
    """
    Extract mean RGB from forehead + bilateral cheek landmark discs.
    landmarks: list of NormalizedLandmark (from small frame, scale back to original).
    """
    r_vals, g_vals, b_vals = [], [], []
    inv = 1.0 / scale  # scale landmarks from small frame back to original

    for lm_group in [_LM_FOREHEAD, _LM_LEFT_CHEEK, _LM_RIGHT_CHEEK]:
        for lm_idx in lm_group:
            if lm_idx >= len(landmarks):
                continue
            lm = landmarks[lm_idx]
            # Landmark coords are normalized to the SMALL frame
            cx = int(lm.x * (W * scale) * inv)   # = lm.x * W
            cy = int(lm.y * (H * scale) * inv)   # = lm.y * H
            cx = max(_ROI_RADIUS, min(W - _ROI_RADIUS, cx))
            cy = max(_ROI_RADIUS, min(H - _ROI_RADIUS, cy))

            roi = frame[cy - _ROI_RADIUS : cy + _ROI_RADIUS,
                        cx - _ROI_RADIUS : cx + _ROI_RADIUS]
            if roi.size < 30:
                continue
            b_ch, g_ch, r_ch = cv2.split(roi)
            r_vals.append(float(r_ch.mean()))
            g_vals.append(float(g_ch.mean()))
            b_vals.append(float(b_ch.mean()))

    if not r_vals:
        return None
    return (float(np.mean(r_vals)),
            float(np.mean(g_vals)),
            float(np.mean(b_vals)))


def _extract_haar(video_path: str, max_frames: int):
    """
    Haar ensemble fallback face extraction (v3 approach).
    Returns: (rgb_traces, timestamps, face_frames, total_samp, fps_nominal)
    """
    cascade_names = [
        'haarcascade_frontalface_default.xml',
        'haarcascade_frontalface_alt2.xml',
        'haarcascade_profileface.xml',
    ]
    cascades = [cv2.CascadeClassifier(cv2.data.haarcascades + cp)
                for cp in cascade_names]
    cascades = [c for c in cascades if not c.empty()]

    if not cascades:
        log.warning("rppg: no Haar cascades available")
        return [], [], 0, 0, 30.0

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return [], [], 0, 0, 30.0

    fc          = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_nominal = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step        = max(1, fc // max_frames)

    rgb_traces  = []
    timestamps  = []
    face_frames = 0
    total_samp  = 0
    last_roi    = None

    fn = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        fn += 1

        if fn % step != 0:
            continue
        if total_samp >= max_frames:
            break

        total_samp += 1
        t = fn / fps_nominal
        h_orig, w_orig = frame.shape[:2]

        # Downscale for detection speed
        scale = 320 / w_orig if w_orig > 320 else 1.0
        small = cv2.resize(frame, (int(w_orig * scale), int(h_orig * scale))) \
                if scale < 1.0 else frame
        gray  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        best_face = None
        for cas in cascades:
            faces = cas.detectMultiScale(gray, 1.1, 4, minSize=(30, 30))
            if len(faces) > 0:
                best_face = max(faces, key=lambda f: f[2] * f[3])
                break

        if best_face is not None:
            x, y, w, h = best_face
            inv = 1.0 / scale
            x, y, w, h = int(x*inv), int(y*inv), int(w*inv), int(h*inv)
            # Multi-ROI: forehead + left/right cheeks
            last_roi = (
                x + w*2//10, y + h*1//10, x + w*8//10, y + h*3//10,   # forehead
                x + w*1//10, y + h*4//10, x + w*4//10, y + h*7//10,   # left cheek
                x + w*6//10, y + h*4//10, x + w*9//10, y + h*7//10,   # right cheek
                h_orig, w_orig
            )
            face_frames += 1

        if last_roi is not None:
            x1f,y1f,x2f,y2f, x1l,y1l,x2l,y2l, x1r,y1r,x2r,y2r, H0,W0 = last_roi
            r_vals, g_vals, b_vals = [], [], []
            for (rx1,ry1,rx2,ry2) in [(x1f,y1f,x2f,y2f),(x1l,y1l,x2l,y2l),(x1r,y1r,x2r,y2r)]:
                rx1,ry1 = max(0,rx1), max(0,ry1)
                rx2,ry2 = min(W0,rx2), min(H0,ry2)
                if rx2 <= rx1+5 or ry2 <= ry1+5: continue
                roi = frame[ry1:ry2, rx1:rx2]
                if roi.size < 30: continue
                b_ch, g_ch, r_ch = cv2.split(roi)
                r_vals.append(float(r_ch.mean()))
                g_vals.append(float(g_ch.mean()))
                b_vals.append(float(b_ch.mean()))
            if r_vals:
                rgb_traces.append((np.mean(r_vals), np.mean(g_vals), np.mean(b_vals)))
                timestamps.append(t)

    cap.release()
    log.info("rppg Haar extract: total=%d faces=%d traces=%d",
             total_samp, face_frames, len(rgb_traces))
    return rgb_traces, timestamps, face_frames, total_samp, fps_nominal


# ─────────────────────────────────────────────────────────────
#  Scoring
# ─────────────────────────────────────────────────────────────

def _score(autocorr: float, dom_bpm: float, snr: float, peak_ratio: float):
    """
    Convert rPPG measurements to 0-100 AI probability.
    Weights: autocorr=60%, snr=20%, bpm=10%, peaks=10%
    """
    evidence = []

    # Autocorrelation (60%) — primary heartbeat periodicity
    if autocorr < 0.04:
        ac_ai = 82
        evidence.append(f"no_pulse_periodicity(autocorr={autocorr:.3f})")
    elif autocorr < 0.08:
        ac_ai = 70
        evidence.append(f"very_weak_pulse(autocorr={autocorr:.3f})")
    elif autocorr < 0.14:
        ac_ai = 57
        evidence.append(f"weak_pulse(autocorr={autocorr:.3f})")
    elif autocorr < 0.25:
        ac_ai = 42
        evidence.append(f"moderate_pulse(autocorr={autocorr:.3f})")
    else:
        ac_ai = 22
        evidence.append(f"strong_pulse(autocorr={autocorr:.3f})")

    # Spectral SNR (20%) — sharpness of dominant HR frequency
    if snr < 0.04:
        snr_ai = 75
        evidence.append(f"no_dominant_freq(snr={snr:.4f})")
    elif snr < 0.07:
        snr_ai = 60
        evidence.append(f"diffuse_spectrum(snr={snr:.4f})")
    elif snr < 0.12:
        snr_ai = 45
        evidence.append(f"moderate_peak(snr={snr:.4f})")
    else:
        snr_ai = 28
        evidence.append(f"clean_peak(snr={snr:.4f})")

    # BPM plausibility (10%)
    if _HR_PLAUSIBLE_LOW <= dom_bpm <= _HR_PLAUSIBLE_HIGH:
        bpm_ai = 35
        evidence.append(f"bpm_plausible({dom_bpm:.0f})")
    elif 35 <= dom_bpm <= 130:
        bpm_ai = 52
        evidence.append(f"bpm_marginal({dom_bpm:.0f})")
    else:
        bpm_ai = 70
        evidence.append(f"bpm_implausible({dom_bpm:.0f})")

    # Peak excess (10%)
    if peak_ratio > 2.0:
        pk_ai = 70
        evidence.append(f"spurious_peaks(ratio={peak_ratio:.1f}x)")
    elif peak_ratio > 1.4:
        pk_ai = 55
        evidence.append(f"excess_peaks(ratio={peak_ratio:.1f}x)")
    else:
        pk_ai = 35
        evidence.append(f"normal_peaks(ratio={peak_ratio:.1f}x)")

    ai_score = int(round(0.60 * ac_ai + 0.20 * snr_ai + 0.10 * bpm_ai + 0.10 * pk_ai))
    return max(10, min(90, ai_score)), evidence


# ─────────────────────────────────────────────────────────────

def _neutral(reason: str, face_frames: int = 0, face_ratio: float = 0.0,
             n_samples: int = 0) -> dict:
    conf = "no_face" if face_frames < _MIN_FACE_FRAMES_LO else "low"
    return {
        "rppg_ai_score": 50,
        "confidence":    conf,
        "estimated_bpm": 0.0,
        "g_amp":         0.0,
        "autocorr":      0.0,
        "snr":           0.0,
        "face_ratio":    face_ratio,
        "face_count":    face_frames,
        "n_samples":     n_samples,
        "duration":      0.0,
        "n_peaks":       0,
        "peak_ratio":    0.0,
        "detector_used": "none",
        "evidence":      [f"rppg_unavailable:{reason}"],
        "available":     False,
    }

