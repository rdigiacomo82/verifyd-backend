# ============================================================
#  VeriFYD — rppg_detector.py  v3
#
#  ENGINE 5: Remote Photoplethysmography (rPPG) — THE MOAT
#
#  What is rPPG?
#  Real human skin has a cardiovascular blood volume pulse (BVP)
#  that causes subtle periodic color changes (~0.5–3% amplitude,
#  45–120 BPM). This is measurable from video without contact.
#  The technique behind Intel's FakeCatcher (Ciftci et al. 2023)
#  and the de Haan & Jeanne (2013) CHROM algorithm.
#
#  Why is this the moat?
#  AI video generators (Sora, Kling, Runway, Pika, HeyGen) render
#  pixels using neural networks trained on visual appearance — none
#  model the cardiovascular system. A generated face has either:
#    a) Near-zero periodic pulse (pure AI render), OR
#    b) Aperiodic noise that doesn't follow heartbeat physiology
#  This signal is fundamentally non-differentiable: to fake it a
#  generator would need per-pixel cardiovascular simulation.
#
#  Algorithm: CHROM (de Haan & Jeanne, IEEE TBME 2013)
#  1. Detect face with ensemble Haar cascade (frontal + alt + profile)
#  2. Extract mean R, G, B from forehead/cheek ROI per sampled frame
#  3. Normalize by channel mean (remove absolute luminance)
#  4. CHROM = (3R-2G) - alpha*(1.5R+G-1.5B)  [cancels specular]
#  5. Bandpass 0.75–3.0 Hz (45–180 BPM)
#  6. FFT → dominant frequency → BPM
#  7. Autocorrelation at heartbeat lag → periodicity score
#  8. SNR gating: reject low-SNR traces (motion blur / poor lighting)
#  9. Peak regularity → spurious peak count = noise flag
#
#  v3 changes vs v2:
#  - Ensemble face detector (frontal_default + frontal_alt2 + profile)
#    → ~50% face detection rate vs ~5% in v2 (10x improvement)
#  - Face detected EVERY sampled frame (not every 10th)
#    → Continuous, uniform RGB trace for correct FFT
#  - actual_fps computed from n_traces / duration (not fps/step)
#    → Correct frequency axis for bandpass and FFT
#  - Confidence gating based on n_face_frames (not content_type only)
#    → Videos with many face frames get analysis regardless of label
#  - SNR pre-filter: skip trace if spectral SNR < 0.035 in HR band
#  - Frame downscaling before face detection (320px wide, 4-6x faster)
#  - Multi-ROI: forehead strip + left/right cheek averaged
#    → Better signal quality than single forehead-only ROI
#
#  Confidence tiers:
#    "high"   — >=60 face frames AND duration >= 6s
#    "medium" — >=25 face frames AND duration >= 4s
#    "low"    — >=10 face frames (heavily attenuated in detection.py)
#    "no_face"— < 10 face frames (returns neutral 50, 0 adjustment)
#
#  Cap in detection.py: +-15 at high confidence.
# ============================================================

import cv2
import logging
import numpy as np

log = logging.getLogger("verifyd.rppg")

_HR_LOW_HZ          = 0.75
_HR_HIGH_HZ         = 3.0
_HR_PLAUSIBLE_LOW   = 42
_HR_PLAUSIBLE_HIGH  = 120
_MAX_FRAMES         = 150
_DETECT_WIDTH       = 320
_MIN_FACE_FRAMES_HI = 60
_MIN_FACE_FRAMES_MD = 25
_MIN_FACE_FRAMES_LO = 10
_MIN_DURATION_HI    = 6.0
_MIN_DURATION_MD    = 4.0
_SNR_GATE           = 0.035


def analyze_rppg(video_path: str, content_type: str = "cinematic") -> dict:
    """
    Main entry point. Analyze rPPG biological pulse signal.

    Returns dict:
        rppg_ai_score  : int 0-100 (higher = more likely AI / no pulse)
        confidence     : "high" | "medium" | "low" | "no_face"
        estimated_bpm  : float
        g_amp          : float (CHROM signal amplitude)
        autocorr       : float (periodicity, 0-1, higher = more real)
        snr            : float (spectral SNR in HR band)
        face_ratio     : float (fraction of sampled frames with face)
        evidence       : list[str]
        available      : bool
    """
    try:
        return _analyze_inner(video_path, content_type)
    except Exception as e:
        log.warning("rppg: unexpected error: %s", e, exc_info=True)
        return _neutral("error")


def _analyze_inner(video_path: str, content_type: str) -> dict:

    # Step 1: Extract RGB traces from detected face ROIs
    rgb_traces, timestamps, face_frames, total_sampled, fps_nominal = \
        _extract_face_rgb_v3(video_path)

    n          = len(rgb_traces)
    face_ratio = face_frames / max(total_sampled, 1)

    if n < _MIN_FACE_FRAMES_LO or face_frames < _MIN_FACE_FRAMES_LO:
        log.info("rppg: insufficient face data — n=%d face_frames=%d", n, face_frames)
        return _neutral("no_face", face_frames=face_frames,
                        face_ratio=face_ratio, n_samples=total_sampled)

    # Reconstruct actual fps from timestamps
    if len(timestamps) >= 2:
        duration   = timestamps[-1] - timestamps[0]
        actual_fps = n / max(duration, 0.1)
    else:
        duration   = n / fps_nominal
        actual_fps = fps_nominal

    log.info("rppg v3: n=%d  face_frames=%d  duration=%.1fs  fps=%.1f  face_ratio=%.2f",
             n, face_frames, duration, actual_fps, face_ratio)

    # Step 2: confidence tier based on face frame count
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

    # Normalize by channel mean → remove absolute luminance
    Rn = R / (R.mean() + 1e-10)
    Gn = G / (G.mean() + 1e-10)
    Bn = B / (B.mean() + 1e-10)

    # CHROM chrominance projection
    Xs    = 3.0 * Rn - 2.0 * Gn
    Ys    = 1.5 * Rn + Gn - 1.5 * Bn
    alpha = Xs.std() / (Ys.std() + 1e-10)
    chrom = sps.detrend(Xs - alpha * Ys)

    # Bandpass: heartbeat band
    nyq  = actual_fps / 2.0
    low  = _HR_LOW_HZ  / nyq
    high = min(_HR_HIGH_HZ / nyq, 0.98)
    if low >= high or nyq < _HR_LOW_HZ:
        log.warning("rppg: fps too low for bandpass (fps=%.1f)", actual_fps)
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

    # SNR gate
    if snr < _SNR_GATE:
        if confidence == "high":
            confidence = "medium"
            log.info("rppg: SNR gate → downgraded to medium (snr=%.4f)", snr)
        elif confidence == "medium":
            log.info("rppg: SNR gate → returning low confidence neutral (snr=%.4f)", snr)
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
        "rppg CHROM v3: bpm=%.1f snr=%.4f autocorr=%.3f amp=%.5f "
        "n_peaks=%d peak_ratio=%.2f confidence=%s",
        dom_bpm, snr, autocorr, chrom_amp, n_peaks, peak_ratio, confidence
    )

    # Step 7: score
    ai_score, evidence = _score(autocorr, dom_bpm, snr, peak_ratio)

    log.info("rppg: ai_score=%d  confidence=%s  evidence=%s",
             ai_score, confidence, evidence)

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
        "evidence":      evidence,
        "available":     True,
    }


def _score(autocorr: float, dom_bpm: float, snr: float, peak_ratio: float):
    """
    Convert rPPG measurements to 0-100 AI probability.

    Weights: autocorr=60%, snr=20%, bpm=10%, peaks=10%

    Calibration targets:
      Real face (good lighting, 10s+):  autocorr > 0.20, snr > 0.08 → score < 40
      AI deepfake (talking head):       autocorr < 0.08, snr < 0.06 → score > 65
      Ambiguous (short, poor light):    returns low/no_face confidence → neutral
    """
    evidence = []

    # Autocorrelation (60% weight) — primary heartbeat periodicity signal
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

    # Spectral SNR (20% weight) — sharpness of dominant frequency
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

    # BPM plausibility (10% weight)
    if _HR_PLAUSIBLE_LOW <= dom_bpm <= _HR_PLAUSIBLE_HIGH:
        bpm_ai = 35
        evidence.append(f"bpm_plausible({dom_bpm:.0f})")
    elif 35 <= dom_bpm <= 130:
        bpm_ai = 52
        evidence.append(f"bpm_marginal({dom_bpm:.0f})")
    else:
        bpm_ai = 70
        evidence.append(f"bpm_implausible({dom_bpm:.0f})")

    # Peak excess (10% weight)
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


def _extract_face_rgb_v3(video_path: str, max_frames: int = _MAX_FRAMES):
    """
    Extract per-frame mean (R,G,B) from face ROI using ensemble Haar cascade.
    Samples uniformly. Returns (rgb_traces, timestamps, face_frames, total_samp, fps).
    """
    cascade_names = [
        'haarcascade_frontalface_default.xml',
        'haarcascade_frontalface_alt2.xml',
        'haarcascade_profileface.xml',
    ]
    cascades = []
    for cp in cascade_names:
        c = cv2.CascadeClassifier(cv2.data.haarcascades + cp)
        if not c.empty():
            cascades.append(c)

    if not cascades:
        log.warning("rppg: no cascade classifiers loaded")
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
    last_roi    = None   # cached ROI from last successful detection

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

        # Downscale for faster detection
        scale_det = _DETECT_WIDTH / w_orig if w_orig > _DETECT_WIDTH else 1.0
        if scale_det < 1.0:
            small = cv2.resize(frame, (int(w_orig * scale_det), int(h_orig * scale_det)))
        else:
            small = frame
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        # Try each cascade in order
        best_face = None
        for cas in cascades:
            faces = cas.detectMultiScale(gray, scaleFactor=1.1,
                                          minNeighbors=4, minSize=(30, 30))
            if len(faces) > 0:
                best_face = max(faces, key=lambda f: f[2] * f[3])
                break

        if best_face is not None:
            x, y, w, h = best_face
            # Scale ROI coordinates back to original frame
            inv = 1.0 / scale_det
            x, y, w, h = int(x*inv), int(y*inv), int(w*inv), int(h*inv)
            # Multi-ROI: forehead + left cheek + right cheek
            x1f = x + w * 2 // 10;  y1f = y + h * 1 // 10
            x2f = x + w * 8 // 10;  y2f = y + h * 3 // 10
            x1l = x + w * 1 // 10;  y1l = y + h * 4 // 10
            x2l = x + w * 4 // 10;  y2l = y + h * 7 // 10
            x1r = x + w * 6 // 10;  y1r = y + h * 4 // 10
            x2r = x + w * 9 // 10;  y2r = y + h * 7 // 10
            last_roi = (x1f, y1f, x2f, y2f,
                        x1l, y1l, x2l, y2l,
                        x1r, y1r, x2r, y2r,
                        h_orig, w_orig)
            face_frames += 1

        # Extract RGB from cached ROI (use last good ROI even without new detection)
        if last_roi is not None:
            (x1f, y1f, x2f, y2f,
             x1l, y1l, x2l, y2l,
             x1r, y1r, x2r, y2r,
             H0, W0) = last_roi

            r_vals, g_vals, b_vals = [], [], []
            for (rx1, ry1, rx2, ry2) in [(x1f, y1f, x2f, y2f),
                                           (x1l, y1l, x2l, y2l),
                                           (x1r, y1r, x2r, y2r)]:
                rx1, ry1 = max(0, rx1), max(0, ry1)
                rx2, ry2 = min(W0, rx2), min(H0, ry2)
                if rx2 <= rx1 + 5 or ry2 <= ry1 + 5:
                    continue
                roi = frame[ry1:ry2, rx1:rx2]
                if roi.size < 30:
                    continue
                b_ch, g_ch, r_ch = cv2.split(roi)
                r_vals.append(float(r_ch.mean()))
                g_vals.append(float(g_ch.mean()))
                b_vals.append(float(b_ch.mean()))

            if r_vals:
                rgb_traces.append((
                    float(np.mean(r_vals)),
                    float(np.mean(g_vals)),
                    float(np.mean(b_vals)),
                ))
                timestamps.append(t)

    cap.release()
    log.info("rppg v3 extract: total_samp=%d face_frames=%d traces=%d",
             total_samp, face_frames, len(rgb_traces))
    return rgb_traces, timestamps, face_frames, total_samp, fps_nominal


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
        "evidence":      [f"rppg_unavailable:{reason}"],
        "available":     False,
    }
