# ============================================================
#  VeriFYD – detector.py  (lightweight, no torch)
#
#  Multi-signal AI video detection using OpenCV + NumPy only.
#  Analyzes: noise patterns, motion consistency, frequency
#  domain artifacts, temporal coherence, edge quality.
# ============================================================

import cv2
import numpy as np
import logging

log = logging.getLogger("verifyd.detector")


def _noise_score(gray: np.ndarray) -> float:
    """Estimate sensor noise via Laplacian variance. Real cameras have more."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _frequency_score(gray: np.ndarray) -> float:
    """
    AI-generated frames often lack high-frequency detail.
    Check ratio of high-freq to total energy in FFT.
    """
    f = np.fft.fft2(gray.astype(np.float32))
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)

    rows, cols = gray.shape
    crow, ccol = rows // 2, cols // 2

    # High-freq = outside center 25%
    radius = min(rows, cols) // 4
    mask = np.ones_like(magnitude, dtype=bool)
    y, x = np.ogrid[:rows, :cols]
    mask[(y - crow) ** 2 + (x - ccol) ** 2 <= radius ** 2] = False

    total = magnitude.sum() + 1e-10
    high_freq = magnitude[mask].sum()

    return float(high_freq / total)


def _edge_quality(gray: np.ndarray) -> float:
    """Real video has sharper, more varied edges. AI tends toward smoother."""
    edges = cv2.Canny(gray, 50, 150)
    return float(np.mean(edges))


def detect_ai(video_path: str) -> int:
    """
    Analyze video for AI-generation signals.
    Returns 0-100 where HIGH = likely AI, LOW = likely real.
    """
    log.info("Primary detector running on %s", video_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        log.error("Could not open video: %s", video_path)
        return 50

    noise_scores = []
    freq_scores = []
    edge_scores = []
    motion_scores = []
    temporal_diffs = []

    prev_gray = None
    prev_hist = None
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Sample every 5th frame, max 60 samples (300 frames)
        if frame_count % 5 != 0:
            continue
        if frame_count > 300:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # --- Noise analysis ---
        noise_scores.append(_noise_score(gray))

        # --- Frequency domain ---
        freq_scores.append(_frequency_score(gray))

        # --- Edge quality ---
        edge_scores.append(_edge_quality(gray))

        # --- Motion consistency ---
        if prev_gray is not None:
            diff = cv2.absdiff(gray, prev_gray)
            motion_scores.append(float(np.mean(diff)))

        # --- Temporal coherence (histogram stability) ---
        hist = cv2.calcHist([gray], [0], None, [64], [0, 256])
        hist = hist.flatten() / (hist.sum() + 1e-10)
        if prev_hist is not None:
            temporal_diffs.append(float(np.sum(np.abs(hist - prev_hist))))

        prev_gray = gray
        prev_hist = hist

    cap.release()

    if not noise_scores:
        log.warning("No frames analyzed")
        return 50

    # --- Compute signals ---
    avg_noise = np.mean(noise_scores)
    avg_freq = np.mean(freq_scores)
    avg_edge = np.mean(edge_scores)
    avg_motion = np.mean(motion_scores) if motion_scores else 0
    motion_var = np.var(motion_scores) if len(motion_scores) > 1 else 0
    temporal_jitter = np.mean(temporal_diffs) if temporal_diffs else 0

    log.info(
        "Signals: noise=%.1f freq=%.3f edge=%.1f motion=%.1f motion_var=%.2f temporal=%.4f",
        avg_noise, avg_freq, avg_edge, avg_motion, motion_var, temporal_jitter,
    )

    # --- Score (start at 50 = uncertain, adjust up/down) ---
    ai_score = 50

    # Low Laplacian noise → likely AI (real cameras have sensor noise)
    if avg_noise < 100:
        ai_score += 20
    elif avg_noise > 500:
        ai_score -= 20

    # Low high-frequency content → likely AI
    if avg_freq < 0.6:
        ai_score += 15
    elif avg_freq > 0.8:
        ai_score -= 15

    # Weak edges → likely AI
    if avg_edge < 10:
        ai_score += 10
    elif avg_edge > 30:
        ai_score -= 10

    # Very low or zero motion variance → synthetic
    if avg_motion < 2:
        ai_score += 10
    if motion_var < 0.5 and len(motion_scores) > 3:
        ai_score += 10

    # Too-smooth temporal transitions → AI
    if temporal_jitter < 0.01 and len(temporal_diffs) > 3:
        ai_score += 10
    elif temporal_jitter > 0.05:
        ai_score -= 10

    ai_score = max(0, min(100, ai_score))

    log.info("Primary AI score: %d", ai_score)
    return ai_score
