# ============================================================
# VeriFYD – detection.py  (v4 STRONG SEPARATION MODEL)
# ============================================================

import cv2
import numpy as np
import logging

log = logging.getLogger("verifyd.detector")

def clamp(v, lo=0, hi=100):
    return max(lo, min(hi, v))

# ------------------------------------------------------------
def run_detection(video_path):
# ------------------------------------------------------------
    """
    Returns:
        score (0-100)
        label ("REAL","UNDETERMINED","AI")
    """

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 50, "UNDETERMINED"

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30

    max_frames = int(fps * 10)

    noise_vals = []
    motion_vals = []
    edge_vals = []
    freq_vals = []
    hist_diffs = []

    prev_gray = None
    prev_hist = None
    frame_count = 0

    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        if frame_count % 2 != 0:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # noise
        noise = cv2.Laplacian(gray, cv2.CV_64F).var()
        noise_vals.append(noise)

        # motion
        if prev_gray is not None:
            diff = cv2.absdiff(gray, prev_gray)
            motion_vals.append(np.mean(diff))

        # edges
        edges = cv2.Canny(gray, 50, 150)
        edge_vals.append(np.mean(edges))

        # frequency
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        mag = np.abs(fshift)

        total = mag.sum() + 1e-6
        center = mag[mag.shape[0]//3:mag.shape[0]*2//3,
                     mag.shape[1]//3:mag.shape[1]*2//3].sum()

        freq_ratio = (total-center)/total
        freq_vals.append(freq_ratio)

        # histogram jitter
        hist = cv2.calcHist([gray],[0],None,[32],[0,256])
        hist = hist.flatten()/(hist.sum()+1e-6)

        if prev_hist is not None:
            hist_diffs.append(np.sum(np.abs(hist-prev_hist)))

        prev_gray = gray
        prev_hist = hist

    cap.release()

    if not noise_vals:
        return 50, "UNDETERMINED"

    avg_noise = np.mean(noise_vals)
    avg_motion = np.mean(motion_vals) if motion_vals else 0
    motion_var = np.var(motion_vals) if len(motion_vals)>1 else 0
    avg_edge = np.mean(edge_vals)
    avg_freq = np.mean(freq_vals)
    avg_hist = np.mean(hist_diffs) if hist_diffs else 0

    log.info(
        "noise %.1f motion %.1f var %.2f edge %.1f freq %.3f hist %.4f",
        avg_noise, avg_motion, motion_var, avg_edge, avg_freq, avg_hist
    )

    # =========================================================
    # SCORING (AGGRESSIVE)
    # =========================================================

    score = 80   # assume real first

    # ---------- NOISE ----------
    if avg_noise < 60:
        score -= 35   # AI clean
    elif avg_noise < 120:
        score -= 20
    elif avg_noise > 350:
        score += 10   # real grain

    # ---------- MOTION ----------
    if motion_var < 0.3:
        score -= 30
    elif motion_var < 1:
        score -= 15
    else:
        score += 10

    # ---------- EDGE DETAIL ----------
    if avg_edge < 6:
        score -= 25
    elif avg_edge > 22:
        score += 8

    # ---------- FREQUENCY ----------
    if avg_freq < 0.55:
        score -= 25
    elif avg_freq > 0.75:
        score += 5

    # ---------- TEMPORAL JITTER ----------
    if avg_hist < 0.008:
        score -= 25
    elif avg_hist > 0.05:
        score += 5

    score = clamp(score)

    # =========================================================
    # FINAL LABEL
    # =========================================================
    if score >= 85:
        label = "REAL"
    elif score >= 60:
        label = "UNDETERMINED"
    else:
        label = "AI"

    log.info("FINAL SCORE %d → %s", score, label)

    return int(score), label








