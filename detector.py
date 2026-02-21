# ============================================================
#  VeriFYD — detector.py  (recalibrated v2)
#
#  Recalibration based on real test data:
#
#  Key finding: The original thresholds assumed high-quality
#  real footage. Compressed mobile real video (.MOV) has LOW
#  noise/texture because compression removes it. Meanwhile
#  high-quality AI renders can score HIGH on those signals.
#
#  Fix: Signals that fired wrong are reweighted or inverted.
#  - noise_score: not reliable alone — removed as primary signal
#  - texture_entropy: now treated as bidirectional signal
#  - optical_flow: real video shows MORE variance (handheld)
#  - saturation: AI renders tend to be oversaturated (confirmed)
#  - temporal_flicker: AI higher (confirmed)
#  - color_corr: both low here, weight reduced
# ============================================================

import cv2
import logging
import numpy as np
from typing import List

log = logging.getLogger("verifyd.detector")


# ── Individual signal functions (unchanged) ──────────────────

def _noise_score(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _frequency_score(gray: np.ndarray) -> float:
    f = np.fft.fft2(gray.astype(np.float32))
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)
    h, w = gray.shape
    cy, cx = h // 2, w // 2
    r = min(h, w) // 4
    total = magnitude.sum() + 1e-10
    y, x = np.ogrid[:h, :w]
    mask = ((y - cy)**2 + (x - cx)**2) > r**2
    high_freq = magnitude[mask].sum()
    return float(high_freq / total)


def _dct_grid_artifact(gray: np.ndarray, block_size: int = 8) -> float:
    h, w = gray.shape
    bh = (h // block_size) * block_size
    bw = (w // block_size) * block_size
    if bh < block_size * 2 or bw < block_size * 2:
        return 1.0
    crop = gray[:bh, :bw].astype(np.float32)
    border_vals, interior_vals = [], []
    for row in range(0, bh, block_size):
        for col in range(0, bw, block_size):
            block = crop[row:row+block_size, col:col+block_size]
            dct_block = cv2.dct(block)
            interior = np.abs(dct_block[1:block_size-1, 1:block_size-1])
            border   = np.concatenate([
                np.abs(dct_block[0, :]).ravel(),
                np.abs(dct_block[-1, :]).ravel(),
                np.abs(dct_block[:, 0]).ravel(),
                np.abs(dct_block[:, -1]).ravel(),
            ])
            border_vals.append(border.mean())
            interior_vals.append(interior.mean() + 1e-10)
    ratio = np.mean(border_vals) / (np.mean(interior_vals) + 1e-10)
    return float(ratio)


def _edge_quality(gray: np.ndarray) -> float:
    edges = cv2.Canny(gray, 50, 150)
    return float(np.mean(edges > 0) * 100)


def _gradient_orientation_entropy(gray: np.ndarray) -> float:
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    angles = np.arctan2(gy, gx)
    hist, _ = np.histogram(angles, bins=36, range=(-np.pi, np.pi))
    hist = hist / (hist.sum() + 1e-10)
    entropy = -np.sum(hist * np.log2(hist + 1e-10))
    return float(entropy)


def _local_texture_entropy(gray: np.ndarray, patch_size: int = 16) -> float:
    h, w = gray.shape
    variances = []
    for y in range(0, h - patch_size, patch_size):
        for x in range(0, w - patch_size, patch_size):
            patch = gray[y:y+patch_size, x:x+patch_size].astype(np.uint8)
            variances.append(float(cv2.Laplacian(patch, cv2.CV_64F).var()))
    if not variances:
        return 0.0
    return float(np.std(variances))


def _color_channel_noise_correlation(frame_bgr: np.ndarray) -> float:
    b, g, r = cv2.split(frame_bgr.astype(np.float32))
    def residual(ch):
        blur = cv2.GaussianBlur(ch, (5, 5), 0)
        return (ch - blur).ravel()
    rb, rg, rr = residual(b), residual(g), residual(r)
    corr_bg = float(np.corrcoef(rb, rg)[0, 1])
    corr_br = float(np.corrcoef(rb, rr)[0, 1])
    corr_gr = float(np.corrcoef(rg, rr)[0, 1])
    avg_abs = (abs(corr_bg) + abs(corr_br) + abs(corr_gr)) / 3.0
    return float(1.0 - avg_abs)


def _saturation_mean(frame_bgr: np.ndarray) -> float:
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    return float(np.mean(hsv[:, :, 1]))


def _optical_flow_regularity(prev_gray: np.ndarray, curr_gray: np.ndarray) -> float:
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
    )
    magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
    return float(np.var(magnitude))


def _temporal_pixel_flicker(frames_gray: List[np.ndarray]) -> float:
    if len(frames_gray) < 3:
        return 0.0
    h, w = frames_gray[0].shape
    stack = np.stack([f.astype(np.float32) for f in frames_gray], axis=0)
    per_pixel_std = np.std(stack, axis=0)
    mean_std = per_pixel_std.mean() + 1e-10
    cov = per_pixel_std.std() / mean_std
    return float(cov)


def _inter_frame_residual_consistency(frames_gray: List[np.ndarray]) -> float:
    if len(frames_gray) < 3:
        return 0.0
    diffs = [
        cv2.absdiff(frames_gray[i], frames_gray[i-1]).astype(np.float32)
        for i in range(1, len(frames_gray))
    ]
    variances = [float(d.var()) for d in diffs]
    return float(np.var(variances))


# ── Main detection function ──────────────────────────────────

def detect_ai(video_path: str) -> int:
    """
    Analyze video for AI-generation signals.
    Returns 0–100 where HIGH = likely AI, LOW = likely real.

    Calibrated on:
    - Real video: compressed mobile .MOV (low noise/texture due to compression)
    - AI video:   high-quality AI render (crisp, oversaturated, unnatural flow)
    """
    log.info("Primary detector running on %s", video_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        log.error("Could not open video: %s", video_path)
        return 50

    noise_scores:       List[float] = []
    freq_scores:        List[float] = []
    edge_scores:        List[float] = []
    dct_grid_scores:    List[float] = []
    grad_entropy_scores:List[float] = []
    texture_entropy_scores: List[float] = []
    color_corr_scores:  List[float] = []
    saturation_scores:  List[float] = []
    flow_regularity_scores: List[float] = []
    motion_scores:      List[float] = []
    gray_buffer:        List[np.ndarray] = []

    prev_gray = None
    prev_hist = None
    temporal_diffs: List[float] = []
    frame_count = 0
    samples = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        if frame_count % 5 != 0:
            continue
        if samples >= 60:
            break
        samples += 1

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        noise_scores.append(_noise_score(gray))
        freq_scores.append(_frequency_score(gray))
        edge_scores.append(_edge_quality(gray))
        dct_grid_scores.append(_dct_grid_artifact(gray))
        grad_entropy_scores.append(_gradient_orientation_entropy(gray))
        texture_entropy_scores.append(_local_texture_entropy(gray))
        color_corr_scores.append(_color_channel_noise_correlation(frame))
        saturation_scores.append(_saturation_mean(frame))

        if prev_gray is not None:
            diff = cv2.absdiff(gray, prev_gray)
            motion_scores.append(float(np.mean(diff)))
            flow_regularity_scores.append(_optical_flow_regularity(prev_gray, gray))

        hist = cv2.calcHist([gray], [0], None, [64], [0, 256])
        hist = hist.flatten() / (hist.sum() + 1e-10)
        if prev_hist is not None:
            temporal_diffs.append(float(np.sum(np.abs(hist - prev_hist))))

        prev_gray = gray.copy()
        prev_hist = hist

        gray_buffer.append(gray)
        if len(gray_buffer) > 10:
            gray_buffer.pop(0)

    cap.release()

    if not noise_scores:
        log.warning("No frames analyzed")
        return 50

    avg_noise           = float(np.mean(noise_scores))
    avg_freq            = float(np.mean(freq_scores))
    avg_edge            = float(np.mean(edge_scores))
    avg_dct_grid        = float(np.mean(dct_grid_scores))
    avg_grad_entropy    = float(np.mean(grad_entropy_scores))
    avg_tex_entropy     = float(np.mean(texture_entropy_scores))
    avg_color_corr      = float(np.mean(color_corr_scores))
    avg_saturation      = float(np.mean(saturation_scores))
    avg_motion          = float(np.mean(motion_scores)) if motion_scores else 0.0
    motion_var          = float(np.var(motion_scores))  if len(motion_scores) > 1 else 0.0
    avg_temporal_jitter = float(np.mean(temporal_diffs)) if temporal_diffs else 0.0
    avg_flow_var        = float(np.mean(flow_regularity_scores)) if flow_regularity_scores else 0.0

    pixel_flicker_cov   = _temporal_pixel_flicker(gray_buffer)
    residual_var_of_var = _inter_frame_residual_consistency(gray_buffer)

    log.info(
        "Signals: noise=%.1f freq=%.3f edge=%.1f dct=%.3f "
        "grad=%.3f tex=%.1f corr=%.3f sat=%.1f "
        "motion=%.1f mvar=%.2f hist=%.4f flow=%.2f "
        "flicker=%.3f rvov=%.2f",
        avg_noise, avg_freq, avg_edge, avg_dct_grid,
        avg_grad_entropy, avg_tex_entropy, avg_color_corr, avg_saturation,
        avg_motion, motion_var, avg_temporal_jitter, avg_flow_var,
        pixel_flicker_cov, residual_var_of_var,
    )

    ai_score = 50.0

    # ── 1. Sensor noise ───────────────────────────────────────────────────────
    # RECALIBRATED: Compressed mobile real video can be LOW (200-500).
    # Very high noise (>1500) is unusual and can indicate AI over-sharpening.
    # Only penalize clearly AI-clean video (< 80).
    if avg_noise < 80:
        ai_score += 15        # extremely clean = likely AI
    elif avg_noise < 150:
        ai_score += 5
    # Note: high noise no longer strongly indicates real — compression confounds

    # ── 2. High-frequency content ─────────────────────────────────────────────
    # Both videos scored 0.41-0.45, close together — weak signal.
    # Reduced weight significantly.
    if avg_freq < 0.30:
        ai_score += 8
    elif avg_freq < 0.40:
        ai_score += 3
    elif avg_freq > 0.80:
        ai_score -= 5

    # ── 3. Edge density ───────────────────────────────────────────────────────
    # RECALIBRATED: AI video had 36, real had 17.
    # Higher edge density from AI rendering/sharpening is suspicious.
    # Low edge from compressed real video no longer means AI.
    if avg_edge < 5:
        ai_score += 6         # extremely blurry — could be AI or very compressed
    elif avg_edge > 30:
        ai_score += 8         # unnaturally sharp — AI over-sharpening
    elif avg_edge > 20:
        ai_score += 3

    # ── 4. DCT grid artifact ──────────────────────────────────────────────────
    if avg_dct_grid > 1.8:
        ai_score += 12
    elif avg_dct_grid > 1.4:
        ai_score += 6
    elif avg_dct_grid < 0.9:
        ai_score -= 4

    # ── 5. Gradient orientation entropy ──────────────────────────────────────
    # Both videos 4.94-5.06 — nearly identical, very weak signal here.
    # Reduced weight.
    if avg_grad_entropy < 3.5:
        ai_score += 8
    elif avg_grad_entropy < 4.2:
        ai_score += 3
    elif avg_grad_entropy > 5.2:
        ai_score -= 5

    # ── 6. Local texture entropy ──────────────────────────────────────────────
    # RECALIBRATED: AI=3217, Real=416.
    # AI renders have EXTREME texture variance (very sharp regions next to
    # very smooth ones = hallucination artifacts). Real compressed video
    # is uniformly mediocre. Both directions now scored.
    if avg_tex_entropy < 100:
        ai_score += 6         # very uniform = possibly AI or very compressed
    elif avg_tex_entropy < 200:
        ai_score += 2
    elif avg_tex_entropy > 2000:
        ai_score += 10        # extreme texture contrast = AI hallucination artifact
    elif avg_tex_entropy > 800:
        ai_score += 5

    # ── 7. Color channel noise correlation ───────────────────────────────────
    # Both videos scored ~0.06-0.09 — both low. Weak discriminator.
    # Only fire on clearly extreme values.
    if avg_color_corr > 0.80:
        ai_score += 6
    elif avg_color_corr < 0.20:
        ai_score -= 4

    # ── 8. Saturation ─────────────────────────────────────────────────────────
    # CONFIRMED SIGNAL: AI=139, Real=67. AI renders are oversaturated.
    # Boosted weight — this is a reliable discriminator.
    if avg_saturation > 120:
        ai_score += 16        # strongly oversaturated = AI
    elif avg_saturation > 90:
        ai_score += 8
    elif avg_saturation > 70:
        ai_score += 3
    elif avg_saturation < 30:
        ai_score -= 5         # very desaturated real footage

    # ── 9. Motion amount ─────────────────────────────────────────────────────
    if avg_motion < 2:
        ai_score += 8
    if motion_var < 0.5 and len(motion_scores) > 3:
        ai_score += 6

    # ── 10. Histogram temporal jitter ─────────────────────────────────────────
    if avg_temporal_jitter < 0.008 and len(temporal_diffs) > 3:
        ai_score += 8
    elif avg_temporal_jitter > 0.06:
        ai_score -= 6

    # ── 11. Optical flow variance ─────────────────────────────────────────────
    # RECALIBRATED: AI=13.1, Real=32.0 (handheld).
    # BUT: real static/tripod shots also have low flow — can't penalize hard.
    # Only fire if also combined with other AI signals (handled by weight).
    if len(flow_regularity_scores) > 3:
        if avg_flow_var < 3.0:
            ai_score += 8     # essentially zero motion — very suspicious
        elif avg_flow_var < 8.0:
            ai_score += 4     # reduced from 12 — real static video can score here
        elif avg_flow_var > 100.0:
            ai_score -= 10    # strong real handheld motion signal

    # ── 12. Temporal pixel flicker ────────────────────────────────────────────
    # Original assumption: AI flicker > real. Test data shows real can also
    # be high (1.04 real vs 0.81 AI on these samples). Use cautiously.
    # Only fire on extreme values to avoid false positives.
    if pixel_flicker_cov > 1.5:
        ai_score += 6     # extremely incoherent flicker
    elif pixel_flicker_cov < 0.15 and len(gray_buffer) >= 5:
        ai_score += 4     # suspiciously uniform = AI temporal smoothing

    # ── 13. Inter-frame residual consistency ──────────────────────────────────
    if residual_var_of_var < 100 and len(gray_buffer) >= 5:
        ai_score += 4
    elif residual_var_of_var > 50000:
        ai_score -= 4

    # ── AUTHENTICITY BOOSTERS ─────────────────────────────────────────────────
    # These signals reward clear real-camera characteristics.
    # They push genuine real videos UP (lower ai_score) without pulling AI up.

    # Booster A: Inter-frame residual variance-of-variance (rvov)
    # Real cameras produce irregular frame-to-frame residuals due to natural
    # sensor noise, micro-motion, and lighting flicker.
    # Real: 10k–60k   AI (static): ~5k   AI (complex): can be high too
    # Only use as a strong negative signal for clearly real-looking values.
    if residual_var_of_var > 40000:
        ai_score -= 12   # very chaotic inter-frame residuals = real camera
    elif residual_var_of_var > 10000:
        ai_score -= 7    # moderately chaotic = likely real

    # Booster B: Motion variance (mvar)
    # AI Cat showed extremely HIGH mvar (147) — erratic AI motion.
    # Real handheld: moderate and consistent (9–50).
    # Already partially handled above; add a downward push for natural range.
    if motion_var > 100:
        ai_score += 10   # erratic motion = AI artifact
    elif 0 < motion_var < 15:
        ai_score -= 10   # very consistent, low motion = real static shot
    elif motion_var < 60:
        ai_score -= 4    # natural moderate motion range = real

    # Booster C: Histogram temporal jitter (scene content change)
    # Real videos: richer scene change (0.15–0.25+)
    # AI videos:   smoother transitions, lower jitter (0.05–0.12)
    if avg_temporal_jitter > 0.18:
        ai_score -= 10   # active real-world scene content change
    elif avg_temporal_jitter > 0.12:
        ai_score -= 5
    elif avg_temporal_jitter < 0.05 and len(temporal_diffs) > 3:
        ai_score += 6    # suspiciously smooth = AI temporal blending

    ai_score = max(0.0, min(100.0, ai_score))
    log.info("Primary AI score: %.0f", ai_score)
    return int(round(ai_score))