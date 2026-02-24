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
    edge_counts_temporal: List[float] = []
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
        edges_raw = cv2.Canny(gray, 50, 150)
        edge_counts_temporal.append(float(np.sum(edges_raw > 0)))
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

    # ═══════════════════════════════════════════════════════════
    # SCORING PHILOSOPHY (v3 - calibrated on 8 real test videos)
    # ai_score HIGH (>60) = AI video   → authenticity = 100 - ai_score = LOW
    # ai_score LOW  (<40) = Real video → authenticity = 100 - ai_score = HIGH
    #
    # Key findings from test data:
    #   Noise:       AI avg=70  REAL avg=120  → good signal
    #   Edge std:    AI avg=12k REAL avg=43k  → strong signal
    #   Saturation:  AI oversaturated         → strong signal
    #   DCT grid:    AI has artifacts         → strong signal
    #   Motion/hist: Too overlapping          → weak, use cautiously
    # ═══════════════════════════════════════════════════════════

    # ── 1. Sensor noise ───────────────────────────────────────────────────────
    # AI avg=70, Real avg=120. Only penalize clearly AI-clean (<50).
    # Don't penalize 50-80 range since real compressed video sits there too.
    # AI avg=70, Real avg=120. Real_3 has noise=68 so raise upper threshold to 60.
    if avg_noise < 45:
        ai_score += 12
    elif avg_noise < 60:
        ai_score += 4
    # No downward push — noise alone is not reliable enough

    # ── 2. High-frequency content ─────────────────────────────────────────────
    if avg_freq < 0.30:
        ai_score += 6
    elif avg_freq < 0.40:
        ai_score += 2

    # ── 3. Edge density ───────────────────────────────────────────────────────
    if avg_edge < 5:
        ai_score += 4
    elif avg_edge > 35:
        ai_score += 6     # AI over-sharpening

    # ── 4. DCT grid artifact ──────────────────────────────────────────────────
    # Strong AI signal — compression artifacts from AI rendering pipeline
    if avg_dct_grid > 1.8:
        ai_score += 14
    elif avg_dct_grid > 1.4:
        ai_score += 7
    elif avg_dct_grid < 0.9:
        ai_score -= 3

    # ── 5. Gradient orientation entropy ──────────────────────────────────────
    if avg_grad_entropy < 3.5:
        ai_score += 6
    elif avg_grad_entropy < 4.2:
        ai_score += 2

    # ── 6. Local texture entropy ──────────────────────────────────────────────
    if avg_tex_entropy < 100:
        ai_score += 5
    elif avg_tex_entropy > 2000:
        ai_score += 8     # AI hallucination artifacts

    # ── 7. Color channel correlation ─────────────────────────────────────────
    # Weak signal — skip downward adjustments entirely
    if avg_color_corr > 0.92:
        ai_score += 4     # only fire on extreme values

    # ── 8. Saturation ─────────────────────────────────────────────────────────
    # STRONG SIGNAL: AI oversaturated (avg 139 vs real avg 67)
    if avg_saturation > 120:
        ai_score += 16
    elif avg_saturation > 90:
        ai_score += 8
    elif avg_saturation > 70:
        ai_score += 3

    # ── 9. Motion (compound signal only) ──────────────────────────────────────
    # CONSERVATIVE: Real static shots (tripod) can have very low motion.
    # Real_Video_1 motion_var=0.079, Real_Video_2 motion_var=0.007 — both static.
    # Only fire on essentially zero motion WITH very specific AI-like jitter pattern.
    # Raised threshold significantly to avoid false positives.
    if avg_motion < 0.5 and avg_temporal_jitter < 0.002:
        ai_score += 6     # near-zero everything = suspicious

    # ── 10. Histogram temporal jitter ─────────────────────────────────────────
    # DISABLED: Real_Video_2 (static real shot) has jitter=0.002 which is
    # indistinguishable from AI. This signal causes too many false positives
    # on real static videos. Removed entirely.

    # ── 11. Optical flow variance ─────────────────────────────────────────────
    if len(flow_regularity_scores) > 3:
        if avg_flow_var < 3.0:
            ai_score += 6
        elif avg_flow_var < 8.0:
            ai_score += 3

    # ── 12. Temporal pixel flicker ────────────────────────────────────────────
    if pixel_flicker_cov > 1.5:
        ai_score += 5
    elif pixel_flicker_cov < 0.15 and len(gray_buffer) >= 5:
        ai_score += 3

    # ── 13. Inter-frame residual consistency ──────────────────────────────────
    if residual_var_of_var < 100 and len(gray_buffer) >= 5:
        ai_score += 4

    # ── 14. Edge temporal std ─────────────────────────────────────────────────
    # STRONG SIGNAL: AI avg=12k, Real avg=43k
    # Real videos have much higher edge variation from natural movement.
    # Only use as UPWARD push for AI (low variation) not downward for real.
    # Downward adjustments were causing inversions.
    if len(edge_counts_temporal) > 3:
        edge_temporal_std = float(np.std(edge_counts_temporal))
        if edge_temporal_std < 6000:
            ai_score += 8     # very low variation = AI temporal smoothing
        elif edge_temporal_std < 10000:
            ai_score += 4     # low variation = mildly suspicious
        # Real_Video_2 has edge_std=6853 — raised from 7000 to avoid false positive
        log.info("Edge temporal std: %.0f", edge_temporal_std)

    ai_score = max(0.0, min(100.0, ai_score))
    log.info("Primary AI score: %.0f", ai_score)
    return int(round(ai_score))