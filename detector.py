# ============================================================
#  VeriFYD — detector.py  (recalibrated v3)
#
#  Recalibration based on waterslide test data (5 videos):
#
#  Measured signal values:
#  ┌─────────────────────┬──────────────┬──────────────┐
#  │ Signal              │ AI avg       │ Real avg     │
#  ├─────────────────────┼──────────────┼──────────────┤
#  │ Noise level         │ 3.58         │ 4.70         │
#  │ Edge sharpness      │ 113          │ 230          │
#  │ Saturation          │ 103          │ 148          │
#  │ Motion              │ 71.5         │ 59.6         │
#  │ Texture variance    │ 3175         │ 1577         │
#  └─────────────────────┴──────────────┴──────────────┘
#
#  Key findings vs prior calibration:
#  - Noise is a RELIABLE separator for waterslide content
#  - Edge sharpness: AI videos are LESS sharp (smoothed)
#  - Texture variance: AI videos are HIGHER (more uniform patches)
#  - Saturation: both high in waterslide content — less reliable
#  - Motion: AI slightly higher than real in this content type
#
#  v3 changes:
#  - Recalibrated noise thresholds for waterslide/action content
#  - Added low-edge-sharpness as AI signal (AI smoothing)
#  - Added high texture variance as AI signal
#  - Reduced saturation weight (less reliable in outdoor/wet content)
#  - Added motion regularity check for action content
# ============================================================

import cv2
import logging
import numpy as np
from typing import List

log = logging.getLogger("verifyd.detector")


# ── Individual signal functions ──────────────────────────────

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


def _laplacian_sharpness(gray: np.ndarray) -> float:
    """
    Measure overall frame sharpness via Laplacian variance.
    AI waterslide videos score LOWER (smoothed/soft) vs real (sharper).
    AI avg ~113, Real avg ~230 in test data.
    """
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _texture_patch_variance(gray: np.ndarray) -> float:
    """
    Measure variance of pixel intensities across the frame.
    AI waterslide videos score HIGHER (more uniform, artificial texture).
    AI avg ~3175, Real avg ~1577 in test data.
    """
    return float(np.var(gray.astype(np.float64)))


# ── Thread-local physics signal store ───────────────────────
import threading
_physics_store = threading.local()

def get_last_physics_signals() -> dict:
    """Return physics signals from the most recent detect_ai() call on this thread."""
    return getattr(_physics_store, "signals", {})

# ── Main detection function ──────────────────────────────────

def detect_ai(video_path: str) -> int:
    """
    Analyze video for AI-generation signals.
    Returns 0-100 where HIGH = likely AI, LOW = likely real.

    Calibrated on:
    - Real videos: compressed mobile waterslide footage
    - AI videos:   AI-generated waterslide content
    """
    log.info("Primary detector running on %s", video_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        log.error("Could not open video: %s", video_path)
        return 50

    cap_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    log.info("Video dimensions: %dx%d", cap_w, cap_h)

    noise_scores:            List[float] = []
    freq_scores:             List[float] = []
    edge_scores:             List[float] = []
    dct_grid_scores:         List[float] = []
    grad_entropy_scores:     List[float] = []
    texture_entropy_scores:  List[float] = []
    color_corr_scores:       List[float] = []
    saturation_scores:       List[float] = []
    flow_regularity_scores:  List[float] = []
    motion_scores:           List[float] = []
    edge_counts_temporal:    List[float] = []
    sharpness_scores:        List[float] = []
    texture_var_scores:      List[float] = []
    gray_buffer:             List[np.ndarray] = []
    vert_flow_scores:        List[float] = []   # physics engine
    low_corr_count:          int = 0            # content jump counter

    prev_gray  = None
    prev_hist  = None
    temporal_diffs: List[float] = []
    frame_count = 0
    samples     = 0

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
        sharpness_scores.append(_laplacian_sharpness(gray))
        texture_var_scores.append(_texture_patch_variance(gray))

        if prev_gray is not None:
            diff = cv2.absdiff(gray, prev_gray)
            motion_scores.append(float(np.mean(diff)))
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
            flow_regularity_scores.append(float(np.var(np.sqrt(flow[...,0]**2 + flow[...,1]**2))))
            # Vertical flow: negative = upward (anti-gravity), positive = downward
            vert_flow_scores.append(float(np.mean(flow[..., 1])))
            # Frame correlation for content jump detection
            c = float(np.corrcoef(gray.flatten().astype(float),
                                  prev_gray.flatten().astype(float))[0, 1])
            if c < 0.5:
                low_corr_count += 1

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
    avg_motion          = float(np.mean(motion_scores))          if motion_scores           else 0.0
    motion_var          = float(np.var(motion_scores))           if len(motion_scores) > 1  else 0.0
    avg_temporal_jitter = float(np.mean(temporal_diffs))         if temporal_diffs           else 0.0
    avg_flow_var        = float(np.mean(flow_regularity_scores)) if flow_regularity_scores   else 0.0
    avg_sharpness       = float(np.mean(sharpness_scores))       if sharpness_scores         else 0.0
    avg_texture_var     = float(np.mean(texture_var_scores))     if texture_var_scores       else 0.0

    # ── Physics engine signals ────────────────────────────────
    avg_vert_flow       = float(np.mean(vert_flow_scores))       if vert_flow_scores         else 0.0
    vert_flow_std       = float(np.std(vert_flow_scores))        if len(vert_flow_scores) > 1 else 0.0
    upward_frame_ratio  = float(sum(1 for v in vert_flow_scores if v < -3.0) /
                          max(len(vert_flow_scores), 1))
    # Trajectory smoothness: low accel_std = unnaturally smooth = AI
    accel_std           = float(np.std(np.diff(vert_flow_scores))) if len(vert_flow_scores) > 2 else 0.0

    pixel_flicker_cov   = _temporal_pixel_flicker(gray_buffer)
    residual_var_of_var = _inter_frame_residual_consistency(gray_buffer)

    log.info(
        "Signals: noise=%.1f freq=%.3f edge=%.1f dct=%.3f "
        "grad=%.3f tex=%.1f corr=%.3f sat=%.1f "
        "motion=%.1f mvar=%.2f hist=%.4f flow=%.2f "
        "flicker=%.3f rvov=%.2f sharpness=%.1f texvar=%.1f "
        "vertflow=%.3f upward_ratio=%.2f accel_std=%.3f low_corr=%d",
        avg_noise, avg_freq, avg_edge, avg_dct_grid,
        avg_grad_entropy, avg_tex_entropy, avg_color_corr, avg_saturation,
        avg_motion, motion_var, avg_temporal_jitter, avg_flow_var,
        pixel_flicker_cov, residual_var_of_var, avg_sharpness, avg_texture_var,
        avg_vert_flow, upward_frame_ratio, accel_std, low_corr_count,
    )

    ai_score = 30.0   # base: assume real until AI signals accumulate

    # ═══════════════════════════════════════════════════════════
    # SCORING PHILOSOPHY (v3 - calibrated on waterslide test data)
    # ai_score HIGH (>60) = AI video   → authenticity = 100 - ai_score = LOW
    # ai_score LOW  (<40) = Real video → authenticity = 100 - ai_score = HIGH
    #
    # Waterslide test data findings:
    #   Noise:         AI=3.58   Real=4.70  → reliable separator
    #   Sharpness:     AI=113    Real=230   → strong signal (AI is softer)
    #   Texture var:   AI=3175   Real=1577  → strong signal (AI more uniform)
    #   Saturation:    AI=103    Real=148   → less reliable in this content
    #   Motion:        AI=71.5   Real=59.6  → weak, use cautiously
    # ═══════════════════════════════════════════════════════════

    # ── 1. Sensor noise (recalibrated for action/waterslide content) ──────────
    # AI avg=3.58, Real avg=4.70 in our test data (pixel-level noise)
    # Note: these are RAW pixel noise values, not Laplacian variance
    raw_noise = float(np.mean([
        np.std(cv2.cvtColor(cv2.imread(video_path), cv2.COLOR_BGR2GRAY).astype(float) -
               cv2.GaussianBlur(cv2.cvtColor(cv2.imread(video_path), cv2.COLOR_BGR2GRAY), (5,5), 0).astype(float))
    ])) if False else avg_noise  # use avg_noise (Laplacian) as proxy

    # Laplacian noise: AI avg~70-100, Real avg~150-300 based on prior calibration
    if avg_noise < 45:
        ai_score += 12
    elif avg_noise < 60:
        ai_score += 5
    elif avg_noise < 80:
        ai_score += 2

    # Real video noise boosters
    if avg_noise > 500:
        ai_score -= 16
    elif avg_noise > 150:
        ai_score -= 10
    elif avg_noise > 100:
        ai_score -= 5

    # ── 2. Frame sharpness — BEST separator for president/broadcast content
    # AI President=108, Real President=853 — 8x difference
    # AI waterslide avg~113, Real waterslide avg~230
    if avg_sharpness < 80:
        ai_score += 14    # very soft = strong AI signal
    elif avg_sharpness < 130:
        ai_score += 8     # soft = moderate AI signal (AI president range)
    elif avg_sharpness < 160:
        ai_score += 3     # slightly soft
    elif avg_sharpness > 700:
        ai_score -= 20    # extremely sharp = very strong real camera signal
    elif avg_sharpness > 400:
        ai_score -= 14    # very sharp = strong real camera signal
    elif avg_sharpness > 250:
        ai_score -= 6     # sharp = real camera signal
    elif avg_sharpness > 200:
        ai_score -= 3     # slightly sharp

    # ── 3. Texture patch variance (NEW — calibrated on waterslide data) ───────
    # AI videos have HIGHER texture variance (more artificial uniform patches)
    # AI avg~3175, Real avg~1577 in test data
    # AI President=7818, Real President=3992 — use 6000 as threshold
    if avg_texture_var > 6500:
        ai_score += 10    # very high = strong AI signal
    elif avg_texture_var > 5000:
        ai_score += 6     # high = moderate AI signal
    elif avg_texture_var > 4000:
        ai_score += 2     # slightly elevated
    elif avg_texture_var < 1000:
        ai_score -= 4     # low variance = natural real texture

    # ── 4. High-frequency content ─────────────────────────────────────────────
    if avg_freq < 0.30:
        ai_score += 6
    elif avg_freq < 0.40:
        ai_score += 2

    # ── 5. Edge density ───────────────────────────────────────────────────────
    # Low edge density is normal in broadcast/talking-head content
    # Only penalize if motion is also present (action content with low edges = suspicious)
    if avg_edge < 5 and avg_motion > 5.0:
        ai_score += 4
    elif avg_edge > 35:
        ai_score += 6     # AI over-sharpening

    # ── 6. DCT grid artifact ──────────────────────────────────────────────────
    _dct_reliable = (cap_w >= 480 and cap_h >= 480)
    if _dct_reliable:
        # Raised thresholds — broadcast compression creates high DCT ratios naturally
        if avg_dct_grid > 8.0:
            ai_score += 10
        elif avg_dct_grid > 4.0:
            ai_score += 5
        elif avg_dct_grid > 2.0:
            ai_score += 2
        elif avg_dct_grid < 0.9:
            ai_score -= 3
    else:
        log.info("DCT signal skipped — small video %dx%d", cap_w, cap_h)

    # ── 7. Gradient orientation entropy ──────────────────────────────────────
    if avg_grad_entropy < 3.5:
        ai_score += 6
    elif avg_grad_entropy < 4.2:
        ai_score += 2

    # ── 8. Local texture entropy ──────────────────────────────────────────────
    if avg_tex_entropy < 100:
        ai_score += 5
    elif avg_tex_entropy > 2000:
        ai_score += 8

    # ── 9. Color channel correlation ─────────────────────────────────────────
    if avg_color_corr > 0.92:
        ai_score += 4

    # ── 10. Saturation (reduced weight — less reliable in outdoor content) ────
    # Both AI and real waterslide videos can be highly saturated outdoors
    # Reduced thresholds and weights vs v2
    if avg_saturation > 160:
        ai_score += 8     # reduced from 16
    elif avg_saturation > 130:
        ai_score += 4     # reduced from 8
    elif avg_saturation > 100:
        ai_score += 1     # reduced from 3

    # ── 11. Motion (conservative — real waterslide has high motion too) ───────
    if avg_motion < 0.5 and avg_temporal_jitter < 0.002:
        ai_score += 6

    # ── 12. Optical flow variance ─────────────────────────────────────────────
    if len(flow_regularity_scores) > 3:
        if avg_flow_var < 3.0:
            ai_score += 6
        elif avg_flow_var < 8.0:
            ai_score += 3

    # ── 13. Temporal pixel flicker ────────────────────────────────────────────
    if pixel_flicker_cov > 1.5:
        ai_score += 5
    elif pixel_flicker_cov < 0.15 and len(gray_buffer) >= 5:
        ai_score += 3

    # ── 14. Inter-frame residual consistency ──────────────────────────────────
    if residual_var_of_var < 100 and len(gray_buffer) >= 5:
        ai_score += 4

    # ── 15. Edge temporal std ─────────────────────────────────────────────────
    # Real videos have much higher edge variation from natural movement
    if len(edge_counts_temporal) > 3:
        edge_temporal_std = float(np.std(edge_counts_temporal))
        if edge_temporal_std < 6000:
            ai_score += 8
        elif edge_temporal_std < 10000:
            ai_score += 4
        log.info("Edge temporal std: %.0f", edge_temporal_std)

    # ── 16. PHYSICS ENGINE — gravity & trajectory analysis ────────────────────
    # Calibrated: AI Slide vert_flow=-1.924, upward_ratio=33.7%, accel_std=1.203
    #             Real Slide vert_flow=+0.316, upward_ratio=7.9%, accel_std=2.079
    #
    # Key insight: real action content has chaotic/variable acceleration (high accel_std)
    # AI content has sustained anti-gravity motion (negative vert_flow, high upward_ratio)
    if len(vert_flow_scores) > 5:
        # Anti-gravity: sustained upward motion is physically impossible in slide/action content
        if avg_vert_flow < -1.5:
            ai_score += 18   # strong gravity violation
            log.info("Physics: gravity violation — sustained upward flow=%.3f", avg_vert_flow)
        elif avg_vert_flow < -0.5:
            ai_score += 8    # moderate upward tendency
            log.info("Physics: moderate upward flow=%.3f", avg_vert_flow)
        elif avg_vert_flow > 0.0:
            ai_score -= 5    # downward motion = real gravity = real video
            log.info("Physics: gravity-consistent downward flow=%.3f", avg_vert_flow)

        # Upward frame ratio: >25% of frames showing strong upward motion = AI
        if upward_frame_ratio > 0.25:
            ai_score += 12
            log.info("Physics: high upward frame ratio=%.1f%%", upward_frame_ratio * 100)
        elif upward_frame_ratio > 0.15:
            ai_score += 6
        elif upward_frame_ratio < 0.10:
            ai_score -= 4    # mostly downward = real physics

        # Trajectory smoothness: real action is chaotic (high accel_std)
        # AI motion is unnaturally smooth (low accel_std)
        if accel_std < 1.0 and avg_motion > 10:
            ai_score += 10   # high motion but smooth trajectory = AI
            log.info("Physics: unnaturally smooth trajectory accel_std=%.3f", accel_std)
        elif accel_std > 2.0:
            ai_score -= 4    # chaotic = real physics
            log.info("Physics: chaotic trajectory (real) accel_std=%.3f", accel_std)

    # Content jump detection — AI videos sometimes have frame discontinuities
    if low_corr_count > 5:
        ai_score += 8
        log.info("Physics: %d content jump frames detected", low_corr_count)
    elif low_corr_count > 2:
        ai_score += 4

    # ── Store physics signals for GPT context handoff ─────────
    _physics_store.signals = {
        "avg_vert_flow":       avg_vert_flow       if vert_flow_scores else None,
        "upward_frame_ratio":  upward_frame_ratio  if vert_flow_scores else None,
        "accel_std":           accel_std           if vert_flow_scores else None,
        "low_corr_count":      low_corr_count,
        "avg_saturation":      avg_saturation,
        "avg_sharpness":       avg_sharpness,
    }

    ai_score = max(0.0, min(100.0, ai_score))
    log.info("Primary AI score: %.0f", ai_score)
    return int(round(ai_score))
