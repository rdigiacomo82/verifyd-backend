# ============================================================
#  VeriFYD — detector.py  (v4 — multi-content calibration)
#
#  v4 adds three new signals specifically targeting
#  cinematic/animal/nature AI-generated content that v3 missed:
#
#  NEW signals calibrated on Bus/Monkey/Moose AI test videos:
#  ┌──────────────────────────┬──────────────┬──────────────┐
#  │ Signal                   │ AI range     │ Real range   │
#  ├──────────────────────────┼──────────────┼──────────────┤
#  │ Saturation frame std     │ <5 or >25    │ 8–20         │
#  │ Background corner drift  │ <3 or >25    │ 5–18         │
#  │ Temporal flicker std     │ >4.0         │ <2.5         │
#  └──────────────────────────┴──────────────┴──────────────┘
#
#  Key findings from new test data:
#  - Dancing Monkey: sat_std=1.32 (frozen AI lighting)  → missed by v3
#  - Bus:            sat_std=32.7 (unstable AI color)   → missed by v3
#  - Moose:          flicker_std=7.2, bg_drift=45       → missed by v3
#  - Real videos:    sat_std=8-20, flicker_std<2.5
#
#  v4 also adds content-type auto-detection to apply the right
#  weights for action/waterslide vs cinematic/animal content.
#
#  Prior calibration data preserved:
#  - Waterslide/action: noise=3.58 AI vs 4.70 real
#  - Sharpness: AI=113, Real=230 (waterslide)
#  - Texture var: AI=3175, Real=1577 (waterslide)
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
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _texture_patch_variance(gray: np.ndarray) -> float:
    return float(np.var(gray.astype(np.float64)))


# ── NEW v4: Background corner drift ─────────────────────────
def _background_corner_drift(frames_gray: List[np.ndarray]) -> float:
    """
    Measures how much background corners shift frame-to-frame.

    AI cinematic videos either have:
      - Frozen backgrounds (drift < 3)  e.g. Dancing Monkey = 2.9
      - Warping/unstable backgrounds (drift > 25)  e.g. Bus = 33.9, Moose = 45.1

    Real videos have natural drift from camera movement (5-18).
    Note: fast-action real videos can have high drift too — use with content-type guard.
    """
    if len(frames_gray) < 2:
        return 0.0
    drifts = []
    for i in range(1, len(frames_gray)):
        f1 = frames_gray[i-1].astype(float)
        f2 = frames_gray[i].astype(float)
        h, w = f1.shape
        corners = [
            f1[:h//5, :w//5]      - f2[:h//5, :w//5],
            f1[:h//5, 4*w//5:]    - f2[:h//5, 4*w//5:],
            f1[4*h//5:, :w//5]    - f2[4*h//5:, :w//5],
            f1[4*h//5:, 4*w//5:]  - f2[4*h//5:, 4*w//5:],
        ]
        drifts.append(np.mean([np.mean(np.abs(c)) for c in corners]))
    return float(np.mean(drifts))


# ── NEW v4: Saturation frame std ────────────────────────────
def _saturation_frame_std(sat_scores: List[float]) -> float:
    """
    Std of per-frame mean saturation across the video.

    AI lighting is either:
      - Unnaturally stable (std < 5)  e.g. Dancing Monkey = 1.32
      - Unstable/flickering (std > 25)  e.g. Bus = 32.7

    Real videos have natural lighting variation: std 8-20.
    """
    if len(sat_scores) < 2:
        return 0.0
    return float(np.std(sat_scores))


# ── NEW v4: Temporal flicker std ────────────────────────────
def _temporal_flicker_std(frames_gray: List[np.ndarray]) -> float:
    """
    Std of frame-to-frame "middle minus neighbors" differences.
    Captures inconsistency in AI generation's temporal rendering.

    AI cinematic: high std e.g. Moose = 7.2
    Real static/slow content: low std e.g. Real President = 1.2
    Note: fast-action real videos also show high values — apply with content-type guard.
    """
    if len(frames_gray) < 3:
        return 0.0
    flicker_scores = []
    for i in range(1, len(frames_gray) - 1):
        prev_f = frames_gray[i-1].astype(float)
        curr_f = frames_gray[i].astype(float)
        next_f = frames_gray[i+1].astype(float)
        flicker = np.mean(np.abs(curr_f - (prev_f + next_f) / 2.0))
        flicker_scores.append(flicker)
    return float(np.std(flicker_scores))


# ── Main detection function ──────────────────────────────────

def detect_ai(video_path: str) -> int:
    """
    Analyze video for AI-generation signals.
    Returns 0-100 where HIGH = likely AI, LOW = likely real.

    v4 calibrated on:
    - Real videos: waterslide, presidential press conference
    - AI videos:   waterslide, presidential deepfake,
                   bus scene, dancing monkey, moose (cinematic)
    """
    log.info("Primary detector v4 running on %s", video_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        log.error("Could not open video: %s", video_path)
        return 50

    cap_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    log.info("Video dimensions: %dx%d  frames=%d  fps=%.1f",
             cap_w, cap_h, total_frames, fps)

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
    gray_buffer:             List[np.ndarray] = []  # rolling 10-frame window for flicker/residual
    v4_gray_frames:          List[np.ndarray] = []  # full sample list for v4 signals

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

        # Resize to fixed size for consistent cross-resolution analysis
        if max(cap_w, cap_h) > 512:
            scale = 512 / max(cap_w, cap_h)
            frame = cv2.resize(frame, (int(cap_w * scale), int(cap_h * scale)))

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

        # v4: keep every 6th frame (max 30) for background/flicker analysis
        if samples % 6 == 0 and len(v4_gray_frames) < 30:
            v4_gray_frames.append(gray)

    cap.release()

    if not noise_scores:
        log.warning("No frames analyzed")
        return 50

    # ── Aggregate signals ───────────────────────────────────
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
    avg_temporal_jitter = float(np.mean(temporal_diffs))         if temporal_diffs          else 0.0
    avg_flow_var        = float(np.mean(flow_regularity_scores)) if flow_regularity_scores  else 0.0
    avg_sharpness       = float(np.mean(sharpness_scores))       if sharpness_scores        else 0.0
    avg_texture_var     = float(np.mean(texture_var_scores))     if texture_var_scores      else 0.0

    pixel_flicker_cov   = _temporal_pixel_flicker(gray_buffer)
    residual_var_of_var = _inter_frame_residual_consistency(gray_buffer)

    # ── NEW v4 signals ──────────────────────────────────────
    # Use v4_gray_frames (full sample) not the rolling gray_buffer
    sat_frame_std  = _saturation_frame_std(saturation_scores)
    bg_drift       = _background_corner_drift(v4_gray_frames)
    flicker_std    = _temporal_flicker_std(v4_gray_frames)

    log.info(
        "Signals: noise=%.1f freq=%.3f edge=%.1f dct=%.3f "
        "grad=%.3f tex=%.1f corr=%.3f sat=%.1f "
        "motion=%.1f mvar=%.2f hist=%.4f flow=%.2f "
        "flicker_cov=%.3f rvov=%.2f sharp=%.1f texvar=%.1f "
        "[v4] sat_std=%.2f bg_drift=%.2f flicker_std=%.3f",
        avg_noise, avg_freq, avg_edge, avg_dct_grid,
        avg_grad_entropy, avg_tex_entropy, avg_color_corr, avg_saturation,
        avg_motion, motion_var, avg_temporal_jitter, avg_flow_var,
        pixel_flicker_cov, residual_var_of_var, avg_sharpness, avg_texture_var,
        sat_frame_std, bg_drift, flicker_std,
    )

    # ── Content type auto-detection ─────────────────────────
    # Used to guard v4 signals that overlap with fast-action real content
    is_action_content = (avg_motion > 8.0 and avg_edge > 3.0)
    is_static_content = (avg_motion < 3.0)
    log.info("Content type: %s (motion=%.1f edge=%.1f)",
             "action" if is_action_content else
             "static" if is_static_content else "cinematic",
             avg_motion, avg_edge)

    ai_score = 30.0   # base: lean real until AI signals accumulate

    # ═══════════════════════════════════════════════════════════
    # SCORING PHILOSOPHY (v4)
    # ai_score HIGH (>60) = AI video   → authenticity = 100 - ai_score = LOW
    # ai_score LOW  (<40) = Real video → authenticity = 100 - ai_score = HIGH
    # ═══════════════════════════════════════════════════════════

    # ── 1. Sensor noise ──────────────────────────────────────
    # High Laplacian variance can also come from high-res AI renders.
    # Reduce negative adjustments to avoid masking other AI signals.
    if avg_noise < 45:
        ai_score += 12
    elif avg_noise < 60:
        ai_score += 5
    elif avg_noise < 80:
        ai_score += 2
    if avg_noise > 500:
        ai_score -= 8    # reduced from -16: high-res AI also scores high here
    elif avg_noise > 150:
        ai_score -= 5    # reduced from -10
    elif avg_noise > 100:
        ai_score -= 2    # reduced from -5

    # ── 2. Frame sharpness ───────────────────────────────────
    # Reduced weight for cinematic content — high-res AI renders are very sharp.
    # Cap the "very sharp = real" negative to avoid drowning out other signals.
    sharpness_weight = 1.0 if is_action_content else 0.6
    if avg_sharpness < 80:
        ai_score += int(14 * sharpness_weight)
    elif avg_sharpness < 130:
        ai_score += int(8 * sharpness_weight)
    elif avg_sharpness < 160:
        ai_score += int(3 * sharpness_weight)
    elif avg_sharpness > 700:
        ai_score -= 10    # reduced from -20: high-res AI is also very sharp
    elif avg_sharpness > 400:
        ai_score -= 6     # reduced from -14
    elif avg_sharpness > 250:
        ai_score -= 3     # reduced from -6
    elif avg_sharpness > 200:
        ai_score -= 1     # reduced from -3

    # ── 3. Texture patch variance ────────────────────────────
    if avg_texture_var > 6500:
        ai_score += 10
    elif avg_texture_var > 5000:
        ai_score += 6
    elif avg_texture_var > 4000:
        ai_score += 2
    elif avg_texture_var < 1000:
        ai_score -= 4

    # ── 4. High-frequency content ────────────────────────────
    if avg_freq < 0.30:
        ai_score += 6
    elif avg_freq < 0.40:
        ai_score += 2

    # ── 5. Edge density ──────────────────────────────────────
    if avg_edge < 5 and avg_motion > 5.0:
        ai_score += 4
    elif avg_edge > 35:
        ai_score += 6

    # ── 6. DCT grid artifact ─────────────────────────────────
    _dct_reliable = (cap_w >= 480 and cap_h >= 480)
    if _dct_reliable:
        if avg_dct_grid > 8.0:
            ai_score += 10
        elif avg_dct_grid > 4.0:
            ai_score += 5
        elif avg_dct_grid > 2.0:
            ai_score += 2
        elif avg_dct_grid < 0.9:
            ai_score -= 3

    # ── 7. Gradient orientation entropy ──────────────────────
    if avg_grad_entropy < 3.5:
        ai_score += 6
    elif avg_grad_entropy < 4.2:
        ai_score += 2

    # ── 8. Local texture entropy ─────────────────────────────
    if avg_tex_entropy < 100:
        ai_score += 5
    elif avg_tex_entropy > 2000:
        ai_score += 8

    # ── 9. Color channel correlation ─────────────────────────
    if avg_color_corr > 0.92:
        ai_score += 4

    # ── 10. Saturation mean ──────────────────────────────────
    if avg_saturation > 160:
        ai_score += 8
    elif avg_saturation > 130:
        ai_score += 4
    elif avg_saturation > 100:
        ai_score += 1

    # ── 11. Motion ───────────────────────────────────────────
    if avg_motion < 0.5 and avg_temporal_jitter < 0.002:
        ai_score += 6

    # ── 12. Optical flow variance ────────────────────────────
    if len(flow_regularity_scores) > 3:
        if avg_flow_var < 3.0:
            ai_score += 6
        elif avg_flow_var < 8.0:
            ai_score += 3

    # ── 13. Temporal pixel flicker CoV ───────────────────────
    if pixel_flicker_cov > 1.5:
        ai_score += 5
    elif pixel_flicker_cov < 0.15 and len(gray_buffer) >= 5:
        ai_score += 3

    # ── 14. Inter-frame residual consistency ─────────────────
    if residual_var_of_var < 100 and len(gray_buffer) >= 5:
        ai_score += 4

    # ── 15. Edge temporal std ────────────────────────────────
    if len(edge_counts_temporal) > 3:
        edge_temporal_std = float(np.std(edge_counts_temporal))
        if edge_temporal_std < 6000:
            ai_score += 8
        elif edge_temporal_std < 10000:
            ai_score += 4
        log.info("Edge temporal std: %.0f", edge_temporal_std)

    # ════════════════════════════════════════════════════════
    # NEW v4 SIGNALS — cinematic / animal / nature AI detection
    # ════════════════════════════════════════════════════════

    # ── 16. Saturation frame std (NEW v4) ────────────────────
    # AI lighting is unnaturally stable (<5) OR chaotically unstable (>35).
    # Real video has natural variation from real-world lighting (8–20).
    # Calibrated: Dancing Monkey AI=1.32, Bus AI=32.7
    # GUARD: avg_saturation > 140 with low std = real broadcast studio, not AI render.
    # GUARD: action content with high sat_std = natural outdoor lighting variation (Real Slide2=32).
    _stable_is_broadcast = (sat_frame_std < 5.0 and avg_saturation > 140)
    _unstable_is_outdoor_action = (sat_frame_std > 22.0 and is_action_content)
    if _stable_is_broadcast:
        log.info("SAT_STD %.2f sat=%.0f → broadcast studio → no penalty", sat_frame_std, avg_saturation)
    elif _unstable_is_outdoor_action:
        log.info("SAT_STD %.2f → outdoor action lighting variation → no penalty", sat_frame_std)
    elif sat_frame_std < 3.0:
        # Frozen lighting on non-broadcast content — strong AI signal (Monkey)
        ai_score += 14
        log.info("SAT_STD %.2f → frozen AI lighting → +14", sat_frame_std)
    elif sat_frame_std < 6.0:
        # Very stable, non-broadcast — moderate AI signal
        ai_score += 8
        log.info("SAT_STD %.2f → stable lighting → +8", sat_frame_std)
    elif sat_frame_std > 35.0 and not is_action_content:
        # Wildly unstable non-action — AI color flickering (Bus-like on static content)
        ai_score += 10
        log.info("SAT_STD %.2f → unstable AI color → +10", sat_frame_std)
    elif 8.0 <= sat_frame_std <= 20.0:
        # Natural range — real video evidence
        ai_score -= 6
        log.info("SAT_STD %.2f → natural range → -6", sat_frame_std)

    # ── 17. Background corner drift (NEW v4) ─────────────────
    # AI backgrounds: frozen (drift<2, Monkey=1.3) OR wildly warping (Bus=25.7, Moose=45.7).
    # Real: natural drift from camera movement (5–18).
    # GUARD: static real content (broadcast/tripod) also has frozen bg — exempt if static+broadcast.
    _static_broadcast = (is_static_content and avg_saturation > 140)
    _bg_warp_threshold = 55.0 if is_action_content else 30.0
    if bg_drift < 2.0 and not _static_broadcast:
        # Frozen background — AI render (not a static broadcast camera)
        ai_score += 12
        log.info("BG_DRIFT %.2f → frozen AI bg → +12", bg_drift)
    elif bg_drift < 4.0 and not is_action_content and not _static_broadcast:
        ai_score += 6
        log.info("BG_DRIFT %.2f → near-frozen bg → +6", bg_drift)
    elif bg_drift > _bg_warp_threshold and not is_action_content:
        # Wildly warping non-action — AI generation artifact
        ai_score += 8
        log.info("BG_DRIFT %.2f → warping bg → +8", bg_drift)
    elif bg_drift > 20.0 and not is_action_content:
        ai_score += 4
        log.info("BG_DRIFT %.2f → unstable bg → +4", bg_drift)
    elif 5.0 <= bg_drift <= 18.0 and not is_action_content:
        ai_score -= 4
        log.info("BG_DRIFT %.2f → natural range → -4", bg_drift)

    # ── 18. Temporal flicker std (NEW v4) ────────────────────
    # AI videos have high flicker_std from inconsistent frame generation.
    # For action content, only flag extreme values — fast motion naturally has high flicker.
    # Calibrated: Moose AI=4.5 (action), Real President=0.9 (static)
    _flicker_high  = 20.0 if is_action_content else 6.0   # extreme = always AI
    _flicker_med   = 10.0 if is_action_content else 4.0
    _flicker_slight= 5.0  if is_action_content else 2.5
    if flicker_std > _flicker_high:
        ai_score += 14
        log.info("FLICKER_STD %.3f → extreme flicker → +14", flicker_std)
    elif flicker_std > _flicker_med:
        ai_score += 8
        log.info("FLICKER_STD %.3f → high flicker → +8", flicker_std)
    elif flicker_std > _flicker_slight and not is_action_content:
        ai_score += 3
        log.info("FLICKER_STD %.3f → slight flicker → +3", flicker_std)
    elif flicker_std < 0.8 and is_static_content:
        ai_score += 5
        log.info("FLICKER_STD %.3f → unnaturally smooth → +5", flicker_std)

    ai_score = max(0.0, min(100.0, ai_score))
    log.info("Primary AI score v4: %.0f  (sat_std=%.1f bg_drift=%.1f flicker_std=%.2f)",
             ai_score, sat_frame_std, bg_drift, flicker_std)
    return int(round(ai_score))
