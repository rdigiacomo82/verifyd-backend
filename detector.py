# ============================================================
#  VeriFYD — detector.py  (v6 — gorilla/uniform-render signals)
#
#  v6 adds two new signals targeting AI animal videos where
#  the render is uniformly sharp (no real camera depth-of-field):
#
#  NEW signals calibrated on Gorilla AI test video:
#  ┌──────────────────────────┬──────────────┬──────────────┐
#  │ Signal                   │ AI range     │ Real range   │
#  ├──────────────────────────┼──────────────┼──────────────┤
#  │ Quad sharpness CoV       │ <0.50        │ 0.58–0.92    │
#  │ sat_std (low-sat guard)  │ <5, mean<100 │ varies       │
#  └──────────────────────────┴──────────────┴──────────────┘
#
#  Root cause of Gorilla miss:
#  - sat_std=3.86 (frozen AI lighting) was guarded by is_action
#  - quad_sharpness_cov=0.365 (uniform render focus, no lens)
#  - Fix: sat_std frozen guard now checks sat_mean<100 as separate
#    path from the broadcast guard (sat_mean>140)
#
#  v5 signals preserved (fg_bg_ratio, motion_sync, hue_entropy).
#  v4 signals preserved (sat_std, bg_drift, flicker_std).
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


# ── NEW v5: Foreground/background sharpness ratio ───────────
def _fg_bg_sharpness_ratio(frames_gray: List[np.ndarray]) -> float:
    """
    Compare sharpness of center (subject) vs top edge (background).
    AI renders have extreme ratios: subject is rendered in perfect
    focus while background is either equally sharp (no lens) or
    synthetically blurred. Real cameras have natural depth falloff.

    Calibrated:
      AI Bus=2145, AI Moose=1525
      Real Slide=372, Real President=335
    Threshold: >1000 = strong AI signal
    """
    if not frames_gray:
        return 0.0
    ratios = []
    for g in frames_gray:
        fh, fw = g.shape
        fg = cv2.Laplacian(g[fh//4:3*fh//4, fw//4:3*fw//4], cv2.CV_64F).var()
        bg = cv2.Laplacian(g[:fh//5, :], cv2.CV_64F).var()
        ratios.append(float(fg / (bg + 1.0)))
    return float(np.mean(ratios))


# ── NEW v5: Crowd / left-right motion sync ───────────────────
def _motion_sync_score(frames_gray: List[np.ndarray]) -> float:
    """
    Measure how synchronised left vs right halves of frame move.
    In real scenes with multiple people, each person's motion is
    independent — left and right differ significantly.
    In AI crowd scenes, all "people" move in scripted lockstep —
    left and right halves show near-identical motion magnitudes.

    Returns the mean absolute fractional difference between
    left-half and right-half motion (LOW = lockstep = AI).

    Calibrated:
      AI Bus=0.088, AI Moose=0.046
      Real Slide1=0.142, Real President=0.113
    Threshold: <0.08 = strong AI signal (lockstep crowd)
    """
    if len(frames_gray) < 2:
        return 1.0
    sync_scores = []
    for i in range(1, len(frames_gray)):
        g1, g2 = frames_gray[i-1], frames_gray[i]
        fw = g1.shape[1]
        left  = np.mean(np.abs(g1[:, :fw//2].astype(float) - g2[:, :fw//2].astype(float)))
        right = np.mean(np.abs(g1[:, fw//2:].astype(float) - g2[:, fw//2:].astype(float)))
        diff_ratio = abs(left - right) / (left + right + 1e-10)
        sync_scores.append(float(diff_ratio))
    return float(np.mean(sync_scores))


# ── NEW v5: Hue entropy (color palette diversity) ────────────
def _hue_entropy(frames_bgr: List[np.ndarray]) -> float:
    """
    Measure diversity of color hues across sampled frames.
    AI videos are generated with a curated/limited color palette
    that shows up as low hue entropy.
    Real cameras capture the full messy spectrum of a real scene.

    Calibrated:
      AI Bus=1.62, AI Monkey=1.29  (low — limited palette)
      Real Slide1=2.90, Real President=2.76  (high — natural)
    Threshold: <2.0 = strong AI signal
    """
    if not frames_bgr:
        return 4.0
    entropies = []
    for f in frames_bgr:
        hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
        h = cv2.calcHist([hsv], [0], None, [36], [0, 180])
        h = h / (h.sum() + 1e-10)
        ent = -float(np.sum(h * np.log2(h + 1e-10)))
        entropies.append(ent)
    return float(np.mean(entropies))


# ── NEW v6: Quadrant sharpness uniformity ───────────────────
def _quad_sharpness_cov(frames_gray: List[np.ndarray]) -> float:
    """
    Measure how uniformly sharp all quadrants of the frame are.
    Real cameras have natural depth-of-field variation — some parts
    of the frame are sharper than others (foreground vs background).
    AI renders every region with equal computational sharpness,
    resulting in unnaturally low CoV across quadrants.

    Calibrated:
      AI Gorilla=0.365, AI Bus=0.433, AI Moose=0.498
      Real Slide1=0.916, Real Slide2=0.581, Real President=0.725
    Threshold: <0.50 = strong AI signal
    """
    if not frames_gray:
        return 1.0
    covs = []
    for g in frames_gray:
        fh, fw = g.shape
        qs = [
            cv2.Laplacian(g[fh//2*a:fh//2*(a+1), fw//2*b:fw//2*(b+1)], cv2.CV_64F).var()
            for a in range(2) for b in range(2)
        ]
        covs.append(float(np.std(qs) / (np.mean(qs) + 1.0)))
    return float(np.mean(covs))


# ── Main detection function ──────────────────────────────────

def detect_ai(video_path: str) -> int:
    """
    Analyze video for AI-generation signals.
    Returns 0-100 where HIGH = likely AI, LOW = likely real.

    v4 calibrated on:
    - Real videos: waterslide, presidential press conference
    - AI videos:   waterslide, presidential deepfake,
                   bus scene, dancing monkey, moose (cinematic)
    v5 adds crowd/reaction, FG/BG depth, and hue entropy signals.
    """
    log.info("Primary detector v5 running on %s", video_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        log.error("Could not open video: %s", video_path)
        return 50

    cap_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    video_duration = total_frames / fps if fps > 0 else 0.0
    _is_short_clip = video_duration < 4.0   # guard: unreliable temporal signals on short clips
    log.info("Video dimensions: %dx%d  frames=%d  fps=%.1f  duration=%.1fs",
             cap_w, cap_h, total_frames, fps, video_duration)

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
    v5_bgr_frames:           List[np.ndarray] = []  # full sample BGR frames for v5 hue entropy
    flow_dir_scores:         List[float] = []       # flow direction entropy (crowd behavior)

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
            # Flow direction entropy — low = unnaturally uniform crowd motion (AI)
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            _, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            ang_hist, _ = np.histogram(ang.flatten(), bins=8, range=(0, 2 * np.pi))
            ang_hist = ang_hist / (ang_hist.sum() + 1e-10)
            dir_entropy = float(-np.sum(ang_hist * np.log2(ang_hist + 1e-10)))
            flow_dir_scores.append(dir_entropy)

        hist = cv2.calcHist([gray], [0], None, [64], [0, 256])
        hist = hist.flatten() / (hist.sum() + 1e-10)
        if prev_hist is not None:
            temporal_diffs.append(float(np.sum(np.abs(hist - prev_hist))))

        prev_gray = gray.copy()
        prev_hist = hist

        gray_buffer.append(gray)
        if len(gray_buffer) > 10:
            gray_buffer.pop(0)

        # v4/v5: keep every 6th frame (max 30) for bg/flicker/crowd/depth/hue analysis
        if samples % 6 == 0 and len(v4_gray_frames) < 30:
            v4_gray_frames.append(gray)
            v5_bgr_frames.append(frame.copy())

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

    # ── NEW v4b: Crowd/scene behavioral signals ─────────────
    avg_flow_dir_entropy = float(np.mean(flow_dir_scores)) if flow_dir_scores else 3.0
    # Peak-to-mean motion ratio — real emergencies have sudden spikes
    if motion_scores:
        arr = np.array(motion_scores)
        peak_to_mean_ratio = float(np.max(arr) / (np.mean(arr) + 0.001))
    else:
        peak_to_mean_ratio = 1.0

    # ── NEW v5/v6 signals ────────────────────────────────────
    fg_bg_ratio  = _fg_bg_sharpness_ratio(v4_gray_frames)
    motion_sync  = _motion_sync_score(v4_gray_frames)
    hue_entropy  = _hue_entropy(v5_bgr_frames)
    quad_cov     = _quad_sharpness_cov(v4_gray_frames)  # v6

    log.info(
        "Signals: noise=%.1f freq=%.3f edge=%.1f dct=%.3f "
        "grad=%.3f tex=%.1f corr=%.3f sat=%.1f "
        "motion=%.1f mvar=%.2f hist=%.4f flow=%.2f "
        "flicker_cov=%.3f rvov=%.2f sharp=%.1f texvar=%.1f "
        "[v4] sat_std=%.2f bg_drift=%.2f flicker_std=%.3f "
        "[v5/v6] fg_bg=%.0f motion_sync=%.3f hue_ent=%.3f quad_cov=%.3f",
        avg_noise, avg_freq, avg_edge, avg_dct_grid,
        avg_grad_entropy, avg_tex_entropy, avg_color_corr, avg_saturation,
        avg_motion, motion_var, avg_temporal_jitter, avg_flow_var,
        pixel_flicker_cov, residual_var_of_var, avg_sharpness, avg_texture_var,
        sat_frame_std, bg_drift, flicker_std,
        fg_bg_ratio, motion_sync, hue_entropy, quad_cov,
    )

    # ── Skin ratio — fraction of pixels in skin-tone HSV range ─
    # Used by single_subject detection to confirm a person is the main subject.
    # Computed from v5_bgr_frames (sampled color frames).
    skin_ratio = 0.0
    if v5_bgr_frames:
        _skin_counts = []
        for _sf in v5_bgr_frames:
            _hsv = cv2.cvtColor(_sf, cv2.COLOR_BGR2HSV)
            _mask = cv2.inRange(_hsv, np.array([0, 20, 70]), np.array([20, 255, 255]))
            _skin_counts.append(_mask.sum() / (_mask.shape[0] * _mask.shape[1] * 255))
        skin_ratio = float(np.mean(_skin_counts))

    # ── Content type auto-detection ─────────────────────────
    # Used to guard v4 signals that overlap with fast-action real content
    # Edge threshold raised from 3 → 12: real action (sports, crowds, waterfalls)
    # has rich edge content (20-40+). AI animal/nature videos have high motion but
    # sparse edges (dark fur, smooth backgrounds) — edge=9 on Gorilla AI, 24 on Real_Video_2.
    is_action_content = (avg_motion > 8.0 and avg_edge > 12.0)
    is_static_content = (avg_motion < 3.0)

    # ── Selfie / talking-head detection (v8) ────────────────
    # Portrait phone selfie: vertical aspect ratio + low motion + moderate sharpness.
    # These videos have naturally stable bg, stable lighting, low sync —
    # signals built for AI crowd detection must be suppressed for them.
    # SKIN GATE: a real selfie always has a person — require skin_ratio > 0.05.
    # This prevents AI animal/nature portrait videos (moose, gorilla in portrait mode)
    # from falsely triggering the selfie guard and receiving a -12 real bonus.
    # Calibrated: AI Moose Snow skin_ratio=0.000 → selfie=False ✓
    _is_portrait        = (cap_h > cap_w * 1.5)
    _is_selfie_content  = (
        _is_portrait                    # vertical phone video
        and avg_motion < 6.0            # mostly static subject
        and avg_edge < 20.0             # not busy/crowded scene
        and avg_sharpness > 100         # real camera sharpness (not AI-smooth 50-80)
        and skin_ratio > 0.05           # must have a person — gates out AI animal videos
    )

    # ── Talking-head / active portrait detection (v9) ────────
    # A real person talking/moving on camera: portrait + active motion + rich edges.
    # Key insight: single person moving as one unit → HIGH motion_sync (not AI signal).
    # Skin-tone dominant → LOW sat_std (not AI signal — it's just skin/neutral clothing).
    # Edge density stays consistent (single subject indoors) → LOW edge_std (not AI signal).
    # These are REAL characteristics that the AI-crowd signals incorrectly penalize.
    _talking_head_skin  = False
    if _is_portrait and avg_motion > 8.0 and avg_edge > 18.0:
        # Quick skin tone check using center of frame
        try:
            sample_frame = gray_buffer[len(gray_buffer)//2]
            # Load original color frame for skin check
            _th_cap = cv2.VideoCapture(video_path)
            _th_cap.set(cv2.CAP_PROP_POS_FRAMES, len(gray_buffer)//2 * max(1, total_frames // len(gray_buffer)))
            _th_ret, _th_frame = _th_cap.read()
            _th_cap.release()
            if _th_ret:
                _th_hsv = cv2.cvtColor(_th_frame, cv2.COLOR_BGR2HSV)
                _th_skin = cv2.inRange(_th_hsv, np.array([0,20,70]), np.array([20,255,255]))
                _talking_head_skin = (_th_skin.sum() / (_th_skin.shape[0]*_th_skin.shape[1]*255)) > 0.06
        except Exception:
            pass
    _is_talking_head = (
        _is_portrait
        and avg_motion > 8.0            # active — person is moving/talking
        and avg_edge > 18.0             # rich edge content (hair, clothing, face detail)
        and not _is_selfie_content      # not a static hold selfie
    )

    # ── Single-subject landscape person detection (v10) ─────────
    # A real person filmed in landscape orientation — phone/camera focused on
    # one subject. Same signals misfire as talking_head but no portrait flag.
    # High skin ratio is the key discriminator — confirms a person is the subject.
    # fg_bg < 200 confirms no extreme AI depth separation (AI renders subject
    # at 1000-2000x background sharpness; real cameras produce 4-400x).
    _is_single_subject = (
        not _is_portrait                # landscape orientation only
        and skin_ratio > 0.10           # significant skin visible — person is main subject
        and avg_motion > 2.0            # some movement
        and avg_motion < 20.0           # not a wild action/crowd scene
        and fg_bg_ratio < 200           # no extreme AI depth separation
        and not is_action_content       # not a crowd/event scene
    )

    log.info("Content type: %s (motion=%.1f edge=%.1f portrait=%s selfie=%s talking_head=%s skin=%.3f single_subject=%s)",
             "action"          if is_action_content   else
             "talking_head"    if _is_talking_head    else
             "single_subject"  if _is_single_subject  else
             "selfie"          if _is_selfie_content  else
             "static"          if is_static_content   else "cinematic",
             avg_motion, avg_edge, _is_portrait, _is_selfie_content, _is_talking_head, skin_ratio, _is_single_subject)

    ai_score = 30.0   # base: lean real until AI signals accumulate

    # ═══════════════════════════════════════════════════════════
    # SCORING PHILOSOPHY (v4)
    # ai_score HIGH (>60) = AI video   → authenticity = 100 - ai_score = LOW
    # ai_score LOW  (<40) = Real video → authenticity = 100 - ai_score = HIGH
    # ═══════════════════════════════════════════════════════════

    # ── 1. Sensor noise ──────────────────────────────────────
    # High Laplacian variance can also come from high-res AI renders.
    # Reduce negative adjustments to avoid masking other AI signals.
    # Resolution-aware: a 720p/1080p video with noise=216 is LOW (AI-smooth).
    # Real HD cameras produce noise 800-5000+. Scale the "real grain" lower
    # bound by pixel count — larger frames naturally have higher Laplacian variance.
    _px_count = cap_w * cap_h
    _noise_real_low  = 300  if _px_count > 500_000 else 150   # HD: 720p+ needs 300+
    _noise_real_high = 1000 if _px_count > 500_000 else 500
    if avg_noise < 45:
        ai_score += 12
    elif avg_noise < 60:
        ai_score += 5
    elif avg_noise < 80:
        ai_score += 2
    elif avg_noise > _noise_real_high:
        ai_score -= 8
        log.info("NOISE %.1f → strong real camera grain → -8", avg_noise)
    elif avg_noise > _noise_real_low:
        ai_score -= 5
        log.info("NOISE %.1f → real camera grain → -5", avg_noise)
    elif avg_noise > 100:
        ai_score -= 2
    else:
        log.info("NOISE %.1f → below real-camera threshold (%.0f) → no bonus", avg_noise, _noise_real_low)

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
    # Require at least one dimension >= 480 for reliable DCT block analysis.
    # Portrait videos (320x568, 360x640) are common AI output formats — use height.
    # Extreme tier added: DCT>20 indicates heavily re-encoded AI generation artifact
    # (AI video → social media compression → re-upload cycle amplifies DCT blocks).
    # Calibrated: AI Moose Snow=29.2, AI Gorilla=19.1, AI Child=8.9 vs Real1=5.8, Real2=5.1
    # Gap between real (~5) and AI (~9+) is clear — raised moderate threshold from 4.0→6.0
    # so real videos with DCT 4-6 get +2 (minor) instead of +5 (moderate).
    _dct_reliable = (cap_w >= 480 or cap_h >= 480)
    if _dct_reliable:
        if avg_dct_grid > 20.0:
            ai_score += 14
            log.info("DCT %.3f → extreme grid artifact → +14", avg_dct_grid)
        elif avg_dct_grid > 8.0:
            ai_score += 10
            log.info("DCT %.3f → strong grid artifact → +10", avg_dct_grid)
        elif avg_dct_grid > 6.0:
            ai_score += 5
            log.info("DCT %.3f → moderate grid artifact → +5", avg_dct_grid)
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

    # ── 9c. Selfie bonus (v8) ────────────────────────────────
    # Portrait phone selfies are overwhelmingly real-world content.
    # Reduce base AI score when all selfie markers are present.
    if _is_selfie_content:
        ai_score -= 12
        log.info("SELFIE portrait+static+sharp → real phone video bonus → -12")

    # ── 9d. Talking-head bonus (v9) ──────────────────────────
    # Active portrait of a real person: strongest real-video signal.
    # Portrait + motion + edges = somebody real talking/moving on camera.
    # Skin presence confirms human subject (not AI animal/cinematic render).
    if _is_talking_head:
        ai_score -= 12
        log.info("TALKING_HEAD portrait+motion+edges → real person video → -12")
    if _is_talking_head and _talking_head_skin:
        ai_score -= 6
        log.info("TALKING_HEAD skin confirmed → real person bonus → -6")

    # ── 9e. Single-subject landscape bonus (v10) ─────────────
    # Landscape video of a real person — skin ratio confirms human subject.
    # Camera naturally focuses on one person → uniform sharpness/low sat variance
    # are real characteristics, not AI signals.
    if _is_single_subject:
        ai_score -= 8
        log.info("SINGLE_SUBJECT landscape person video → real subject bonus → -8")

    # ── 9f. Compound real-evidence bonus ─────────────────────
    # When multiple independent real-camera signals fire simultaneously, the
    # probability of AI generation is extremely low. Each signal alone is
    # explainable, but 4+ together is a very strong real-camera fingerprint.
    # Criteria: strong sensor noise + high sharpness + real person + natural palette.
    # Calibrated: Real Video 1 hits all 4; AI videos have at most 1-2.
    _compound_real = (
        avg_noise       > 2000   and   # strong real sensor noise (AI renders: <500)
        avg_sharpness   > 1000   and   # high camera sharpness (AI soft renders: <500)
        (_is_talking_head or _is_selfie_content or _is_single_subject) and  # real person confirmed
        hue_entropy     > 2.6          # natural color palette (AI: <2.5)
    )
    if _compound_real:
        ai_score -= 8
        log.info("COMPOUND_REAL noise+sharp+person+palette → strong real fingerprint → -8")

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
        if _is_selfie_content or _is_talking_head or _is_single_subject:
            # Talking heads / single-subject portraits naturally have very low edge std
            # (consistent scene, consistent clothing) — not an AI signal
            log.info("Edge temporal std: %.0f → portrait content guard → skip", edge_temporal_std)
        elif edge_temporal_std < 6000:
            ai_score += 8
        elif edge_temporal_std < 10000:
            ai_score += 4
        if not _is_selfie_content and not _is_talking_head:
            log.info("Edge temporal std: %.0f", edge_temporal_std)

    # ════════════════════════════════════════════════════════
    # NEW v4 SIGNALS — cinematic / animal / nature AI detection
    # ════════════════════════════════════════════════════════

    # ── 16. Saturation frame std (NEW v4, updated v6) ────────
    # AI lighting is unnaturally stable (<5) OR chaotically unstable (>35).
    # Real video has natural variation from real-world lighting (8–20).
    # THREE cases for low sat_std:
    #   A) Broadcast studio: sat_std<5, sat_mean>140 → NOT AI, skip
    #   B) AI render (low-sat): sat_std<5, sat_mean<100 → AI frozen lighting (Gorilla=3.86/50)
    #   C) Mid-range (100-140): sat_std<5 → moderate AI signal
    # GUARD: action content with high sat_std = natural outdoor lighting variation
    # GUARD: action content with low sat naturally (overcast/indoor) — not an AI signal
    _stable_is_broadcast = (sat_frame_std < 5.0 and avg_saturation > 140)
    _stable_is_ai_render = (sat_frame_std < 5.0 and avg_saturation < 100 and not is_action_content)
    _unstable_is_outdoor = (sat_frame_std > 22.0 and is_action_content)

    if _stable_is_broadcast:
        log.info("SAT_STD %.2f sat=%.0f → broadcast studio → no penalty", sat_frame_std, avg_saturation)
    elif _is_selfie_content and sat_frame_std < 8.0:
        # Indoor selfie lighting is naturally stable — not an AI signal
        log.info("SAT_STD %.2f → selfie indoor lighting → no penalty", sat_frame_std)
    elif (_is_talking_head or _is_single_subject) and sat_frame_std < 8.0:
        # Talking head indoors: skin + neutral clothing = naturally low sat variance
        # Emily video: sat_std=1.09, 83% low-sat pixels — REAL characteristic
        log.info("SAT_STD %.2f → talking_head skin/neutral dominant → no penalty", sat_frame_std)
    elif is_action_content and sat_frame_std < 6.0:
        # Action video with stable sat — natural (overcast sky, indoor sport, etc.)
        log.info("SAT_STD %.2f → action content stable sat → no penalty", sat_frame_std)
    elif _stable_is_ai_render:
        # Frozen lighting on low-sat NON-action content = AI animal/nature render (Gorilla, Monkey)
        ai_score += 14
        log.info("SAT_STD %.2f sat=%.0f → frozen AI render → +14", sat_frame_std, avg_saturation)
    elif _unstable_is_outdoor:
        log.info("SAT_STD %.2f → outdoor action variation → no penalty", sat_frame_std)
    elif sat_frame_std < 3.0 and not _is_short_clip:
        ai_score += 14
        log.info("SAT_STD %.2f → frozen AI lighting → +14", sat_frame_std)
    elif sat_frame_std < 6.0 and not _is_short_clip:
        ai_score += 8
        log.info("SAT_STD %.2f → stable lighting → +8 (skipped for short clip)", sat_frame_std)
    elif sat_frame_std > 35.0 and not is_action_content:
        ai_score += 10
        log.info("SAT_STD %.2f → unstable AI color → +10", sat_frame_std)
    elif 8.0 <= sat_frame_std <= 20.0:
        ai_score -= 6
        log.info("SAT_STD %.2f → natural range → -6", sat_frame_std)

    # ── 16b. Absolute saturation level — hyperreal AI rendering ──
    # AI video generators (Sora, Kling, RunwayML) render with hyperreal oversaturated
    # color palettes far beyond what real camera sensors produce.
    # Real cameras (outdoor bright): sat mean ~70-105
    # Real cameras (indoor/skin): sat mean ~50-90
    # AI hyperreal style (children, nature, animals): sat mean ~120-160
    # Key: HIGH absolute saturation + FROZEN std = AI hyperreal rendering signature.
    # Calibrated: AI_Child sat=138 sat_std=1.78 → fires; Real_Video_2 sat=89.8 → safe.
    if avg_saturation > 130 and sat_frame_std < 8.0:
        ai_score += 12
        log.info("SAT_HYPERREAL sat=%.0f std=%.2f → frozen hyperreal AI color → +12",
                 avg_saturation, sat_frame_std)
    elif avg_saturation > 120 and sat_frame_std < 8.0:
        ai_score += 7
        log.info("SAT_HYPERREAL sat=%.0f std=%.2f → elevated frozen saturation → +7",
                 avg_saturation, sat_frame_std)

    # ── 17. Background corner drift (NEW v4) ─────────────────
    # AI backgrounds: frozen (drift<2, Monkey=1.3) OR wildly warping (Bus=25.7, Moose=45.7).
    # Real: natural drift from camera movement (5–18).
    # GUARD: static real content (broadcast/tripod) also has frozen bg — exempt if static+broadcast.
    _static_broadcast = (is_static_content and avg_saturation > 140)
    _bg_warp_threshold = 55.0 if is_action_content else 30.0
    if _is_selfie_content and bg_drift < 6.0:
        # Selfie held steady or on table — naturally low bg drift
        log.info("BG_DRIFT %.2f → selfie static hold → no penalty", bg_drift)
    elif bg_drift < 2.0 and not _static_broadcast:
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

    # ── 19. Flow direction entropy (NEW v4b) ─────────────────
    # Measures how varied optical flow directions are across the scene.
    # AI crowds have unnaturally uniform movement (low entropy).
    # Real emergency/crowd footage has chaotic multi-directional movement (high entropy).
    # Calibrated: Bus AI=1.73, Moose AI=2.34 vs Real action ~2.8+
    # Only meaningful when significant motion is present.
    # Action guard: real action/sport videos have coordinated directional movement
    # (e.g. a single athlete moving across frame) — suppress minor penalties for them.
    if avg_motion > 3.0 and len(flow_dir_scores) > 5 and not _is_short_clip:
        if avg_flow_dir_entropy < 1.5:
            # Very uniform flow — strong AI crowd signal (fires even for action)
            ai_score += 12
            log.info("FLOW_ENTROPY %.3f → uniform AI motion → +12", avg_flow_dir_entropy)
        elif avg_flow_dir_entropy < 2.0 and not is_action_content:
            ai_score += 7
            log.info("FLOW_ENTROPY %.3f → somewhat uniform → +7", avg_flow_dir_entropy)
        elif avg_flow_dir_entropy < 2.5 and not is_action_content:
            ai_score += 3
            log.info("FLOW_ENTROPY %.3f → slightly uniform → +3", avg_flow_dir_entropy)
        elif avg_flow_dir_entropy > 2.8:
            # Natural chaotic movement — real video signal
            ai_score -= 4
            log.info("FLOW_ENTROPY %.3f → chaotic natural motion → -4", avg_flow_dir_entropy)

    # ── 20. Peak-to-mean motion ratio (NEW v4b) ───────────────
    # Real emergency/event videos have sudden motion bursts (panic, reactions).
    # AI videos have unnaturally smooth motion curves — no dramatic spikes.
    # Calibrated: Bus AI=21 (has 1 big cut spike), Moose AI=1.5 (no spikes at all)
    # Real emergencies: typically 5-30+ with multiple spikes throughout
    # Guard: only meaningful for non-static content.
    # Action guard: continuous action/sport videos have sustained high motion throughout
    # — no "reaction spikes" is expected and normal, not an AI signal for them.
    # Require motion_var > 5 for action content: truly flat AI motion has near-zero variance.
    if avg_motion > 3.0 and not is_static_content and not _is_short_clip and not _is_talking_head and not _is_single_subject and not (_is_portrait and avg_edge < 30.0):
        if peak_to_mean_ratio < 2.0 and not (is_action_content and motion_var > 3.0):
            # Completely flat motion — no reactions at all (Moose-style)
            ai_score += 10
            log.info("PEAK_RATIO %.2f → no reaction spikes → +10", peak_to_mean_ratio)
        elif peak_to_mean_ratio < 3.5 and not is_action_content:
            ai_score += 5
            log.info("PEAK_RATIO %.2f → weak reaction spikes → +5", peak_to_mean_ratio)

    # ── 21. FG/BG sharpness ratio (NEW v5) ───────────────────
    # AI renders: subject is 1000–2000x sharper than background.
    # Real cameras: natural depth-of-field gives ratio 300–800.
    # Calibrated: Bus AI=2145, Moose AI=1525 vs Real=335–798
    if fg_bg_ratio > 1200:
        ai_score += 14
        log.info("FG_BG %.0f → extreme AI depth → +14", fg_bg_ratio)
    elif fg_bg_ratio > 900:
        ai_score += 8
        log.info("FG_BG %.0f → high AI depth → +8", fg_bg_ratio)
    elif fg_bg_ratio > 600:
        ai_score += 3
        log.info("FG_BG %.0f → elevated depth → +3", fg_bg_ratio)
    elif fg_bg_ratio < 400:
        ai_score -= 4
        log.info("FG_BG %.0f → natural depth → -4", fg_bg_ratio)

    # ── 22. Crowd motion sync (NEW v5) ───────────────────────
    # AI crowds: left/right halves move in lockstep (low diff ratio).
    # Real crowds: independent motion, higher variance between halves.
    # Calibrated: Bus AI=0.088, Moose AI=0.046 vs Real=0.10–0.14
    # Guard: only meaningful when multiple people / crowd is present
    # Selfie/talking-head guard: single person moves as one unit → naturally high sync
    # Action guard: single-subject action videos naturally have correlated L/R motion
    # EXCEPTION: portrait action videos — these are often AI-generated person/child videos
    # where the subject + background are both synthetic. Re-enable with tighter threshold.
    _sync_thresh_strong = 0.05 if is_action_content else 0.06
    _sync_thresh_med    = 0.07 if is_action_content else 0.09
    _sync_thresh_slight = 0.09 if is_action_content else 0.105
    _portrait_action    = is_action_content and _is_portrait
    # Portrait action: suppress the broad action guard — use tighter portrait-specific thresholds
    _sync_guard = (
        is_static_content or
        _is_selfie_content or
        _is_talking_head or
        _is_single_subject or
        (is_action_content and not _is_portrait and avg_edge < 30.0)  # landscape action with low edges only
    )
    if avg_motion > 3.0 and not _sync_guard:
        if _portrait_action:
            # Portrait action: tighter thresholds — suppress mild sync, catch extreme lockstep
            if motion_sync < 0.07:
                ai_score += 10
                log.info("MOTION_SYNC %.3f → portrait action lockstep (AI subject) → +10", motion_sync)
            elif motion_sync < 0.09:
                ai_score += 5
                log.info("MOTION_SYNC %.3f → portrait action moderate sync → +5", motion_sync)
        else:
            if motion_sync < _sync_thresh_strong:
                ai_score += 14
                log.info("MOTION_SYNC %.3f → extreme lockstep crowd → +14", motion_sync)
            elif motion_sync < _sync_thresh_med:
                ai_score += 8
                log.info("MOTION_SYNC %.3f → lockstep crowd → +8", motion_sync)
            elif motion_sync < _sync_thresh_slight:
                ai_score += 3
                log.info("MOTION_SYNC %.3f → slightly synchronized → +3", motion_sync)
            elif motion_sync > 0.13:
                ai_score -= 4
                log.info("MOTION_SYNC %.3f → natural independent motion → -4", motion_sync)

    # ── 23. Hue entropy — color palette diversity (NEW v5) ───
    # AI videos use a limited/curated color palette (low hue entropy).
    # Real cameras capture the full natural spectrum (high entropy).
    # Calibrated: Bus AI=1.62, Monkey AI=1.29 vs Real=2.76–3.19
    if hue_entropy < 1.8:
        ai_score += 12
        log.info("HUE_ENT %.3f → very limited AI palette → +12", hue_entropy)
    elif hue_entropy < 2.2:
        ai_score += 7
        log.info("HUE_ENT %.3f → limited AI palette → +7", hue_entropy)
    elif hue_entropy < 2.5:
        ai_score += 3
        log.info("HUE_ENT %.3f → somewhat limited palette → +3", hue_entropy)
    elif hue_entropy > 2.6:
        # Rich natural palette — real video signal
        ai_score -= 5
        log.info("HUE_ENT %.3f → natural palette → -5", hue_entropy)

    # ── 24. Quadrant sharpness uniformity (NEW v6) ───────────
    # Real cameras have natural depth-of-field — sharpness varies
    # significantly across the frame. AI renders every quadrant
    # with equal computational precision (low CoV = too uniform).
    # Calibrated: Gorilla=0.365, Bus=0.433, Moose=0.498
    #             Real Slide1=0.916, Real Slide2=0.581, Real Pres=0.725
    # Portrait phone videos (h > w*1.5) have naturally lower quad CoV — relax thresholds.
    # Action guard: fast motion causes global motion blur across the entire frame,
    # making all quadrants uniformly blurry — identical artifact to AI render uniformity.
    # Real_Video_2: quad_cov=0.165 with is_action=True — all quadrants ~4000-6000 (sharp+blur mixed).
    _is_portrait = (cap_h > cap_w * 1.5)
    _quad_thresh_strong = 0.18 if _is_portrait else 0.40
    _quad_thresh_med    = 0.30 if _is_portrait else 0.50
    _quad_thresh_slight = 0.55 if _is_portrait else 0.55
    if quad_cov < _quad_thresh_strong and not _is_talking_head and not _is_single_subject and not is_action_content:
        ai_score += 14
        log.info("QUAD_COV %.3f → very uniform render focus → +14", quad_cov)
    elif quad_cov < _quad_thresh_med and not _is_talking_head and not _is_single_subject and not is_action_content:
        ai_score += 8
        log.info("QUAD_COV %.3f → uniform render focus → +8", quad_cov)
    elif quad_cov < _quad_thresh_slight and not is_action_content:
        ai_score += 3
        log.info("QUAD_COV %.3f → slightly uniform → +3", quad_cov)
    elif quad_cov > 0.75:
        # Strong depth-of-field variation = real camera signal
        ai_score -= 5
        log.info("QUAD_COV %.3f → natural lens depth-of-field → -5", quad_cov)

    ai_score = max(0.0, min(100.0, ai_score))
    log.info("Primary AI score v6: %.0f  (quad_cov=%.3f fg_bg=%.0f sync=%.3f hue=%.2f sat_std=%.1f)",
             ai_score, quad_cov, fg_bg_ratio, motion_sync, hue_entropy, sat_frame_std)

    # Build content_type string for GPT context
    _content_type = (
        "action"          if is_action_content  else
        "talking_head"    if _is_talking_head   else
        "single_subject"  if _is_single_subject else
        "selfie"          if _is_selfie_content else
        "static"          if is_static_content  else
        "cinematic"
    )

    signal_context = {
        "signal_score":   int(round(ai_score)),
        "content_type":   _content_type,
        "avg_saturation": avg_saturation,
        "avg_sharpness":  avg_sharpness,
        "sat_frame_std":  sat_frame_std,
        "bg_drift":       bg_drift,
        "flicker_std":    flicker_std,
        "quad_cov":       quad_cov,
        "fg_bg_ratio":    fg_bg_ratio,
        "motion_sync":    motion_sync,
    }
    return int(round(ai_score)), signal_context