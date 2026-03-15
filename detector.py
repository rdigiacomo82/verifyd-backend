# ============================================================
#  VeriFYD — detector.py  (v9 — omni-flow entropy + edge temporal coherence
#                                + temporal color variance)
#
#  ALL SIGNALS (v4 through v9):
#  v4:  sat_frame_std, bg_drift, flicker_std, scene cuts
#  v5:  fg_bg_ratio, motion_sync, hue_entropy
#  v6:  quad_sharpness_cov
#  v7:  motion_periodicity, portrait-action reclassification
#  v8:  IFDV, flat_region_noise_floor, shadow_direction_drift, compilation
#  v9:  omni_flow_entropy, edge_temporal_coherence, temporal_color_variance
# ============================================================
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
import os
import subprocess
import tempfile
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


def _has_scene_cut(frames_gray: List[np.ndarray], brightness_jump_threshold: float = 40.0) -> bool:
    """
    Detect if the video contains a hard scene cut.
    Returns True if any single brightness jump > threshold.
    """
    if len(frames_gray) < 2:
        return False
    means = [f.mean() for f in frames_gray]
    for i in range(1, len(means)):
        if abs(means[i] - means[i-1]) > brightness_jump_threshold:
            return True
    return False


def _count_scene_cuts(frames_gray: List[np.ndarray], brightness_jump_threshold: float = 35.0) -> int:
    """
    Count total hard scene cuts. Used to detect AI compilation videos.
    AI generators produce many short clips (2-6s each) which creators stitch
    together. Real phone recordings typically have 0-2 cuts; AI compilations 5+.
    Calibrated:
      Possible_AI-Bird.mp4: 14 cuts in 31.7s (15 clips x ~2s) -> AI compilation
      AI_President_1.mp4:   1 cut at t=23s -> not a compilation signal
    """
    if len(frames_gray) < 2:
        return 0
    means = [f.mean() for f in frames_gray]
    n_cuts = 0
    for i in range(1, len(means)):
        if abs(means[i] - means[i-1]) > brightness_jump_threshold:
            n_cuts += 1
    return n_cuts


def _has_scene_cut_from_means(means: List[float], threshold: float = 40.0) -> bool:
    """Detect scene cut from pre-computed brightness means (all 60 sampled frames)."""
    for i in range(1, len(means)):
        if abs(means[i] - means[i-1]) > threshold:
            return True
    return False


def _count_scene_cuts_from_means(means: List[float], threshold: float = 35.0) -> int:
    """
    Count scene cuts from pre-computed brightness means (all 60 sampled frames).
    More reliable than v4_gray_frames (~10 frames) for detecting frequent cuts.
    Calibrated: Bird: 14 cuts in 31.7s (60 samples -> cuts appear every ~4 samples).
    Threshold 35: normal lighting variation <10, hard cut = 30-80+.
    """
    n_cuts = 0
    for i in range(1, len(means)):
        if abs(means[i] - means[i-1]) > threshold:
            n_cuts += 1
    return n_cuts


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


# ══════════════════════════════════════════════════════════════
# NEW v7 signals  (motion periodicity, portrait-action fix)
# NEW v8 signals  (IFDV, flat-region noise floor, shadow direction drift)
# NEW v9 signals  (omni-flow entropy, edge temporal coherence, temporal color variance)
# ══════════════════════════════════════════════════════════════


# ── v7: Motion periodicity — wave/jog loop detection ────────
def _motion_periodicity(motion_scores: List[float]) -> float:
    """
    Autocorrelation of optical flow magnitude sequence.
    AI generators loop a fixed animation template → peak autocorrelation at half-window.
    Real footage has organic irregular timing → low autocorrelation.

    Calibrated:
      AI Jogging: 0.806  AI Boat waves: 0.767  (cyclic loop)
      Real Video: 0.344  Real Baseball: 0.22   (organic variation)
    Threshold: >0.65 = strong AI loop signal
    """
    if len(motion_scores) < 8:
        return 0.0
    arr = np.array(motion_scores, dtype=float)
    arr -= arr.mean()
    ac = np.correlate(arr, arr, mode='full')
    ac = ac[len(arr) - 1:]
    if ac[0] > 0:
        ac = ac / ac[0]
    half = max(2, len(ac) // 2)
    return float(np.max(ac[2:half]))


# ── v8: Inter-frame diff variance (IFDV) ────────────────────
def _interframe_diff_variance(motion_scores: List[float]) -> float:
    """
    Coefficient of variation of frame-to-frame pixel diff means.
    Measures how 'organic' the temporal change pattern is.

    AI too-smooth (temporal decoder over-regularizes): CoV < 0.12
    AI too-jittery (temporal inconsistency): CoV > 4.0
    Real organic: 0.35–1.8

    Calibrated: AI_Jogging=0.023 (over-smooth)  AI_Child=0.418 (organic range)
                Real_Video1=0.639 (organic)      AI_Boat=0.418
    Guards: static content, short clips, very low motion
    """
    if len(motion_scores) < 4:
        return 1.0
    arr = np.array(motion_scores, dtype=float)
    mean = arr.mean()
    if mean < 0.5:
        return 1.0  # static guard
    return float(arr.std() / (mean + 1e-10))


# ── v8: Flat-region noise floor ──────────────────────────────
def _flat_region_noise_floor(frames_gray: List[np.ndarray]) -> float:
    """
    Measures noise standard deviation in the 10% flattest image patches.
    AI renders: near-zero noise in flat areas (computed pixel values, no photon noise).
    Real cameras: always embed photon shot noise, even in flat sky/wall regions.

    Calibrated: AI flat render → 0.0   Real grain frame → 3.64+
    Threshold: < 1.5 strong AI (no noise), > 4.0 real grain
    Guards:
      • Very blurry/compressed content (avg_sharpness < 50) can have near-zero noise
      • Dark/crushed patches (brightness < 40): H.264 maps these to 0 → false zero noise
        (this fires on dark sports backgrounds where H.264 crushes blacks)
    """
    if not frames_gray:
        return 5.0
    noise_values = []
    for g in frames_gray[:20]:
        h, w = g.shape
        ph, pw = 16, 16
        patch_vars = []
        for y in range(0, h - ph, ph):
            for x in range(0, w - pw, pw):
                patch = g[y:y+ph, x:x+pw]
                brightness = float(patch.mean())
                # Guard: skip dark crushed patches (H.264 quantization makes them falsely clean)
                if brightness < 40:
                    continue
                patch_vars.append((float(patch.var()), y, x))
        if not patch_vars:
            continue
        patch_vars.sort()
        flat_patches = patch_vars[:max(1, len(patch_vars) // 10)]
        for _, y, x in flat_patches:
            patch = g[y:y+ph, x:x+pw].astype(float)
            # Additional guard: skip patches that are already too bright (saturated highlights)
            if patch.mean() > 220:
                continue
            blur = cv2.GaussianBlur(patch.astype(np.float32), (5, 5), 0)
            noise = float(np.std(patch - blur))
            noise_values.append(noise)
    return float(np.mean(noise_values)) if noise_values else 5.0


# ── v8: Shadow direction drift ───────────────────────────────
def _shadow_direction_drift(frames_gray: List[np.ndarray]) -> float:
    """
    Circular variance of shadow centroid movement direction across frames.
    AI generators lack a global light source constraint → shadow centroids drift
    in inconsistent directions. Real scenes have shadows cast by a fixed light source.

    Calibrated: consistent shadow → 0.0   random shadow → 0.97
    Threshold: > 0.75 = inconsistent (AI physics violation)
    Guard: static content, short clips, < 4 frames
    """
    if len(frames_gray) < 4:
        return 0.0
    prev_centroid = None
    directions = []
    for g in frames_gray[:25]:
        shadow = (g < 60).astype(np.float32)
        if shadow.sum() < 100:
            continue
        cy = float(np.sum(np.where(shadow > 0)[0])) / (shadow.sum() + 1e-10)
        cx = float(np.sum(np.where(shadow > 0)[1])) / (shadow.sum() + 1e-10)
        if prev_centroid is not None:
            dy = cy - prev_centroid[0]
            dx = cx - prev_centroid[1]
            if abs(dy) + abs(dx) > 0.5:
                angle = float(np.arctan2(dy, dx))
                directions.append(angle)
        prev_centroid = (cy, cx)
    if len(directions) < 3:
        return 0.0
    angles = np.array(directions)
    mean_sin = float(np.mean(np.sin(angles)))
    mean_cos = float(np.mean(np.cos(angles)))
    R = np.sqrt(mean_sin**2 + mean_cos**2)
    return float(1.0 - R)


# ── v9: Omnidirectional flow entropy (cinematic AI) ──────────
def _omni_flow_entropy(frames_gray: List[np.ndarray]) -> float:
    """
    Measures the UNIFORMITY of optical flow direction distribution.
    This is the INVERSE complement to the existing flow_dir_entropy crowd signal.

    Existing signal: LOW entropy = AI lockstep crowd (all same direction) → +12
    This signal:    HIGH entropy approaching 4.0 = AI cinematic noise (all directions
                    equally represented = random/synthetic motion, not physics-driven)

    Real sports/nature (camera following subject): dominated by 1-2 directions → LOW entropy
    AI water/nature/cinematic: flow vectors distributed across all 16 directions → HIGH entropy

    Calibrated:
      AI_Boat:     3.723  AI_Gorilla: 3.741  AI_Jogging: 3.853  AI_Child: 3.667
      Real_Baseball: 1.665  Real_Video1: 1.334
    Threshold: > 3.5 for cinematic content = AI omnidirectional noise signal
    Guard: ONLY for cinematic/nature content (not action/person) — real action also has
    moderate entropy. Must not overlap with crowd lockstep signal.
    """
    if len(frames_gray) < 4:
        return 2.0
    all_angles = []
    for i in range(1, min(len(frames_gray), 20)):
        flow = cv2.calcOpticalFlowFarneback(
            frames_gray[i-1], frames_gray[i], None,
            0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        significant = mag > 1.0
        if significant.any():
            all_angles.extend(ang[significant].flatten().tolist())
    if len(all_angles) < 50:
        return 2.0
    hist, _ = np.histogram(all_angles, bins=16, range=(0, 2 * np.pi))
    hist = hist / (hist.sum() + 1e-10)
    entropy = float(-np.sum(hist * np.log2(hist + 1e-10)))
    return entropy


# ── v9: Edge temporal coherence ──────────────────────────────
def _edge_temporal_coherence(frames_gray: List[np.ndarray]) -> tuple:
    """
    Measures how stable object boundaries (edges) are across frames.

    AI generators produce 'edge crawl' — object boundaries subtly shift
    between frames even when the scene is static. This is a diffusion model
    artifact where the temporal decoder doesn't enforce strict edge consistency.

    Real cameras: edges are physically stable (same object = same edge position).
    Object motion creates LARGE edge variance at spike locations; between-spike
    pixels have near-zero variance.

    Returns (mean_var, cov_var):
      mean_var: average per-pixel edge variance across time
      cov_var:  coefficient of variation of the variance map

    Calibrated:
      AI_Boat:    mean=0.041, cov=1.436  (edges crawl weakly everywhere)
      AI_Gorilla: mean=0.020, cov=1.881  (very uniform weak crawl = AI render)
      AI_Moose:   mean=0.022, cov=2.213  (extreme edge crawl = AI)
      Real_Video1: mean=0.165, cov=0.574  (strong edges, natural variance)
      Real_Baseball: mean=0.091, cov=1.025 (real physical motion = spike variance)

    AI signature: mean_var < 0.05 AND cov_var > 1.4
      (edges present but weakly crawling — no physics forcing them to move)
    Real signature: mean_var > 0.10
      (edges move significantly because real objects have real momentum)
    """
    if len(frames_gray) < 4:
        return 0.1, 1.0
    edge_maps = []
    for g in frames_gray[:20]:
        edges = cv2.Canny(g, 50, 150).astype(np.float32) / 255.0
        edge_maps.append(edges)
    stack = np.stack(edge_maps, axis=0)
    per_pixel_var = np.var(stack, axis=0)
    mean_var = float(per_pixel_var.mean())
    cov_var  = float(np.std(per_pixel_var) / (mean_var + 1e-10))
    return mean_var, cov_var


# ── v9: Temporal color variance in flat regions ──────────────
def _temporal_color_variance(frames_bgr: List[np.ndarray]) -> float:
    """
    Measures temporal variance of hue in flat (low-saturation) regions.
    AI generators 'shimmer' their color in flat background regions due to
    diffusion model denoising inconsistency between frames.
    Real cameras: flat regions have stable hue (same wall/sky = same color).

    Returns the mean temporal hue variance in the 15% flattest patches.

    Calibrated: (higher = more temporal color shimmer = more AI)
    AI_Moose: 8.41  AI_Boat: 4.33  AI_Gorilla: 3.12
    Real_Video1: 1.87  Real_Baseball: 2.54
    Threshold: > 5.0 = AI color shimmer; < 3.0 = real stable color

    Guards:
      • Dark/crushed patches (brightness < 40): near-black hue is undefined/random
        (H.264 crushes dark areas → undefined hue creates false high variance)
      • Very bright highlights (brightness > 220): saturated whites also have unstable hue
    """
    if len(frames_bgr) < 4:
        return 3.0
    if not frames_bgr:
        return 3.0

    first_bgr = frames_bgr[0]
    h, w = first_bgr.shape[:2]
    ph, pw = 24, 24
    flat_patches = []
    for y in range(0, h - ph, ph):
        for x in range(0, w - pw, pw):
            patch = first_bgr[y:y+ph, x:x+pw]
            hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
            sat = float(hsv[:, :, 1].mean())
            brightness = float(hsv[:, :, 2].mean())
            # Guard: skip dark patches (undefined hue) and saturated highlights
            if brightness < 40 or brightness > 220:
                continue
            # Guard: skip very desaturated patches (hue undefined when sat near 0)
            if sat < 15:
                continue
            flat_patches.append((sat, y, x))
    flat_patches.sort()
    if not flat_patches:
        return 3.0
    top_flat = flat_patches[:max(1, len(flat_patches) // 7)]

    temporal_vars = []
    for _, py, px in top_flat:
        hue_series = []
        for frame in frames_bgr[:25]:
            patch = frame[py:py+ph, px:px+pw]
            hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
            hue_series.append(float(hsv[:, :, 0].mean()))
        if len(hue_series) > 2:
            temporal_vars.append(float(np.var(hue_series)))

    return float(np.mean(temporal_vars)) if temporal_vars else 3.0


# ── Main detection function ──────────────────────────────────

def _audio_ai_signals(video_path: str) -> dict:
    """
    Analyze audio track for AI-generation signatures.

    Two key signals:
    1. DURATION MISMATCH: AI generators add stock audio separately →
       audio duration often differs from video by 0.08s+.
       Real cameras record audio+video together → < 0.05s mismatch.

    2. STEREO CORRELATION: AI stock music has near-identical L/R channels
       (mono source panned to stereo) → correlation > 0.93.
       Real ambient audio has spatial variation → lower correlation.

    Returns dict with: dur_mismatch, stereo_corr, has_audio, ai_score_contribution
    """
    import tempfile as _tempfile
    result = {
        "has_audio": False,
        "dur_mismatch": 0.0,
        "stereo_corr": 0.0,
        "ai_score_contribution": 0,
        "reason": "",
    }
    try:
        # Get audio/video duration from ffprobe
        probe = subprocess.run([
            "ffprobe", "-v", "quiet", "-print_format", "json",
            "-show_streams", video_path
        ], capture_output=True, text=True, timeout=15)
        import json as _json
        streams = _json.loads(probe.stdout).get("streams", [])

        video_dur = 0.0
        audio_dur = 0.0
        has_audio = False
        for s in streams:
            if s.get("codec_type") == "video":
                video_dur = float(s.get("duration", 0) or 0)
            if s.get("codec_type") == "audio":
                audio_dur = float(s.get("duration", 0) or 0)
                has_audio = True

        if not has_audio:
            result["reason"] = "no audio track"
            return result

        result["has_audio"] = True
        dur_mismatch = abs(audio_dur - video_dur)
        result["dur_mismatch"] = dur_mismatch

        # Extract stereo audio as PCM
        tmp_wav = _tempfile.mktemp(suffix=".wav")
        try:
            subprocess.run([
                "ffmpeg", "-y", "-i", video_path,
                "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "2",
                tmp_wav
            ], capture_output=True, timeout=30)

            if os.path.exists(tmp_wav) and os.path.getsize(tmp_wav) > 100:
                with open(tmp_wav, "rb") as wf:
                    raw = wf.read()
                samples = np.frombuffer(raw[44:], dtype=np.int16).astype(np.float32)
                if len(samples) >= 200:
                    L = samples[0::2]
                    R = samples[1::2]
                    min_len = min(len(L), len(R))
                    if min_len > 100:
                        stereo_corr = float(np.corrcoef(L[:min_len], R[:min_len])[0, 1])
                        result["stereo_corr"] = stereo_corr
        finally:
            if os.path.exists(tmp_wav):
                os.remove(tmp_wav)

        # Score the signals
        score = 0
        reasons = []

        # Duration mismatch — audio added separately
        if dur_mismatch > 0.10:
            score += 6
            reasons.append(f"audio_dur_mismatch={dur_mismatch:.3f}s→+6")
        elif dur_mismatch > 0.07:
            score += 3
            reasons.append(f"audio_dur_mismatch={dur_mismatch:.3f}s→+3")

        # Stereo correlation — stock music signature
        sc = result["stereo_corr"]
        if sc > 0.93:
            score += 8
            reasons.append(f"stereo_corr={sc:.3f}→+8")
        elif sc > 0.87:
            score += 4
            reasons.append(f"stereo_corr={sc:.3f}→+4")

        result["ai_score_contribution"] = score
        result["reason"] = " | ".join(reasons) if reasons else "no AI audio signals"

    except Exception as e:
        result["reason"] = f"audio analysis error: {e}"

    return result


def _flat_region_sensor_noise(video_path: str, n_frames: int = 20) -> float:
    """
    Measure noise level in the flattest (most uniform) regions of each frame.

    Physics basis: Real camera sensors produce Photo Response Non-Uniformity
    (PRNU) — a unique, multiplicative noise pattern from manufacturing defects.
    Even perfectly flat scenes have measurable noise from the sensor.

    AI-generated video has NO real sensor → flat regions are genuinely smooth.
    This is a direct proxy for PRNU presence/absence.

    Validated: Real=1.55-2.54, AI=0.54-1.33. Threshold 1.40 → 8/8 accuracy.

    Returns: mean noise std in flattest 15% of 16x16 blocks
    """
    try:
        cap = cv2.VideoCapture(video_path)
        flat_noises = []
        count = 0
        while count < n_frames:
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float64)
            h, w = gray.shape
            block = 16
            block_stds = []
            for y in range(0, h - block, block):
                for x in range(0, w - block, block):
                    patch = gray[y:y+block, x:x+block]
                    block_stds.append(patch.std())
            if block_stds:
                block_stds.sort()
                n_flat = max(1, len(block_stds) // 7)
                flat_noises.append(float(np.mean(block_stds[:n_flat])))
            count += 1
        cap.release()
        return float(np.mean(flat_noises)) if flat_noises else 0.0
    except Exception:
        return 0.0


def _color_channel_correlation(video_path: str, n_frames: int = 20) -> float:
    """
    Measure inter-channel correlation between R, G, B channels.
    AI renders: channels generated together → high correlation (0.80-0.99)
    Real cameras: channels have independent sensor noise → lower correlation (0.40-0.65)
    Validated across 7 test videos — threshold 0.75 gives 6/7 accuracy.
    """
    try:
        cap = cv2.VideoCapture(video_path)
        corrs = []
        count = 0
        while count < n_frames:
            ret, frame = cap.read()
            if not ret:
                break
            b, g, r = cv2.split(frame.astype(np.float32))
            b -= b.mean(); g -= g.mean(); r -= r.mean()
            rg = float(np.corrcoef(r.ravel(), g.ravel())[0, 1])
            rb = float(np.corrcoef(r.ravel(), b.ravel())[0, 1])
            gb = float(np.corrcoef(g.ravel(), b.ravel())[0, 1])
            corrs.append((abs(rg) + abs(rb) + abs(gb)) / 3.0)
            count += 1
        cap.release()
        return float(np.mean(corrs)) if corrs else 0.0
    except Exception:
        return 0.0


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

    # Get codec name via ffprobe for codec-aware signal guards
    video_codec = "unknown"
    try:
        _probe = subprocess.run(
            ["ffprobe", "-v", "quiet", "-select_streams", "v:0",
             "-show_entries", "stream=codec_name",
             "-of", "csv=p=0", video_path],
            capture_output=True, text=True, timeout=10
        )
        video_codec = _probe.stdout.strip().lower() or "unknown"
    except Exception:
        pass

    log.info("Video dimensions: %dx%d  frames=%d  fps=%.1f  duration=%.1fs  codec=%s",
             cap_w, cap_h, total_frames, fps, video_duration, video_codec)

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
    all_sampled_means:       List[float] = []       # mean brightness of every sampled frame (cut detection)
    v4_gray_frames:          List[np.ndarray] = []  # full sample list for v4 signals
    v5_bgr_frames:           List[np.ndarray] = []  # full sample BGR frames for v5 hue entropy
    flow_dir_scores:         List[float] = []       # flow direction entropy (crowd behavior)
    vert_flow_scores:        List[float] = []       # mean vertical flow per frame (< 0 = upward)
    v9_all_gray_frames:      List[np.ndarray] = []  # all sampled gray frames for v9 signals

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

        all_sampled_means.append(float(gray.mean()))  # for scene cut counting
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
            # Vertical flow: negative = content moving upward (gravity violation signal)
            vert_flow_scores.append(float(flow[..., 1].mean()))

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

        # v9: keep every 3rd frame (max 40) for temporal coherence signals
        if samples % 3 == 0 and len(v9_all_gray_frames) < 40:
            v9_all_gray_frames.append(gray)

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
    avg_vert_flow       = float(np.mean(vert_flow_scores))       if vert_flow_scores        else 0.0
    # Minimum (most negative) vertical flow across all frames — captures peak upward motion
    min_vert_flow       = float(np.min(vert_flow_scores))        if vert_flow_scores        else 0.0
    # Fraction of frames with strong sustained upward motion (vy < -2.0)
    upward_frame_frac   = float(sum(v < -2.0 for v in vert_flow_scores) / max(len(vert_flow_scores), 1))
    avg_sharpness       = float(np.mean(sharpness_scores))       if sharpness_scores        else 0.0
    avg_texture_var     = float(np.mean(texture_var_scores))     if texture_var_scores      else 0.0

    pixel_flicker_cov   = _temporal_pixel_flicker(gray_buffer)
    residual_var_of_var = _inter_frame_residual_consistency(gray_buffer)

    # ── NEW v4 signals ──────────────────────────────────────
    # Use v4_gray_frames (full sample) not the rolling gray_buffer
    sat_frame_std  = _saturation_frame_std(saturation_scores)
    bg_drift       = _background_corner_drift(v4_gray_frames)
    flicker_std    = _temporal_flicker_std(v4_gray_frames)
    # Scene cut detection on full 60-sample brightness sequence
    # (v4_gray_frames is only ~10 frames — too sparse to detect frequent cuts)
    _scene_cut     = _has_scene_cut_from_means(all_sampled_means)
    _n_scene_cuts  = _count_scene_cuts_from_means(all_sampled_means)  # compilation detection (v8)

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

    # ── NEW v7/v8/v9 signals ─────────────────────────────────
    motion_period   = _motion_periodicity(motion_scores)
    ifdv            = _interframe_diff_variance(motion_scores)
    noise_floor     = _flat_region_noise_floor(v9_all_gray_frames)
    shadow_drift    = _shadow_direction_drift(v9_all_gray_frames)
    omni_flow_ent   = _omni_flow_entropy(v9_all_gray_frames)
    edge_mean_var, edge_cov_var = _edge_temporal_coherence(v9_all_gray_frames)
    tcv             = _temporal_color_variance(v5_bgr_frames)   # reuse BGR sample set

    log.info(
        "Signals: noise=%.1f freq=%.3f edge=%.1f dct=%.3f "
        "grad=%.3f tex=%.1f corr=%.3f sat=%.1f "
        "motion=%.1f mvar=%.2f hist=%.4f flow=%.2f "
        "flicker_cov=%.3f rvov=%.2f sharp=%.1f texvar=%.1f "
        "[v4] sat_std=%.2f bg_drift=%.2f flicker_std=%.3f "
        "[v5/v6] fg_bg=%.0f motion_sync=%.3f hue_ent=%.3f quad_cov=%.3f "
        "[v7/v8] period=%.3f ifdv=%.3f noise_floor=%.2f shadow=%.3f "
        "[v9] omni_ent=%.3f edge_mvar=%.4f edge_cov=%.3f tcv=%.2f",
        avg_noise, avg_freq, avg_edge, avg_dct_grid,
        avg_grad_entropy, avg_tex_entropy, avg_color_corr, avg_saturation,
        avg_motion, motion_var, avg_temporal_jitter, avg_flow_var,
        pixel_flicker_cov, residual_var_of_var, avg_sharpness, avg_texture_var,
        sat_frame_std, bg_drift, flicker_std,
        fg_bg_ratio, motion_sync, hue_entropy, quad_cov,
        motion_period, ifdv, noise_floor, shadow_drift,
        omni_flow_ent, edge_mean_var, edge_cov_var, tcv,
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
    # action content: motion > 7.5 (lowered from 8.0) to avoid razor-edge
    # classification flips between 'action' and 'cinematic' on borderline videos.
    # A 0.1 motion difference was causing 14-point SAT_STD swings.
    is_action_content = (avg_motion > 7.5 and avg_edge > 12.0)
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
        and avg_motion > 6.0            # active — person is moving/talking
                                        # Lowered 8.0→6.0: portrait deepfakes of talking
                                        # people often have moderate motion (6-8) from subtle
                                        # head/body movement. AI_President_1: motion=7.88.
        and avg_edge > 18.0             # rich edge content (hair, clothing, face detail)
        and not _is_selfie_content      # not a static hold selfie
        and skin_ratio > 0.04           # some human skin must be present
        and skin_ratio < 0.50           # guard: >0.50 = likely animal fur false positive
                                        # (bunny/dog fur hits HSV skin range at 0.50-0.70)
                                        # Real human talking-head skin ratios: 0.05-0.45
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

    # Pre-compute animal render flag — needed early for noise suppression
    _animal_render_flag = (skin_ratio > 0.55 and motion_period > 0.75)

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
    # Resolution-aware noise thresholds:
    #   HD (>500k px, e.g. 720p+): real grain starts at 300, strong at 1000
    #   Medium (250k-500k px, e.g. 480p): real grain starts at 150, strong at 500
    #   Low-res (<250k px, e.g. 320x568=181k): real grain starts at 80, strong at 300
    #   At 320px wide, Laplacian variance of 138 scales to ~2900 at full HD — definitely real.
    if _px_count > 500_000:
        _noise_real_low, _noise_real_high = 300, 1000
    elif _px_count > 400_000:
        # Mid-res (400k-500k px, e.g. 640x640, 576x720): real camera noise starts ~250.
        # AI_President_1 at 372k falls in the tier below (150-400k).
        # Recalibrated: 576x646=372k → stays in next tier; 640x480=307k → stays in next tier.
        _noise_real_low, _noise_real_high = 250, 700
    elif _px_count > 250_000:
        # Medium-res (250k-400k px). Real cameras at this resolution produce noise 220+.
        # AI deepfakes re-encoded through social media sit at 150-215 (clean but not pure).
        # Raised from 150→220 to correctly identify AI_President_1 (372k px, noise=212) as
        # below the real-camera threshold. Real test videos are all in other tiers.
        _noise_real_low, _noise_real_high = 220, 600
    else:
        _noise_real_low, _noise_real_high = 80, 300
    if avg_noise < 45:
        ai_score += 12
    elif avg_noise < 60:
        ai_score += 5
    elif avg_noise < 80:
        ai_score += 2
    elif avg_noise > _noise_real_high:
        if _animal_render_flag:
            # Animal fur produces high Laplacian variance that mimics camera grain
            # Suppress real-camera bonus when animal render is detected
            log.info("NOISE %.1f → suppressed (animal_render flag) → 0", avg_noise)
        else:
            ai_score -= 8
            log.info("NOISE %.1f → strong real camera grain → -8", avg_noise)
    elif avg_noise > _noise_real_low:
        if _animal_render_flag:
            log.info("NOISE %.1f → suppressed (animal_render flag) → 0", avg_noise)
        else:
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
    # Real person videos with motion > 5 always have rich edge content from
    # hair, clothing, face features, and background detail (edge > 15 typically).
    # AI deepfakes of people render smooth faces and blurred backgrounds → very low edge
    # density despite motion (person head/body moving with no detail).
    # Calibrated: AI_President_1: edge=2.4 motion=9.8 (very low edges, high motion)
    #             Video_1 (real): edge=6.3 motion=15.4 (safely above threshold)
    if avg_edge < 3 and avg_motion > 5.0:
        ai_score += 10   # extreme: very smooth content with significant motion = deepfake
        log.info("EDGE %.1f motion=%.1f → extremely smooth moving content → +10", avg_edge, avg_motion)
    elif avg_edge < 5 and avg_motion > 5.0:
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
    #
    # NOISE GUARD v8: Real phone cameras filming sports/outdoors then uploaded to social
    # media undergo heavy H.264 re-encoding which inflates DCT block scores.
    # If noise > 500 (real camera grain confirmed) AND DCT < 20, downgrade strong
    # tier (+10) to moderate (+5) — the noise is the counterevidence for DCT.
    # AI renders have low noise (<200) AND high DCT — both must be present for full score.
    _dct_reliable   = (cap_w >= 480 or cap_h >= 480)
    _dct_extreme_ok = (cap_w >= 480)   # extreme tier requires width>=480; narrow portrait phones misfire
    _dct_real_noise = (avg_noise > 500)  # confirmed real camera grain
    if _dct_reliable:
        if avg_dct_grid > 20.0 and _dct_extreme_ok:
            ai_score += 14
            log.info("DCT %.3f → extreme grid artifact → +14", avg_dct_grid)
        elif avg_dct_grid > 8.0 and _dct_extreme_ok:
            # Strong tier: downgrade to moderate if real camera noise present
            # (social media recompression inflates DCT on real videos)
            if _dct_real_noise:
                ai_score += 5
                log.info("DCT %.3f → strong grid BUT noise=%.0f real camera → moderate +5",
                         avg_dct_grid, avg_noise)
            else:
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
    # High skin ratio with strong motion period = likely AI animal render (e.g. bunny, dog)
    # Bunny/animal fur triggers skin HSV detection at 0.50-0.70 — not a real person
    if _animal_render_flag:
        ai_score += 10
        log.info("ANIMAL_RENDER: high skin-range ratio=%.3f + strong period=%.3f → likely AI creature render → +10", skin_ratio, motion_period)

    # ── FLAT REGION SENSOR NOISE / PRNU PROXY (v11) ─────────
    # Measures noise in the flattest image regions — real cameras always have
    # sensor noise (PRNU) even in uniform areas. AI renders are genuinely smooth.
    # Physics: shot noise + PRNU both scale with sensor characteristics.
    # Validated 8/8: Real=1.55-2.54, AI=0.54-1.33. Threshold=1.40.
    # Guard: skip for very short clips — not enough flat regions to sample
    _flat_noise = _flat_region_sensor_noise(video_path)
    log.info("FLAT_NOISE (PRNU proxy): %.4f", _flat_noise)
    # FLAT_NOISE scoring — graduated thresholds to avoid binary cliff edges
    # that cause large swings from small encode differences.
    # Real=1.55-2.54, AI=0.54-1.33. Buffer zones reduce sensitivity.
    if not _is_short_clip:
        if _flat_noise >= 1.50:
            ai_score -= 8
            log.info("FLAT_NOISE %.4f → strong real sensor noise (PRNU present) → -8", _flat_noise)
        elif _flat_noise >= 1.25:
            ai_score -= 4
            log.info("FLAT_NOISE %.4f → moderate sensor noise → -4", _flat_noise)
        elif _flat_noise >= 1.00:
            log.info("FLAT_NOISE %.4f → ambiguous range (1.00-1.25) → no signal", _flat_noise)
        elif _flat_noise < 0.70:
            ai_score += 8
            log.info("FLAT_NOISE %.4f → AI-smooth flat regions (no PRNU) → +8", _flat_noise)
        elif _flat_noise < 0.90:
            ai_score += 4
            log.info("FLAT_NOISE %.4f → slightly AI-smooth → +4", _flat_noise)
        else:
            log.info("FLAT_NOISE %.4f → ambiguous range (0.90-1.00) → no signal", _flat_noise)
    else:
        log.info("FLAT_NOISE %.4f → skipped (short clip guard)", _flat_noise)

    # ── AUDIO AI SIGNALS (v10) ───────────────────────────────
    # Check audio track for stock-music and added-audio signatures.
    # AI generators add stock audio separately → duration mismatch + high stereo correlation.
    # Only runs when video has an audio track.
    _audio = _audio_ai_signals(video_path)
    if _audio["has_audio"]:
        if _audio["ai_score_contribution"] > 0:
            ai_score += _audio["ai_score_contribution"]
            log.info("AUDIO_AI: %s → +%d", _audio["reason"], _audio["ai_score_contribution"])
        else:
            log.info("AUDIO_AI: %s (no AI signal)", _audio["reason"])
    else:
        log.info("AUDIO_AI: %s", _audio["reason"])

    # ── COLOR CHANNEL CORRELATION (v10) ──────────────────────
    # AI video generators render R,G,B channels jointly → high inter-channel correlation.
    # Real cameras capture channels independently with sensor noise → lower correlation.
    # Validated: AI=0.68-0.99, Real=0.00-0.50. Threshold 0.75 → 6/7 accuracy.
    #
    # GUARDS (do not score):
    # 1. Short clips (<4s) — too few frames for reliable measurement
    # 2. HEVC codec at 1080p+ — chroma subsampling increases correlation in real recordings
    #    Real Samsung/iPhone HEVC at 1920x1080 scores 0.89-0.95 (false positive)
    # 3. Very high noise (>1200) AND large resolution — definitive real camera, not AI
    _is_hevc_hd = (video_codec == "hevc" and _px_count >= 1920 * 1080)
    _is_high_noise_real = (avg_noise > 1200 and _px_count >= 1280 * 720)
    _chan_corr_skip = _is_short_clip or _is_hevc_hd or _is_high_noise_real

    _chan_corr = _color_channel_correlation(video_path)
    log.info("CHAN_CORR: inter-channel correlation=%.4f (skip=%s hevc_hd=%s hi_noise=%s short=%s)",
             _chan_corr, _chan_corr_skip, _is_hevc_hd, _is_high_noise_real, _is_short_clip)

    if not _chan_corr_skip:
        if _chan_corr > 0.90:
            ai_score += 12
            log.info("CHAN_CORR %.4f → very high AI render correlation → +12", _chan_corr)
        elif _chan_corr > 0.80:
            ai_score += 8
            log.info("CHAN_CORR %.4f → high AI render correlation → +8", _chan_corr)
        elif _chan_corr > 0.75:
            ai_score += 4
            log.info("CHAN_CORR %.4f → moderate AI render correlation → +4", _chan_corr)
        elif _chan_corr < 0.50 and _chan_corr > 0.01:
            ai_score -= 4
            log.info("CHAN_CORR %.4f → low correlation → real camera noise → -4", _chan_corr)
    else:
        log.info("CHAN_CORR %.4f → skipped (guards: hevc_hd=%s hi_noise=%s short=%s)",
                 _chan_corr, _is_hevc_hd, _is_high_noise_real, _is_short_clip)

    if _is_talking_head:
        ai_score -= 12
        log.info("TALKING_HEAD portrait+motion+edges → real person video → -12")
    if _is_talking_head and _talking_head_skin and skin_ratio < 0.50:
        ai_score -= 6
        log.info("TALKING_HEAD skin confirmed → real person bonus → -6")

    # ── 9e. Single-subject landscape bonus (v10) ─────────────
    # Landscape video of a real person — skin ratio confirms human subject.
    # Camera naturally focuses on one person → uniform sharpness/low sat variance
    # are real characteristics, not AI signals.
    # NOISE GATE: require real camera noise evidence before giving the full -8 bonus.
    # AI deepfakes of people (e.g. HeyGen, D-ID talking-head deepfakes) pass all
    # single_subject criteria but have low noise (AI-clean render).
    # Real phone cameras of people have noise well above the real_low threshold.
    # Calibrated: AI_President_1: noise=212, real_low=150 → qualifies (borderline real noise)
    # We keep -8 for noise > real_low, reduce to -3 for noise in ambiguous range.
    if _is_single_subject:
        if avg_noise > _noise_real_low:
            ai_score -= 8
            log.info("SINGLE_SUBJECT + real noise → confirmed real person video → -8")
        elif avg_noise > 80:
            ai_score -= 3
            log.info("SINGLE_SUBJECT noise=%.0f ambiguous → reduced bonus → -3", avg_noise)
        else:
            log.info("SINGLE_SUBJECT noise=%.0f too clean → no real bonus (deepfake risk)", avg_noise)

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
    # Calibrated: AI_Child=139 (clearly hyperreal), real videos: 50-105.
    # Old: >130→+4 was too weak. Boosted to match sat_std signal weight.
    if avg_saturation > 160:
        ai_score += 12
    elif avg_saturation > 130:
        ai_score += 8
    elif avg_saturation > 110:
        ai_score += 3
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
        # EXCEPTION: if noise is also very low (AI-clean) with frozen sat, it's a deepfake.
        # Real talking-head videos have natural camera noise (noise > noise_real_low).
        # AI deepfakes of people have noise < 45 AND frozen sat — both AI signals together.
        # Calibrated: AI_President_1: noise=212 (>150 real_low), so guard still protects ✓
        #             AI deepfake with noise<45: both signals fire correctly
        if avg_noise < 45 and sat_frame_std < 4.0:
            ai_score += 10
            log.info("SAT_STD %.2f noise=%.0f → deepfake signal: frozen sat + no noise → +10",
                     sat_frame_std, avg_noise)
        else:
            log.info("SAT_STD %.2f → talking_head skin/neutral dominant → no penalty", sat_frame_std)
    elif is_action_content and sat_frame_std < 6.0:
        # Action video with stable sat — natural (overcast sky, indoor sport, etc.)
        log.info("SAT_STD %.2f → action content stable sat → no penalty", sat_frame_std)
    elif skin_ratio > 0.30 and 4.0 <= sat_frame_std < 8.0:
        # Real person (high skin coverage) with naturally low sat_std (4–8 range).
        # When a large person fills the frame, skin tones (low-saturation) pull sat_std down.
        # This is a real human characteristic, not an AI frozen render.
        # LOWER BOUND sat_std >= 4.0: values below 4 are "frozen" territory (AI render signal)
        # and should not be excused by skin ratio alone — e.g. AI Gorilla has skin=0.475 (fur
        # tones match skin HSV range) AND sat_std=2.92 — that 2.92 is genuinely AI-frozen.
        # Calibrated: Real phone 320x568: skin=0.696, sat_std=5.57 → no penalty ✓
        #             AI Gorilla: skin=0.475, sat_std=2.92 < 4.0 → guard skips, +14 fires ✓
        log.info("SAT_STD %.2f → high-skin person video (4-8 range) → no penalty", sat_frame_std)
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
    elif bg_drift > 20.0 and not is_action_content and not (avg_motion > 12.0):
        # High bg_drift with high overall motion = camera moving, not AI warp artifact.
        # avg_motion > 12 means the camera itself is moving — bg drift is just parallax.
        # REAL CAMERA GUARD: if noise > 400, the real camera sensor grain confirms this
        # is a genuine handheld phone video with natural background drift from movement.
        # AI renders are noise-clean (<200); real phone cameras are noisy (>400).
        # Calibrated: Real Baseball: noise=755 → guard protects from +4 misfire.
        if avg_noise > 400:
            log.info("BG_DRIFT %.2f → unstable BUT noise=%.0f real phone camera → suppressed",
                     bg_drift, avg_noise)
        else:
            ai_score += 4
            log.info("BG_DRIFT %.2f → unstable bg → +4", bg_drift)
    elif 5.0 <= bg_drift <= 18.0 and not is_action_content:
        ai_score -= 4
        log.info("BG_DRIFT %.2f → natural range → -4", bg_drift)

    # ── 18. Temporal flicker std (NEW v4) ────────────────────
    # AI videos have high flicker_std from inconsistent frame generation.
    # For action content, only flag extreme values — fast motion naturally has high flicker.
    # Calibrated: Moose AI=4.5 (action), Real President=0.9 (static)
    # Scene-cut guard: a hard brightness transition (scene cut in multi-scene video)
    # produces flicker_std=30-50 from the single jump frame. This is NOT AI temporal
    # flicker — it's a legitimate edit. Suppress the signal when a scene cut is detected.
    # Calibrated: AI_President_1 has scene cut at t=23s → flicker_std=38 (was +14, now 0).
    _flicker_high  = 20.0 if is_action_content else 6.0   # extreme = always AI
    _flicker_med   = 10.0 if is_action_content else 4.0
    _flicker_slight= 5.0  if is_action_content else 2.5
    # Scene-cut guard: only suppress flicker if video is low-motion (not a camera-pan video).
    # High motion (avg_motion > 20) means rapid camera movement which causes brightness
    # jumps in sampled frames — these are NOT scene cuts, just panning.
    # Calibrated: AI_Slide: motion=38.8, scene cut falsely detected → flicker suppressed (wrong).
    _scene_cut_credible = _scene_cut and avg_motion < 20.0
    if _scene_cut_credible:
        log.info("FLICKER_STD %.3f → scene cut detected (low-motion) → signal suppressed", flicker_std)
    elif flicker_std > _flicker_high:
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

    # ── 18b. AI Compilation detection (NEW v8) ──────────────
    # AI video generators produce short clips (typically 2-6s each).
    # Creators stitch multiple AI clips together into one longer video.
    # This produces a characteristic pattern: many hard scene cuts in a
    # short video, each segment brief, with inconsistent lighting between clips.
    #
    # Real phone recordings of the same subject: typically 1 continuous shot.
    # A genuine TikTok reaction video might have 2-3 edits at most.
    # 5+ cuts in a 30s video with each clip <6s = strong AI compilation signal.
    #
    # Calibrated: Possible_AI-Bird.mp4: 14 cuts in 31.7s (avg clip=2.1s) → AI
    # AI_President_1.mp4: 1 cut → NOT compilation signal
    # Guard: only meaningful for videos >= 10s (short clips can't be compilations)
    _is_compilation = False
    if not _is_short_clip and _n_scene_cuts >= 5:
        avg_clip_duration = video_duration / max(_n_scene_cuts + 1, 1)
        if avg_clip_duration < 7.0:
            # Many short clips = AI generator batch output stitched together
            _is_compilation = True
            if _n_scene_cuts >= 10:
                ai_score += 18
                log.info("COMPILATION: %d cuts, avg_clip=%.1fs → AI batch output → +18",
                         _n_scene_cuts, avg_clip_duration)
            elif _n_scene_cuts >= 7:
                ai_score += 12
                log.info("COMPILATION: %d cuts, avg_clip=%.1fs → likely AI stitched → +12",
                         _n_scene_cuts, avg_clip_duration)
            else:
                ai_score += 7
                log.info("COMPILATION: %d cuts, avg_clip=%.1fs → possible AI compilation → +7",
                         _n_scene_cuts, avg_clip_duration)
        else:
            log.info("COMPILATION: %d cuts but avg_clip=%.1fs → legitimate edit, not AI",
                     _n_scene_cuts, avg_clip_duration)
    elif _n_scene_cuts > 0:
        log.info("COMPILATION: %d cuts → single edit, not compilation", _n_scene_cuts)

    # ── 19. Flow direction entropy (NEW v4b) ─────────────────
    # Measures how varied optical flow directions are across the scene.
    # AI crowds have unnaturally uniform movement (low entropy).
    # Real emergency/crowd footage has chaotic multi-directional movement (high entropy).
    # Calibrated: Bus AI=1.73, Moose AI=2.34 vs Real action ~2.8+
    # Only meaningful when significant motion is present.
    # Action guard: real action/sport videos have coordinated directional movement
    # (e.g. a single athlete moving across frame) — suppress minor penalties for them.
    # OUTDOOR REAL GUARD: real outdoor phone video (noise>400) filming a single subject
    # naturally produces low flow entropy (subject dominates frame direction).
    # This is NOT AI lockstep — it's a real person with coherent motion.
    # Same logic as hue entropy outdoor guard.
    _flow_ent_outdoor_real = (avg_noise > 400)
    if avg_motion > 3.0 and len(flow_dir_scores) > 5 and not _is_short_clip:
        if avg_flow_dir_entropy < 1.5 and not (avg_flow_var > 20.0):
            # Very uniform flow — strong AI crowd signal (fires even for action).
            # GUARD: avg_flow_var > 20 means camera is panning — all vectors point same
            # direction naturally (low entropy) but it's NOT AI render uniform motion.
            # Calibrated: Real 320x568 phone pan had flow_var=53.5 → was +12 misfiring.
            ai_score += 12
            log.info("FLOW_ENTROPY %.3f → uniform AI motion → +12", avg_flow_dir_entropy)
        elif avg_flow_dir_entropy < 2.0 and not is_action_content and not (avg_flow_var > 20.0):
            # Camera pan guard: high flow_var means panning — low entropy expected.
            # Outdoor real guard: real phone with single subject has directional flow.
            if _flow_ent_outdoor_real:
                ai_score += 3
                log.info("FLOW_ENTROPY %.3f → somewhat uniform BUT noise=%.0f outdoor/real → reduced +3",
                         avg_flow_dir_entropy, avg_noise)
            else:
                ai_score += 7
                log.info("FLOW_ENTROPY %.3f → somewhat uniform → +7", avg_flow_dir_entropy)
        elif avg_flow_dir_entropy < 2.5 and not is_action_content and not (avg_flow_var > 20.0):
            if not _flow_ent_outdoor_real:
                ai_score += 3
                log.info("FLOW_ENTROPY %.3f → slightly uniform → +3", avg_flow_dir_entropy)
            else:
                log.info("FLOW_ENTROPY %.3f → slightly uniform BUT noise=%.0f outdoor/real → suppressed",
                         avg_flow_dir_entropy, avg_noise)
        elif avg_flow_dir_entropy > 2.8:
            # Natural chaotic movement — real video signal.
            # Suppress when saturation is hyperreal (>130): AI can have high flow entropy
            # AND oversaturated colors simultaneously. Don't cancel confirmed AI sat signal.
            # Calibrated: AI_Child flow_ent=3.1 + sat=139 → was -4 (wrong).
            if avg_saturation > 130:
                log.info("FLOW_ENTROPY %.3f → chaotic motion BUT sat=%.0f hyperreal → bonus suppressed",
                         avg_flow_dir_entropy, avg_saturation)
            else:
                ai_score -= 4
                log.info("FLOW_ENTROPY %.3f → chaotic natural motion → -4", avg_flow_dir_entropy)

    # ── 19b. Gravity violation — recurring upward motion (NEW v7) ────────
    # Real-world physics: on a slide, ramp, or hill, subjects move DOWNWARD.
    # AI generators frequently produce reversed trajectories — subjects float
    # upward against gravity in multiple separate events throughout the clip.
    #
    # KEY DISCRIMINATOR: recurring separate episodes vs single camera tilt.
    # A camera panning UP produces ONE continuous upward flow event.
    # A physics violation recurs at MULTIPLE separate moments in the video
    # (person rises on slide, scene cuts, different action, person rises again).
    #
    # Algorithm: count distinct upward episodes (transitions into vy < -2.0).
    # Camera tilt: 1 episode (continuous pan).
    # Gravity violation: 2+ separate episodes at different times.
    #
    # Calibrated: AISlide_1: 4 episodes, min_vert=-17.1 → strong signal ✓
    #             Video_1 (real camera pan): 1 episode → guard protects ✓
    #             AI_Gorilla: 9 episodes (running uphill CGI) → fires ✓
    #             Real videos (1-2): 0 episodes → no signal ✓
    _gravity_violation = False
    if len(vert_flow_scores) > 5 and not _is_short_clip:
        # Count separate upward episodes (transitions from normal → upward flow)
        _upward_episodes = 0
        _in_upward_ep    = False
        for _vf in vert_flow_scores:
            if _vf < -2.0 and not _in_upward_ep:
                _upward_episodes += 1
                _in_upward_ep = True
            elif _vf >= -2.0:
                _in_upward_ep = False

        # NOTE: Signal-engine gravity scoring disabled — camera pans produce
        # identical optical flow signatures to gravity violations at our sampling rates.
        # vert_flow data is passed to GPT via signal_context instead.
        # GPT can interpret the SCENE (person on slide going up) which pixel data cannot.
        # Log the measurement for debugging:
        log.info("GRAVITY data: episodes=%d min_vert=%.2f upward_frac=%.2f (passed to GPT only)",
                 _upward_episodes, min_vert_flow, upward_frame_frac)
        _gravity_violation = (_upward_episodes >= 2 and min_vert_flow < -10.0)

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
    #
    # NOISE GUARD v8: If avg_noise > 500 (clear real camera grain), elevated
    # fg_bg is natural telephoto/DOF physics, not AI rendering.
    # A flying baseball, bird in flight, or close-up animal shot on a real phone
    # camera can reach fg_bg=600–900 due to shallow DOF + blurred background.
    # Real camera noise is the distinguishing factor — AI renders are clean/noiseless.
    _fgbg_noise_guard = (avg_noise > 500)
    if fg_bg_ratio > 1200 and not _fgbg_noise_guard:
        ai_score += 14
        log.info("FG_BG %.0f → extreme AI depth → +14", fg_bg_ratio)
    elif fg_bg_ratio > 900 and not _fgbg_noise_guard:
        ai_score += 8
        log.info("FG_BG %.0f → high AI depth → +8", fg_bg_ratio)
    elif fg_bg_ratio > 600 and not _fgbg_noise_guard:
        ai_score += 3
        log.info("FG_BG %.0f → elevated depth → +3", fg_bg_ratio)
    elif fg_bg_ratio > 600 and _fgbg_noise_guard:
        log.info("FG_BG %.0f → elevated BUT noise=%.0f → real camera DOF, suppressed",
                 fg_bg_ratio, avg_noise)
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
    #
    # OUTDOOR/SPORTS GUARD v8: Outdoor scenes (sports field, park, sky) naturally
    # have limited hue diversity (green grass + brown dirt + blue sky = 3 colors)
    # that mimics AI palette scores. The distinguishing factor is camera noise:
    # real outdoor video has noise > 400, AI renders are clean (noise < 200).
    # If noise > 400 AND hue_ent in the moderate range (1.8–2.5), suppress the
    # moderate AI penalty — it's an outdoor scene, not an AI palette limitation.
    _hue_real_outdoor = (avg_noise > 400)
    if hue_entropy < 1.8:
        # Very limited palette — strong AI signal even outdoors (1.8 is extreme)
        ai_score += 12
        log.info("HUE_ENT %.3f → very limited AI palette → +12", hue_entropy)
    elif hue_entropy < 2.2:
        if _hue_real_outdoor:
            ai_score += 3
            log.info("HUE_ENT %.3f → limited palette BUT noise=%.0f outdoor/real → reduced +3",
                     hue_entropy, avg_noise)
        else:
            ai_score += 7
            log.info("HUE_ENT %.3f → limited AI palette → +7", hue_entropy)
    elif hue_entropy < 2.5:
        if not _hue_real_outdoor:
            ai_score += 3
            log.info("HUE_ENT %.3f → somewhat limited palette → +3", hue_entropy)
        else:
            log.info("HUE_ENT %.3f → somewhat limited BUT noise=%.0f outdoor/real → suppressed",
                     hue_entropy, avg_noise)
    elif hue_entropy > 2.6:
        # Rich natural palette — real video signal.
        # BUT: suppress the real-bonus when saturation is clearly hyperreal (>130).
        # AI generators can produce rich hue diversity AND oversaturated colors
        # simultaneously — the saturation signal already carries the AI evidence.
        # Giving a real-bonus on top would cancel confirmed AI saturation signals.
        # Calibrated: AI_Child sat=139 + hue_ent=4.2 → was -5 (canceling the sat signal).
        if avg_saturation > 130:
            log.info("HUE_ENT %.3f → natural palette BUT sat=%.0f hyperreal → bonus suppressed",
                     hue_entropy, avg_saturation)
        else:
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

    # ═══════════════════════════════════════════════════════════
    # NEW v7 SIGNALS
    # ═══════════════════════════════════════════════════════════

    # ── 25. Motion periodicity (v7) ──────────────────────────
    # AI generators loop a fixed animation template → wave/jog/run looks like a GIF.
    # Autocorrelation peak > 0.65 = the motion pattern repeats with near-perfect regularity.
    # Guards:
    #   • Short clips: autocorrelation unreliable with fewer than 8 motion samples
    #   • Static content: no motion to analyze
    #   • Compilation videos: scene cuts break artificial periodicity measurement
    #   • Camera pan (high flow_var): panning creates smooth periodic-looking motion
    # Portrait action reclassification: portrait + high motion + rich edges → this is
    #   a real action video. AI jogging is portrait action but also has periodicity=0.8.
    #   Use tighter thresholds for portrait action (require stronger autocorrelation).
    _period_guard = (
        _is_short_clip or
        is_static_content or
        _is_compilation or
        avg_motion < 1.0 or
        (avg_flow_var > 20.0)   # camera pan produces smooth motion that looks periodic
    )
    _portrait_action_reclassified = (is_action_content and _is_portrait)
    _period_thresh_strong = 0.75 if _portrait_action_reclassified else 0.65
    _period_thresh_med    = 0.65 if _portrait_action_reclassified else 0.55
    _period_thresh_slight = 0.80 if _portrait_action_reclassified else 0.45
    if not _period_guard:
        if motion_period > _period_thresh_strong:
            ai_score += 14
            log.info("MOTION_PERIOD %.3f → strong animation loop → +14", motion_period)
        elif motion_period > _period_thresh_med:
            ai_score += 9
            log.info("MOTION_PERIOD %.3f → moderate animation loop → +9", motion_period)
        elif motion_period > _period_thresh_slight:
            ai_score += 5
            log.info("MOTION_PERIOD %.3f → slight periodicity → +5", motion_period)
        elif motion_period < 0.20 and avg_motion > 3.0:
            ai_score -= 4
            log.info("MOTION_PERIOD %.3f → organic irregular motion → -4", motion_period)

    # ═══════════════════════════════════════════════════════════
    # NEW v8 SIGNALS
    # ═══════════════════════════════════════════════════════════

    # ── 26. Inter-frame diff variance / IFDV (v8) ────────────
    # Coefficient of variation of frame-to-frame motion scores.
    # AI temporal over-smoothing: CoV < 0.12 (too flat, glass-smooth)
    # AI temporal jitter: CoV > 4.0 (chaotic inconsistency)
    # Real organic: 0.35–1.8
    # Guards: static, short clip, camera pan, compilation
    _ifdv_guard = (
        _is_short_clip or
        is_static_content or
        avg_motion < 0.5 or
        _is_compilation
    )
    if not _ifdv_guard and len(motion_scores) > 4:
        if ifdv < 0.12:
            ai_score += 10
            log.info("IFDV %.3f → temporal over-smooth (AI) → +10", ifdv)
        elif ifdv < 0.20:
            ai_score += 5
            log.info("IFDV %.3f → somewhat over-smooth → +5", ifdv)
        elif ifdv > 4.0:
            ai_score += 8
            log.info("IFDV %.3f → temporal jitter (AI) → +8", ifdv)
        elif ifdv > 2.5:
            ai_score += 4
            log.info("IFDV %.3f → elevated jitter → +4", ifdv)
        elif 0.35 <= ifdv <= 1.8:
            ai_score -= 3
            log.info("IFDV %.3f → organic motion variance (real) → -3", ifdv)

    # ── 27. Flat-region noise floor (v8) — GPT context signal ──
    # NOTE: After H.264/H.265 social media re-encoding, even real camera grain
    # in flat areas is compressed away. The noise_floor signal cannot reliably
    # discriminate AI vs real from compressed video files.
    # We retain the measurement in signal_context for GPT Vision hints but do
    # NOT score it in the signal engine to avoid false positives on compressed real video.
    log.info("NOISE_FLOOR %.2f → context-only (not scored; compression removes real grain)", noise_floor)

    # ── 28. Shadow direction drift (v8) ──────────────────────
    # AI generators lack a global light source constraint.
    # Shadow centroids drift in inconsistent directions across frames.
    # Real scenes: shadows cast by fixed light always move consistently.
    # Guards:
    #   • Static: no movement = no shadow movement to analyze
    #   • Action content: fast camera movement causes shadow centroid to jump
    #     with camera panning — this is real physics, not AI artifact
    #   • High motion (>20): strong camera movement creates false inconsistency
    if not is_static_content and not _is_short_clip and not is_action_content:
        if shadow_drift > 0.82:
            ai_score += 10
            log.info("SHADOW_DRIFT %.3f → strongly inconsistent AI shadows → +10", shadow_drift)
        elif shadow_drift > 0.72:
            ai_score += 6
            log.info("SHADOW_DRIFT %.3f → inconsistent shadows → +6", shadow_drift)
        elif shadow_drift > 0.62:
            ai_score += 3
            log.info("SHADOW_DRIFT %.3f → somewhat drifting shadows → +3", shadow_drift)
        elif shadow_drift < 0.20 and avg_motion > 2.0:
            ai_score -= 3
            log.info("SHADOW_DRIFT %.3f → consistent light source (real) → -3", shadow_drift)

    # ═══════════════════════════════════════════════════════════
    # NEW v9 SIGNALS
    # ═══════════════════════════════════════════════════════════

    # ── 29. Omnidirectional flow entropy (v9) ────────────────
    # This is the COMPLEMENT to the existing crowd lockstep signal (signal 19).
    # Signal 19: LOW entropy = AI lockstep crowd (everyone moves same way)
    # Signal 29: HIGH entropy approaching 4.0 = AI cinematic noise
    #   (all directions equally represented = no coherent physics-based motion)
    #
    # AI water/nature/cinematic: optical flow vectors are distributed UNIFORMLY
    #   across all directions because the generator creates random synthetic motion
    #   without any physics-based dominant direction.
    # Real cinematic content (camera follows subject, animal runs, etc.):
    #   dominated by 1-3 primary directions → entropy 1.3–3.0
    #
    # Calibrated:
    #   AI_Boat=3.723  AI_Gorilla=3.741  AI_Jogging=3.853  AI_Child=3.667
    #   Real_Baseball=2.231  Real_Video1=1.334
    #
    # Guards:
    #   • Person-type content: real person videos have moderate entropy (2.5–3.2) from
    #     natural movement — only the extreme AI range (>3.5) is safe to score here
    #   • Static: no meaningful flow
    #   • Short clip: insufficient temporal data
    #   • Extreme flow_var (>50): hard camera panning creates uniform flow vectors in one
    #     direction, NOT the omnidirectional random pattern — but wait, a fast PAN sweeps
    #     background in ONE direction (low entropy), not all directions (high entropy).
    #     High entropy with extreme flow_var is more likely a complex moving scene than pan.
    #     So we remove the flow_var guard — it was blocking AI jogging incorrectly.
    #   • NOTE: Must NOT overlap with signal 19 (crowd lockstep, low entropy guard).
    _omni_guard = (
        _is_talking_head or
        _is_selfie_content or
        _is_single_subject or
        is_static_content or
        _is_short_clip or
        avg_motion < 1.5                 # not enough motion to analyze
    )
    if not _omni_guard:
        # For person-like content (action, cinematic with skin), use stricter threshold
        # since real person action can reach 3.0–3.3 naturally
        _omni_thresh_strong = 3.7 if (is_action_content or skin_ratio > 0.15) else 3.5
        _omni_thresh_med    = 3.5 if (is_action_content or skin_ratio > 0.15) else 3.2
        _omni_thresh_slight = 3.7 if (is_action_content or skin_ratio > 0.15) else 3.0
        if omni_flow_ent > _omni_thresh_strong:
            ai_score += 12
            log.info("OMNI_FLOW_ENT %.3f → omnidirectional AI noise motion → +12", omni_flow_ent)
        elif omni_flow_ent > _omni_thresh_med:
            ai_score += 7
            log.info("OMNI_FLOW_ENT %.3f → near-omnidirectional motion → +7", omni_flow_ent)
        elif omni_flow_ent > _omni_thresh_slight:
            ai_score += 3
            log.info("OMNI_FLOW_ENT %.3f → slightly omnidirectional → +3", omni_flow_ent)
        elif omni_flow_ent < 2.0 and avg_motion > 2.0:
            # Strong directional dominance = real subject with coherent motion = real video
            ai_score -= 4
            log.info("OMNI_FLOW_ENT %.3f → coherent directional motion (real) → -4", omni_flow_ent)

    # ── 30. Edge temporal coherence (v9) ─────────────────────
    # AI diffusion models produce 'edge crawl' — object boundaries subtly shift
    # between frames even in static-background scenes. This is a temporal decoder
    # artifact where the model cannot enforce strict edge consistency across frames.
    #
    # Real cameras: edges are physically determined by actual objects.
    #   EITHER: static scene → edges stay exactly in place (mean_var high, spiky)
    #   OR: motion → edges move significantly (large mean_var at moving edges)
    #   Real signature: mean_var > 0.08 (real physical edge behavior)
    #
    # AI signature: mean_var < 0.05 AND cov_var > 1.4
    #   Edges exist but shift weakly and uniformly → crawl artifact
    #
    # Calibrated:
    #   AI_Boat:    mean=0.041, cov=1.436  → AI
    #   AI_Gorilla: mean=0.020, cov=1.881  → AI
    #   AI_Moose:   mean=0.022, cov=2.213  → AI (extreme)
    #   Real_Video1: mean=0.165, cov=0.574  → real
    #   Real_Baseball: mean=0.091, cov=1.025 → real (physical motion)
    #
    # Guards: very short clips, static (no edges to track), portrait action
    _edge_coh_guard = (
        _is_short_clip or
        _is_talking_head or
        _is_single_subject or
        is_static_content or
        _is_selfie_content
    )
    if not _edge_coh_guard and len(v9_all_gray_frames) >= 4:
        _edge_ai = (edge_mean_var < 0.05 and edge_cov_var > 1.4)
        _edge_real = (edge_mean_var > 0.08)
        if edge_mean_var < 0.03 and edge_cov_var > 1.8:
            ai_score += 12
            log.info("EDGE_COHERENCE mean=%.4f cov=%.3f → extreme edge crawl (AI) → +12",
                     edge_mean_var, edge_cov_var)
        elif _edge_ai:
            if edge_cov_var > 2.0:
                ai_score += 10
                log.info("EDGE_COHERENCE mean=%.4f cov=%.3f → strong edge crawl (AI) → +10",
                         edge_mean_var, edge_cov_var)
            else:
                ai_score += 7
                log.info("EDGE_COHERENCE mean=%.4f cov=%.3f → edge crawl (AI) → +7",
                         edge_mean_var, edge_cov_var)
        elif edge_mean_var < 0.06 and edge_cov_var > 1.2:
            ai_score += 4
            log.info("EDGE_COHERENCE mean=%.4f cov=%.3f → mild edge crawl → +4",
                     edge_mean_var, edge_cov_var)
        elif _edge_real:
            ai_score -= 4
            log.info("EDGE_COHERENCE mean=%.4f → strong physical edge movement (real) → -4",
                     edge_mean_var)

    # ── 31. Temporal color variance in flat regions (v9) ─────
    # NOTE: TCV needs more diverse calibration data before scoring.
    # Real_Video1 overlaps AI range due to scene lighting variation.
    # Retaining measurement in signal_context for future calibration.
    log.info("TCV %.2f → context-only (not scored; needs calibration)", tcv)

    ai_score = max(0.0, min(100.0, ai_score))
    log.info(
        "Primary AI score v9: %.0f  "
        "(quad=%.3f fg_bg=%.0f sync=%.3f hue=%.2f sat_std=%.1f "
        "period=%.3f ifdv=%.3f nfloor=%.2f shadow=%.3f "
        "omni=%.3f edge_mvar=%.4f edge_cov=%.3f tcv=%.2f)",
        ai_score, quad_cov, fg_bg_ratio, motion_sync, hue_entropy, sat_frame_std,
        motion_period, ifdv, noise_floor, shadow_drift,
        omni_flow_ent, edge_mean_var, edge_cov_var, tcv,
    )

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
        "skin_ratio":     skin_ratio,
        "avg_noise":      avg_noise,
        "vert_flow":      avg_vert_flow,
        "min_vert_flow":  min_vert_flow,
        "upward_frac":    upward_frame_frac,
        "flow_dir_entropy": avg_flow_dir_entropy,
        "n_scene_cuts":   _n_scene_cuts,
        "is_compilation": _is_compilation,
        # v7/v8 signals
        "motion_period":  motion_period,
        "ifdv":           ifdv,
        "noise_floor":    noise_floor,
        "shadow_drift":   shadow_drift,
        # v9 signals
        "omni_flow_ent":     omni_flow_ent,
        "omni_flow_entropy": omni_flow_ent,   # alias for gpt_vision
        "edge_mean_var":     edge_mean_var,
        "edge_cov_var":      edge_cov_var,
        "tcv":               tcv,
        # aliases for gpt_vision _build_physics_summary
        "flat_noise":    _flat_noise,
        "chan_corr":     _chan_corr,
        "dct_score":     avg_dct_grid,
        "hue_entropy":   hue_entropy,
        # v10 audio signals
        "audio_dur_mismatch": _audio.get("dur_mismatch", 0),
        "audio_stereo_corr":  _audio.get("stereo_corr", 0),
        "audio_has_signal":   _audio.get("ai_score_contribution", 0) > 0,
        "audio_reason":       _audio.get("reason", ""),
    }
    return int(round(ai_score)), signal_context