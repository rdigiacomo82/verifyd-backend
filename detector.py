# ============================================================
#  VeriFYD — detector.py  (lightweight, no torch)
#
#  Multi-signal AI video detection using OpenCV + NumPy only.
#
#  Signal categories (research-backed):
#    1. Noise / sensor texture
#    2. Spatial frequency domain
#    3. Edge quality & gradient orientation
#    4. Temporal coherence & inter-frame consistency
#    5. Optical flow regularity
#    6. Local block DCT artifacts (upsampling grid)
#    7. Color channel statistics
#    8. Temporal pixel-wise flicker
#    9. Local texture entropy (Laplacian-of-Gaussian patch variance)
#
#  Returns 0–100 where HIGH = likely AI, LOW = likely real.
# ============================================================

import cv2
import numpy as np
import logging
from typing import List

log = logging.getLogger("verifyd.detector")


# ─────────────────────────────────────────────
#  Frame-level feature extractors
# ─────────────────────────────────────────────

def _noise_score(gray: np.ndarray) -> float:
    """
    Laplacian variance = measure of sensor noise / fine detail.
    Real cameras: higher variance from photon shot noise.
    AI frames: smoother, lower variance.
    """
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _frequency_score(gray: np.ndarray) -> float:
    """
    Ratio of high-frequency energy to total FFT magnitude.
    AI-generated frames tend to lack natural high-freq texture detail.
    """
    f = np.fft.fft2(gray.astype(np.float32))
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)

    rows, cols = gray.shape
    crow, ccol = rows // 2, cols // 2
    radius = min(rows, cols) // 4
    mask = np.ones_like(magnitude, dtype=bool)
    y, x = np.ogrid[:rows, :cols]
    mask[(y - crow) ** 2 + (x - ccol) ** 2 <= radius ** 2] = False

    total = magnitude.sum() + 1e-10
    high_freq = magnitude[mask].sum()
    return float(high_freq / total)


def _dct_grid_artifact(gray: np.ndarray, block_size: int = 8) -> float:
    """
    AI video generators (diffusion / GAN upsamplers) produce subtle periodic
    grid patterns at the block level, detectable as peaks in the DCT spectrum.
    We compute average DCT energy at DC+border vs interior coefficients.
    Higher ratio → more grid-like structure → more likely AI.
    """
    h, w = gray.shape
    h = (h // block_size) * block_size
    w = (w // block_size) * block_size
    if h == 0 or w == 0:
        return 0.0

    img = gray[:h, :w].astype(np.float32)
    border_energy = 0.0
    interior_energy = 0.0
    n_blocks = 0

    for r in range(0, h, block_size):
        for c in range(0, w, block_size):
            block = img[r:r+block_size, c:c+block_size]
            dct = cv2.dct(block)
            # Border coefficients (first row + first col, excluding DC)
            border = np.abs(dct[0, 1:]).sum() + np.abs(dct[1:, 0]).sum()
            # Interior coefficients
            interior = np.abs(dct[1:, 1:]).sum()
            border_energy += border
            interior_energy += interior
            n_blocks += 1

    if interior_energy < 1e-10:
        return 0.0
    return float(border_energy / (interior_energy + 1e-10))


def _edge_quality(gray: np.ndarray) -> float:
    """
    Canny edge density. Real video: sharper, denser edges from real-world
    textures. AI: softer, fewer genuine edges.
    """
    edges = cv2.Canny(gray, 50, 150)
    return float(np.mean(edges))


def _gradient_orientation_entropy(gray: np.ndarray) -> float:
    """
    Real scenes produce a wide distribution of gradient orientations
    (many textures, random angles). AI-generated content tends toward
    dominant orientation clusters — lower entropy.
    Returns Shannon entropy of the gradient orientation histogram.
    """
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    angles = np.arctan2(gy, gx)  # -pi to pi
    hist, _ = np.histogram(angles, bins=36, range=(-np.pi, np.pi))
    hist = hist.astype(np.float64) + 1e-10
    hist /= hist.sum()
    entropy = -np.sum(hist * np.log2(hist))
    return float(entropy)


def _local_texture_entropy(gray: np.ndarray, patch_size: int = 16) -> float:
    """
    Measures how varied local patch textures are across the frame.
    Uses Laplacian variance per patch: real images have diverse local
    sharpness; AI images are uniformly smooth or sharp.
    Returns std-dev of per-patch Laplacian variance (lower = more uniform = AI).
    """
    h, w = gray.shape
    variances = []
    step = patch_size
    for r in range(0, h - patch_size, step):
        for c in range(0, w - patch_size, step):
            patch = gray[r:r+patch_size, c:c+patch_size]
            v = cv2.Laplacian(patch, cv2.CV_64F).var()
            variances.append(v)
    if not variances:
        return 0.0
    return float(np.std(variances))


def _color_channel_noise_correlation(frame_bgr: np.ndarray) -> float:
    """
    Real camera sensors have correlated noise across channels due to demosaicing
    and optics. AI generators produce channels more independently.
    Returns Pearson correlation between noise residuals of B and R channels.
    Higher absolute correlation → more camera-like → less AI.
    We return the (1 - |corr|) so higher = more AI-like.
    """
    b, g, r = cv2.split(frame_bgr.astype(np.float32))

    def noise_residual(ch):
        blurred = cv2.GaussianBlur(ch, (5, 5), 0)
        return (ch - blurred).flatten()

    nb = noise_residual(b)
    nr = noise_residual(r)

    if nb.std() < 1e-6 or nr.std() < 1e-6:
        return 0.5

    corr = float(np.corrcoef(nb, nr)[0, 1])
    return float(1.0 - abs(corr))  # higher = channels more independent = AI


def _saturation_mean(frame_bgr: np.ndarray) -> float:
    """Mean HSV saturation. AI video can have unrealistic saturation levels."""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    return float(hsv[:, :, 1].mean())


# ─────────────────────────────────────────────
#  Temporal (multi-frame) feature extractors
# ─────────────────────────────────────────────

def _optical_flow_regularity(prev_gray: np.ndarray, curr_gray: np.ndarray) -> float:
    """
    Compute dense optical flow and return the spatial variance of flow magnitude.
    Real video: chaotic, varied local motion. AI: too-uniform or patchy flow.
    Returns variance of flow magnitude (low = suspiciously smooth = AI-like).
    """
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2,
        flags=0
    )
    magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
    return float(np.var(magnitude))


def _temporal_pixel_flicker(frames_gray: List[np.ndarray]) -> float:
    """
    Pixel-wise temporal flicker: compute std-dev of pixel values over time,
    then look at the distribution of per-pixel variance.
    AI videos show either too-stable pixels OR high-frequency flicker that is
    spatially incoherent. We return the coefficient of variation of per-pixel
    temporal std-devs (high CoV = inconsistent flicker across pixels = AI).
    """
    if len(frames_gray) < 3:
        return 0.0
    stack = np.stack(frames_gray, axis=0).astype(np.float32)
    pixel_std = np.std(stack, axis=0)  # (H, W)
    mean_std = pixel_std.mean() + 1e-10
    cov = float(pixel_std.std() / mean_std)
    return cov


def _inter_frame_residual_consistency(frames_gray: List[np.ndarray]) -> float:
    """
    Measure how consistent inter-frame difference patterns are over time.
    Real video: diff magnitudes vary naturally with scene motion.
    AI video: differences are sometimes too regular or have characteristic
    artifacts repeated at fixed intervals.
    Returns variance-of-variance of frame differences (low = too regular = AI).
    """
    if len(frames_gray) < 4:
        return 0.0
    diff_vars = []
    for i in range(1, len(frames_gray)):
        diff = cv2.absdiff(frames_gray[i], frames_gray[i-1]).astype(np.float32)
        diff_vars.append(float(np.var(diff)))
    return float(np.var(diff_vars))


# ─────────────────────────────────────────────
#  Main detection function
# ─────────────────────────────────────────────

def detect_ai(video_path: str) -> int:
    """
    Analyze video for AI-generation signals.
    Returns 0–100 where HIGH = likely AI, LOW = likely real.
    """
    log.info("Primary detector running on %s", video_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        log.error("Could not open video: %s", video_path)
        return 50

    # Per-frame accumulators
    noise_scores: List[float] = []
    freq_scores: List[float] = []
    edge_scores: List[float] = []
    dct_grid_scores: List[float] = []
    grad_entropy_scores: List[float] = []
    texture_entropy_scores: List[float] = []
    color_corr_scores: List[float] = []
    saturation_scores: List[float] = []
    flow_regularity_scores: List[float] = []
    motion_scores: List[float] = []

    # Buffers for temporal multi-frame analysis
    gray_buffer: List[np.ndarray] = []

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

        # Sample every 5th frame, max 60 samples (~300 frames)
        if frame_count % 5 != 0:
            continue
        if samples >= 60:
            break

        samples += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ── Spatial frame-level signals ──
        noise_scores.append(_noise_score(gray))
        freq_scores.append(_frequency_score(gray))
        edge_scores.append(_edge_quality(gray))
        dct_grid_scores.append(_dct_grid_artifact(gray))
        grad_entropy_scores.append(_gradient_orientation_entropy(gray))
        texture_entropy_scores.append(_local_texture_entropy(gray))
        color_corr_scores.append(_color_channel_noise_correlation(frame))
        saturation_scores.append(_saturation_mean(frame))

        # ── Temporal: motion & histogram ──
        if prev_gray is not None:
            diff = cv2.absdiff(gray, prev_gray)
            motion_scores.append(float(np.mean(diff)))
            flow_var = _optical_flow_regularity(prev_gray, gray)
            flow_regularity_scores.append(flow_var)

        hist = cv2.calcHist([gray], [0], None, [64], [0, 256])
        hist = hist.flatten() / (hist.sum() + 1e-10)
        if prev_hist is not None:
            temporal_diffs.append(float(np.sum(np.abs(hist - prev_hist))))

        prev_gray = gray.copy()
        prev_hist = hist

        # ── Rolling buffer for pixel-flicker analysis (last 10 frames) ──
        gray_buffer.append(gray)
        if len(gray_buffer) > 10:
            gray_buffer.pop(0)

    cap.release()

    if not noise_scores:
        log.warning("No frames analyzed")
        return 50

    # ─────────────────────────────────────────────
    #  Aggregate signals
    # ─────────────────────────────────────────────

    avg_noise            = float(np.mean(noise_scores))
    avg_freq             = float(np.mean(freq_scores))
    avg_edge             = float(np.mean(edge_scores))
    avg_dct_grid         = float(np.mean(dct_grid_scores))
    avg_grad_entropy     = float(np.mean(grad_entropy_scores))
    avg_tex_entropy      = float(np.mean(texture_entropy_scores))
    avg_color_corr       = float(np.mean(color_corr_scores))
    avg_saturation       = float(np.mean(saturation_scores))
    avg_motion           = float(np.mean(motion_scores)) if motion_scores else 0.0
    motion_var           = float(np.var(motion_scores))  if len(motion_scores) > 1 else 0.0
    avg_temporal_jitter  = float(np.mean(temporal_diffs)) if temporal_diffs else 0.0
    avg_flow_var         = float(np.mean(flow_regularity_scores)) if flow_regularity_scores else 0.0

    # Multi-frame signals
    pixel_flicker_cov    = _temporal_pixel_flicker(gray_buffer)
    residual_var_of_var  = _inter_frame_residual_consistency(gray_buffer)

    log.info(
        "Signals: noise=%.1f freq=%.3f edge=%.1f dct_grid=%.3f "
        "grad_ent=%.3f tex_ent=%.1f color_corr=%.3f sat=%.1f "
        "motion=%.1f motion_var=%.2f temporal=%.4f flow_var=%.2f "
        "flicker_cov=%.3f residual_vov=%.2f",
        avg_noise, avg_freq, avg_edge, avg_dct_grid,
        avg_grad_entropy, avg_tex_entropy, avg_color_corr, avg_saturation,
        avg_motion, motion_var, avg_temporal_jitter, avg_flow_var,
        pixel_flicker_cov, residual_var_of_var,
    )

    # ─────────────────────────────────────────────
    #  Scoring  (start at 50 = uncertain)
    #  Each signal contributes a bounded adjustment.
    #  Total max swing: ±85 before clamp → clamp to [0,100].
    # ─────────────────────────────────────────────

    ai_score = 50.0

    # ── 1. Sensor noise (Laplacian variance) ──────────────────────────────
    # Real cameras: higher noise floor (sensor + demosaicing).
    # AI: very clean output. Thresholds tuned to typical resolutions.
    if avg_noise < 80:
        ai_score += 18
    elif avg_noise < 200:
        ai_score += 8
    elif avg_noise > 600:
        ai_score -= 15
    elif avg_noise > 350:
        ai_score -= 7

    # ── 2. High-frequency content ─────────────────────────────────────────
    # AI generation (especially diffusion) suppresses fine-scale texture.
    if avg_freq < 0.55:
        ai_score += 14
    elif avg_freq < 0.65:
        ai_score += 6
    elif avg_freq > 0.82:
        ai_score -= 12
    elif avg_freq > 0.75:
        ai_score -= 5

    # ── 3. Edge density ───────────────────────────────────────────────────
    if avg_edge < 8:
        ai_score += 10
    elif avg_edge < 15:
        ai_score += 4
    elif avg_edge > 35:
        ai_score -= 8

    # ── 4. DCT grid artifact ──────────────────────────────────────────────
    # High border/interior DCT ratio = upsampling grid artifacts common in
    # GAN/diffusion generators.
    if avg_dct_grid > 1.8:
        ai_score += 12
    elif avg_dct_grid > 1.4:
        ai_score += 6
    elif avg_dct_grid < 0.9:
        ai_score -= 6

    # ── 5. Gradient orientation entropy ──────────────────────────────────
    # Real scenes: rich diversity of angles → high entropy.
    # AI: tends to cluster orientations → lower entropy.
    if avg_grad_entropy < 4.0:
        ai_score += 10
    elif avg_grad_entropy < 4.5:
        ai_score += 4
    elif avg_grad_entropy > 5.0:
        ai_score -= 8

    # ── 6. Local texture entropy (patch Laplacian std-dev) ───────────────
    # Real images: highly varied local sharpness.
    # AI: more spatially uniform sharpness → lower std of patch variances.
    if avg_tex_entropy < 50:
        ai_score += 10
    elif avg_tex_entropy < 120:
        ai_score += 4
    elif avg_tex_entropy > 400:
        ai_score -= 8

    # ── 7. Color channel noise correlation ───────────────────────────────
    # Real cameras: correlated noise between channels (demosaicing).
    # AI: channels generated somewhat independently → lower correlation.
    # avg_color_corr = (1 - |corr|): high = independent = AI
    if avg_color_corr > 0.75:
        ai_score += 8
    elif avg_color_corr > 0.60:
        ai_score += 3
    elif avg_color_corr < 0.35:
        ai_score -= 6

    # ── 8. Motion consistency ─────────────────────────────────────────────
    # Very low / zero motion → likely static synthetic background.
    # Unnaturally low motion variance → AI temporal smoothing.
    if avg_motion < 2:
        ai_score += 8
    if motion_var < 0.5 and len(motion_scores) > 3:
        ai_score += 8

    # ── 9. Histogram temporal stability ──────────────────────────────────
    # Too-smooth histogram changes → AI temporal blending artifacts.
    if avg_temporal_jitter < 0.008 and len(temporal_diffs) > 3:
        ai_score += 8
    elif avg_temporal_jitter > 0.06:
        ai_score -= 8

    # ── 10. Optical flow spatial variance ─────────────────────────────────
    # Real motion: varied local flow (objects at different depths/speeds).
    # AI: flow either too uniform (no parallax) or spatially disjointed.
    if len(flow_regularity_scores) > 3:
        if avg_flow_var < 5.0:
            ai_score += 8   # too-uniform motion → AI
        elif avg_flow_var < 15.0:
            ai_score += 3
        elif avg_flow_var > 200.0:
            ai_score -= 6   # chaotic, camera-shake-style real motion

    # ── 11. Pixel-wise temporal flicker (CoV of per-pixel std) ───────────
    # Incoherent flicker pattern = AI compression / generation artifact.
    # Very low CoV = all pixels flicker identically = also suspicious.
    if pixel_flicker_cov > 2.5:
        ai_score += 6   # spatially incoherent flicker
    elif pixel_flicker_cov < 0.3 and len(gray_buffer) >= 5:
        ai_score += 4   # suspiciously uniform flicker

    # ── 12. Inter-frame residual variance-of-variance ─────────────────────
    # Very low variance-of-variance = frame diffs are too regular = AI.
    if residual_var_of_var < 100 and len(gray_buffer) >= 5:
        ai_score += 5
    elif residual_var_of_var > 50000:
        ai_score -= 5   # highly irregular = real camera motion

    ai_score = max(0.0, min(100.0, ai_score))

    log.info("Primary AI score: %.0f", ai_score)
    return int(round(ai_score))
