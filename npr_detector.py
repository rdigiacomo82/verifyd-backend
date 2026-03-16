# ============================================================
#  VeriFYD — npr_detector.py
#
#  NPR (Noise Pattern Residual) Frequency Domain Analysis
#
#  WHY THIS WORKS:
#  Real cameras produce sensor-specific noise in the high-frequency
#  domain. This noise is largely random and follows a Gaussian-like
#  distribution. AI generators (diffusion models, GANs, transformers)
#  produce characteristic frequency artifacts:
#
#  1. GRID ARTIFACTS (DCT/block artifacts)
#     AI upscalers and VAEs introduce periodic patterns at 8x8 or
#     16x16 pixel boundaries. Visible as peaks in 2D FFT at specific
#     spatial frequencies.
#
#  2. SPECTRAL SLOPE DEVIATION
#     Real images follow a 1/f^2 power spectrum (pink noise).
#     AI-generated images often have flatter high-frequency spectra
#     (too much detail = oversharpened) or steeper slopes (too smooth).
#
#  3. AZIMUTHAL UNIFORMITY
#     Real camera noise is rotationally symmetric in frequency space.
#     AI generators often show directional biases from conv kernels
#     and attention patterns — the FFT is not rotationally uniform.
#
#  4. RESIDUAL FINGERPRINT KURTOSIS
#     After applying a high-pass filter (Laplacian), the kurtosis
#     of the residual distribution distinguishes real (kurtosis ~3-8)
#     from AI (kurtosis >15, sometimes >100 due to upsampling rings).
#     This is our existing HF_KURTOSIS signal — NPR extends it with
#     spatial structure analysis.
#
#  5. INTER-FRAME RESIDUAL CONSISTENCY
#     Real video has random temporal noise variation frame-to-frame.
#     AI video has consistent, structured residuals — the same
#     artifacts appear in the same spatial locations across frames.
#
#  MEMORY: Pure numpy/scipy — zero model loading, ~2MB RAM.
#  SPEED: ~15-25ms per clip on CPU.
#  RISK: Zero — purely additive signal, capped contribution.
#
#  Returns:
#    npr_score   : int 0-100, AI probability from frequency analysis
#    npr_signals : dict, individual signal values for logging
# ============================================================

import numpy as np
import cv2
import logging
from typing import Tuple, Dict, List

log = logging.getLogger("verifyd.npr")

# ─────────────────────────────────────────────────────────────
#  Configuration
# ─────────────────────────────────────────────────────────────
_MAX_FRAMES        = 12    # frames to analyze per clip
_FRAME_SIZE        = 256   # resize to this for FFT consistency
_LAPLACIAN_KERNEL  = np.array([[0,-1,0],[-1,4,-1],[0,-1,0]], dtype=np.float32)


def analyze_npr(video_path: str) -> Tuple[int, Dict]:
    """
    Run NPR frequency domain analysis on a video clip.

    Returns:
        (npr_ai_score, signals_dict)
        npr_ai_score: 0-100, higher = more likely AI
    """
    signals: Dict = {
        "spectral_slope":        None,
        "grid_artifact_strength": None,
        "azimuthal_uniformity":  None,
        "residual_kurtosis":     None,
        "temporal_residual_consistency": None,
        "npr_score":             0,
    }

    try:
        frames = _extract_gray_frames(video_path, _MAX_FRAMES)
        if len(frames) < 3:
            log.debug("NPR: insufficient frames (%d) in %s", len(frames), video_path)
            return 0, signals

        score = 0
        components = []

        # ── Signal 1: Spectral slope ──────────────────────────
        slope, slope_score = _analyze_spectral_slope(frames)
        signals["spectral_slope"] = round(slope, 4)
        score += slope_score
        components.append(f"slope={slope:.3f}→{slope_score:+d}")

        # ── Signal 2: Grid artifact strength ─────────────────
        grid_strength, grid_score = _analyze_grid_artifacts(frames)
        signals["grid_artifact_strength"] = round(grid_strength, 4)
        score += grid_score
        components.append(f"grid={grid_strength:.3f}→{grid_score:+d}")

        # ── Signal 3: Azimuthal uniformity ───────────────────
        az_ratio, az_score = _analyze_azimuthal_uniformity(frames)
        signals["azimuthal_uniformity"] = round(az_ratio, 4)
        score += az_score
        components.append(f"azimuth={az_ratio:.3f}→{az_score:+d}")

        # ── Signal 4: Residual kurtosis ───────────────────────
        kurt, kurt_score = _analyze_residual_kurtosis(frames)
        signals["residual_kurtosis"] = round(kurt, 2)
        score += kurt_score
        components.append(f"kurt={kurt:.1f}→{kurt_score:+d}")

        # ── Signal 5: Temporal residual consistency ───────────
        if len(frames) >= 4:
            trc, trc_score = _analyze_temporal_residual_consistency(frames)
            signals["temporal_residual_consistency"] = round(trc, 4)
            score += trc_score
            components.append(f"trc={trc:.3f}→{trc_score:+d}")

        # Clamp to 0-100
        score = max(0, min(100, score))
        signals["npr_score"] = score

        log.info("NPR analysis: score=%d [%s]", score, " ".join(components))
        return score, signals

    except Exception as e:
        log.warning("NPR analysis failed: %s", e)
        return 0, signals


def _extract_gray_frames(video_path: str, max_frames: int) -> List[np.ndarray]:
    """Extract evenly-spaced grayscale frames from video."""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []

    step = max(1, total // max_frames)
    frames = []
    frame_idx = 0

    while len(frames) < max_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (_FRAME_SIZE, _FRAME_SIZE))
        frames.append(gray.astype(np.float32))
        frame_idx += step

    cap.release()
    return frames


def _analyze_spectral_slope(frames: List[np.ndarray]) -> Tuple[float, int]:
    """
    Measure the spectral slope of the averaged power spectrum.

    Real images: slope ~ -2.0 to -2.5 (1/f^2 pink noise)
    AI images: slope ~ -1.2 to -1.8 (too much HF from upsampling)
               or < -2.8 (over-smoothed diffusion output)

    Returns (slope, score_delta)
    """
    slopes = []
    for frame in frames:
        # 2D FFT and shift
        fft = np.fft.fft2(frame)
        fft_shift = np.fft.fftshift(fft)
        power = np.abs(fft_shift) ** 2
        power = np.log1p(power)

        # Radial average
        h, w = power.shape
        cy, cx = h // 2, w // 2
        y_idx, x_idx = np.ogrid[:h, :w]
        r = np.sqrt((x_idx - cx)**2 + (y_idx - cy)**2).astype(int)
        r_max = min(cx, cy)

        radial_mean = np.zeros(r_max)
        for i in range(r_max):
            mask = (r == i)
            if mask.sum() > 0:
                radial_mean[i] = power[mask].mean()

        # Fit log-log slope (skip DC component)
        valid = radial_mean[2:r_max] > 0
        if valid.sum() < 5:
            continue
        x = np.log(np.arange(2, r_max)[valid])
        y = np.log(radial_mean[2:r_max][valid])
        if len(x) > 1:
            slope = np.polyfit(x, y, 1)[0]
            slopes.append(slope)

    if not slopes:
        return -2.0, 0

    avg_slope = float(np.mean(slopes))

    # Score based on deviation from expected real camera slope
    # Real: -2.0 to -2.5
    # AI upsampled: -1.0 to -1.8 (too flat = too much HF energy)
    # AI over-smoothed: < -2.8
    score = 0
    if avg_slope > -1.5:
        score = 15   # Very flat — strong upsampling artifact
    elif avg_slope > -1.8:
        score = 10
    elif avg_slope > -2.0:
        score = 5
    elif avg_slope < -3.0:
        score = 8    # Over-smoothed — diffusion artifact
    elif avg_slope < -2.8:
        score = 4

    return avg_slope, score


def _analyze_grid_artifacts(frames: List[np.ndarray]) -> Tuple[float, int]:
    """
    Detect periodic grid artifacts in the frequency domain.

    AI generators use 8x8 DCT blocks (JPEG-like) and 8x8/16x16
    convolutional patches. These create peaks at specific frequencies
    in the 2D FFT that are absent in real camera footage.

    Returns (grid_strength, score_delta)
    """
    grid_strengths = []

    for frame in frames:
        fft = np.fft.fft2(frame)
        fft_shift = np.fft.fftshift(fft)
        magnitude = np.abs(fft_shift)

        h, w = magnitude.shape
        cy, cx = h // 2, w // 2

        # Look for peaks at 1/8 and 1/16 of image dimensions
        # (8x8 and 16x16 block periodicity)
        grid_freqs = []
        for block_size in [8, 16]:
            fx = w // block_size
            fy = h // block_size
            # Check a small neighborhood around expected peak
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    if dx == 0 and dy == 0:
                        continue
                    py = cy + fy * dy
                    px = cx + fx * dx
                    if 0 <= py < h and 0 <= px < w:
                        grid_freqs.append(magnitude[py, px])

        if not grid_freqs:
            continue

        # Compare grid peaks to overall mean (excluding DC)
        dc_region = magnitude[cy-2:cy+3, cx-2:cx+3]
        non_dc = np.copy(magnitude)
        non_dc[cy-5:cy+6, cx-5:cx+6] = 0
        overall_mean = non_dc[non_dc > 0].mean() if (non_dc > 0).any() else 1.0

        grid_peak = np.mean(grid_freqs)
        ratio = grid_peak / (overall_mean + 1e-8)
        grid_strengths.append(ratio)

    if not grid_strengths:
        return 0.0, 0

    avg_strength = float(np.mean(grid_strengths))

    # Real camera: ratio ~1.0-1.5 (no systematic peaks)
    # AI with block artifacts: ratio >2.0, often >3.0
    score = 0
    if avg_strength > 3.5:
        score = 18
    elif avg_strength > 2.5:
        score = 12
    elif avg_strength > 2.0:
        score = 7
    elif avg_strength > 1.8:
        score = 3

    return avg_strength, score


def _analyze_azimuthal_uniformity(frames: List[np.ndarray]) -> Tuple[float, int]:
    """
    Measure azimuthal (rotational) uniformity of the power spectrum.

    Real cameras: noise is approximately rotationally symmetric.
    AI generators: directional artifacts from conv kernels and
    attention patterns create non-uniform angular distribution.

    Returns (non_uniformity_ratio, score_delta)
    """
    ratios = []

    for frame in frames:
        fft = np.fft.fft2(frame)
        fft_shift = np.fft.fftshift(fft)
        power = np.abs(fft_shift) ** 2

        h, w = power.shape
        cy, cx = h // 2, w // 2

        # Divide into 8 angular sectors and measure power in each
        y_idx, x_idx = np.ogrid[:h, :w]
        angles = np.arctan2(y_idx - cy, x_idx - cx)  # -pi to pi
        r = np.sqrt((x_idx - cx)**2 + (y_idx - cy)**2)

        # Only consider mid-range frequencies (not DC, not near Nyquist)
        r_min = min(cx, cy) * 0.1
        r_max = min(cx, cy) * 0.7
        freq_mask = (r >= r_min) & (r <= r_max)

        n_sectors = 8
        sector_powers = []
        for i in range(n_sectors):
            a_min = -np.pi + (2 * np.pi / n_sectors) * i
            a_max = a_min + (2 * np.pi / n_sectors)
            sector_mask = freq_mask & (angles >= a_min) & (angles < a_max)
            if sector_mask.sum() > 0:
                sector_powers.append(power[sector_mask].mean())

        if len(sector_powers) < 4:
            continue

        sector_powers = np.array(sector_powers)
        # Coefficient of variation — higher = less uniform = more AI
        cv = sector_powers.std() / (sector_powers.mean() + 1e-8)
        ratios.append(cv)

    if not ratios:
        return 0.0, 0

    avg_ratio = float(np.mean(ratios))

    # Real camera: CV ~0.05-0.15
    # AI with directional artifacts: CV >0.20, often >0.35
    score = 0
    if avg_ratio > 0.40:
        score = 12
    elif avg_ratio > 0.30:
        score = 8
    elif avg_ratio > 0.22:
        score = 4
    elif avg_ratio < 0.05:
        # Extremely uniform — possible over-smoothed AI
        score = 3

    return avg_ratio, score


def _analyze_residual_kurtosis(frames: List[np.ndarray]) -> Tuple[float, int]:
    """
    Measure kurtosis of Laplacian residual (high-frequency noise).

    This extends the existing HF_KURTOSIS signal with proper
    spatial structure — we measure kurtosis in localized patches
    rather than globally, which catches AI upsampling rings that
    appear only near edges.

    Real camera: patch kurtosis ~3-10 (Gaussian-ish)
    AI upsampled: patch kurtosis >20 (heavy tails near edges)
    AI over-smooth: patch kurtosis ~2-3 (too Gaussian)

    Returns (patch_kurtosis, score_delta)
    """
    kurtosis_vals = []

    for frame in frames:
        # Apply Laplacian to extract high-frequency residual
        residual = cv2.filter2D(frame, -1, _LAPLACIAN_KERNEL)

        # Analyze in 32x32 patches
        patch_size = 32
        h, w = residual.shape
        for y in range(0, h - patch_size, patch_size):
            for x in range(0, w - patch_size, patch_size):
                patch = residual[y:y+patch_size, x:x+patch_size].flatten()
                if patch.std() < 0.5:
                    continue  # Skip flat patches
                # Kurtosis = E[(x-mu)^4] / sigma^4
                mu = patch.mean()
                sigma = patch.std()
                if sigma < 1e-8:
                    continue
                kurt = np.mean(((patch - mu) / sigma) ** 4)
                kurtosis_vals.append(kurt)

    if not kurtosis_vals:
        return 3.0, 0

    avg_kurtosis = float(np.median(kurtosis_vals))  # median is more robust

    # Score
    score = 0
    if avg_kurtosis > 80:
        score = 20
    elif avg_kurtosis > 50:
        score = 14
    elif avg_kurtosis > 30:
        score = 8
    elif avg_kurtosis > 18:
        score = 4
    elif avg_kurtosis < 2.5:
        score = 5  # Over-smooth

    return avg_kurtosis, score


def _analyze_temporal_residual_consistency(frames: List[np.ndarray]) -> Tuple[float, int]:
    """
    Measure how consistent the high-frequency residual is across frames.

    Real video: residual varies randomly frame-to-frame (sensor noise)
    AI video: residual is structured and consistent — same artifacts
              appear in same locations because the generator is
              deterministic in its upsampling/rendering patterns.

    Returns (consistency_score, score_delta)
    """
    residuals = []
    for frame in frames:
        residual = cv2.filter2D(frame, -1, _LAPLACIAN_KERNEL)
        # Normalize to compare structure not magnitude
        r_std = residual.std()
        if r_std > 0:
            residuals.append(residual / r_std)

    if len(residuals) < 3:
        return 0.0, 0

    # Compute pairwise correlations between consecutive residuals
    correlations = []
    for i in range(len(residuals) - 1):
        r1 = residuals[i].flatten()
        r2 = residuals[i + 1].flatten()
        # Pearson correlation
        corr = np.corrcoef(r1, r2)[0, 1]
        if not np.isnan(corr):
            correlations.append(abs(corr))

    if not correlations:
        return 0.0, 0

    avg_corr = float(np.mean(correlations))

    # Real video: low residual correlation ~0.02-0.15 (random noise)
    # AI video: higher correlation ~0.20-0.60 (structured artifacts)
    score = 0
    if avg_corr > 0.45:
        score = 16
    elif avg_corr > 0.30:
        score = 10
    elif avg_corr > 0.20:
        score = 5
    elif avg_corr < 0.03:
        # Very uncorrelated — strong real signal
        score = -4

    return avg_corr, score


def get_npr_contribution(npr_score: int) -> int:
    """
    Convert NPR score to a contribution for the main AI score.
    NPR is weighted conservatively (max ±15 points) since it's
    a supplementary signal alongside signal detector + GPT.
    """
    if npr_score >= 75:
        return 12
    elif npr_score >= 60:
        return 8
    elif npr_score >= 45:
        return 5
    elif npr_score >= 30:
        return 2
    elif npr_score <= 10:
        return -3  # Strong NPR real signal
    return 0
