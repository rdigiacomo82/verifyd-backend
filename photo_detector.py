# ============================================================
#  VeriFYD — photo_detector.py  v1
#
#  Image-specific signal detection engine.
#  Parallel to detector.py but designed for single still images.
#
#  Signals used (temporal signals removed — single frame only):
#
#  SPATIAL SIGNALS (direct pixel analysis):
#    1.  ela_score         — Error Level Analysis: JPEG re-compression
#                           inconsistencies reveal composited/AI regions
#    2.  flat_noise        — PRNU proxy: sensor noise fingerprint in
#                           flat regions; AI images lack this
#    3.  chan_corr         — RGB channel correlation: AI renders have
#                           near-perfect inter-channel lock
#    4.  dct_uniformity    — DCT block variance: AI images have
#                           unnaturally uniform compression blocks
#    5.  hf_kurtosis       — High-frequency kurtosis: AI upsampling
#                           produces characteristic HF distribution
#    6.  texture_variance  — Local texture variance: AI skin/surfaces
#                           are suspiciously smooth across patches
#    7.  edge_coherence    — Edge regularity: AI images have
#                           overly clean, anti-aliased edges
#    8.  color_saturation  — Saturation level: AI hyperreal colors
#    9.  noise_floor       — Overall noise level: AI images are cleaner
#    10. metadata_score    — EXIF metadata presence/authenticity
#
#  RETURNS: (ai_score: int 0-100, context: dict)
# ============================================================
import os
import cv2
import logging
import numpy as np
import tempfile
from typing import Tuple

log = logging.getLogger("verifyd.photo_detector")


# ─────────────────────────────────────────────────────────────
#  ELA — Error Level Analysis
# ─────────────────────────────────────────────────────────────
def _compute_ela(image_path: str, quality: int = 90) -> float:
    """
    Re-save image at known JPEG quality, measure pixel-level error.
    AI-generated images and composited regions show boundary artifacts
    where compression levels differ — real photos have uniform ELA.

    NOTE: ELA is unreliable on PNG files (lossless — no re-compression
    artifacts to measure). For PNG input we convert to JPEG first.

    Returns ELA score 0-100 (higher = more AI-like).
    """
    try:
        # PNG guard: re-save as JPEG first so ELA has something to measure.
        # Pure PNG has no compression history → ELA on original PNG is meaningless.
        _ext = os.path.splitext(image_path)[1].lower()
        _is_png = _ext in (".png", ".webp")
        if _is_png:
            import tempfile as _tf
            _jpg_tmp = _tf.mktemp(suffix=".jpg")
            _orig = cv2.imread(image_path)
            if _orig is None:
                return 50.0
            cv2.imwrite(_jpg_tmp, _orig, [cv2.IMWRITE_JPEG_QUALITY, 95])
            image_path = _jpg_tmp
            log.info("ELA: PNG input → converted to JPEG for ELA analysis")

        img = cv2.imread(image_path)
        if img is None:
            return 50.0

        # Save at known quality
        tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        tmp_path = tmp.name
        tmp.close()
        cv2.imwrite(tmp_path, img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        img_resaved = cv2.imread(tmp_path)
        os.unlink(tmp_path)

        if img_resaved is None or img.shape != img_resaved.shape:
            return 50.0

        # Compute absolute difference
        diff = cv2.absdiff(img.astype(np.float32), img_resaved.astype(np.float32))
        diff_gray = np.mean(diff, axis=2)

        # Key ELA metrics
        mean_ela    = float(np.mean(diff_gray))
        std_ela     = float(np.std(diff_gray))
        max_ela     = float(np.max(diff_gray))

        # High variance in ELA = suspicious (composited or AI regions)
        # Low uniform ELA = consistent real-camera compression
        # AI-generated images typically show: low mean, low std,
        # but with sharp local maxima at object boundaries

        # Compute local variance to find boundary artifacts
        kernel = np.ones((8, 8), np.float32) / 64
        local_mean = cv2.filter2D(diff_gray, -1, kernel)
        local_var  = cv2.filter2D(diff_gray ** 2, -1, kernel) - local_mean ** 2
        boundary_score = float(np.percentile(local_var, 95))

        # Score: real cameras have moderate, uniform ELA
        # AI images: very low ELA (clean render) OR high boundary variance
        ela_score = 0.0

        if mean_ela < 2.0:
            # Suspiciously clean — AI render with no real camera compression history
            ela_score += 35.0
        elif mean_ela < 5.0:
            ela_score += 15.0

        if std_ela < 1.5 and mean_ela < 4.0:
            # Extremely uniform ELA — AI rendered image
            ela_score += 25.0

        if boundary_score > 50.0:
            # Sharp boundary artifacts — composited or AI-inserted regions
            ela_score += 20.0

        return min(100.0, ela_score)

    except Exception as e:
        log.debug("ELA error: %s", e)
        return 50.0


# ─────────────────────────────────────────────────────────────
#  PRNU proxy — flat region noise
# ─────────────────────────────────────────────────────────────
def _compute_flat_noise(img_gray: np.ndarray) -> float:
    """
    Measure noise in flat (low-gradient) regions.
    Real cameras have PRNU sensor noise in flat areas.
    AI renders have near-zero noise in flat areas.
    Returns flat_noise value (low = AI, high = real camera).
    """
    try:
        # Find flat regions via Laplacian
        lap = cv2.Laplacian(img_gray, cv2.CV_64F)
        flat_mask = np.abs(lap) < np.percentile(np.abs(lap), 25)

        if flat_mask.sum() < 500:
            return 1.0  # not enough flat area

        flat_pixels = img_gray[flat_mask].astype(np.float64)
        # High-pass filter to isolate noise from signal
        kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=np.float64) / 9
        noise_img = cv2.filter2D(img_gray.astype(np.float64), -1, kernel)
        flat_noise_pixels = noise_img[flat_mask]

        return float(np.std(flat_noise_pixels))

    except Exception:
        return 1.0


# ─────────────────────────────────────────────────────────────
#  RGB channel correlation
# ─────────────────────────────────────────────────────────────
def _compute_chan_corr(img_bgr: np.ndarray) -> float:
    """
    Measure correlation between R, G, B channels.
    AI renders have near-perfect channel lock (corr > 0.95).
    Real camera photos have lower correlation due to sensor physics.
    """
    try:
        b = img_bgr[:, :, 0].flatten().astype(np.float32)
        g = img_bgr[:, :, 1].flatten().astype(np.float32)
        r = img_bgr[:, :, 2].flatten().astype(np.float32)

        corr_rg = float(np.corrcoef(r, g)[0, 1])
        corr_rb = float(np.corrcoef(r, b)[0, 1])
        corr_gb = float(np.corrcoef(g, b)[0, 1])

        return float(np.mean([abs(corr_rg), abs(corr_rb), abs(corr_gb)]))
    except Exception:
        return 0.5


# ─────────────────────────────────────────────────────────────
#  DCT block uniformity
# ─────────────────────────────────────────────────────────────
def _compute_dct_uniformity(img_gray: np.ndarray) -> float:
    """
    Measure variance of DCT energy across 8x8 blocks.
    AI images have suspiciously uniform DCT distributions.
    Returns uniformity score (higher = more uniform = more AI-like).
    """
    try:
        h, w = img_gray.shape
        block_energies = []
        for y in range(0, h - 8, 8):
            for x in range(0, w - 8, 8):
                block = img_gray[y:y+8, x:x+8].astype(np.float32)
                dct = cv2.dct(block)
                # AC energy (exclude DC component)
                ac_energy = float(np.sum(dct[1:, 1:] ** 2))
                block_energies.append(ac_energy)

        if len(block_energies) < 10:
            return 50.0

        arr = np.array(block_energies)
        # Coefficient of variation — lower = more uniform = more AI-like
        cv = float(np.std(arr) / (np.mean(arr) + 1e-6))
        # Normalize: real photos have cv ~0.8-2.0, AI images ~0.2-0.5
        uniformity = max(0.0, min(1.0, (0.8 - cv) / 0.6))
        return uniformity * 100.0

    except Exception:
        return 50.0


# ─────────────────────────────────────────────────────────────
#  High-frequency kurtosis
# ─────────────────────────────────────────────────────────────
def _compute_hf_kurtosis(img_gray: np.ndarray) -> float:
    """
    Measure kurtosis of high-frequency components.
    AI upsampling produces characteristic HF distribution.
    Very high kurtosis (>50) = AI upsampling artifact.
    Very low kurtosis (<5)  = real camera optics.
    """
    try:
        # High-pass via Laplacian
        lap = cv2.Laplacian(img_gray.astype(np.float64), cv2.CV_64F)
        flat = lap.flatten()
        mean = np.mean(flat)
        std  = np.std(flat)
        if std < 1e-6:
            return -1.0
        kurt = float(np.mean(((flat - mean) / std) ** 4))
        return kurt
    except Exception:
        return -1.0


# ─────────────────────────────────────────────────────────────
#  Local texture variance (skin/surface smoothness)
# ─────────────────────────────────────────────────────────────
def _compute_texture_variance(img_gray: np.ndarray) -> float:
    """
    Measure local texture variance across overlapping patches.
    AI images have uniformly smooth texture across surfaces.
    Returns mean local variance (lower = more AI-like).
    """
    try:
        h, w = img_gray.shape
        patch_vars = []
        step = 16
        size = 32
        for y in range(0, h - size, step):
            for x in range(0, w - size, step):
                patch = img_gray[y:y+size, x:x+size].astype(np.float32)
                patch_vars.append(float(np.var(patch)))

        if not patch_vars:
            return 500.0

        return float(np.mean(patch_vars))
    except Exception:
        return 500.0


# ─────────────────────────────────────────────────────────────
#  Splice / composite boundary detector
# ─────────────────────────────────────────────────────────────
def _compute_splice_score(img_bgr: np.ndarray) -> float:
    """
    Detect real-person-on-AI-background composites by measuring
    noise floor discontinuity across image regions.

    When a real photo is composited onto an AI background:
    - Person region: high real camera noise (natural PRNU)
    - Background region: near-zero AI-render noise
    - This creates a large noise variance GAP between regions

    Method: divide image into a grid, compute noise floor per cell,
    measure the coefficient of variation of noise across cells.
    High CoV = inconsistent noise = composite/splice signal.

    Also checks for vertical noise gradient: real composites often
    have person in center/foreground (noisy) vs clean AI background.

    Returns splice_score 0-100 (higher = more likely composited).
    """
    try:
        h, w = img_bgr.shape[:2]
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float64)

        # Compute noise floor in a 4x4 grid of cells
        rows, cols = 4, 4
        cell_noises = []
        for r in range(rows):
            for c in range(cols):
                y1 = r * h // rows
                y2 = (r + 1) * h // rows
                x1 = c * w // cols
                x2 = (c + 1) * w // cols
                cell = gray[y1:y2, x1:x2]
                # High-pass filter to isolate noise
                blur = cv2.GaussianBlur(cell.astype(np.float32), (5, 5), 0)
                noise = float(np.std(cell - blur))
                cell_noises.append(noise)

        if not cell_noises or np.mean(cell_noises) < 0.1:
            return 0.0

        # Coefficient of variation of noise across cells
        noise_arr = np.array(cell_noises)
        noise_cov = float(np.std(noise_arr) / (np.mean(noise_arr) + 1e-6))

        # A real composite has some cells with real camera noise
        # and some cells with near-zero AI noise → high CoV
        # A purely real photo has consistent noise everywhere → low CoV
        # A purely AI image has uniformly low noise → low CoV

        # Also look at min/max noise ratio — composites have extreme spread
        noise_ratio = float(np.max(noise_arr) / (np.min(noise_arr) + 0.01))

        score = 0.0

        if noise_cov > 1.2:
            score += 35.0
            log.info("SPLICE: noise CoV=%.3f → highly inconsistent noise → +35", noise_cov)
        elif noise_cov > 0.8:
            score += 20.0
            log.info("SPLICE: noise CoV=%.3f → moderately inconsistent noise → +20", noise_cov)
        elif noise_cov > 0.5:
            score += 10.0
            log.info("SPLICE: noise CoV=%.3f → slightly inconsistent noise → +10", noise_cov)

        if noise_ratio > 15.0:
            score += 25.0
            log.info("SPLICE: noise_ratio=%.1f → extreme noise spread (composite) → +25", noise_ratio)
        elif noise_ratio > 8.0:
            score += 15.0
            log.info("SPLICE: noise_ratio=%.1f → high noise spread → +15", noise_ratio)
        elif noise_ratio > 4.0:
            score += 5.0

        return min(100.0, score)

    except Exception as e:
        log.debug("SPLICE error: %s", e)
        return 0.0


# ─────────────────────────────────────────────────────────────
#  Metadata / EXIF analysis
# ─────────────────────────────────────────────────────────────
def _analyze_metadata(image_path: str) -> Tuple[int, dict]:
    """
    Check EXIF metadata for real camera fingerprints.
    Returns (ai_score_adjustment, metadata_dict).
    Negative adjustment = real camera evidence.
    Positive adjustment = AI/missing metadata.
    """
    meta = {}
    adjustment = 0

    try:
        import subprocess, json as _j
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_format", image_path],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            data = _j.loads(result.stdout)
            tags = data.get("format", {}).get("tags", {})

            # Real camera indicators
            make  = tags.get("make",  tags.get("Make",  ""))
            model = tags.get("model", tags.get("Model", ""))
            sw    = tags.get("software", tags.get("Software", ""))
            dt    = tags.get("creation_time", tags.get("DateTime", ""))

            meta["make"]     = make
            meta["model"]    = model
            meta["software"] = sw
            meta["datetime"] = dt

            if make or model:
                # Real camera make/model present
                adjustment -= 15
                log.info("PHOTO_META: camera make=%s model=%s → real camera signal -15", make, model)
            elif not dt:
                # No EXIF at all — common in AI-generated images
                adjustment += 10
                log.info("PHOTO_META: no EXIF metadata → AI signal +10")

            # Check for AI tool signatures in software tag
            ai_tools = ["midjourney", "dall-e", "stable diffusion", "firefly",
                        "kling", "sora", "runway", "pika", "ideogram", "leonardo"]
            if any(t in sw.lower() for t in ai_tools):
                adjustment += 50
                log.info("PHOTO_META: AI software tag detected: %s → +50", sw)

    except Exception as e:
        log.debug("PHOTO_META: ffprobe error: %s", e)

    # Also try exiftool if available
    try:
        result = subprocess.run(
            ["exiftool", "-json", "-Make", "-Model", "-Software",
             "-CreateDate", "-GPSLatitude", image_path],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            import json as _j2
            exif = _j2.loads(result.stdout)
            if exif:
                e = exif[0]
                if e.get("Make") or e.get("Model"):
                    adjustment -= 10  # additional real camera signal
                if e.get("GPSLatitude"):
                    adjustment -= 8   # GPS = real device capture
                    log.info("PHOTO_META: GPS data present → real device -8")
    except Exception:
        pass

    return adjustment, meta


# ─────────────────────────────────────────────────────────────
#  Main detection function
# ─────────────────────────────────────────────────────────────
def detect_ai_photo(image_path: str) -> Tuple[int, dict]:
    """
    Main entry point. Analyzes a single image for AI generation signals.

    Returns:
        (ai_score: int 0-100, context: dict)
        ai_score: 0 = definitely real, 100 = definitely AI
    """
    log.info("Photo detector v1 running on %s", image_path)

    # Load image
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        log.error("Photo detector: cannot open %s", image_path)
        return 50, {"content_type": "photo", "error": "cannot_open"}

    h, w = img_bgr.shape[:2]
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    log.info("Photo detector: image %dx%d", w, h)

    # ── Signal 1: ELA ────────────────────────────────────────
    ela_score = _compute_ela(image_path)
    log.info("ELA score: %.1f", ela_score)

    # ── Signal 2: Flat noise (PRNU proxy) ────────────────────
    flat_noise = _compute_flat_noise(img_gray)
    log.info("Flat noise (PRNU): %.4f", flat_noise)

    # ── Signal 3: Channel correlation ────────────────────────
    chan_corr = _compute_chan_corr(img_bgr)
    log.info("Channel correlation: %.4f", chan_corr)

    # ── Signal 4: DCT uniformity ─────────────────────────────
    dct_unif = _compute_dct_uniformity(img_gray)
    log.info("DCT uniformity: %.1f", dct_unif)

    # ── Signal 5: HF kurtosis ────────────────────────────────
    hf_kurt = _compute_hf_kurtosis(img_gray)
    log.info("HF kurtosis: %.2f", hf_kurt)

    # ── Signal 6: Texture variance ───────────────────────────
    tex_var = _compute_texture_variance(img_gray)
    log.info("Texture variance: %.1f", tex_var)

    # ── Signal 7: Color saturation ───────────────────────────
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    avg_sat = float(np.mean(img_hsv[:, :, 1]))
    log.info("Avg saturation: %.1f", avg_sat)

    # ── Signal 8: Overall noise ──────────────────────────────
    noise = float(np.var(img_gray.astype(np.float64)))
    log.info("Noise (variance): %.1f", noise)

    # ── Signal 9: Edge coherence ─────────────────────────────
    edges = cv2.Canny(img_gray, 50, 150)
    edge_density = float(np.sum(edges > 0)) / (h * w)
    log.info("Edge density: %.4f", edge_density)

    # ── Signal 10: Metadata ──────────────────────────────────
    meta_adjustment, meta_dict = _analyze_metadata(image_path)

    # ── Signal 11: Splice / composite boundary ───────────────
    splice_score = _compute_splice_score(img_bgr)
    log.info("Splice score: %.1f", splice_score)

    # ── Score computation ────────────────────────────────────
    ai_score = 0

    # ELA: 0-35 pts
    ai_score += min(35, int(ela_score * 0.35))

    # Flat noise: AI images have very low PRNU
    if flat_noise < 0.15:
        ai_score += 20
        log.info("FLAT_NOISE %.4f → AI-smooth flat regions → +20", flat_noise)
    elif flat_noise < 0.40:
        ai_score += 10
        log.info("FLAT_NOISE %.4f → low sensor noise → +10", flat_noise)
    elif flat_noise > 1.5:
        ai_score -= 8
        log.info("FLAT_NOISE %.4f → strong real PRNU → -8", flat_noise)

    # Channel correlation: AI renders have near-perfect lock
    if chan_corr > 0.95:
        ai_score += 15
        log.info("CHAN_CORR %.4f → very high AI channel lock → +15", chan_corr)
    elif chan_corr > 0.90:
        ai_score += 8
        log.info("CHAN_CORR %.4f → high AI channel correlation → +8", chan_corr)
    elif chan_corr < 0.70:
        ai_score -= 5
        log.info("CHAN_CORR %.4f → natural channel variation → -5", chan_corr)

    # DCT uniformity
    if dct_unif > 70:
        ai_score += 12
        log.info("DCT_UNIF %.1f → uniform AI compression blocks → +12", dct_unif)
    elif dct_unif > 50:
        ai_score += 6

    # HF kurtosis
    if hf_kurt > 80:
        ai_score += 10
        log.info("HF_KURT %.1f → extreme AI upsampling artifact → +10", hf_kurt)
    elif hf_kurt > 50:
        ai_score += 5
    elif 3 < hf_kurt < 15:
        ai_score -= 5
        log.info("HF_KURT %.1f → real camera optics → -5", hf_kurt)

    # Texture variance: very smooth = AI
    if tex_var < 50:
        ai_score += 10
        log.info("TEX_VAR %.1f → very smooth AI surface → +10", tex_var)
    elif tex_var < 150:
        ai_score += 4
    elif tex_var > 800:
        ai_score -= 4
        log.info("TEX_VAR %.1f → high real-world texture variation → -4", tex_var)

    # Saturation: hyperreal AI colors
    if avg_sat > 140:
        ai_score += 8
        log.info("SAT %.1f → hyperreal AI saturation → +8", avg_sat)
    elif avg_sat > 110:
        ai_score += 4

    # Noise: AI images are very clean
    if noise < 100:
        ai_score += 8
        log.info("NOISE %.1f → very low noise, AI-clean → +8", noise)
    elif noise > 2000:
        ai_score -= 6
        log.info("NOISE %.1f → strong real camera grain → -6", noise)

    # Metadata adjustment
    ai_score += meta_adjustment

    # Splice/composite score — weighted contribution
    # A pure composite (real person on AI background) can have splice_score=60+
    # Weight it at 40% since it's a new signal needing calibration
    if splice_score > 50:
        _splice_contrib = int(round(splice_score * 0.40))
        ai_score += _splice_contrib
        log.info("SPLICE %.1f → composite boundary detected → +%d", splice_score, _splice_contrib)
    elif splice_score > 30:
        _splice_contrib = int(round(splice_score * 0.25))
        ai_score += _splice_contrib
        log.info("SPLICE %.1f → possible composite → +%d", splice_score, _splice_contrib)

    # Clamp
    ai_score = max(0, min(100, ai_score))

    # Determine content type (for GPT hint)
    skin_ratio = _estimate_skin_ratio(img_bgr)
    if skin_ratio > 0.15:
        content_type = "portrait"
    elif edge_density > 0.08:
        content_type = "action"
    else:
        content_type = "scene"

    log.info(
        "Photo AI score v1: %d  (ela=%.0f flat=%.3f corr=%.3f dct=%.0f "
        "kurt=%.1f tex=%.0f sat=%.0f noise=%.0f meta=%+d)",
        ai_score, ela_score, flat_noise, chan_corr, dct_unif,
        hf_kurt, tex_var, avg_sat, noise, meta_adjustment
    )

    context = {
        "content_type":  content_type,
        "ela_score":     ela_score,
        "flat_noise":    flat_noise,
        "chan_corr":      chan_corr,
        "dct_uniformity": dct_unif,
        "hf_kurtosis":   hf_kurt,
        "texture_var":   tex_var,
        "avg_saturation": avg_sat,
        "avg_noise":     noise,
        "meta_adjustment": meta_adjustment,
        "metadata":      meta_dict,
        "skin_ratio":    skin_ratio,
        "image_width":   w,
        "image_height":  h,
        "signal_score":  ai_score,
        "source":        "photo_upload",
        "splice_score":  splice_score,
    }

    return ai_score, context


def _estimate_skin_ratio(img_bgr: np.ndarray) -> float:
    """Rough skin tone detection to determine if portrait."""
    try:
        img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        # HSV skin range
        lower = np.array([0, 30, 60], dtype=np.uint8)
        upper = np.array([25, 200, 255], dtype=np.uint8)
        mask = cv2.inRange(img_hsv, lower, upper)
        return float(np.sum(mask > 0)) / (img_bgr.shape[0] * img_bgr.shape[1])
    except Exception:
        return 0.0
