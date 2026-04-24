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

    Returns ELA score 0-100 (higher = more AI-like).
    """
    try:
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
#  HEIC / HEIF conversion helper
# ─────────────────────────────────────────────────────────────
def _convert_heic_to_jpg(image_path: str) -> str:
    """
    Convert HEIC/HEIF to JPEG so OpenCV can process it.
    iPhone photos are HEIC by default — cv2.imread cannot open them natively.
    
    Tries three methods in order:
      1. ffmpeg (already installed on Render for video processing)
      2. pillow-heif / pillow if available
      3. ImageMagick convert if available
    
    Returns path to converted JPEG, or original path if conversion fails.
    The caller is responsible for cleaning up the temp file.
    """
    import subprocess, tempfile as _tf

    ext = os.path.splitext(image_path)[1].lower()
    if ext not in ('.heic', '.heif'):
        return image_path  # not HEIC, nothing to do

    tmp_jpg = _tf.mktemp(suffix='.jpg')

    # Method 1: ffmpeg (fastest, already on the server)
    try:
        result = subprocess.run([
            'ffmpeg', '-y', '-i', image_path,
            '-q:v', '2',   # high quality JPEG
            tmp_jpg
        ], capture_output=True, timeout=30)
        if result.returncode == 0 and os.path.exists(tmp_jpg) and os.path.getsize(tmp_jpg) > 1000:
            log.info("HEIC: converted via ffmpeg → %s", tmp_jpg)
            return tmp_jpg
    except Exception as e:
        log.debug("HEIC ffmpeg conversion failed: %s", e)

    # Method 2: pillow-heif
    try:
        import pillow_heif
        from PIL import Image as _PILImage
        pillow_heif.register_heif_opener()
        img = _PILImage.open(image_path)
        img.save(tmp_jpg, 'JPEG', quality=95)
        if os.path.exists(tmp_jpg) and os.path.getsize(tmp_jpg) > 1000:
            log.info("HEIC: converted via pillow-heif → %s", tmp_jpg)
            return tmp_jpg
    except Exception as e:
        log.debug("HEIC pillow-heif conversion failed: %s", e)

    # Method 3: ImageMagick
    try:
        result = subprocess.run([
            'convert', image_path, tmp_jpg
        ], capture_output=True, timeout=30)
        if result.returncode == 0 and os.path.exists(tmp_jpg) and os.path.getsize(tmp_jpg) > 1000:
            log.info("HEIC: converted via ImageMagick → %s", tmp_jpg)
            return tmp_jpg
    except Exception as e:
        log.debug("HEIC ImageMagick conversion failed: %s", e)

    log.error("HEIC: all conversion methods failed for %s", image_path)
    if os.path.exists(tmp_jpg):
        os.remove(tmp_jpg)
    return image_path  # return original — will fail gracefully downstream


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

    # ── HEIC/HEIF conversion ─────────────────────────────────
    # iPhone photos are HEIC by default. OpenCV cannot open them.
    # Convert to JPEG first so the full pipeline can run.
    _heic_tmp = None
    _orig_path = image_path
    ext = os.path.splitext(image_path)[1].lower()
    if ext in ('.heic', '.heif'):
        converted = _convert_heic_to_jpg(image_path)
        if converted != image_path:
            _heic_tmp = converted   # track for cleanup
            image_path = converted
            log.info("HEIC: using converted file %s", image_path)
        else:
            log.error("HEIC: conversion failed — cannot process %s", _orig_path)
            return 50, {"content_type": "photo", "error": "heic_conversion_failed"}

    # Load image
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        log.error("Photo detector: cannot open %s", image_path)
        if _heic_tmp and os.path.exists(_heic_tmp):
            os.remove(_heic_tmp)
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
    }

    # ── HEIC temp file cleanup ───────────────────────────────
    if _heic_tmp and os.path.exists(_heic_tmp):
        try:
            os.remove(_heic_tmp)
        except Exception:
            pass

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
