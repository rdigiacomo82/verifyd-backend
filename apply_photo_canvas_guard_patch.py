from pathlib import Path

PHOTO = Path("photo_detector.py")
ORTHO = Path("orthogonal_photo_detector.py")

MARKER = "VERIFYD_PHOTO_CANVAS_GUARD_V1"

if not PHOTO.exists():
    raise SystemExit("ERROR: photo_detector.py not found")

if not ORTHO.exists():
    raise SystemExit("ERROR: orthogonal_photo_detector.py not found")

photo = PHOTO.read_text(encoding="utf-8")
ortho = ORTHO.read_text(encoding="utf-8")

if MARKER in photo:
    raise SystemExit("ERROR: photo canvas guard already appears installed")

# ============================================================
# 1. Add reusable active-content / letterbox detector
# ============================================================

anchor = '''
# ─────────────────────────────────────────────────────────────
#  ELA — Error Level Analysis
# ─────────────────────────────────────────────────────────────
'''

helper = r'''
# ============================================================
# VERIFYD_PHOTO_CANVAS_GUARD_V1
#
# Detect highly uniform display/gallery/letterbox bars around a
# photograph. These areas must not participate in image forensic
# measurements because they can artificially:
#   - raise RGB channel correlation,
#   - create extreme HF boundaries,
#   - dilute skin/content ratios,
#   - make a normal photo appear "tall mobile".
#
# Conservative by design:
#   * only top/bottom edge-connected regions are considered;
#   * rows must be both very dark and highly uniform;
#   * both bars must be present;
#   * at least 40% of the original image must remain.
#
# Returns:
#   (analysis_image, info_dict)
# ============================================================

def _extract_active_photo_content(img_bgr: np.ndarray):
    info = {
        "canvas_detected": False,
        "canvas_fraction": 0.0,
        "top_crop_rows": 0,
        "bottom_crop_rows": 0,
        "original_height": 0,
        "original_width": 0,
        "active_height": 0,
        "active_width": 0,
    }

    try:
        if img_bgr is None or img_bgr.size == 0:
            return img_bgr, info

        h, w = img_bgr.shape[:2]

        info["original_height"] = int(h)
        info["original_width"] = int(w)
        info["active_height"] = int(h)
        info["active_width"] = int(w)

        if h < 200 or w < 200:
            return img_bgr, info

        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        row_mean = np.mean(gray.astype(np.float32), axis=1)
        row_std = np.std(gray.astype(np.float32), axis=1)

        # Dark + extremely uniform rows.
        # Real night scenes generally still contain texture/noise and
        # therefore do not form long edge-connected regions satisfying
        # both conditions.
        uniform_dark = (
            (row_mean <= 12.0) &
            (row_std <= 5.0)
        )

        top = 0
        while top < h and uniform_dark[top]:
            top += 1

        bottom = 0
        while bottom < h and uniform_dark[h - 1 - bottom]:
            bottom += 1

        top_fraction = top / float(h)
        bottom_fraction = bottom / float(h)
        total_fraction = (top + bottom) / float(h)

        active_h = h - top - bottom

        # Require bars on BOTH ends. This deliberately avoids cropping
        # normal photographs that merely contain a dark sky/floor.
        qualifies = (
            top_fraction >= 0.05 and
            bottom_fraction >= 0.05 and
            total_fraction >= 0.15 and
            active_h >= max(200, int(h * 0.40))
        )

        if not qualifies:
            return img_bgr, info

        cropped = img_bgr[top:h-bottom, :]

        if cropped is None or cropped.size == 0:
            return img_bgr, info

        info.update({
            "canvas_detected": True,
            "canvas_fraction": float(total_fraction),
            "top_crop_rows": int(top),
            "bottom_crop_rows": int(bottom),
            "active_height": int(cropped.shape[0]),
            "active_width": int(cropped.shape[1]),
        })

        log.info(
            "PHOTO_CANVAS_GUARD: uniform gallery/letterbox canvas detected "
            "top=%d bottom=%d canvas_fraction=%.3f original=%dx%d active=%dx%d",
            top,
            bottom,
            total_fraction,
            w,
            h,
            cropped.shape[1],
            cropped.shape[0],
        )

        return cropped, info

    except Exception as exc:
        log.debug("PHOTO_CANVAS_GUARD: detection error: %s", exc)
        return img_bgr, info


'''

if anchor not in photo:
    raise SystemExit("ERROR: helper insertion anchor not found")

photo = photo.replace(anchor, helper + anchor, 1)

# ============================================================
# 2. After loading the image, establish the active analysis view
# ============================================================

anchor = '''    h, w = img_bgr.shape[:2]
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    log.info("Photo detector: image %dx%d", w, h)

    # ── Signal 1: ELA ────────────────────────────────────────
'''

replacement = '''    original_h, original_w = img_bgr.shape[:2]

    # VERIFYD_PHOTO_CANVAS_GUARD_V1
    # Pixel forensic measurements should describe the photograph,
    # not uniform black gallery/player bars surrounding it.
    analysis_bgr, canvas_info = _extract_active_photo_content(img_bgr)

    h, w = analysis_bgr.shape[:2]
    img_gray = cv2.cvtColor(analysis_bgr, cv2.COLOR_BGR2GRAY)

    log.info(
        "Photo detector: image %dx%d analysis=%dx%d canvas=%s",
        original_w,
        original_h,
        w,
        h,
        canvas_info.get("canvas_detected", False),
    )

    # ── Signal 1: ELA ────────────────────────────────────────
'''

if anchor not in photo:
    raise SystemExit("ERROR: image analysis-view anchor not found")

photo = photo.replace(anchor, replacement, 1)

# ============================================================
# 3. Use active photograph for RGB / HSV / pixel measurements
# ============================================================

photo = photo.replace(
    'chan_corr = _compute_chan_corr(img_bgr)',
    'chan_corr = _compute_chan_corr(analysis_bgr)',
    1,
)

photo = photo.replace(
    'img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)',
    'img_hsv = cv2.cvtColor(analysis_bgr, cv2.COLOR_BGR2HSV)',
    1,
)

photo = photo.replace(
    'skin_ratio = _estimate_skin_ratio(img_bgr)',
    'skin_ratio = _estimate_skin_ratio(analysis_bgr)',
    1,
)

# ============================================================
# 4. Missing EXIF is expected for gallery/screenshot containers
# ============================================================

anchor = '''    meta_adjustment, meta_dict = _analyze_metadata(image_path)

    # ── Score computation ────────────────────────────────────
'''

replacement = '''    meta_adjustment, meta_dict = _analyze_metadata(image_path)

    # VERIFYD_PHOTO_CANVAS_GUARD_V1
    #
    # A screenshot/gallery export naturally strips camera EXIF.
    # If we positively detected a surrounding display canvas,
    # "no EXIF" is not independent evidence of AI generation.
    if (
        canvas_info.get("canvas_detected", False) and
        meta_adjustment == 10 and
        not (meta_dict.get("make") or meta_dict.get("model"))
    ):
        log.info(
            "PHOTO_CANVAS_GUARD: suppressing no-EXIF +10 because "
            "display/gallery canvas was detected"
        )
        meta_adjustment = 0

    # ── Score computation ────────────────────────────────────
'''

if anchor not in photo:
    raise SystemExit("ERROR: metadata anchor not found")

photo = photo.replace(anchor, replacement, 1)

# ============================================================
# 5. Add guard telemetry to returned context
# ============================================================

# Find the final context construction by locating common existing
# entries. We add only observability fields and do not change its
# existing scoring interface.

needle = '''        "no_camera_metadata": no_camera_metadata,
'''

if needle in photo:
    photo = photo.replace(
        needle,
        needle +
        '''        "display_canvas_detected": canvas_info.get("canvas_detected", False),
        "display_canvas_fraction": canvas_info.get("canvas_fraction", 0.0),
        "analysis_width": int(w),
        "analysis_height": int(h),
''',
        1,
    )
else:
    # Fallback: adding the fields is optional for scoring; the guard
    # itself remains functional if the context key layout changed.
    print("NOTE: context telemetry anchor not found; scoring guard still patched.")

# ============================================================
# 6. DINO orthogonal shadow should inspect active photo too
# ============================================================

anchor = '''        image = cv2.imread(image_path)

        if image is None:
'''

replacement = '''        image = cv2.imread(image_path)

        if image is not None:
            try:
                from photo_detector import _extract_active_photo_content
                image, _canvas_info = _extract_active_photo_content(image)

                if _canvas_info.get("canvas_detected", False):
                    log.info(
                        "ORTHO_PHOTO_SHADOW: using active photo content "
                        "after canvas crop fraction=%.3f",
                        _canvas_info.get("canvas_fraction", 0.0),
                    )
            except Exception as _canvas_exc:
                log.debug(
                    "ORTHO_PHOTO_SHADOW: canvas crop unavailable: %s",
                    _canvas_exc,
                )

        if image is None:
'''

if anchor not in ortho:
    raise SystemExit("ERROR: orthogonal image-load anchor not found")

ortho = ortho.replace(anchor, replacement, 1)

# ============================================================
# Backups + write
# ============================================================

photo_backup = Path("photo_detector.py.before_canvas_guard")
ortho_backup = Path("orthogonal_photo_detector.py.before_canvas_guard")

if not photo_backup.exists():
    photo_backup.write_text(
        PHOTO.read_text(encoding="utf-8"),
        encoding="utf-8",
    )

if not ortho_backup.exists():
    ortho_backup.write_text(
        ORTHO.read_text(encoding="utf-8"),
        encoding="utf-8",
    )

PHOTO.write_text(photo, encoding="utf-8")
ORTHO.write_text(ortho, encoding="utf-8")

print("")
print("SUCCESS: PHOTO_CANVAS_GUARD_V1 applied.")
print("")
print("Expected behavior:")
print("  - uniform top/bottom gallery bars removed from pixel analysis")
print("  - active photo dimensions used for tall/mobile aspect checks")
print("  - RGB correlation calculated on actual photograph")
print("  - skin/content estimation calculated on actual photograph")
print("  - no-EXIF +10 suppressed only for positively detected canvas images")
print("  - orthogonal DINO shadow analyzes actual photograph")
print("  - existing AI thresholds otherwise unchanged")
print("  - existing portrait composite protection remains enabled")