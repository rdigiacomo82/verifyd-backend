from pathlib import Path

p = Path("photo_detector.py")

if not p.exists():
    raise SystemExit("ERROR: photo_detector.py not found")

text = p.read_text(encoding="utf-8")

old = '''    # ── Signal 1: ELA ────────────────────────────────────────
    ela_score = _compute_ela(image_path)
    log.info("ELA score: %.1f", ela_score)
'''

new = '''    # ── Signal 1: ELA ────────────────────────────────────────
    # VERIFYD_PHOTO_CANVAS_GUARD_V1
    # ELA must analyze the same active photograph used by the
    # remaining pixel-forensic signals. If gallery/letterbox
    # canvas was removed, create a temporary JPEG containing only
    # the active photo for ELA analysis.
    _ela_analysis_path = image_path
    _ela_tmp_path = None

    if canvas_info.get("canvas_detected", False):
        try:
            _ela_tmp = tempfile.NamedTemporaryFile(
                suffix=".jpg",
                delete=False,
            )
            _ela_tmp_path = _ela_tmp.name
            _ela_tmp.close()

            cv2.imwrite(
                _ela_tmp_path,
                analysis_bgr,
                [cv2.IMWRITE_JPEG_QUALITY, 95],
            )

            if (
                os.path.exists(_ela_tmp_path) and
                os.path.getsize(_ela_tmp_path) > 1000
            ):
                _ela_analysis_path = _ela_tmp_path
                log.info(
                    "PHOTO_CANVAS_GUARD: ELA using active photo content"
                )
        except Exception as _ela_canvas_exc:
            log.debug(
                "PHOTO_CANVAS_GUARD: active-content ELA preparation failed: %s",
                _ela_canvas_exc,
            )

    ela_score = _compute_ela(_ela_analysis_path)

    if _ela_tmp_path and os.path.exists(_ela_tmp_path):
        try:
            os.remove(_ela_tmp_path)
        except Exception:
            pass

    log.info("ELA score: %.1f", ela_score)
'''

if old not in text:
    raise SystemExit("ERROR: ELA anchor not found; no changes made")

text = text.replace(old, new, 1)

p.write_text(text, encoding="utf-8")

print("SUCCESS: canvas-aware ELA fix applied.")