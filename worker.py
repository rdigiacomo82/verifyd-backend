# ============================================================
#  VeriFYD — worker.py
#
#  RQ background worker for async video/photo analysis.
#  Performance update:
#  - For uploaded REAL videos, store the visible detection result immediately
#    after analysis/database insert, before watermark stamping/email.
#  - Stamping, R2 upload, and email still run afterward in the same job.
#  - Detection logic and stamping logic are unchanged.
# ============================================================

import os
import logging
import hashlib
import json
import tempfile

log = logging.getLogger("verifyd.worker")
logging.basicConfig(level=logging.INFO)

# ── DINOv2 pre-warm ───────────────────────────────────────────
# RQ forks a new child process per job. Module-level globals reset
# in each fork so lazy-loading reloads the model every job (~10s).
# Loading HERE at import time means the parent process caches the
# model and all forked children inherit it copy-on-write — zero
# reload cost per job.
try:
    from dinov2_detector import _load_model as _dino_prewarm
    log.info("Worker startup: pre-warming DINOv2 ViT-Small...")
    _dino_prewarm()
    log.info("Worker startup: DINOv2 pre-warm complete — model cached in parent process")
except Exception as _dino_err:
    log.warning("Worker startup: DINOv2 pre-warm failed (%s) — will lazy-load per job", _dino_err)

try:
    from deepfake_detector import _load_model as _deepfake_prewarm
    log.info("Worker startup: pre-warming ViT Deepfake Detector...")
    _deepfake_prewarm()
    log.info("Worker startup: Deepfake Detector pre-warm complete — model cached in parent process")
except Exception as _df_err:
    log.warning("Worker startup: Deepfake Detector pre-warm failed (%s) — will lazy-load per job", _df_err)

RESULT_TTL = 1800   # 30 min


def _get_redis():
    import redis
    url = os.environ.get("REDIS_URL", "redis://localhost:6379")
    return redis.from_url(url, decode_responses=False)


def _sha256_file(path: str) -> str:
    """Compute SHA256 of a file without loading it all into RAM."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _store_result(redis_conn, job_id: str, result: dict) -> None:
    redis_conn.setex(f"result:{job_id}", RESULT_TTL, json.dumps(result))
    log.info("Stored result: job=%s label=%s auth=%s video_ready=%s",
             job_id, result.get("label"), result.get("authenticity_score"), result.get("video_ready"))


def process_upload_job(
    file_key: str,
    filename: str,
    email:    str,
) -> dict:
    """
    Background job: retrieve uploaded video, analyze it, store result.

    Performance behavior:
    - Detection result is stored immediately after detection + DB insert.
    - If REAL, certification/stamping then continues after the UI-visible
      result is already available.
    - After stamping/R2/email completes, result is updated with video_ready=True.
    """
    from rq        import get_current_job
    from detection import run_detection_multiclip
    from video     import clip_first_6_seconds, stamp_video
    from database  import insert_certificate, increment_user_uses
    from config    import BASE_URL
    from emailer   import send_certification_email

    rq_job = get_current_job()
    job_id = rq_job.id if rq_job else file_key.replace("file:", "")

    r = _get_redis()

    LABEL_UI = {
        "REAL":         ("REAL VIDEO VERIFIED", "green",  True),
        "UNDETERMINED": ("VIDEO UNDETERMINED",  "blue",   False),
        "AI":           ("AI DETECTED",         "red",    False),
    }

    suffix   = os.path.splitext(filename)[1] or ".mp4"
    tmp_path = os.path.join(tempfile.gettempdir(), f"{job_id}{suffix}")

    if file_key.startswith("r2:"):
        r2_key = file_key[3:]
        try:
            from storage import download_video, delete_video
            download_video(r2_key, tmp_path)
            delete_video(r2_key)
            file_size = os.path.getsize(tmp_path)
            log.info("Retrieved file from R2: job=%s key=%s size=%d bytes",
                     job_id, r2_key, file_size)
        except Exception as e:
            log.error("R2 download failed: key=%s job=%s error=%s", r2_key, job_id, e)
            result = {"job_status": "error", "error": "File could not be retrieved. Please re-upload."}
            _store_result(r, job_id, result)
            return result
        sha256 = _sha256_file(tmp_path)
    else:
        file_bytes = r.get(file_key)
        if not file_bytes:
            log.error("File not found in Redis: key=%s job=%s", file_key, job_id)
            result = {"job_status": "error", "error": "File expired from queue. Please re-upload."}
            _store_result(r, job_id, result)
            return result
        with open(tmp_path, "wb") as f:
            f.write(file_bytes)
        r.delete(file_key)
        log.info("Retrieved file from Redis: job=%s size=%d bytes", job_id, len(file_bytes))
        sha256 = hashlib.sha256(file_bytes).hexdigest()

    try:
        log.info("Worker: starting detection for job=%s email=%s", job_id, email)

        authenticity, label, detail = run_detection_multiclip(tmp_path)
        ui_text, color, certify = LABEL_UI.get(label, ("VIDEO UNDETERMINED", "blue", False))

        log.info("Worker: detection complete job=%s label=%s auth=%d",
                 job_id, label, authenticity)

        increment_user_uses(email)
        insert_certificate(
            cert_id       = job_id,
            email         = email,
            original_file = filename,
            label         = label,
            authenticity  = authenticity,
            ai_score      = detail["ai_score"],
            sha256        = sha256,
        )

        result = {
            "status":             ui_text,
            "authenticity_score": authenticity,
            "color":              color,
            "label":              label,
            "gpt_reasoning":      detail.get("gpt_reasoning", ""),
            "gpt_flags":          detail.get("gpt_flags", []),
            "signal_score":       detail.get("signal_ai_score", 0),
            "gpt_score":          detail.get("gpt_ai_score", 0),
            "job_status":         "complete",
            "video_ready":        False,
        }

        if certify:
            result["certificate_id"] = job_id
            result["download_url"]   = f"{BASE_URL}/download/{job_id}"
            result["certification_status"] = "processing"

        # PERFORMANCE OPTIMIZATION:
        # Store the analysis result before ffmpeg stamping/R2/email so the UI can
        # show the result as soon as detection completes. Stamping still continues.
        _store_result(r, job_id, result)

        if certify:
            certified_path = os.path.join(tempfile.gettempdir(), f"cert_{job_id}.mp4")
            download_url   = f"{BASE_URL}/download/{job_id}"
            try:
                clip_path = clip_first_6_seconds(tmp_path)
                stamp_video(clip_path, certified_path, job_id)
                if os.path.exists(clip_path):
                    os.remove(clip_path)

                if os.path.exists(certified_path):
                    from database import get_user_status as _gus
                    try:
                        _plan = _gus(email).get("plan", "free")
                    except Exception:
                        _plan = "free"

                    _stored_in_r2 = False
                    try:
                        from storage import upload_certified, r2_available
                        if r2_available():
                            upload_certified(job_id, certified_path, _plan)
                            os.remove(certified_path)
                            _stored_in_r2 = True
                            log.info("Worker: certified video stored in R2: job=%s plan=%s",
                                     job_id, _plan)
                    except Exception as _r2e:
                        log.warning("R2 cert upload failed, falling back to Redis: %s", _r2e)

                    if not _stored_in_r2 and os.path.exists(certified_path):
                        _cert_ttl = {
                            "free":       86400,
                            "creator":    259200,
                            "pro":        604800,
                            "enterprise": 2592000,
                        }.get(_plan, 86400)
                        with open(certified_path, "rb") as vf:
                            cert_bytes = vf.read()
                        r.setex(f"cert:{job_id}", _cert_ttl, cert_bytes)
                        os.remove(certified_path)
                        log.info("Worker: certified video stored in Redis (fallback): job=%s size=%d bytes plan=%s ttl=%dh",
                                 job_id, len(cert_bytes), _plan, _cert_ttl // 3600)

                if email and "@" in email:
                    try:
                        send_certification_email(email, job_id, authenticity, filename, download_url)
                    except Exception as e:
                        log.warning("Worker: email failed for %s: %s", job_id, e)

                result["video_ready"] = True
                result["certification_status"] = "ready"
                _store_result(r, job_id, result)

            except Exception as e:
                log.error("Worker: stamp failed for %s: %s", job_id, e)
                result["video_ready"] = False
                result["certification_status"] = "failed"
                result["certification_error"] = "Certified video generation failed, but analysis completed."
                _store_result(r, job_id, result)

        return result

    except Exception as e:
        log.exception("Worker: job %s failed", job_id)
        result = {"job_status": "error", "error": str(e)[:300]}
        _store_result(r, job_id, result)
        return result

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
            log.info("Worker: cleaned up temp file %s", tmp_path)


# NOTE:
# The remaining functions below are unchanged from your current worker.py.
# They are included in full so this file can be used as a direct replacement.

def process_photo_upload_job(
    file_key: str,
    filename: str,
    email: str,
) -> dict:
    """
    Background job: retrieve photo from R2/Redis, analyze it,
    stamp if REAL, store result. Called by RQ.
    """
    import os as _os
    import tempfile
    import hashlib
    from rq import get_current_job
    from photo_detection import run_photo_detection
    from database import insert_certificate, increment_user_uses, get_user_status as _gus
    from config import BASE_URL
    from emailer import send_certification_email

    rq_job = get_current_job()
    job_id = rq_job.id if rq_job else file_key.replace("file:", "")

    r = _get_redis()

    LABEL_UI = {
        "REAL":         ("REAL PHOTO VERIFIED",  "green", True),
        "UNDETERMINED": ("PHOTO UNDETERMINED",    "blue",  False),
        "AI":           ("AI DETECTED",           "red",   False),
    }

    ext = _os.path.splitext(filename)[1].lower() or ".jpg"
    if ext not in (".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif"):
        ext = ".jpg"
    tmp_path = _os.path.join(tempfile.gettempdir(), f"{job_id}{ext}")

    try:
        if file_key.startswith("r2:"):
            r2_key = file_key[3:]
            from storage import download_video, delete_video
            download_video(r2_key, tmp_path)
            delete_video(r2_key)
            log.info("Retrieved photo from R2: job=%s key=%s size=%d bytes",
                     job_id, r2_key, _os.path.getsize(tmp_path))
            sha256 = _sha256_file(tmp_path)
        else:
            file_bytes = r.get(file_key)
            if not file_bytes:
                result = {"job_status": "error", "error": "Photo expired from queue. Please re-upload."}
                _store_result(r, job_id, result)
                return result
            with open(tmp_path, "wb") as fh:
                fh.write(file_bytes)
            r.delete(file_key)
            sha256 = hashlib.sha256(file_bytes).hexdigest()

        log.info("Worker: starting photo detection job=%s email=%s", job_id, email)

        authenticity, label, detail = run_photo_detection(tmp_path)
        ui_text, color, certify = LABEL_UI.get(label, ("PHOTO UNDETERMINED", "blue", False))

        log.info("Worker: photo detection complete job=%s label=%s auth=%d", job_id, label, authenticity)

        increment_user_uses(email)
        insert_certificate(
            cert_id       = job_id,
            email         = email,
            original_file = filename,
            label         = label,
            authenticity  = authenticity,
            ai_score      = detail["ai_score"],
            sha256        = sha256,
        )

        result = {
            "status":             ui_text,
            "authenticity_score": authenticity,
            "color":              color,
            "label":              label,
            "gpt_reasoning":      detail.get("gpt_reasoning", ""),
            "gpt_flags":          detail.get("gpt_flags", []),
            "signal_score":       detail.get("signal_ai_score", 0),
            "gpt_score":          detail.get("gpt_ai_score", 0),
            "ela_score":          detail.get("ela_score", 0),
            "generator_guess":    detail.get("generator_guess", "Unknown"),
            "media_type":         "photo",
            "job_status":         "complete",
        }

        if certify:
            certified_path = _os.path.join(tempfile.gettempdir(), f"cert_{job_id}{ext}")
            download_url = f"{BASE_URL}/download-photo/{job_id}"
            try:
                from video import stamp_photo

                _stamp_src = tmp_path
                _heic_converted = None
                if ext in (".heic", ".heif"):
                    import subprocess as _sp, tempfile as _tf
                    _jpg_tmp = _tf.mktemp(suffix=".jpg")
                    _converted_ok = False

                    try:
                        _r = _sp.run(["convert", tmp_path, _jpg_tmp], capture_output=True, timeout=30)
                        if _r.returncode == 0 and _os.path.exists(_jpg_tmp) and _os.path.getsize(_jpg_tmp) > 1000:
                            _converted_ok = True
                            log.info("Worker: HEIC→JPEG via ImageMagick: %s", _jpg_tmp)
                    except Exception as _e1:
                        log.debug("Worker: ImageMagick HEIC conversion failed: %s", _e1)

                    if not _converted_ok:
                        try:
                            _r2 = _sp.run(["ffmpeg", "-y", "-i", tmp_path, "-q:v", "2", _jpg_tmp], capture_output=True, timeout=30)
                            if _r2.returncode == 0 and _os.path.exists(_jpg_tmp) and _os.path.getsize(_jpg_tmp) > 1000:
                                _converted_ok = True
                                log.info("Worker: HEIC→JPEG via ffmpeg: %s", _jpg_tmp)
                        except Exception as _e2:
                            log.debug("Worker: ffmpeg HEIC conversion failed: %s", _e2)

                    if _converted_ok:
                        _stamp_src = _jpg_tmp
                        _heic_converted = _jpg_tmp
                        certified_path = _os.path.join(tempfile.gettempdir(), f"cert_{job_id}.jpg")
                        ext = ".jpg"
                    else:
                        log.warning("Worker: HEIC→JPEG all conversion methods failed, skipping stamp")
                        _stamp_src = None

                if _stamp_src:
                    stamp_photo(_stamp_src, certified_path, job_id)
                if _heic_converted and _os.path.exists(_heic_converted):
                    _os.remove(_heic_converted)

                if _os.path.exists(certified_path):
                    try:
                        _plan = _gus(email).get("plan", "free")
                    except Exception:
                        _plan = "free"

                    _stored = False
                    try:
                        from storage import r2_available, upload_certified_photo
                        if r2_available():
                            upload_certified_photo(job_id, certified_path, _plan, ext)
                            _os.remove(certified_path)
                            _stored = True
                            log.info("Worker: certified photo stored in R2: job=%s plan=%s", job_id, _plan)
                    except Exception as _r2e:
                        log.warning("R2 photo cert upload failed, falling back to Redis: %s", _r2e)

                    if not _stored and _os.path.exists(certified_path):
                        _cert_ttl = {
                            "free": 86400, "creator": 259200,
                            "pro": 604800, "enterprise": 2592000,
                        }.get(_plan, 86400)
                        with open(certified_path, "rb") as pf:
                            cert_bytes = pf.read()
                        r.setex(f"cert:{job_id}", _cert_ttl, cert_bytes)
                        _os.remove(certified_path)
                        log.info("Worker: certified photo in Redis: job=%s size=%d", job_id, len(cert_bytes))

                result["certificate_id"] = job_id
                result["download_url"]   = download_url

                if email and "@" in email:
                    try:
                        send_certification_email(email, job_id, authenticity, filename, download_url, is_photo=True)
                    except Exception as em:
                        log.warning("Worker: email failed for %s: %s", job_id, em)

            except Exception as stamp_err:
                log.error("Worker: photo stamp failed for %s: %s", job_id, stamp_err)

        _store_result(r, job_id, result)
        return result

    except Exception as e:
        log.exception("Worker: photo job %s failed", job_id)
        result = {"job_status": "error", "error": str(e)[:300]}
        _store_result(r, job_id, result)
        return result

    finally:
        if _os.path.exists(tmp_path):
            _os.remove(tmp_path)
            log.info("Worker: cleaned up photo temp file %s", tmp_path)


def process_photo_link_job(job_id: str, image_url: str, email: str) -> dict:
    """Background job: download image from URL, analyze it, store result."""
    import os
    import tempfile
    import urllib.request
    import urllib.error

    r = _get_redis()
    tmp_path = os.path.join(tempfile.gettempdir(), f"{job_id}.jpg")

    try:
        log.info("Worker: downloading image for job=%s url=%s", job_id, image_url[:80])

        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Accept": "image/webp,image/apng,image/*,*/*;q=0.8",
            }
            req = urllib.request.Request(image_url, headers=headers)
            with urllib.request.urlopen(req, timeout=30) as resp:
                content_type = resp.headers.get("Content-Type", "")
                data = resp.read()

            ext = ".jpg"
            if "png" in content_type:
                ext = ".png"
            elif "webp" in content_type:
                ext = ".webp"
            elif data[:4] == b"\x89PNG":
                ext = ".png"
            elif data[:4] == b"RIFF" and data[8:12] == b"WEBP":
                ext = ".webp"

            if ext != ".jpg":
                tmp_path = tmp_path.replace(".jpg", ext)

            with open(tmp_path, "wb") as fh:
                fh.write(data)

            log.info("Worker: downloaded image job=%s size=%d ext=%s", job_id, len(data), ext)

        except urllib.error.HTTPError as e:
            err = f"Could not download image: HTTP {e.code}. The image may be private or require login."
            result = {"job_status": "error", "error": err}
            _store_result(r, job_id, result)
            return result
        except Exception as e:
            err = f"Could not download image: {str(e)[:150]}"
            result = {"job_status": "error", "error": err}
            _store_result(r, job_id, result)
            return result

        try:
            from database import get_or_create_user
            get_or_create_user(email)
        except Exception:
            pass

        from photo_detection import run_photo_detection
        from database import insert_certificate, increment_user_uses

        authenticity, label, detail = run_photo_detection(tmp_path)

        LABEL_UI = {
            "REAL":         ("REAL PHOTO VERIFIED",  "green", True),
            "UNDETERMINED": ("PHOTO UNDETERMINED",    "blue",  False),
            "AI":           ("AI DETECTED",           "red",   False),
        }
        ui_text, color, _ = LABEL_UI.get(label, ("PHOTO UNDETERMINED", "blue", False))

        increment_user_uses(email)
        insert_certificate(
            cert_id       = job_id,
            email         = email,
            original_file = image_url,
            label         = label,
            authenticity  = authenticity,
            ai_score      = detail["ai_score"],
            sha256        = None,
        )

        result = {
            "status":             ui_text,
            "authenticity_score": authenticity,
            "color":              color,
            "label":              label,
            "gpt_reasoning":      detail.get("gpt_reasoning", ""),
            "gpt_flags":          detail.get("gpt_flags", []),
            "signal_score":       detail.get("signal_ai_score", 0),
            "gpt_score":          detail.get("gpt_ai_score", 0),
            "ela_score":          detail.get("ela_score", 0),
            "generator_guess":    detail.get("generator_guess", "Unknown"),
            "media_type":         "photo",
            "job_status":         "complete",
        }

        try:
            import hashlib as _hl, json as _jc
            _key = "urlcache:photo:v1:" + _hl.md5(image_url.strip().encode()).hexdigest()
            _cache = {k: v for k, v in result.items() if k != "job_status"}
            r.setex(_key, 3600, _jc.dumps(_cache))
            log.info("Worker: cached photo URL result for %s", image_url[:60])
        except Exception as _ce:
            log.warning("Worker: photo URL cache write failed: %s", _ce)

        _store_result(r, job_id, result)
        return result

    except Exception as e:
        log.exception("Worker: photo link job %s failed", job_id)
        result = {"job_status": "error", "error": str(e)[:300]}
        _store_result(r, job_id, result)
        return result

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def process_link_job(job_id: str, video_url: str, email: str, double_count: bool = False) -> dict:
    """Background job: download and analyze a video from a URL."""
    from detection import run_detection_multiclip
    from video     import download_video_ytdlp, clip_first_6_seconds, stamp_video
    from database  import insert_certificate, increment_user_uses
    from config    import BASE_URL
    from emailer   import send_certification_email

    r = _get_redis()

    LABEL_UI = {
        "REAL":         ("REAL VIDEO VERIFIED", "green",  True),
        "UNDETERMINED": ("VIDEO UNDETERMINED",  "blue",   False),
        "AI":           ("AI DETECTED",         "red",    False),
    }

    tmp_path = os.path.join(tempfile.gettempdir(), f"{job_id}.mp4")

    try:
        log.info("Worker: downloading url for job=%s", job_id)
        download_video_ytdlp(video_url, tmp_path)

        if not os.path.exists(tmp_path):
            raise RuntimeError(f"Download produced no file: {video_url}")

        log.info("Worker: starting detection for link job=%s", job_id)

        try:
            from database import get_or_create_user
            get_or_create_user(email)
            log.info("Worker: ensured user record exists for %s", email)
        except Exception as _ue:
            log.warning("Worker: could not create user record for %s: %s", email, _ue)

        authenticity, label, detail = run_detection_multiclip(tmp_path)
        ui_text, color, certify = LABEL_UI.get(label, ("VIDEO UNDETERMINED", "blue", False))

        uses = 2 if double_count else 1
        for _ in range(uses):
            increment_user_uses(email)

        insert_certificate(
            cert_id       = job_id,
            email         = email,
            original_file = video_url,
            label         = label,
            authenticity  = authenticity,
            ai_score      = detail["ai_score"],
            sha256        = None,
        )

        result = {
            "status":             ui_text,
            "authenticity_score": authenticity,
            "color":              color,
            "label":              label,
            "gpt_reasoning":      detail.get("gpt_reasoning", ""),
            "gpt_flags":          detail.get("gpt_flags", []),
            "signal_score":       detail.get("signal_ai_score", 0),
            "gpt_score":          detail.get("gpt_ai_score", 0),
            "job_status":         "complete",
        }

        certify = False
        if False:
            certified_path = os.path.join(tempfile.gettempdir(), f"cert_{job_id}.mp4")
            download_url   = f"{BASE_URL}/download/{job_id}"
            try:
                clip_path = clip_first_6_seconds(tmp_path)
                stamp_video(clip_path, certified_path, job_id)
                if os.path.exists(clip_path):
                    os.remove(clip_path)
                if os.path.exists(certified_path):
                    with open(certified_path, "rb") as vf:
                        cert_bytes = vf.read()
                    r.setex(f"cert:{job_id}", 3600, cert_bytes)
                    os.remove(certified_path)
                    log.info("Worker: certified video stored in Redis: link job=%s size=%d bytes", job_id, len(cert_bytes))
                result["certificate_id"] = job_id
                result["download_url"]   = download_url
                if email and "@" in email:
                    try:
                        send_certification_email(email, job_id, authenticity, video_url, download_url)
                    except Exception as e:
                        log.warning("Worker: email failed for %s: %s", job_id, e)
            except Exception as e:
                log.error("Worker: stamp failed for link job %s: %s", job_id, e)

        _store_result(r, job_id, result)

        try:
            import hashlib as _hl, json as _jc
            _url_cache_key = "urlcache:v2:" + _hl.md5(video_url.strip().encode()).hexdigest()
            _cache_result = {k: v for k, v in result.items() if k != "job_status"}
            r.setex(_url_cache_key, 3600, _jc.dumps(_cache_result))
            log.info("Worker: cached URL result for %s", video_url[:80])
        except Exception as _ce:
            log.warning("Worker: URL cache write failed: %s", _ce)

        return result

    except Exception as e:
        log.exception("Worker: link job %s failed", job_id)
        err_str = str(e).lower()
        if "comfortable for some audiences" in err_str or "not comfortable" in err_str:
            user_error = "This TikTok video has restricted access and cannot be downloaded for analysis. To analyze it: open TikTok, save the video to your device, then upload the file directly using the Upload button."
        elif "sign in to confirm" in err_str or "login" in err_str or "bot" in err_str:
            user_error = "This video requires sign-in to access and cannot be analyzed via link. This happens with age-restricted or restricted videos. Try downloading the video to your device and uploading the file directly."
        elif "403" in err_str or "forbidden" in err_str:
            user_error = "Access to this video was blocked (403 Forbidden). The video may be region-locked, private, or restricted. Try uploading the video file directly instead."
        elif "private" in err_str:
            user_error = "This video is private and cannot be accessed."
        elif "not available" in err_str or "unavailable" in err_str:
            user_error = "This video is unavailable or has been removed."
        elif "download produced no file" in err_str:
            user_error = "The video could not be downloaded. YouTube may be blocking automated access to this video. Try uploading the video file directly instead."
        else:
            user_error = "This video could not be analyzed. YouTube may be blocking access, or the link may be invalid. Try uploading the video file directly instead."

        result = {"job_status": "error", "error": user_error, "error_detail": str(e)[:300]}
        _store_result(r, job_id, result)
        return result

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


# ─────────────────────────────────────────────────────────────
#  Keepalive job — prevents worker cold starts
# ─────────────────────────────────────────────────────────────
def keepalive_ping():
    """
    Lightweight no-op job enqueued every 8 minutes by main.py scheduler.

    PURPOSE: Render's worker service suspends after inactivity, causing
    cold starts that add ~15-20s to the next real job (model pre-warm).
    This job keeps the worker process alive between real user submissions
    so models stay cached in memory at all times.

    The job does nothing except log a heartbeat. Total execution time: <1ms.
    Redis TTL for result: 60 seconds (no need to keep longer).
    """
    import time as _time
    _ts = _time.strftime("%Y-%m-%d %H:%M:%S UTC", _time.gmtime())
    log.info("Keepalive ping: worker alive at %s — models cached in memory", _ts)
    return {"status": "alive", "ts": _ts}



