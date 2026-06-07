# ============================================================
#  VeriFYD — worker.py
#
#  RQ background worker for async video/photo/document analysis.
#
#  Phase 9A-1 update:
#  - For REAL documents, create the existing certified PDF exactly as before.
#  - Also create a universal VeriFYD certified file package ZIP containing
#    the original file, certified PDF report, hashes, signed seal, and metadata.
#  - Existing video/photo behavior is preserved.
# ============================================================

import os
import logging
import hashlib
import json
import tempfile
import shutil

log = logging.getLogger("verifyd.worker")
logging.basicConfig(level=logging.INFO)

# IMPORTANT:
# Do NOT pre-warm HuggingFace / transformer models at module import time.
# Detection modules lazy-load/cache models only when an actual job needs them.

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


# ─────────────────────────────────────────────────────────────
#  Runtime diagnostics / document-rendering dependencies
# ─────────────────────────────────────────────────────────────
_OFFICE_DIAGNOSTICS_LOGGED = False

_OFFICE_EXTRA_PATHS = (
    "/usr/lib/libreoffice/program",
    "/usr/local/lib/libreoffice/program",
    "/opt/libreoffice/program",
)


def _ensure_office_paths() -> None:
    """
    Make common LibreOffice executable folders visible to child modules.

    Some Linux images install soffice under /usr/lib/libreoffice/program
    even when /usr/bin/soffice is not present. doc_certifier.py uses
    shutil.which("soffice") / shutil.which("libreoffice"), so updating PATH
    here allows the existing conversion logic to work without changing
    doc_certifier.py again.
    """
    current_path = os.environ.get("PATH", "")
    parts = [p for p in current_path.split(os.pathsep) if p]
    changed = False

    for extra in _OFFICE_EXTRA_PATHS:
        if os.path.isdir(extra) and extra not in parts:
            parts.insert(0, extra)
            changed = True

    if changed:
        os.environ["PATH"] = os.pathsep.join(parts)


def _which_office_binary() -> str:
    """Return the best LibreOffice/soffice executable path available."""
    _ensure_office_paths()

    found = shutil.which("soffice") or shutil.which("libreoffice")
    if found:
        return found

    for extra in _OFFICE_EXTRA_PATHS:
        for name in ("soffice", "libreoffice"):
            candidate = os.path.join(extra, name)
            if os.path.exists(candidate) and os.access(candidate, os.X_OK):
                return candidate

    return ""


def _office_version(binary: str) -> str:
    """Best-effort LibreOffice version string for Render logs."""
    if not binary:
        return ""
    try:
        import subprocess
        proc = subprocess.run([binary, "--version"], capture_output=True, text=True, timeout=20)
        output = (proc.stdout or proc.stderr or "").strip()
        return output[:300]
    except Exception as e:
        return f"version_check_failed: {e}"[:300]


def _log_worker_runtime_dependencies(context: str = "startup", force: bool = False) -> None:
    """
    Log the exact worker runtime state needed to diagnose document rendering.

    This intentionally logs only paths/binary availability, not secrets.
    """
    global _OFFICE_DIAGNOSTICS_LOGGED
    if _OFFICE_DIAGNOSTICS_LOGGED and not force:
        return

    _ensure_office_paths()
    office_bin = _which_office_binary()
    log.info("Worker runtime diagnostics [%s]: PATH=%s", context, os.environ.get("PATH", ""))
    log.info("Worker runtime diagnostics [%s]: shutil.which('soffice')=%s", context, shutil.which("soffice"))
    log.info("Worker runtime diagnostics [%s]: shutil.which('libreoffice')=%s", context, shutil.which("libreoffice"))
    log.info("Worker runtime diagnostics [%s]: selected_office_binary=%s", context, office_bin or "NOT_FOUND")
    if office_bin:
        log.info("Worker runtime diagnostics [%s]: office_version=%s", context, _office_version(office_bin))

    _OFFICE_DIAGNOSTICS_LOGGED = True


# Log once when the RQ worker imports this module.
_log_worker_runtime_dependencies("module_import")



def process_upload_job(file_key: str, filename: str, email: str) -> dict:
    """Background job: retrieve uploaded video, analyze it, store result."""
    from rq import get_current_job
    from detection import run_detection_multiclip
    from video import clip_first_6_seconds, stamp_video
    from database import insert_certificate, increment_user_uses
    from config import BASE_URL
    from emailer import send_certification_email

    rq_job = get_current_job()
    job_id = rq_job.id if rq_job else file_key.replace("file:", "")
    r = _get_redis()

    LABEL_UI = {
        "REAL": ("REAL VIDEO VERIFIED", "green", True),
        "UNDETERMINED": ("VIDEO UNDETERMINED", "blue", False),
        "AI": ("AI DETECTED", "red", False),
    }

    suffix = os.path.splitext(filename)[1] or ".mp4"
    tmp_path = os.path.join(tempfile.gettempdir(), f"{job_id}{suffix}")

    if file_key.startswith("r2:"):
        r2_key = file_key[3:]
        try:
            from storage import download_video, delete_video
            download_video(r2_key, tmp_path)
            delete_video(r2_key)
            log.info("Retrieved file from R2: job=%s key=%s size=%d bytes", job_id, r2_key, os.path.getsize(tmp_path))
        except Exception as e:
            log.error("R2 download failed: key=%s job=%s error=%s", r2_key, job_id, e)
            result = {"job_status": "error", "error": "File could not be retrieved. Please re-upload."}
            _store_result(r, job_id, result)
            return result
        sha256 = _sha256_file(tmp_path)
    else:
        file_bytes = r.get(file_key)
        if not file_bytes:
            result = {"job_status": "error", "error": "File expired from queue. Please re-upload."}
            _store_result(r, job_id, result)
            return result
        with open(tmp_path, "wb") as f:
            f.write(file_bytes)
        r.delete(file_key)
        sha256 = hashlib.sha256(file_bytes).hexdigest()

    try:
        log.info("Worker: starting detection for job=%s email=%s", job_id, email)
        authenticity, label, detail = run_detection_multiclip(tmp_path)
        ui_text, color, certify = LABEL_UI.get(label, ("VIDEO UNDETERMINED", "blue", False))

        increment_user_uses(email)
        insert_certificate(
            cert_id=job_id,
            email=email,
            original_file=filename,
            label=label,
            authenticity=authenticity,
            ai_score=detail["ai_score"],
            sha256=sha256,
        )

        result = {
            "status": ui_text,
            "authenticity_score": authenticity,
            "color": color,
            "label": label,
            "gpt_reasoning": detail.get("gpt_reasoning", ""),
            "gpt_flags": detail.get("gpt_flags", []),
            "signal_score": detail.get("signal_ai_score", 0),
            "gpt_score": detail.get("gpt_ai_score", 0),
            "audio_score": detail.get("audio_ai_score", 50),
            "audio_confidence": detail.get("audio_confidence", "unavailable"),
            "audio_contribution": detail.get("audio_contribution", 0),
            "audio_evidence": detail.get("audio_evidence", []),
            "job_status": "complete",
            "video_ready": False,
        }

        if certify:
            result["certificate_id"] = job_id
            result["download_url"] = f"{BASE_URL}/download/{job_id}"
            result["certification_status"] = "processing"
        _store_result(r, job_id, result)

        if certify:
            certified_path = os.path.join(tempfile.gettempdir(), f"cert_{job_id}.mp4")
            download_url = f"{BASE_URL}/download/{job_id}"
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
                            log.info("Worker: certified video stored in R2: job=%s plan=%s", job_id, _plan)
                    except Exception as _r2e:
                        log.warning("R2 cert upload failed, falling back to Redis: %s", _r2e)

                    if not _stored_in_r2 and os.path.exists(certified_path):
                        _cert_ttl = {"free": 86400, "creator": 259200, "pro": 604800, "enterprise": 2592000}.get(_plan, 86400)
                        with open(certified_path, "rb") as vf:
                            cert_bytes = vf.read()
                        r.setex(f"cert:{job_id}", _cert_ttl, cert_bytes)
                        os.remove(certified_path)

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



def process_photo_upload_job(file_key: str, filename: str, email: str) -> dict:
    """Background job: retrieve photo from R2/Redis, analyze it, stamp if REAL, store result."""
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
        "REAL": ("REAL PHOTO VERIFIED", "green", True),
        "UNDETERMINED": ("PHOTO UNDETERMINED", "blue", False),
        "AI": ("AI DETECTED", "red", False),
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
            sha256 = _sha256_file(tmp_path)
            log.info("Worker: retrieved photo from R2: job=%s key=%s size=%d bytes", job_id, r2_key, _os.path.getsize(tmp_path))
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

        authenticity, label, detail = run_photo_detection(tmp_path)
        ui_text, color, certify = LABEL_UI.get(label, ("PHOTO UNDETERMINED", "blue", False))

        increment_user_uses(email)
        insert_certificate(
            cert_id=job_id,
            email=email,
            original_file=filename,
            label=label,
            authenticity=authenticity,
            ai_score=detail["ai_score"],
            sha256=sha256,
        )

        result = {
            "status": ui_text,
            "authenticity_score": authenticity,
            "color": color,
            "label": label,
            "gpt_reasoning": detail.get("gpt_reasoning", ""),
            "gpt_flags": detail.get("gpt_flags", []),
            "signal_score": detail.get("signal_ai_score", 0),
            "gpt_score": detail.get("gpt_ai_score", 0),
            "ela_score": detail.get("ela_score", 0),
            "generator_guess": detail.get("generator_guess", "Unknown"),
            "media_type": "photo",
            "job_status": "complete",
            "photo_ready": False,
        }

        if certify:
            result["certificate_id"] = job_id
            result["download_url"] = f"{BASE_URL}/download-photo/{job_id}"
            result["certification_status"] = "processing"

        _store_result(r, job_id, result)

        if not certify:
            return result

        certified_ext = ext
        certified_path = _os.path.join(tempfile.gettempdir(), f"cert_{job_id}{certified_ext}")
        download_url = f"{BASE_URL}/download-photo/{job_id}"
        _heic_converted = None

        try:
            from video import stamp_photo

            _stamp_src = tmp_path
            if ext in (".heic", ".heif"):
                import subprocess as _sp
                import tempfile as _tf

                _jpg_tmp = _tf.mktemp(suffix=".jpg")
                _converted_ok = False

                try:
                    _r = _sp.run(["ffmpeg", "-y", "-i", tmp_path, "-q:v", "2", _jpg_tmp], capture_output=True, timeout=60)
                    _converted_ok = _r.returncode == 0 and _os.path.exists(_jpg_tmp) and _os.path.getsize(_jpg_tmp) > 1000
                except Exception:
                    pass

                if not _converted_ok:
                    try:
                        _r2 = _sp.run(["convert", tmp_path, _jpg_tmp], capture_output=True, timeout=60)
                        _converted_ok = _r2.returncode == 0 and _os.path.exists(_jpg_tmp) and _os.path.getsize(_jpg_tmp) > 1000
                    except Exception:
                        pass

                if not _converted_ok:
                    raise RuntimeError("HEIC/HEIF photo conversion failed during certification stamping.")

                _stamp_src = _jpg_tmp
                _heic_converted = _jpg_tmp
                certified_ext = ".jpg"
                certified_path = _os.path.join(tempfile.gettempdir(), f"cert_{job_id}.jpg")

            stamp_photo(_stamp_src, certified_path, job_id)

            if _heic_converted and _os.path.exists(_heic_converted):
                _os.remove(_heic_converted)
                _heic_converted = None

            if not _os.path.exists(certified_path) or _os.path.getsize(certified_path) < 1000:
                raise RuntimeError("Certified photo was not created or was too small.")

            try:
                _plan = _gus(email).get("plan", "free")
            except Exception:
                _plan = "free"

            _stored = False
            try:
                from storage import r2_available, upload_certified_photo
                if r2_available():
                    upload_certified_photo(job_id, certified_path, _plan, certified_ext)
                    _os.remove(certified_path)
                    _stored = True
                    log.info("Worker: certified photo stored in R2: job=%s plan=%s ext=%s", job_id, _plan, certified_ext)
            except Exception as _r2e:
                log.warning("R2 photo cert upload failed, falling back to Redis: %s", _r2e)

            if not _stored and _os.path.exists(certified_path):
                _cert_ttl = {"free": 86400, "creator": 259200, "pro": 604800, "enterprise": 2592000}.get(_plan, 86400)
                with open(certified_path, "rb") as pf:
                    cert_bytes = pf.read()
                r.setex(f"cert:{job_id}", _cert_ttl, cert_bytes)
                _os.remove(certified_path)
                log.info("Worker: certified photo stored in Redis fallback: job=%s ttl=%s", job_id, _cert_ttl)

            if email and "@" in email:
                try:
                    sent = send_certification_email(email, job_id, authenticity, filename, download_url, is_photo=True)
                    log.info("Worker: photo certification email sent=%s job=%s email=%s", sent, job_id, email)
                except Exception as em:
                    log.warning("Worker: photo certification email failed for %s: %s", job_id, em)

            result["photo_ready"] = True
            result["certification_status"] = "ready"
            result["download_url"] = download_url
            _store_result(r, job_id, result)

        except Exception as stamp_err:
            log.error("Worker: photo certification failed for %s: %s", job_id, stamp_err)
            result["photo_ready"] = False
            result["certification_status"] = "failed"
            result["certification_error"] = "Certified photo generation failed, but analysis completed."
            _store_result(r, job_id, result)
        finally:
            if _heic_converted and _os.path.exists(_heic_converted):
                try:
                    _os.remove(_heic_converted)
                except Exception:
                    pass

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
    """Background job: download image from URL, analyze it, certify if REAL, store result."""
    import os
    import tempfile
    import urllib.request
    import urllib.error
    import hashlib as _hashlib

    r = _get_redis()
    tmp_path = os.path.join(tempfile.gettempdir(), f"{job_id}.jpg")

    try:
        try:
            headers = {"User-Agent": "Mozilla/5.0", "Accept": "image/webp,image/apng,image/*,*/*;q=0.8"}
            req = urllib.request.Request(image_url, headers=headers)
            with urllib.request.urlopen(req, timeout=30) as resp:
                content_type = resp.headers.get("Content-Type", "")
                data = resp.read()
            ext = ".jpg"
            if "png" in content_type or data[:4] == b"\x89PNG":
                ext = ".png"
            elif "webp" in content_type or (data[:4] == b"RIFF" and data[8:12] == b"WEBP"):
                ext = ".webp"
            if ext != ".jpg":
                tmp_path = tmp_path.replace(".jpg", ext)
            with open(tmp_path, "wb") as fh:
                fh.write(data)
            sha256 = _hashlib.sha256(data).hexdigest()
        except urllib.error.HTTPError as e:
            result = {"job_status": "error", "error": f"Could not download image: HTTP {e.code}. The image may be private or require login."}
            _store_result(r, job_id, result)
            return result
        except Exception as e:
            result = {"job_status": "error", "error": f"Could not download image: {str(e)[:150]}"}
            _store_result(r, job_id, result)
            return result

        try:
            from database import get_or_create_user
            get_or_create_user(email)
        except Exception:
            pass

        from photo_detection import run_photo_detection
        from database import insert_certificate, increment_user_uses, get_user_status as _gus
        from config import BASE_URL
        from emailer import send_certification_email

        authenticity, label, detail = run_photo_detection(tmp_path)
        LABEL_UI = {
            "REAL": ("REAL PHOTO VERIFIED", "green", True),
            "UNDETERMINED": ("PHOTO UNDETERMINED", "blue", False),
            "AI": ("AI DETECTED", "red", False),
        }
        ui_text, color, certify = LABEL_UI.get(label, ("PHOTO UNDETERMINED", "blue", False))
        increment_user_uses(email)
        insert_certificate(
            cert_id=job_id,
            email=email,
            original_file=image_url,
            label=label,
            authenticity=authenticity,
            ai_score=detail["ai_score"],
            sha256=sha256,
        )

        result = {
            "status": ui_text,
            "authenticity_score": authenticity,
            "color": color,
            "label": label,
            "gpt_reasoning": detail.get("gpt_reasoning", ""),
            "gpt_flags": detail.get("gpt_flags", []),
            "signal_score": detail.get("signal_ai_score", 0),
            "gpt_score": detail.get("gpt_ai_score", 0),
            "ela_score": detail.get("ela_score", 0),
            "generator_guess": detail.get("generator_guess", "Unknown"),
            "media_type": "photo",
            "job_status": "complete",
            "photo_ready": False,
        }

        if certify:
            result["certificate_id"] = job_id
            result["download_url"] = f"{BASE_URL}/download-photo/{job_id}"
            result["certification_status"] = "processing"

        _store_result(r, job_id, result)

        if certify:
            certified_path = os.path.join(tempfile.gettempdir(), f"cert_{job_id}{ext}")
            download_url = f"{BASE_URL}/download-photo/{job_id}"
            try:
                from video import stamp_photo
                stamp_photo(tmp_path, certified_path, job_id)

                if not os.path.exists(certified_path) or os.path.getsize(certified_path) < 1000:
                    raise RuntimeError("Certified linked photo was not created or was too small.")

                try:
                    _plan = _gus(email).get("plan", "free")
                except Exception:
                    _plan = "free"

                _stored = False
                try:
                    from storage import r2_available, upload_certified_photo
                    if r2_available():
                        upload_certified_photo(job_id, certified_path, _plan, ext)
                        os.remove(certified_path)
                        _stored = True
                        log.info("Worker: certified linked photo stored in R2: job=%s plan=%s ext=%s", job_id, _plan, ext)
                except Exception as _r2e:
                    log.warning("R2 linked photo cert upload failed, falling back to Redis: %s", _r2e)

                if not _stored and os.path.exists(certified_path):
                    _cert_ttl = {"free": 86400, "creator": 259200, "pro": 604800, "enterprise": 2592000}.get(_plan, 86400)
                    with open(certified_path, "rb") as pf:
                        cert_bytes = pf.read()
                    r.setex(f"cert:{job_id}", _cert_ttl, cert_bytes)
                    os.remove(certified_path)
                    log.info("Worker: certified linked photo stored in Redis fallback: job=%s ttl=%s", job_id, _cert_ttl)

                if email and "@" in email:
                    try:
                        sent = send_certification_email(email, job_id, authenticity, image_url, download_url, is_photo=True)
                        log.info("Worker: linked photo certification email sent=%s job=%s email=%s", sent, job_id, email)
                    except Exception as em:
                        log.warning("Worker: linked photo certification email failed for %s: %s", job_id, em)

                result["photo_ready"] = True
                result["certification_status"] = "ready"
                result["download_url"] = download_url
                _store_result(r, job_id, result)
            except Exception as stamp_err:
                log.error("Worker: linked photo certification failed for %s: %s", job_id, stamp_err)
                result["photo_ready"] = False
                result["certification_status"] = "failed"
                result["certification_error"] = "Certified linked photo generation failed, but analysis completed."
                _store_result(r, job_id, result)

        try:
            import hashlib as _hl, json as _jc
            _key = "urlcache:photo:v1:" + _hl.md5(image_url.strip().encode()).hexdigest()
            _cache = {k: v for k, v in result.items() if k != "job_status"}
            r.setex(_key, 3600, _jc.dumps(_cache))
        except Exception:
            pass

        return result
    except Exception as e:
        log.exception("Worker: photo link job %s failed", job_id)
        result = {"job_status": "error", "error": str(e)[:300]}
        _store_result(r, job_id, result)
        return result
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass


def process_link_job(job_id: str, video_url: str, email: str, double_count: bool = False) -> dict:
    """Background job: download and analyze a video from a URL."""
    from detection import run_detection_multiclip
    from video import download_video_ytdlp
    from database import insert_certificate, increment_user_uses

    r = _get_redis()
    LABEL_UI = {
        "REAL": ("REAL VIDEO VERIFIED", "green", True),
        "UNDETERMINED": ("VIDEO UNDETERMINED", "blue", False),
        "AI": ("AI DETECTED", "red", False),
    }
    tmp_path = os.path.join(tempfile.gettempdir(), f"{job_id}.mp4")
    try:
        download_video_ytdlp(video_url, tmp_path)
        if not os.path.exists(tmp_path):
            raise RuntimeError(f"Download produced no file: {video_url}")
        try:
            from database import get_or_create_user
            get_or_create_user(email)
        except Exception:
            pass
        authenticity, label, detail = run_detection_multiclip(tmp_path)
        ui_text, color, _ = LABEL_UI.get(label, ("VIDEO UNDETERMINED", "blue", False))
        uses = 2 if double_count else 1
        for _ in range(uses):
            increment_user_uses(email)
        insert_certificate(cert_id=job_id, email=email, original_file=video_url, label=label, authenticity=authenticity, ai_score=detail["ai_score"], sha256=None)
        result = {
            "status": ui_text,
            "authenticity_score": authenticity,
            "color": color,
            "label": label,
            "gpt_reasoning": detail.get("gpt_reasoning", ""),
            "gpt_flags": detail.get("gpt_flags", []),
            "signal_score": detail.get("signal_ai_score", 0),
            "gpt_score": detail.get("gpt_ai_score", 0),
            "audio_score": detail.get("audio_ai_score", 50),
            "audio_confidence": detail.get("audio_confidence", "unavailable"),
            "audio_contribution": detail.get("audio_contribution", 0),
            "audio_evidence": detail.get("audio_evidence", []),
            "job_status": "complete",
        }
        _store_result(r, job_id, result)
        try:
            import hashlib as _hl, json as _jc
            _url_cache_key = "urlcache:v2:" + _hl.md5(video_url.strip().encode()).hexdigest()
            _cache_result = {k: v for k, v in result.items() if k != "job_status"}
            r.setex(_url_cache_key, 3600, _jc.dumps(_cache_result))
        except Exception:
            pass
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
#  Pro/Enterprise ZIP expanded-content certification helpers
# ─────────────────────────────────────────────────────────────
ZIP_EXPANDED_CERT_EXTENSIONS = {
    ".pdf", ".docx", ".doc", ".xlsx", ".xls", ".pptx", ".ppt",
    ".odt", ".ods", ".odp",
    ".txt", ".md", ".csv", ".rtf", ".eml", ".msg",
    ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tif", ".tiff", ".webp", ".heic", ".heif",
    ".html", ".htm", ".mhtml", ".mht", ".xml", ".json", ".svg", ".vsdx",
    ".yaml", ".yml", ".ini", ".log", ".sql",
    ".dwg", ".dxf",
}

ZIP_EXPANDED_CERT_MAX_FILES = int(os.environ.get("VERIFYD_ZIP_CHILD_CERT_MAX_FILES", "75"))
ZIP_EXPANDED_CERT_MAX_TOTAL_BYTES = int(os.environ.get("VERIFYD_ZIP_CHILD_CERT_MAX_TOTAL_BYTES", str(150 * 1024 * 1024)))


def _zip_safe_member_name(name: str) -> str:
    import posixpath
    raw = str(name or "").replace("\\", "/").strip()
    if not raw or raw.endswith("/"):
        return ""
    norm = posixpath.normpath(raw).lstrip("/")
    if not norm or norm.startswith("../") or "/../" in norm or norm in (".", ".."):
        return ""
    return norm[:240]


def _zip_child_safe_arc_part(value: str, fallback: str = "file") -> str:
    import re
    base = str(value or "").replace("\\", "/").split("/")[-1].strip() or fallback
    base = re.sub(r"[^A-Za-z0-9._ -]+", "_", base)
    base = base.strip(" ._") or fallback
    return base[:120]


def _create_zip_child_certified_artifacts(zip_path: str, *, parent_cert_id: str, label: str,
                                          authenticity: int, certified_to: str,
                                          detail: dict | None = None) -> list[dict]:
    import hashlib
    import os as _os
    import shutil
    import tempfile
    import zipfile

    artifacts: list[dict] = []
    detail = dict(detail or {})

    if not zipfile.is_zipfile(zip_path):
        return artifacts

    from doc_certifier import stamp_document

    work_dir = tempfile.mkdtemp(prefix=f"verifyd_zip_children_{parent_cert_id[:8]}_")
    total_selected = 0
    selected_count = 0

    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            infos = []
            for info in zf.infolist():
                safe_name = _zip_safe_member_name(info.filename)
                if not safe_name:
                    continue
                ext = _os.path.splitext(safe_name)[1].lower()
                if ext not in ZIP_EXPANDED_CERT_EXTENSIONS:
                    continue
                if ext == ".zip":
                    continue
                if selected_count >= ZIP_EXPANDED_CERT_MAX_FILES:
                    break
                if info.file_size < 0:
                    continue
                if total_selected + int(info.file_size or 0) > ZIP_EXPANDED_CERT_MAX_TOTAL_BYTES:
                    break
                infos.append((info, safe_name, ext))
                total_selected += int(info.file_size or 0)
                selected_count += 1

            log.info(
                "Worker: ZIP expanded certification selected %d file(s) from %s total_bytes=%d",
                len(infos), zip_path, total_selected
            )

            for idx, (info, safe_name, ext) in enumerate(infos, start=1):
                child_id = f"{parent_cert_id}-{idx:03d}"
                safe_base = _zip_child_safe_arc_part(safe_name, fallback=f"file_{idx:03d}{ext}")
                extract_path = _os.path.join(work_dir, f"child_{idx:03d}_{safe_base}")
                cert_path = _os.path.join(work_dir, f"child_cert_{idx:03d}.pdf")

                with zf.open(info, "r") as src, open(extract_path, "wb") as dst:
                    shutil.copyfileobj(src, dst, length=1024 * 1024)

                h = hashlib.sha256()
                with open(extract_path, "rb") as fh:
                    for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                        h.update(chunk)
                child_sha = h.hexdigest()

                child_detail = dict(detail)
                child_detail["parent_zip_certificate_id"] = parent_cert_id
                child_detail["parent_zip_member"] = safe_name
                child_detail["sha256"] = child_sha

                try:
                    stamp_document(
                        src_path=extract_path,
                        dest_path=cert_path,
                        cert_id=child_id,
                        authenticity=authenticity,
                        label=label,
                        filename=safe_name,
                        sha256=child_sha,
                        detail=child_detail,
                    )
                    if _os.path.exists(cert_path) and _os.path.getsize(cert_path) > 1000:
                        folder = f"{idx:03d}_{_zip_child_safe_arc_part(_os.path.splitext(safe_base)[0], fallback='file')}"
                        artifacts.append({
                            "path": cert_path,
                            "arcname": f"zip_contents_certified/{folder}/VeriFYD_Certified_Report_{child_id}.pdf",
                            "kind": "zip_child_certified_report",
                            "source_member": safe_name,
                            "child_certificate_id": child_id,
                        })
                        artifacts.append({
                            "path": extract_path,
                            "arcname": f"zip_contents_original/{folder}/{safe_base}",
                            "kind": "zip_child_original_file",
                            "source_member": safe_name,
                            "child_certificate_id": child_id,
                        })
                        log.info("Worker: ZIP child certified report created parent=%s child=%s member=%s", parent_cert_id, child_id, safe_name)
                    else:
                        log.warning("Worker: ZIP child certified report missing/small parent=%s member=%s", parent_cert_id, safe_name)
                except Exception as child_err:
                    log.warning("Worker: ZIP child certification failed parent=%s member=%s error=%s", parent_cert_id, safe_name, child_err)

        if artifacts:
            manifest = {
                "parent_certificate_id": parent_cert_id,
                "certified_to": certified_to or "",
                "mode": "pro_expanded_zip_certification",
                "selected_file_limit": ZIP_EXPANDED_CERT_MAX_FILES,
                "selected_total_bytes_limit": ZIP_EXPANDED_CERT_MAX_TOTAL_BYTES,
                "artifact_count": len(artifacts),
                "child_reports": [
                    {
                        "child_certificate_id": a.get("child_certificate_id"),
                        "source_member": a.get("source_member"),
                        "package_path": a.get("arcname"),
                        "kind": a.get("kind"),
                    }
                    for a in artifacts
                ],
            }
            manifest_path = _os.path.join(work_dir, "zip_child_manifest.json")
            import json
            with open(manifest_path, "w", encoding="utf-8") as mf:
                json.dump(manifest, mf, indent=2, sort_keys=True)
            artifacts.append({
                "path": manifest_path,
                "arcname": "verifyd/zip_child_certification_manifest.json",
                "kind": "zip_child_manifest",
                "source_member": "",
                "child_certificate_id": "",
            })

        return artifacts
    except Exception as e:
        log.warning("Worker: ZIP expanded certification failed parent=%s error=%s", parent_cert_id, e)
        return artifacts

def process_document_upload_job(file_key: str, filename: str, email: str) -> dict:
    """
    Background job: retrieve document from R2/Redis, analyze it, store result,
    create the existing certified PDF, and create a universal certified-file ZIP.
    """
    import os as _os
    import tempfile as _tempfile
    import hashlib as _hashlib
    from rq import get_current_job
    from document_detection import run_document_detection
    from database import insert_certificate, increment_user_uses, get_user_status as _gus
    from config import BASE_URL
    from emailer import send_certification_email

    rq_job = get_current_job()
    job_id = rq_job.id if rq_job else file_key.replace("file:", "")
    r = _get_redis()

    _log_worker_runtime_dependencies("document_job_start")

    LABEL_UI = {
        "REAL": ("REAL DOCUMENT VERIFIED", "green", True),
        "UNDETERMINED": ("DOCUMENT UNDETERMINED", "blue", False),
        "AI": ("AI / TAMPERING DETECTED", "red", False),
    }

    ext = _os.path.splitext(filename)[1].lower() or ".pdf"
    if ext not in (
        ".pdf", ".docx", ".doc", ".xlsx", ".xls", ".pptx", ".ppt",
        ".odt", ".ods", ".odp",
        ".txt", ".md", ".csv", ".rtf", ".eml", ".msg",
        ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tif", ".tiff", ".webp", ".heic", ".heif",
        ".zip",
        ".vsdx", ".html", ".htm", ".mhtml", ".mht", ".xml", ".json", ".svg",
        ".yaml", ".yml", ".ini", ".log", ".sql",
        ".pst", ".ost", ".dwg", ".dxf",
    ):
        ext = ".pdf"
    tmp_path = _os.path.join(_tempfile.gettempdir(), f"{job_id}{ext}")

    try:
        if file_key.startswith("r2:"):
            r2_key = file_key[3:]
            from storage import download_video, delete_video
            download_video(r2_key, tmp_path)
            delete_video(r2_key)
            log.info("Retrieved document from R2: job=%s key=%s size=%d bytes", job_id, r2_key, _os.path.getsize(tmp_path))
            sha256 = _sha256_file(tmp_path)
        else:
            file_bytes = r.get(file_key)
            if not file_bytes:
                result = {"job_status": "error", "error": "Document expired from queue. Please re-upload."}
                _store_result(r, job_id, result)
                return result
            with open(tmp_path, "wb") as fh:
                fh.write(file_bytes)
            r.delete(file_key)
            sha256 = _hashlib.sha256(file_bytes).hexdigest()

        log.info("Worker: starting document detection job=%s email=%s filename=%s", job_id, email, filename)

        image_document_exts = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".heic", ".heif")
        if ext in image_document_exts:
            from photo_detection import run_photo_detection
            authenticity, label, detail = run_photo_detection(tmp_path)
            detail = dict(detail or {})
            detail.setdefault("content_type", "document")
            detail["document_type"] = ext.lstrip(".")
            detail["media_type"] = "document"
            detail.setdefault("pages", 1)
            detail.setdefault("embedded_images", 1)
            detail.setdefault("metadata_score", detail.get("signal_ai_score", 0))
            detail.setdefault("text_score", 0)
            detail.setdefault("sha256", sha256)
            log.info("Worker: image-document detection used photo engine job=%s ext=%s label=%s auth=%d", job_id, ext, label, authenticity)
        else:
            authenticity, label, detail = run_document_detection(tmp_path)

        ui_text, color, certify = LABEL_UI.get(label, ("DOCUMENT UNDETERMINED", "blue", False))

        increment_user_uses(email)
        insert_certificate(
            cert_id=job_id,
            email=email,
            original_file=filename,
            label=label,
            authenticity=authenticity,
            ai_score=detail["ai_score"],
            sha256=sha256,
        )

        risk_report = detail.get("document_risk_report") or detail.get("risk_report", {})
        result = {
            "status": ui_text,
            "authenticity_score": authenticity,
            "color": color,
            "label": label,
            "gpt_reasoning": detail.get("gpt_reasoning", ""),
            "gpt_flags": detail.get("gpt_flags", []),
            "signal_score": detail.get("signal_ai_score", 0),
            "gpt_score": detail.get("gpt_ai_score", 0),
            "metadata_score": detail.get("metadata_score", 0),
            "text_score": detail.get("text_score", 0),
            "document_risk_report": risk_report,
            "risk_report": risk_report,
            "overall_risk": risk_report.get("overall_risk") if isinstance(risk_report, dict) else None,
            "risk_score": risk_report.get("risk_score") if isinstance(risk_report, dict) else None,
            "metadata_integrity": risk_report.get("metadata_integrity") if isinstance(risk_report, dict) else None,
            "external_metadata_tool": detail.get("external_metadata_tool", {}),
            "secure_seal": "pending",
            "document_type": detail.get("document_type", ext.lstrip(".")),
            "pages": detail.get("pages", 0),
            "embedded_images": detail.get("embedded_images", 0),
            "sha256": detail.get("sha256", sha256),
            "media_type": "document",
            "job_status": "complete",
            "document_ready": False,
            "universal_certified_file": "pending" if certify else "not_created",
        }

        if certify:
            result["certificate_id"] = job_id
            result["download_url"] = f"{BASE_URL}/download-document/{job_id}"
            result["certification_status"] = "processing"

        _store_result(r, job_id, result)

        if certify:
            certified_path = _os.path.join(_tempfile.gettempdir(), f"cert_doc_{job_id}.pdf")
            package_path = _os.path.join(_tempfile.gettempdir(), f"verifyd_certified_file_{job_id}.zip")
            download_url = f"{BASE_URL}/download-document/{job_id}"
            certified_file_package_url = f"{BASE_URL}/download-certified-file/{job_id}"
            try:
                _log_worker_runtime_dependencies("before_stamp_document", force=True)
                from doc_certifier import stamp_document

                log.info("Worker: creating stamped certified document: job=%s src=%s dest=%s label=%s auth=%d", job_id, tmp_path, certified_path, label, authenticity)
                stamp_document(
                    src_path=tmp_path,
                    dest_path=certified_path,
                    cert_id=job_id,
                    authenticity=authenticity,
                    label=label,
                    filename=filename,
                    sha256=sha256,
                    detail=detail,
                )

                _cert_exists = _os.path.exists(certified_path)
                _cert_size = _os.path.getsize(certified_path) if _cert_exists else 0
                _src_size = _os.path.getsize(tmp_path) if _os.path.exists(tmp_path) else 0
                log.info("Worker: stamped certified document created: job=%s exists=%s size=%d original_size=%d path=%s", job_id, _cert_exists, _cert_size, _src_size, certified_path)

                if _cert_exists and _cert_size > 1000:
                    try:
                        _plan = _gus(email).get("plan", "free")
                    except Exception:
                        _plan = "free"

                    # Phase 9A-1: create universal certified-file package BEFORE the
                    # certified PDF is uploaded/removed. The package includes both the
                    # exact original source file and exact issued certified PDF report.
                    #
                    # Pro/Enterprise ZIP expansion: when the uploaded source is a ZIP,
                    # create child certified reports for supported internal files and
                    # embed them in the same universal certified package.
                    zip_child_artifacts = []
                    if ext == ".zip" and str(_plan).lower() in ("pro", "enterprise"):
                        zip_child_artifacts = _create_zip_child_certified_artifacts(
                            tmp_path,
                            parent_cert_id=job_id,
                            label=label,
                            authenticity=authenticity,
                            certified_to=email,
                            detail=detail,
                        )
                        if zip_child_artifacts:
                            result["zip_child_certification"] = "created"
                            result["zip_child_artifact_count"] = len(zip_child_artifacts)
                        else:
                            result["zip_child_certification"] = "none_created"

                    try:
                        from universal_certifier import create_universal_certified_package
                        create_universal_certified_package(
                            original_path=tmp_path,
                            certified_pdf_path=certified_path,
                            package_path=package_path,
                            cert_id=job_id,
                            original_filename=filename,
                            certified_to=email,
                            label=label,
                            authenticity=authenticity,
                            ai_score=detail.get("ai_score", ""),
                            original_sha256=sha256,
                            detail=detail,
                            extra_artifacts=zip_child_artifacts,
                        )
                        result["universal_certified_file"] = "created"
                        result["certified_file_package"] = "present"
                        log.info("Worker: universal certified file package created: job=%s path=%s size=%d", job_id, package_path, _os.path.getsize(package_path))
                    except Exception as pkg_e:
                        result["universal_certified_file"] = "failed"
                        result["certified_file_package_error"] = str(pkg_e)[:200]
                        log.warning("Worker: universal certified package creation failed for %s: %s", job_id, pkg_e)

                    _stored_in_r2 = False
                    _package_stored_in_r2 = False
                    try:
                        from storage import upload_certified_document, upload_certified_file_package, r2_available
                        if r2_available():
                            log.info("Worker: uploading STAMPED certified document to R2: job=%s path=%s size=%d plan=%s", job_id, certified_path, _os.path.getsize(certified_path), _plan)
                            upload_certified_document(job_id, certified_path, _plan)
                            _os.remove(certified_path)
                            _stored_in_r2 = True
                            log.info("Worker: certified document stored in R2: job=%s plan=%s", job_id, _plan)

                            if _os.path.exists(package_path):
                                package_key = upload_certified_file_package(job_id, package_path, _plan)
                                _os.remove(package_path)
                                _package_stored_in_r2 = True
                                result["universal_certified_file"] = "stored"
                                result["certified_file_storage"] = "r2"
                                result["certified_file_key"] = package_key
                                log.info("Worker: universal certified file package stored in R2: job=%s plan=%s key=%s", job_id, _plan, package_key)
                    except Exception as _r2e:
                        log.warning("R2 document/package upload failed, falling back to Redis where needed: %s", _r2e)

                    _cert_ttl = {"free": 86400, "creator": 259200, "pro": 604800, "enterprise": 2592000}.get(_plan, 86400)

                    if not _stored_in_r2 and _os.path.exists(certified_path):
                        with open(certified_path, "rb") as df:
                            doc_bytes = df.read()
                        r.setex(f"doccert:{job_id}", _cert_ttl, doc_bytes)
                        _os.remove(certified_path)
                        log.info("Worker: certified document stored in Redis: job=%s size=%d bytes plan=%s ttl=%dh", job_id, len(doc_bytes), _plan, _cert_ttl // 3600)

                    if not _package_stored_in_r2 and _os.path.exists(package_path):
                        with open(package_path, "rb") as pf:
                            pkg_bytes = pf.read()
                        r.setex(f"filecert:{job_id}", _cert_ttl, pkg_bytes)
                        _os.remove(package_path)
                        result["universal_certified_file"] = "stored"
                        result["certified_file_storage"] = "redis"
                        result["certified_file_key"] = f"filecert:{job_id}"
                        log.info("Worker: universal certified file package stored in Redis: job=%s size=%d bytes plan=%s ttl=%dh", job_id, len(pkg_bytes), _plan, _cert_ttl // 3600)

                    result["document_ready"] = True
                    result["certification_status"] = "ready"
                    result["summary_pdf_url"] = download_url
                    result["secure_seal"] = "present"

                    # Default document behavior returns the certified PDF. For Pro/Enterprise
                    # ZIPs with child reports, make the primary returned download the full
                    # universal package so the user receives every certified internal file.
                    email_download_url = download_url
                    if ext == ".zip" and result.get("zip_child_certification") == "created":
                        result["download_url"] = certified_file_package_url
                        result["download_type"] = "certified_file_package_zip"
                        result["certified_file_package_url"] = certified_file_package_url
                        result["download_all_certified_files_url"] = certified_file_package_url
                        result["zip_child_certification_message"] = (
                            "Pro ZIP certification completed. Download the full certified package "
                            "to access the individually certified internal files."
                        )
                        email_download_url = certified_file_package_url
                    else:
                        result["download_url"] = download_url
                        if result.get("universal_certified_file") in ("created", "stored"):
                            result["certified_file_package_url"] = certified_file_package_url

                    _store_result(r, job_id, result)

                    if email and "@" in email:
                        try:
                            _sent = send_certification_email(email, job_id, authenticity, filename, email_download_url, is_document=True)
                            log.info("Worker: document certification email sent=%s job=%s email=%s", _sent, job_id, email)
                        except Exception as _em:
                            log.warning("Worker: document certification email failed for %s: %s", job_id, _em)
                else:
                    raise RuntimeError("Stamped document output missing or too small")
            except Exception as stamp_err:
                log.error("Worker: document stamp failed for %s: %s", job_id, stamp_err)
                result["document_ready"] = False
                result["certification_status"] = "failed"
                result["certification_error"] = "Certified document generation failed, but analysis completed."
                _store_result(r, job_id, result)
            finally:
                for _cleanup_path in (certified_path, package_path):
                    try:
                        if _os.path.exists(_cleanup_path):
                            _os.remove(_cleanup_path)
                    except Exception:
                        pass

        log.info("Worker: document detection complete job=%s label=%s auth=%d", job_id, label, authenticity)
        return result
    except Exception as e:
        log.exception("Worker: document job %s failed", job_id)
        result = {"job_status": "error", "error": str(e)[:300]}
        _store_result(r, job_id, result)
        return result
    finally:
        if _os.path.exists(tmp_path):
            _os.remove(tmp_path)
            log.info("Worker: cleaned up document temp file %s", tmp_path)


# ─────────────────────────────────────────────────────────────
#  Keepalive job — prevents worker cold starts
# ─────────────────────────────────────────────────────────────
def keepalive_ping():
    """Lightweight no-op job enqueued every 8 minutes by main.py scheduler."""
    import time as _time
    _log_worker_runtime_dependencies("keepalive", force=True)
    _ts = _time.strftime("%Y-%m-%d %H:%M:%S UTC", _time.gmtime())
    log.info("Keepalive ping: worker alive at %s — models cached in memory", _ts)
    return {"status": "alive", "ts": _ts}






