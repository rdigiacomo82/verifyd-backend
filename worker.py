# ============================================================
#  VeriFYD — worker.py
#
#  RQ background worker for async video analysis.
#  Run via: rq worker verifyd --url $REDIS_URL
#
#  File transfer: main.py stores video bytes in Redis under
#  key "file:{job_id}". Worker retrieves bytes, writes to
#  local /tmp, processes, then cleans up.
#  No shared disk needed between web service and worker.
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
    log.info("Stored result: job=%s label=%s auth=%s",
             job_id, result.get("label"), result.get("authenticity_score"))


def process_upload_job(
    file_key: str,   # Redis key where file bytes are stored
    filename: str,
    email:    str,
) -> dict:
    """
    Background job: retrieve video from Redis, analyze it, store result.
    Called by RQ. job_id retrieved from RQ job context.
    file_key = 'file:{job_id}' stored by main.py.
    """
    from rq        import get_current_job
    from detection import run_detection, run_detection_multiclip
    from video     import clip_first_6_seconds, stamp_video
    from database  import insert_certificate, increment_user_uses
    from config    import CERT_DIR, BASE_URL
    from emailer   import send_certification_email

    # Get job_id from RQ context
    rq_job = get_current_job()
    job_id = rq_job.id if rq_job else file_key.replace("file:", "")

    r = _get_redis()

    LABEL_UI = {
        "REAL":         ("REAL VIDEO VERIFIED", "green",  True),
        "UNDETERMINED": ("VIDEO UNDETERMINED",  "blue",   False),
        "AI":           ("AI DETECTED",         "red",    False),
    }

    # ── Retrieve file from storage (R2 or Redis fallback) ───────
    suffix   = os.path.splitext(filename)[1] or ".mp4"
    tmp_path = os.path.join(tempfile.gettempdir(), f"{job_id}{suffix}")

    if file_key.startswith("r2:"):
        # ── R2 path ───────────────────────────────────────────
        r2_key = file_key[3:]   # strip "r2:" prefix
        try:
            from storage import download_video, delete_video
            download_video(r2_key, tmp_path)
            delete_video(r2_key)   # clean up R2 immediately
            file_size = os.path.getsize(tmp_path)
            log.info("Retrieved file from R2: job=%s key=%s size=%d bytes",
                     job_id, r2_key, file_size)
        except Exception as e:
            log.error("R2 download failed: key=%s job=%s error=%s", r2_key, job_id, e)
            result = {"job_status": "error", "error": "File could not be retrieved. Please re-upload."}
            _store_result(r, job_id, result)
            return result
        # SHA256 from disk (avoids loading whole file into RAM)
        sha256 = _sha256_file(tmp_path)
    else:
        # ── Redis fallback (original behaviour) ───────────────
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

        # ── Run detection ─────────────────────────────────────
        authenticity, label, detail = run_detection_multiclip(tmp_path)
        ui_text, color, certify = LABEL_UI.get(label, ("VIDEO UNDETERMINED", "blue", False))

        log.info("Worker: detection complete job=%s label=%s auth=%d",
                 job_id, label, authenticity)

        # ── Record use + persist certificate ─────────────────
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

        # ── Build result ──────────────────────────────────────
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

        # ── Stamp certified videos ────────────────────────────
        if certify:
            certified_path = os.path.join(tempfile.gettempdir(), f"cert_{job_id}.mp4")
            download_url   = f"{BASE_URL}/download/{job_id}"
            try:
                clip_path = clip_first_6_seconds(tmp_path)
                stamp_video(clip_path, certified_path, job_id)
                if os.path.exists(clip_path):
                    os.remove(clip_path)
                # Store certified video bytes in Redis so backend can serve download
                # (worker and backend are separate containers with no shared disk)
                if os.path.exists(certified_path):
                    from database import get_user_status as _gus
                    try:
                        _plan = _gus(email).get("plan", "free")
                    except Exception:
                        _plan = "free"

                    # ── Try R2 first ──────────────────────────
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

                    # ── Redis fallback ────────────────────────
                    if not _stored_in_r2:
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
                                 job_id, len(cert_bytes), _plan, _cert_ttl//3600)
                result["certificate_id"] = job_id
                result["download_url"]   = download_url
                # Send email
                if email and "@" in email:
                    try:
                        send_certification_email(email, job_id, authenticity,
                                                 filename, download_url)
                    except Exception as e:
                        log.warning("Worker: email failed for %s: %s", job_id, e)
            except Exception as e:
                log.error("Worker: stamp failed for %s: %s", job_id, e)

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


def process_link_job(
    job_id:       str,
    video_url:    str,
    email:        str,
    double_count: bool = False,
) -> dict:
    """
    Background job: download and analyze a video from a URL.
    """
    from detection import run_detection, run_detection_multiclip
    from video     import download_video_ytdlp, clip_first_6_seconds, stamp_video
    from database  import insert_certificate, increment_user_uses
    from config    import CERT_DIR, BASE_URL
    from emailer   import send_certification_email

    r = _get_redis()

    LABEL_UI = {
        "REAL":         ("REAL VIDEO VERIFIED", "green",  True),
        "UNDETERMINED": ("VIDEO UNDETERMINED",  "blue",   False),
        "AI":           ("AI DETECTED",         "red",    False),
    }

    tmp_path = os.path.join(tempfile.gettempdir(), f"{job_id}.mp4")

    # ── Blocked domains — reject immediately with clear error ────
    _BLOCKED_DOMAINS = [
        "pornhub.com", "xvideos.com", "xhamster.com", "redtube.com",
        "youporn.com", "tube8.com", "xnxx.com", "spankbang.com",
        "onlyfans.com", "fansly.com",
    ]
    _url_lower = video_url.lower()
    if any(domain in _url_lower for domain in _BLOCKED_DOMAINS):
        _blocked_result = {
            "job_status": "error",
            "error": "This platform is not supported. VeriFYD analyzes YouTube, TikTok, and direct video uploads.",
            "error_detail": f"Blocked domain: {video_url[:100]}",
        }
        _store_result(r, job_id, _blocked_result)
        log.warning("Worker: blocked domain rejected for job=%s url=%s", job_id, video_url[:80])
        return _blocked_result

    try:
        log.info("Worker: downloading url for job=%s", job_id)
        download_video_ytdlp(video_url, tmp_path)

        if not os.path.exists(tmp_path):
            raise RuntimeError(f"Download produced no file: {video_url}")

        log.info("Worker: starting detection for link job=%s", job_id)

        # Register user if not already in DB — link jobs may skip email
        # verification flow so we ensure they exist for admin visibility
        try:
            from database import get_or_create_user
            get_or_create_user(email)
            log.info("Worker: ensured user record exists for %s", email)
        except Exception as _ue:
            log.warning("Worker: could not create user record for %s: %s", email, _ue)

        # Pass full downloaded video directly to multiclip detection.
        # extract_clips_for_detection() handles scaling internally (max 720px)
        # so no separate normalization step needed — saves 5-10s on large videos.
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

        # Link analysis: results only — no certified video stamp
        # The submitter may not own the linked video, so we don't
        # create a certified download or watermark it.
        certify = False  # override regardless of label
        if False:  # disabled for link jobs
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
                    log.info("Worker: certified video stored in Redis: link job=%s size=%d bytes",
                             job_id, len(cert_bytes))
                result["certificate_id"] = job_id
                result["download_url"]   = download_url
                if email and "@" in email:
                    try:
                        send_certification_email(email, job_id, authenticity,
                                                 video_url, download_url)
                    except Exception as e:
                        log.warning("Worker: email failed for %s: %s", job_id, e)
            except Exception as e:
                log.error("Worker: stamp failed for link job %s: %s", job_id, e)

        _store_result(r, job_id, result)

        # Write URL cache so repeat analyses of same link are instant
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

        # Provide specific, user-friendly error messages for known failure modes
        err_str = str(e).lower()
        if "sign in to confirm" in err_str or "login" in err_str or "bot" in err_str:
            user_error = (
                "This YouTube video requires sign-in to access and cannot be analyzed. "
                "This happens with age-restricted videos or videos with strict privacy settings. "
                "Try uploading the video file directly instead."
            )
        elif "403" in err_str or "forbidden" in err_str:
            user_error = (
                "Access to this video was blocked (403 Forbidden). "
                "The video may be region-locked, private, or restricted. "
                "Try uploading the video file directly instead."
            )
        elif "private" in err_str:
            user_error = "This video is private and cannot be accessed."
        elif "copyright" in err_str or "content id" in err_str:
            user_error = (
                "This video cannot be analyzed because it is copyright-protected. "
                "The content owner has restricted automated access to this video. "
                "Try uploading the video file directly instead."
            )
        elif "not available" in err_str or "unavailable" in err_str:
            user_error = "This video is unavailable or has been removed."
        elif "download produced no file" in err_str:
            user_error = (
                "This video could not be downloaded. It may be copyright-protected, "
                "age-restricted, region-locked, or set to private. "
                "Try uploading the video file directly instead."
            )
        else:
            user_error = (
                "This video could not be analyzed. "
                "YouTube may be blocking access, or the link may be invalid. "
                "Try uploading the video file directly instead."
            )

        result = {
            "job_status": "error",
            "error": user_error,
            "error_detail": str(e)[:300],
        }
        _store_result(r, job_id, result)
        return result

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

