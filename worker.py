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

RESULT_TTL = 1800   # 30 min


def _get_redis():
    import redis
    url = os.environ.get("REDIS_URL", "redis://localhost:6379")
    return redis.from_url(url, decode_responses=False)


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

    # ── Retrieve file bytes from Redis ────────────────────────
    file_bytes = r.get(file_key)
    if not file_bytes:
        log.error("File not found in Redis: key=%s job=%s", file_key, job_id)
        result = {"job_status": "error", "error": "File expired from queue. Please re-upload."}
        _store_result(r, job_id, result)
        return result

    # Write bytes to local temp file
    suffix   = os.path.splitext(filename)[1] or ".mp4"
    tmp_path = os.path.join(tempfile.gettempdir(), f"{job_id}{suffix}")
    with open(tmp_path, "wb") as f:
        f.write(file_bytes)

    # Delete file bytes from Redis immediately to free memory
    r.delete(file_key)
    log.info("Retrieved file from Redis: job=%s size=%d bytes", job_id, len(file_bytes))

    # SHA256 for certificate
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
                    with open(certified_path, "rb") as vf:
                        cert_bytes = vf.read()
                    # Plan-aware TTL:
                    # free=24h, creator=72h, pro=7d, enterprise=30d
                    from database import get_user_status as _gus
                    try:
                        _plan = _gus(email).get("plan", "free")
                    except Exception:
                        _plan = "free"
                    _cert_ttl = {
                        "free":       86400,      # 24 hours
                        "creator":    259200,     # 72 hours
                        "pro":        604800,     # 7 days
                        "enterprise": 2592000,    # 30 days
                    }.get(_plan, 86400)
                    r.setex(f"cert:{job_id}", _cert_ttl, cert_bytes)
                    os.remove(certified_path)
                    log.info("Worker: certified video stored in Redis: job=%s size=%d bytes plan=%s ttl=%dh",
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

    try:
        log.info("Worker: downloading url for job=%s", job_id)
        download_video_ytdlp(video_url, tmp_path)

        if not os.path.exists(tmp_path):
            raise RuntimeError(f"Download produced no file: {video_url}")

        log.info("Worker: starting detection for link job=%s", job_id)

        # Normalize resolution to 576px wide — matches standard TikTok manual download
        # SMVD serves 704x1280 or 720x1280; users upload 576x1024 or 576x1048
        # Normalizing ensures signal scores (noise, chan_corr, flat_noise) are consistent
        from config import FFMPEG_BIN, TMP_DIR
        detect_path = tmp_path
        norm_path = None
        try:
            _probe = subprocess.run([
                "ffprobe", "-v", "quiet", "-select_streams", "v:0",
                "-show_entries", "stream=width",
                "-of", "csv=p=0", tmp_path
            ], capture_output=True, text=True, timeout=10)
            dl_width = int(_probe.stdout.strip()) if _probe.stdout.strip().isdigit() else 0
            log.info("Worker: downloaded video width=%d for link job=%s", dl_width, job_id)

            if dl_width > 576:
                norm_path = os.path.join(TMP_DIR, f"{job_id}_norm.mp4")
                norm_result = subprocess.run([
                    FFMPEG_BIN, "-y", "-i", tmp_path,
                    "-vf", "scale=576:-2",
                    "-c:v", "libx264", "-crf", "23", "-preset", "fast",
                    "-c:a", "copy",
                    norm_path
                ], capture_output=True, timeout=120)
                if norm_result.returncode == 0 and os.path.exists(norm_path):
                    log.info("Worker: normalized to 576px wide for link job=%s", job_id)
                    detect_path = norm_path
                else:
                    log.warning("Worker: normalization failed for job=%s, using original", job_id)
                    norm_path = None
        except Exception as norm_err:
            log.warning("Worker: normalization error for job=%s: %s", job_id, norm_err)

        # Clip first 6 seconds — same pipeline as upload job
        try:
            clip_path = clip_first_6_seconds(detect_path)
        except Exception as clip_err:
            log.warning("Worker: clip failed for link job=%s, using full video: %s", job_id, clip_err)
            clip_path = detect_path
        authenticity, label, detail = run_detection_multiclip(clip_path)
        if clip_path != detect_path and os.path.exists(clip_path):
            os.remove(clip_path)
        if norm_path and os.path.exists(norm_path):
            try: os.remove(norm_path)
            except Exception: pass
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
        return result

    except Exception as e:
        log.exception("Worker: link job %s failed", job_id)
        result = {"job_status": "error", "error": str(e)[:300]}
        _store_result(r, job_id, result)
        return result

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
