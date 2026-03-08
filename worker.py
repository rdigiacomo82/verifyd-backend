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
    job_id:   str,
    file_key: str,   # Redis key where file bytes are stored
    filename: str,
    email:    str,
) -> dict:
    """
    Background job: retrieve video from Redis, analyze it, store result.
    Called by RQ. file_key = 'file:{job_id}' stored by main.py.
    """
    from detection import run_detection
    from video     import clip_first_6_seconds, stamp_video
    from database  import insert_certificate, increment_user_uses
    from config    import CERT_DIR, BASE_URL
    from emailer   import send_certification_email

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
        authenticity, label, detail = run_detection(tmp_path)
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
            certified_path = os.path.join(CERT_DIR, f"{job_id}.mp4")
            download_url   = f"{BASE_URL}/download/{job_id}"
            try:
                os.makedirs(CERT_DIR, exist_ok=True)
                clip_path = clip_first_6_seconds(tmp_path)
                stamp_video(clip_path, certified_path, job_id)
                if os.path.exists(clip_path):
                    os.remove(clip_path)
                result["certificate_id"] = job_id
                result["download_url"]   = download_url
                log.info("Worker: certified video stamped: %s", certified_path)
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
    from detection import run_detection
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
        authenticity, label, detail = run_detection(tmp_path)
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

        if certify:
            certified_path = os.path.join(CERT_DIR, f"{job_id}.mp4")
            download_url   = f"{BASE_URL}/download/{job_id}"
            try:
                os.makedirs(CERT_DIR, exist_ok=True)
                clip_path = clip_first_6_seconds(tmp_path)
                stamp_video(clip_path, certified_path, job_id)
                if os.path.exists(clip_path):
                    os.remove(clip_path)
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
