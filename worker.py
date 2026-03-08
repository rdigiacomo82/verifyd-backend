# ============================================================
#  VeriFYD — worker.py
#
#  RQ background worker for async video analysis.
#  Run this as a separate Render Background Worker service.
#
#  This file contains:
#    1. process_upload_job()    — handles file upload analysis
#    2. process_link_job()      — handles link analysis
#    3. Helper to store results in Redis
#
#  Jobs are queued by main.py and picked up by this worker.
#  Results stored in Redis with 30-minute TTL.
# ============================================================

import os
import uuid
import logging
import tempfile
import hashlib

log = logging.getLogger("verifyd.worker")
logging.basicConfig(level=logging.INFO)

# ── Result TTL ────────────────────────────────────────────────
RESULT_TTL = 1800   # 30 minutes in seconds


def _store_result(redis_conn, job_id: str, result: dict) -> None:
    """Store job result in Redis with TTL."""
    import json
    redis_conn.setex(f"result:{job_id}", RESULT_TTL, json.dumps(result))
    log.info("Stored result for job %s: label=%s", job_id, result.get("label"))


def process_upload_job(
    job_id:    str,
    file_path: str,
    filename:  str,
    email:     str,
) -> dict:
    """
    Background job: analyze an uploaded video file.
    Called by RQ worker. Stores result in Redis when done.
    """
    import redis
    import json
    from detection import run_detection
    from video     import clip_first_6_seconds, stamp_video
    from database  import (insert_certificate, increment_user_uses,
                           get_user_status)
    from config    import CERT_DIR, BASE_URL
    from emailer   import send_certification_email

    redis_url  = os.environ.get("REDIS_URL", "redis://localhost:6379")
    redis_conn = redis.from_url(redis_url)

    LABEL_UI = {
        "REAL":         ("REAL VIDEO VERIFIED", "green",  True),
        "UNDETERMINED": ("VIDEO UNDETERMINED",  "blue",   False),
        "AI":           ("AI DETECTED",         "red",    False),
    }

    try:
        log.info("Worker: processing upload job %s for %s", job_id, email)

        # Hash the file
        h = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        sha256 = h.hexdigest()

        # Run detection
        authenticity, label, detail = run_detection(file_path)
        ui_text, color, certify = LABEL_UI.get(label, ("UNKNOWN", "grey", False))

        # Count the use
        increment_user_uses(email)

        # Persist certificate record
        cid = job_id   # reuse job_id as cert_id for simplicity
        insert_certificate(
            cert_id       = cid,
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
        }

        if certify:
            certified_path = f"{CERT_DIR}/{cid}.mp4"
            download_url   = f"{BASE_URL}/download/{cid}"
            try:
                clip_path = clip_first_6_seconds(file_path)
                stamp_video(clip_path, certified_path, cid)
                if os.path.exists(clip_path):
                    os.remove(clip_path)
                result["certificate_id"] = cid
                result["download_url"]   = download_url
                # Send certification email
                if email and email != "anonymous@verifyd.com":
                    send_certification_email(
                        email, cid, authenticity, filename, download_url
                    )
            except Exception as e:
                log.error("Worker: stamp failed for %s: %s", cid, e)

        _store_result(redis_conn, job_id, result)
        return result

    except Exception as e:
        log.exception("Worker: job %s failed: %s", job_id, e)
        error_result = {
            "job_status": "error",
            "error":      str(e)[:200],
        }
        _store_result(redis_conn, job_id, error_result)
        return error_result

    finally:
        # Always clean up the uploaded file
        if os.path.exists(file_path):
            os.remove(file_path)
            log.info("Worker: cleaned up %s", file_path)


def process_link_job(
    job_id:       str,
    video_url:    str,
    email:        str,
    double_count: bool = False,
) -> dict:
    """
    Background job: download and analyze a video from a URL.
    Called by RQ worker. Stores result in Redis when done.
    """
    import redis
    from detection import run_detection
    from video     import download_video_ytdlp
    from database  import insert_certificate, increment_user_uses

    redis_url  = os.environ.get("REDIS_URL", "redis://localhost:6379")
    redis_conn = redis.from_url(redis_url)

    LABEL_UI = {
        "REAL":         ("REAL VIDEO VERIFIED", "green",  True),
        "UNDETERMINED": ("VIDEO UNDETERMINED",  "blue",   False),
        "AI":           ("AI DETECTED",         "red",    False),
    }

    tmp_path = tempfile.mktemp(suffix=".mp4")

    try:
        log.info("Worker: processing link job %s url=%s", job_id, video_url)

        download_video_ytdlp(video_url, tmp_path)

        if not os.path.exists(tmp_path) or os.path.getsize(tmp_path) < 1024:
            raise RuntimeError("Downloaded file is too small or missing.")

        authenticity, label, detail = run_detection(tmp_path)
        ui_text, color, certify = LABEL_UI.get(label, ("UNKNOWN", "grey", False))

        # Count use(s)
        if email and email != "anonymous@verifyd.com":
            increment_user_uses(email)
            if double_count:
                increment_user_uses(email)

        cid = job_id
        insert_certificate(
            cert_id       = cid,
            email         = email,
            original_file = video_url[:100],
            label         = label,
            authenticity  = authenticity,
            ai_score      = detail["ai_score"],
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

        _store_result(redis_conn, job_id, result)
        return result

    except Exception as e:
        log.exception("Worker: link job %s failed: %s", job_id, e)
        error_result = {
            "job_status": "error",
            "error":      str(e)[:200],
        }
        _store_result(redis_conn, job_id, error_result)
        return error_result

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
