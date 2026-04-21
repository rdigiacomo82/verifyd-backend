# ============================================================
#  VeriFYD — queue_helper.py
#
#  Redis/RQ queue interface for background job processing.
#  Web service enqueues jobs; worker processes them.
#
#  File transfer strategy:
#    - If R2 is configured: upload file to R2, pass object key
#    - Fallback: store file bytes in Redis (original behaviour)
#
#  This means the system works identically with or without R2.
# ============================================================

import os
import json
import logging

log = logging.getLogger("verifyd.queue")

RESULT_TTL  = 1800   # 30 min — how long results stay in Redis
FILE_TTL    = 3600   # 1 hr  — how long raw file bytes stay in Redis (fallback)
QUEUE_NAME  = "verifyd"


def get_redis():
    import redis
    url = os.environ.get("REDIS_URL", "redis://localhost:6379")
    return redis.from_url(url, decode_responses=False)


def enqueue_upload(job_id: str, file_path: str, filename: str, email: str) -> None:
    """
    Transfer file to storage (R2 if configured, Redis fallback),
    then enqueue a processing job.
    """
    from rq import Queue

    r = get_redis()

    # ── Try R2 first ─────────────────────────────────────────
    try:
        from storage import upload_video, r2_available
        if r2_available():
            r2_key = upload_video(job_id, file_path, filename)
            log.info("Stored file in R2: key=%s", r2_key)
            q = Queue(QUEUE_NAME, connection=r)
            q.enqueue(
                "worker.process_upload_job",
                kwargs={
                    "file_key": f"r2:{r2_key}",   # prefix tells worker to use R2
                    "filename": filename,
                    "email":    email,
                },
                job_id=job_id,
                job_timeout=600,
                result_ttl=RESULT_TTL,
            )
            log.info("Enqueued upload job (R2): job_id=%s email=%s", job_id, email)
            return
    except Exception as e:
        log.warning("R2 upload failed, falling back to Redis: %s", e)

    # ── Redis fallback (original behaviour) ──────────────────
    file_key = f"file:{job_id}"
    with open(file_path, "rb") as f:
        file_bytes = f.read()
    r.setex(file_key, FILE_TTL, file_bytes)
    log.info("Stored file in Redis (fallback): key=%s size=%d", file_key, len(file_bytes))

    q = Queue(QUEUE_NAME, connection=r)
    q.enqueue(
        "worker.process_upload_job",
        kwargs={
            "file_key": file_key,
            "filename": filename,
            "email":    email,
        },
        job_id=job_id,
        job_timeout=600,
        result_ttl=RESULT_TTL,
    )
    log.info("Enqueued upload job (Redis): job_id=%s email=%s file=%s", job_id, email, filename)


def enqueue_link(job_id: str, video_url: str, email: str, double_count: bool = False) -> None:
    """Enqueue a link analysis job (unchanged — no file transfer needed)."""
    from rq import Queue

    r = get_redis()
    q = Queue(QUEUE_NAME, connection=r)
    q.enqueue(
        "worker.process_link_job",
        kwargs={
            "job_id":       job_id,
            "video_url":    video_url,
            "email":        email,
            "double_count": double_count,
        },
        job_id=job_id,
        job_timeout=600,
        result_ttl=RESULT_TTL,
    )
    log.info("Enqueued link job: job_id=%s email=%s url=%s", job_id, email, video_url)


def enqueue_photo_upload(job_id: str, file_path: str,
                         filename: str, email: str) -> None:
    """
    Transfer photo to storage (R2 if configured, Redis fallback),
    then enqueue a photo processing job.
    Mirrors enqueue_upload() but calls process_photo_upload_job.
    """
    from rq import Queue

    r = get_redis()

    # ── Try R2 first ─────────────────────────────────────────
    try:
        from storage import upload_video, r2_available
        if r2_available():
            r2_key = upload_video(job_id, file_path, filename)
            log.info("Stored photo in R2: key=%s", r2_key)
            q = Queue(QUEUE_NAME, connection=r)
            q.enqueue(
                "worker.process_photo_upload_job",
                kwargs={
                    "file_key": f"r2:{r2_key}",
                    "filename": filename,
                    "email":    email,
                },
                job_id=job_id,
                job_timeout=300,   # photos are faster than video — 5 min max
                result_ttl=RESULT_TTL,
            )
            log.info("Enqueued photo upload job (R2): job_id=%s email=%s", job_id, email)
            return
    except Exception as e:
        log.warning("R2 photo upload failed, falling back to Redis: %s", e)

    # ── Redis fallback ────────────────────────────────────────
    file_key = f"file:{job_id}"
    with open(file_path, "rb") as f:
        file_bytes = f.read()
    r.setex(file_key, FILE_TTL, file_bytes)
    log.info("Stored photo in Redis (fallback): key=%s size=%d", file_key, len(file_bytes))

    q = Queue(QUEUE_NAME, connection=r)
    q.enqueue(
        "worker.process_photo_upload_job",
        kwargs={
            "file_key": file_key,
            "filename": filename,
            "email":    email,
        },
        job_id=job_id,
        job_timeout=300,
        result_ttl=RESULT_TTL,
    )
    log.info("Enqueued photo upload job (Redis): job_id=%s email=%s file=%s",
             job_id, email, filename)


def enqueue_photo_link(job_id: str, image_url: str, email: str) -> None:
    """
    Enqueue a photo link analysis job.
    Mirrors enqueue_link() but calls process_photo_link_job.
    """
    from rq import Queue

    r = get_redis()
    q = Queue(QUEUE_NAME, connection=r)
    q.enqueue(
        "worker.process_photo_link_job",
        kwargs={
            "job_id":    job_id,
            "image_url": image_url,
            "email":     email,
        },
        job_id=job_id,
        job_timeout=300,
        result_ttl=RESULT_TTL,
    )
    log.info("Enqueued photo link job: job_id=%s email=%s url=%s",
             job_id, email, image_url[:80])


def get_job_result(job_id: str) -> dict:
    """
    Check Redis for a completed job result.
    Returns None if not found, or dict with job_status field.
    """
    try:
        r = get_redis()

        result_key = f"result:{job_id}"
        raw = r.get(result_key)
        if raw:
            result = json.loads(raw)
            result["job_status"] = "complete"
            return result

        from rq.job import Job
        try:
            job = Job.fetch(job_id, connection=r)
            if job.is_failed:
                return {"job_status": "error", "error": str(job.exc_info)[:200]}
            elif job.is_started:
                return {"job_status": "processing"}
            elif job.is_queued:
                from rq import Queue
                q = Queue(QUEUE_NAME, connection=r)
                position = q.job_ids.index(job_id) + 1 if job_id in q.job_ids else 1
                return {"job_status": "queued", "position": position}
        except Exception:
            pass

        return {"job_status": "not_found"}

    except Exception as e:
        log.error("get_job_result error: %s", e)
        return {"job_status": "not_found"}