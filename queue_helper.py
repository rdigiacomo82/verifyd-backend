# ============================================================
#  VeriFYD — queue_helper.py
#
#  Redis/RQ queue interface for background job processing.
#  Web service enqueues jobs; worker processes them.
#  File bytes transferred through Redis (no shared disk needed).
# ============================================================

import os
import json
import logging

log = logging.getLogger("verifyd.queue")

RESULT_TTL  = 1800   # 30 min — how long results stay in Redis
FILE_TTL    = 3600   # 1 hr  — how long raw file bytes stay in Redis
QUEUE_NAME  = "verifyd"


def get_redis():
    import redis
    url = os.environ.get("REDIS_URL", "redis://localhost:6379")
    return redis.from_url(url, decode_responses=False)


def enqueue_upload(job_id: str, file_path: str, filename: str, email: str) -> None:
    """
    Store file bytes in Redis, then enqueue a processing job.
    The worker retrieves bytes from Redis (no shared disk needed).
    """
    from rq import Queue

    r = get_redis()

    # Store file bytes in Redis so worker can retrieve them
    file_key = f"file:{job_id}"
    with open(file_path, "rb") as f:
        file_bytes = f.read()
    r.setex(file_key, FILE_TTL, file_bytes)
    log.info("Stored file in Redis: key=%s size=%d", file_key, len(file_bytes))

    # Enqueue job
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
    log.info("Enqueued upload job: job_id=%s email=%s file=%s", job_id, email, filename)


def enqueue_link(job_id: str, video_url: str, email: str, double_count: bool = False) -> None:
    """Enqueue a link analysis job."""
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


def get_job_result(job_id: str) -> dict:
    """
    Check Redis for a completed job result.
    Returns None if not found, or dict with job_status field.
    """
    try:
        r = get_redis()

        # Check for completed result stored by worker
        result_key = f"result:{job_id}"
        raw = r.get(result_key)
        if raw:
            result = json.loads(raw)
            result["job_status"] = "complete"
            return result

        # Check RQ job status
        from rq.job import Job
        try:
            job = Job.fetch(job_id, connection=r)
            if job.is_failed:
                return {"job_status": "error", "error": str(job.exc_info)[:200]}
            elif job.is_started:
                return {"job_status": "processing"}
            elif job.is_queued:
                # Get position in queue
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