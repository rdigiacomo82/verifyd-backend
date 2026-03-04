# ============================================================
#  VeriFYD — queue_helper.py  v2
#
#  Passes file content through Redis instead of file path
#  since web service and background worker don't share disk.
# ============================================================

import os
import json
import base64
import logging

log = logging.getLogger("verifyd.queue")

REDIS_URL  = os.environ.get("REDIS_URL", "redis://localhost:6379")
RESULT_TTL = 1800   # 30 minutes


def get_redis():
    import redis
    return redis.from_url(REDIS_URL)


def get_queue():
    from rq import Queue
    return Queue("verifyd", connection=get_redis(), default_timeout=600)


def enqueue_upload(job_id: str, file_path: str,
                   filename: str, email: str) -> str:
    """
    Read file content, store in Redis, enqueue job with redis key.
    Worker retrieves file content from Redis instead of disk.
    """
    from worker import process_upload_job

    # Store file content in Redis with TTL
    r = get_redis()
    with open(file_path, "rb") as f:
        file_bytes = f.read()

    # Store raw bytes in Redis under a separate key
    file_key = f"upload_file:{job_id}"
    r.setex(file_key, RESULT_TTL, file_bytes)
    log.info("Stored file in Redis: %s (%d bytes)", file_key, len(file_bytes))

    # Clean up local file immediately after storing in Redis
    if os.path.exists(file_path):
        os.remove(file_path)
        log.info("Cleaned up local file: %s", file_path)

    q = get_queue()
    q.enqueue(
        process_upload_job,
        kwargs={
            "job_id":   job_id,
            "file_key": file_key,
            "filename": filename,
            "email":    email,
        },
        job_id      = job_id,
        result_ttl  = RESULT_TTL,
        job_timeout = 600,
    )
    log.info("Enqueued upload job %s for %s", job_id, email)
    return job_id


def enqueue_link(job_id: str, video_url: str,
                 email: str, double_count: bool = False) -> str:
    from worker import process_link_job
    q = get_queue()
    q.enqueue(
        process_link_job,
        kwargs={
            "job_id":       job_id,
            "video_url":    video_url,
            "email":        email,
            "double_count": double_count,
        },
        job_id      = job_id,
        result_ttl  = RESULT_TTL,
        job_timeout = 600,
    )
    log.info("Enqueued link job %s for %s", job_id, email)
    return job_id


def get_job_result(job_id: str) -> dict:
    """
    Check job status. Returns dict with job_status field.
    """
    try:
        r = get_redis()

        # Check for completed result stored by worker
        raw = r.get(f"result:{job_id}")
        if raw:
            return json.loads(raw)

        # Check live RQ job status
        from rq.job import Job
        try:
            job    = Job.fetch(job_id, connection=r)
            status = job.get_status()
            if str(status) in ("queued",):
                try:
                    q   = get_queue()
                    ids = q.job_ids
                    pos = ids.index(job_id) + 1 if job_id in ids else 1
                except Exception:
                    pos = 1
                return {"job_status": "queued", "position": pos}
            elif str(status) in ("started", "deferred", "scheduled"):
                return {"job_status": "processing"}
            elif str(status) == "failed":
                return {"job_status": "error",
                        "error": "Analysis failed — please try again."}
        except Exception:
            pass

        return {"job_status": "not_found"}

    except Exception as e:
        log.error("get_job_result error: %s", e)
        return {"job_status": "error", "error": str(e)[:100]}
