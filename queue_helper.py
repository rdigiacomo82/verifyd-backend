# ============================================================
#  VeriFYD — queue_helper.py
#
#  Redis/RQ queue management.
#  Single source of truth for queue connection and job ops.
# ============================================================

import os
import json
import logging

log = logging.getLogger("verifyd.queue")

REDIS_URL  = os.environ.get("REDIS_URL", "redis://localhost:6379")
RESULT_TTL = 1800   # 30 minutes


def get_redis():
    import redis
    return redis.from_url(REDIS_URL)


def get_queue():
    from rq import Queue
    return Queue("verifyd", connection=get_redis(), default_timeout=300)


def enqueue_upload(job_id: str, file_path: str,
                   filename: str, email: str) -> str:
    from rq import Queue
    from worker import process_upload_job
    q = get_queue()
    q.enqueue(
        process_upload_job,
        kwargs={
            "job_id":    job_id,
            "file_path": file_path,
            "filename":  filename,
            "email":     email,
        },
        job_id      = job_id,
        result_ttl  = RESULT_TTL,
        job_timeout = 300,
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
        job_timeout = 300,
    )
    log.info("Enqueued link job %s for %s", job_id, email)
    return job_id


def get_job_result(job_id: str) -> dict:
    """
    Check job status. Returns dict with job_status field:
      queued      — waiting in queue (includes position)
      processing  — currently running
      complete    — done (includes full result)
      error       — failed
      not_found   — unknown job_id
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
                    q       = get_queue()
                    ids     = q.job_ids
                    pos     = ids.index(job_id) + 1 if job_id in ids else 1
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
