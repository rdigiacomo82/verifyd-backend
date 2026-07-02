# ============================================================
#  VeriFYD — queue_helper.py
#
#  Redis/RQ enqueue helpers for video, photo, link, and document
#  analysis jobs.
#
#  This file is safe as a full replacement. It preserves the
#  existing helper names used by main.py:
#    - enqueue_upload
#    - enqueue_link
#    - enqueue_photo_upload
#    - enqueue_photo_link
#    - get_job_result
#
#  New helper added for VeriFYD Docs MVP:
#    - enqueue_document_upload
# ============================================================

import json
import logging
import os
from typing import Any, Dict, Optional

log = logging.getLogger("verifyd.queue")

QUEUE_NAME = os.environ.get("RQ_QUEUE", "verifyd")
DOCUMENT_QUEUE_NAME = os.environ.get("RQ_DOCUMENT_QUEUE", "verifyd-documents")
FILE_TTL_SECONDS = int(os.environ.get("VERIFYD_FILE_TTL", "1800"))
RESULT_TTL_SECONDS = int(os.environ.get("VERIFYD_RESULT_TTL", "1800"))


def _get_redis(decode_responses: bool = False):
    """Return a Redis connection using REDIS_URL."""
    import redis

    return redis.from_url(
        os.environ.get("REDIS_URL", "redis://localhost:6379"),
        decode_responses=decode_responses,
    )


def _get_queue(redis_conn=None, queue_name: Optional[str] = None):
    """
    Return an RQ queue used by VeriFYD workers.

    Default queue:
      verifyd

    Document/Office/CAD queue:
      verifyd-documents

    Queue names can be overridden with:
      RQ_QUEUE
      RQ_DOCUMENT_QUEUE
    """
    import rq

    selected_queue = queue_name or QUEUE_NAME
    return rq.Queue(selected_queue, connection=redis_conn or _get_redis())


def _safe_remove(path: str) -> None:
    try:
        if path and os.path.exists(path):
            os.remove(path)
    except Exception as exc:
        log.debug("Could not remove temp file %s: %s", path, exc)


def _store_file_in_redis(redis_conn, job_id: str, raw_path: str) -> str:
    """
    Store a file in Redis for the worker to retrieve.
    Returns the Redis file key.
    """
    file_key = f"file:{job_id}"
    with open(raw_path, "rb") as fh:
        file_bytes = fh.read()
    redis_conn.setex(file_key, FILE_TTL_SECONDS, file_bytes)
    log.info("Stored file in Redis: key=%s size=%d bytes", file_key, len(file_bytes))
    return file_key


def _try_store_file_in_r2(job_id: str, raw_path: str, filename: str) -> Optional[str]:
    """
    Try to store an upload in Cloudflare R2.
    Returns an R2 object key if successful, otherwise None.

    NOTE: storage.upload_video is used as a generic binary upload helper in
    the current VeriFYD codebase. The worker downloads the object by key and
    then deletes it after analysis.
    """
    try:
        from storage import r2_available, upload_video

        if not r2_available():
            return None
        r2_key = upload_video(job_id, raw_path, filename)
        log.info("Stored file in R2: key=%s", r2_key)
        return r2_key
    except Exception as exc:
        log.warning("R2 store failed; falling back to Redis: %s", exc)
        return None


def _enqueue_file_job(
    *,
    job_id: str,
    raw_path: str,
    filename: str,
    email: str,
    worker_func,
    job_timeout: int = 900,
    queue_name: Optional[str] = None,
    suppress_email: bool = False,
):
    """
    Common upload helper:
    1. Prefer R2 for file storage.
    2. Fall back to Redis if R2 is unavailable.
    3. Enqueue the requested worker function with a file reference.
    4. Remove the local upload temp file after it is stored.
    """
    r = _get_redis(decode_responses=False)
    selected_queue = queue_name or QUEUE_NAME
    q = _get_queue(r, selected_queue)

    log.info(
        "Enqueue file job: job_id=%s filename=%s worker_func=%s queue=%s",
        job_id,
        filename,
        getattr(worker_func, "__name__", str(worker_func)),
        selected_queue,
    )

    r2_key = _try_store_file_in_r2(job_id, raw_path, filename)
    if r2_key:
        _safe_remove(raw_path)
        return q.enqueue(
            worker_func,
            f"r2:{r2_key}",
            filename,
            email,
            suppress_email,
            job_id=job_id,
            job_timeout=job_timeout,
            result_ttl=RESULT_TTL_SECONDS,
        )

    file_key = _store_file_in_redis(r, job_id, raw_path)
    _safe_remove(raw_path)
    return q.enqueue(
        worker_func,
        file_key,
        filename,
        email,
        suppress_email,
        job_id=job_id,
        job_timeout=job_timeout,
        result_ttl=RESULT_TTL_SECONDS,
    )


# ─────────────────────────────────────────────────────────────
#  Video helpers
# ─────────────────────────────────────────────────────────────
def enqueue_upload(job_id: str, raw_path: str, filename: str, email: str, suppress_email: bool = False):
    """Store uploaded video and enqueue video analysis."""
    from worker import process_upload_job

    return _enqueue_file_job(
        job_id=job_id,
        raw_path=raw_path,
        filename=filename,
        email=email,
        worker_func=process_upload_job,
        job_timeout=1200,
        suppress_email=suppress_email,
    )


def enqueue_link(job_id: str, video_url: str, email: str, double_count: bool = False):
    """Enqueue URL/video-link analysis."""
    from worker import process_link_job

    r = _get_redis(decode_responses=False)
    q = _get_queue(r, QUEUE_NAME)
    log.info("Enqueue link job: job_id=%s queue=%s url=%s", job_id, QUEUE_NAME, video_url)
    return q.enqueue(
        process_link_job,
        job_id,
        video_url,
        email,
        double_count,
        job_id=job_id,
        job_timeout=1200,
        result_ttl=RESULT_TTL_SECONDS,
    )


# ─────────────────────────────────────────────────────────────
#  Photo helpers
# ─────────────────────────────────────────────────────────────
def enqueue_photo_upload(job_id: str, raw_path: str, filename: str, email: str, suppress_email: bool = False):
    """Store uploaded photo and enqueue photo analysis."""
    from worker import process_photo_upload_job

    return _enqueue_file_job(
        job_id=job_id,
        raw_path=raw_path,
        filename=filename,
        email=email,
        worker_func=process_photo_upload_job,
        job_timeout=600,
        suppress_email=suppress_email,
    )


def enqueue_photo_link(job_id: str, image_url: str, email: str):
    """Enqueue photo/image URL analysis."""
    from worker import process_photo_link_job

    r = _get_redis(decode_responses=False)
    q = _get_queue(r)
    return q.enqueue(
        process_photo_link_job,
        job_id,
        image_url,
        email,
        job_id=job_id,
        job_timeout=600,
        result_ttl=RESULT_TTL_SECONDS,
    )


# ─────────────────────────────────────────────────────────────
#  Audio helpers
# ─────────────────────────────────────────────────────────────
def enqueue_audio_upload(job_id: str, raw_path: str, filename: str, email: str, suppress_email: bool = False):
    """Store uploaded standalone audio and enqueue audio analysis."""
    from worker import process_audio_upload_job

    return _enqueue_file_job(
        job_id=job_id,
        raw_path=raw_path,
        filename=filename,
        email=email,
        worker_func=process_audio_upload_job,
        job_timeout=600,
        suppress_email=suppress_email,
    )


# ─────────────────────────────────────────────────────────────
#  Document helpers — VeriFYD Docs MVP
# ─────────────────────────────────────────────────────────────
def enqueue_document_upload(job_id: str, raw_path: str, filename: str, email: str, suppress_email: bool = False):
    """
    Store uploaded document and enqueue document authentication.

    Supported by the MVP route:
      /upload-document/

    Worker function expected:
      worker.process_document_upload_job(file_key, filename, email)
    """
    from worker import process_document_upload_job

    return _enqueue_file_job(
        job_id=job_id,
        raw_path=raw_path,
        filename=filename,
        email=email,
        worker_func=process_document_upload_job,
        job_timeout=300,
        queue_name=DOCUMENT_QUEUE_NAME,
        suppress_email=suppress_email,
    )




# ─────────────────────────────────────────────────────────────
#  Trust Desk helpers — ZIP intake skeleton
# ─────────────────────────────────────────────────────────────
def enqueue_trust_desk_zip(
    job_id: str,
    raw_path: str,
    filename: str,
    email: str,
    organization: str = "",
    submitter_name: str = "",
    case_number: str = "",
    notes: str = "",
):
    """Store a Trust Desk ZIP and enqueue ZIP intake/inventory processing."""
    from worker import process_trust_desk_zip_job

    r = _get_redis(decode_responses=False)
    # IMPORTANT: keep the Trust Desk parent/finalizer job off the document queue.
    # The parent waits for document child jobs to finish; if it runs on the same
    # verifyd-documents queue, it can occupy the only document worker and block
    # its own PDF/YAML child jobs. The parent itself does not need LibreOffice,
    # so run it on the general verifyd queue and let document children run on
    # verifyd-documents.
    q = _get_queue(r, QUEUE_NAME)

    log.info(
        "Enqueue Trust Desk ZIP job: job_id=%s filename=%s queue=%s organization=%s case=%s",
        job_id, filename, QUEUE_NAME, organization, case_number
    )

    r2_key = _try_store_file_in_r2(job_id, raw_path, filename)
    if r2_key:
        _safe_remove(raw_path)
        file_ref = f"r2:{r2_key}"
    else:
        file_ref = _store_file_in_redis(r, job_id, raw_path)
        _safe_remove(raw_path)

    return q.enqueue(
        process_trust_desk_zip_job,
        file_ref,
        filename,
        email,
        organization or "",
        submitter_name or "",
        case_number or "",
        notes or "",
        job_id=job_id,
        job_timeout=3600,
        result_ttl=RESULT_TTL_SECONDS,
    )


# ─────────────────────────────────────────────────────────────
#  Result helper
# ─────────────────────────────────────────────────────────────
def get_job_result(job_id: str) -> Dict[str, Any]:
    """
    Return a stored job result from Redis.

    Worker stores results at:
      result:{job_id}

    Returns {"job_status": "not_found"} when no result exists.
    """
    if not job_id:
        return {"job_status": "not_found"}

    try:
        r = _get_redis(decode_responses=False)
        data = r.get(f"result:{job_id}")
        if not data:
            return {"job_status": "not_found"}
        if isinstance(data, bytes):
            data = data.decode("utf-8", errors="replace")
        result = json.loads(data)
        if isinstance(result, dict):
            return result
        return {"job_status": "error", "error": "Invalid job result format."}
    except Exception as exc:
        log.warning("get_job_result failed for %s: %s", job_id, exc)
        return {"job_status": "error", "error": "Could not retrieve job result."}

# ─────────────────────────────────────────────────────────────
#  Phase 3B-1 — Certified Web & Social Capture
# ─────────────────────────────────────────────────────────────
def enqueue_web_capture(job_id: str, url: str, email: str):
    """Enqueue a certified public URL screenshot/evidence capture job."""
    from worker import process_web_capture_job

    r = _get_redis(decode_responses=False)
    q = _get_queue(r, QUEUE_NAME)
    log.info("Enqueue web capture job: job_id=%s queue=%s url=%s", job_id, QUEUE_NAME, (url or '')[:120])
    return q.enqueue(
        process_web_capture_job,
        job_id,
        url,
        email,
        job_id=job_id,
        job_timeout=900,
        result_ttl=RESULT_TTL_SECONDS,
    )
