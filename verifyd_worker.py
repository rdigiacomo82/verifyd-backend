# VERIFYD_DINOV2_PERSISTENT_WORKER_V1
"""RQ launcher that preloads DINOv2 once before RQ forks job processes.

The parent keeps the model resident. Linux forked job processes inherit the
loaded weights through copy-on-write memory, preserving normal RQ job
isolation/timeouts without downloading and constructing DINOv2 for each job.
"""

import logging
import os
import sys


logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper())
log = logging.getLogger("verifyd.worker_launcher")


def _preload_dinov2() -> bool:
    """Load the existing detector cache in the long-lived RQ parent."""
    if os.getenv("VERIFYD_PRELOAD_DINOV2", "1").strip().lower() in {
        "0", "false", "no", "off"
    }:
        log.info("DINOv2 parent preload disabled by VERIFYD_PRELOAD_DINOV2")
        return False

    try:
        import dinov2_detector

        loaded = bool(dinov2_detector._load_model())
        if loaded:
            log.info("DINOv2 preloaded once in RQ parent pid=%s", os.getpid())
        else:
            log.warning("DINOv2 preload unavailable; jobs will continue without it")
        return loaded
    except Exception:
        # Preserve the detector's existing graceful-degradation behavior. A model
        # or dependency problem must not prevent the queue worker from starting.
        log.exception("DINOv2 parent preload failed; starting worker normally")
        return False


def main() -> int:
    from redis import Redis
    from rq import Queue, Worker

    redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379")
    queue_names = [
        name.strip()
        for name in os.environ.get("VERIFYD_RQ_QUEUES", "verifyd").split(",")
        if name.strip()
    ]
    if not queue_names:
        queue_names = ["verifyd"]

    _preload_dinov2()

    connection = Redis.from_url(redis_url)
    queues = [Queue(name, connection=connection) for name in queue_names]
    log.info("Starting RQ worker for queues=%s pid=%s", queue_names, os.getpid())
    worker = Worker(queues, connection=connection)
    worker.work(with_scheduler=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
