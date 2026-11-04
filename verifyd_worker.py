# VERIFYD_DINOV2_PERSISTENT_WORKER_V1
"""RQ launcher that preloads DINOv2 once before RQ forks job processes.

The parent keeps the model resident. Linux forked job processes inherit the
loaded weights through copy-on-write memory, preserving normal RQ job
isolation/timeouts without downloading and constructing DINOv2 for each job.
"""

import logging
import os
import sys
import time


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


# VERIFYD_DEEPFAKE_PERSISTENT_WORKER_WARMUP_V1
def _preload_deepfake() -> bool:
    """Load the existing deepfake detector cache in the long-lived RQ parent."""
    if os.getenv("VERIFYD_PRELOAD_DEEPFAKE", "1").strip().lower() in {
        "0", "false", "no", "off"
    }:
        log.info("Deepfake parent preload disabled by VERIFYD_PRELOAD_DEEPFAKE")
        return False

    started = time.perf_counter()
    try:
        import deepfake_detector

        loaded = bool(deepfake_detector._load_model())
        elapsed = time.perf_counter() - started
        if loaded:
            log.info(
                "Deepfake model preloaded once in RQ parent pid=%s duration=%.2fs",
                os.getpid(), elapsed,
            )
        else:
            log.warning(
                "Deepfake preload unavailable after %.2fs; jobs will continue normally",
                elapsed,
            )
        return loaded
    except Exception:
        # Warmup must never prevent the queue worker from starting. The detector
        # retains its existing graceful-degradation behavior inside each job.
        log.exception(
            "Deepfake parent preload failed after %.2fs; starting worker normally",
            time.perf_counter() - started,
        )
        return False


def _resolve_voicemail_audio_tools() -> None:
    """Resolve ffmpeg bundled by build.sh even when runtime PATH omits it.

    Only sets paths if an operator hasn't explicitly configured FFMPEG_BIN or
    FFPROBE_BIN and the respective executable isn't otherwise on PATH.
    """
    import shutil

    directory = "/opt/render/project/.render/ffmpeg"
    for env_key, binary_name in (
        ("FFMPEG_BIN", "ffmpeg"),
        ("FFPROBE_BIN", "ffprobe"),
    ):
        if os.environ.get(env_key) or shutil.which(binary_name):
            continue
        binary = os.path.join(directory, binary_name)
        if os.path.isfile(binary) and os.access(binary, os.X_OK):
            os.environ[env_key] = binary


def _log_voicemail_readiness(queue_names: list[str]) -> None:
    """Report capability flags only — never log API credentials or voicemail data.

    The service can be healthy while its separate RQ worker lacks ffmpeg,
    speech-to-text credentials, or the queue the API uses.
    """
    import importlib.util
    import shutil

    required_queue = os.getenv("VERIFYD_TRUST_VOICE_ANALYSIS_QUEUE", "verifyd").strip() or "verifyd"
    checks = {
        "enabled_on_worker": os.getenv("VERIFYD_VOICEMAIL_INTELLIGENCE_ENABLED", "0").strip().lower() in ("1", "true", "yes", "on"),
        "transcription_key_present": bool(os.getenv("OPENAI_API_KEY")),
        "ffmpeg_available": bool(shutil.which(os.getenv("FFMPEG_BIN", "ffmpeg"))),
        "openai_library_present": importlib.util.find_spec("openai") is not None,
        "audio_detector_dependencies_present": all(
            importlib.util.find_spec(name) is not None
            for name in ("numpy", "librosa", "soundfile", "boto3")
        ),
        "analysis_queue_consumed": required_queue in queue_names,
    }
    log.info("Voicemail Intelligence worker readiness (no user data): %s", checks)


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

    _resolve_voicemail_audio_tools()

    _preload_dinov2()
    _preload_deepfake()

    try:
        _log_voicemail_readiness(queue_names)
    except Exception:
        log.exception("Optional voicemail worker readiness probe failed; continuing RQ worker")

    connection = Redis.from_url(redis_url)
    queues = [Queue(name, connection=connection) for name in queue_names]
    log.info("Starting RQ worker for queues=%s pid=%s", queue_names, os.getpid())
    worker = Worker(queues, connection=connection)
    worker.work(with_scheduler=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
