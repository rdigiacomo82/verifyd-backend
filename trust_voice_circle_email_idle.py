"""Optional low-latency Hostinger IMAP IDLE watcher for VeriFYD Circle.

A separate *persistent* worker process is required. This script is deliberately
inert unless VERIFYD_CIRCLE_EMAIL_IDLE_ENABLED=true and the already-secured
Hostinger gateway is configured. Keep the five-minute cron enabled as fallback.

The watcher never changes IMAP messages. Its only effect is to invoke the
existing verified-recipient importer (which uses the same advisory lock, UID
checkpoint, idempotent provider IDs, and rate limits as the cron).
Python >=3.14 is required for imaplib.IMAP4.idle().
"""
from __future__ import annotations

import imaplib
import logging
import os
import signal
import ssl
import sys
import threading
import time

from trust_voice_circle_email_apex_gateway import (
    _IMAP_HOST, _on, configured, ingest_batch,
)

log = logging.getLogger("verifyd.circle_email.idle")
_STOP = threading.Event()
_IDLE_SECONDS = 180
_RETRY_SECONDS = 15
_MAX_RETRY_SECONDS = 90


def _stop(*_unused):
    _STOP.set()


def _ingest() -> None:
    result = ingest_batch()
    status = result["status"]
    if status not in ("ok", "busy"):
        raise RuntimeError("gateway_not_ready")
    log.info("Hostinger IDLE ingest status=%s imported=%d quarantined=%d skipped=%d",
             status, result.get("processed", 0), result.get("quarantined", 0),
             result.get("skipped", 0))


def _watch_connection() -> None:
    # The separate listener connection MUST NOT alter message read flags,
    # delete mail, rewind the saved UID cursor or forward raw message headers.
    connection = imaplib.IMAP4_SSL(
        _IMAP_HOST, 993, ssl_context=ssl.create_default_context(), timeout=30)
    try:
        connection.login(os.environ["VERIFYD_CIRCLE_EMAIL_HOSTINGER_IMAP_USER"].strip().lower(),
                         os.environ["VERIFYD_CIRCLE_EMAIL_HOSTINGER_IMAP_PASSWORD"])
        status, _ = connection.select("INBOX", readonly=True)
        if status != "OK":
            raise RuntimeError("imap_select_failed")
        if not (b"IDLE" in connection.capabilities or "IDLE" in connection.capabilities):
            raise RuntimeError("hostinger_idle_not_supported")
        if not callable(getattr(connection, "idle", None)):
            raise RuntimeError("python_imap_idle_unavailable")

        # Catch up before entering IDLE; no mail can be lost across restarts.
        _ingest()
        while not _STOP.is_set():
            # Exit IDLE regularly and rescan to cover a notification lost
            # between the independent importer and listener connections.
            notified = False
            with connection.idle(duration=_IDLE_SECONDS) as idler:
                for kind, _ in idler:
                    if kind in ("EXISTS", "RECENT"):
                        notified = True
                        break
                    if _STOP.is_set():
                        break
            if _STOP.is_set():
                break
            _ingest()
            if notified:
                log.info("Hostinger IDLE wakeup processed")
    finally:
        try:
            connection.logout()
        except (OSError, imaplib.IMAP4.error):
            pass


def _next_cron_cutoff_utc(now: float, *, reserve_seconds: int = 40,
                          max_seconds: int = 235) -> float:
    """UTC five-minute slot with a safety margin for Render cron shutdown.

    Avoids stretching a scheduled run into the next cron slot. The ordinary
    five-minute batch still runs even if this optional listening fails.
    """
    next_slot = (int(now // 300) + 1) * 300
    return min(now + max_seconds, next_slot - reserve_seconds)


def watch_current_cron_window() -> None:
    """Bounded IMAP IDLE on the EXISTING five-minute cron, without a new service.

    Never attempts to run indefinitely or modify messages. While this
    optional listener is waiting, Render charges for active cron runtime,
    so it is guarded by an opt-in env flag and a hard five-minute-slot
    cutoff. On any failure the cron still exits normally and retries on
    its next scheduled run.
    """
    if not _on("VERIFYD_CIRCLE_EMAIL_APEX_FAST_WINDOW_ENABLED"):
        return
    if sys.version_info < (3, 14):
        log.warning("Hostinger cron fast listener requires Python 3.14+")
        return
    expected = os.getenv("VERIFYD_CIRCLE_EMAIL_IDLE_EXPECTED_USER",
                         "circleinbound@vfvid.com").strip().lower()
    actual = os.getenv("VERIFYD_CIRCLE_EMAIL_HOSTINGER_IMAP_USER",
                       "").strip().lower()
    if not expected or actual != expected:
        log.warning("Hostinger cron fast listener mailbox mismatch; disabled")
        return
    seconds = _next_cron_cutoff_utc(time.time()) - time.time()
    if seconds < 8:
        log.info("Hostinger cron fast listener skipped; too near next run")
        return
    deadline = time.monotonic() + seconds
    connection = None
    try:
        connection = imaplib.IMAP4_SSL(
            _IMAP_HOST, 993, ssl_context=ssl.create_default_context(), timeout=30)
        connection.login(actual,
                         os.environ["VERIFYD_CIRCLE_EMAIL_HOSTINGER_IMAP_PASSWORD"])
        status, _ = connection.select("INBOX", readonly=True)
        if status != "OK" or not (b"IDLE" in connection.capabilities or "IDLE" in connection.capabilities):
            raise RuntimeError("cron_idle_not_available")
        if not callable(getattr(connection, "idle", None)):
            raise RuntimeError("cron_idle_python_unavailable")
        log.info("Hostinger cron fast listener active")
        while not _STOP.is_set():
            remaining = deadline - time.monotonic()
            if remaining < 2:
                break
            notified = False
            # Recheck every ~30 seconds even if a provider notification is
            # lost. Only a short amount of time is billed while idling.
            with connection.idle(duration=min(30, remaining)) as idler:
                for kind, _ in idler:
                    if kind in ("EXISTS", "RECENT", b"EXISTS", b"RECENT"):
                        notified = True
                        break
            if _STOP.is_set() or deadline - time.monotonic() < 2:
                break
            _ingest()
            if notified:
                log.info("Hostinger cron fast listener received new-mail signal")
        log.info("Hostinger cron fast listener complete")
    except (OSError, imaplib.IMAP4.error, RuntimeError, ValueError, KeyError) as exc:
        # Never log raw IMAP details, sender data or credentials.
        log.warning("Hostinger cron fast listener unavailable reason=%s",
                    type(exc).__name__)
    finally:
        if connection is not None:
            try:
                connection.logout()
            except (OSError, imaplib.IMAP4.error):
                pass


def main() -> int:
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper())
    if not _on("VERIFYD_CIRCLE_EMAIL_IDLE_ENABLED") or not configured():
        log.error("Hostinger IDLE listener not enabled/configured; refusing to start")
        return 2
    if sys.version_info < (3, 14):
        log.error("Hostinger IDLE requires Python 3.14+")
        return 2
    expected = os.getenv("VERIFYD_CIRCLE_EMAIL_IDLE_EXPECTED_USER", "").strip().lower()
    actual = os.getenv("VERIFYD_CIRCLE_EMAIL_HOSTINGER_IMAP_USER", "").strip().lower()
    if not expected or actual != expected:
        # A dedicated worker must NEVER silently attach to the unrelated
        # operational/support inbox if the wrong Cron env is referenced.
        log.error("Hostinger IDLE expected mailbox mismatch; refusing to start")
        return 2
    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    delay = _RETRY_SECONDS
    while not _STOP.is_set():
        try:
            _watch_connection()
            delay = _RETRY_SECONDS
        except (OSError, imaplib.IMAP4.error, RuntimeError, ValueError) as exc:
            # Don't log the exception message: it could contain credentials,
            # a mailbox name, or a sender address.
            log.warning("Hostinger IDLE reconnect reason=%s", type(exc).__name__)
            if _STOP.wait(delay):
                break
            delay = min(_MAX_RETRY_SECONDS, delay * 2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
