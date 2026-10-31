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
        if b"IDLE" not in connection.capabilities:
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


def main() -> int:
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper())
    if not _on("VERIFYD_CIRCLE_EMAIL_IDLE_ENABLED") or not configured():
        log.error("Hostinger IDLE listener not enabled/configured; refusing to start")
        return 2
    if sys.version_info < (3, 14):
        log.error("Hostinger IDLE requires Python 3.14+")
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
