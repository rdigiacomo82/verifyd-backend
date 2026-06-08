"""
VeriFYD certification email outbox/diagnostic compatibility patch.
Run from repo root:
    py verifyd_email_outbox_compat_patch.py

This patch avoids brittle worker block rewrites. It routes existing
send_certification_email(...) calls through a safe compatibility wrapper that:
- confirms the certified artifact exists when possible
- records Redis email/outbox status
- captures Resend response IDs
- provides manual admin resend/status endpoints
"""
from __future__ import annotations
import pathlib, re, textwrap

ROOT = pathlib.Path.cwd()

NOTIFICATION_HELPER = r'''# ============================================================
# VeriFYD — notification_helper.py
# Durable certification-email helper / outbox diagnostics
# ============================================================
from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

log = logging.getLogger("verifyd.notification")

EMAIL_STATUS_TTL = int(os.environ.get("VERIFYD_CERT_EMAIL_STATUS_TTL", str(60 * 60 * 24 * 30)))


def _redis_conn(redis_conn=None):
    if redis_conn is not None:
        return redis_conn
    import redis
    return redis.from_url(os.environ.get("REDIS_URL", "redis://localhost:6379"), decode_responses=False)


def _status_key(cert_id: str, field: str) -> str:
    return f"cert_email:{cert_id}:{field}"


def _rget_text(r, key: str) -> str:
    try:
        val = r.get(key)
        if val is None:
            return ""
        if isinstance(val, bytes):
            return val.decode("utf-8", errors="replace")
        return str(val)
    except Exception:
        return ""


def _rset_text(r, key: str, value: Any, ttl: int = EMAIL_STATUS_TTL) -> None:
    try:
        r.setex(key, ttl, str(value if value is not None else ""))
    except Exception as e:
        log.warning("cert-email: Redis status write failed key=%s error=%s", key, e)


def _get_attempts(r, cert_id: str) -> int:
    try:
        return int(_rget_text(r, _status_key(cert_id, "attempts")) or "0")
    except Exception:
        return 0


def _set_email_status(
    r,
    cert_id: str,
    *,
    status: str,
    attempts: int | None = None,
    error: str = "",
    email: str = "",
    media_type: str = "",
    download_url: str = "",
    artifact_confirmed: bool | None = None,
    resend_id: str = "",
    resend_response: Any = None,
) -> None:
    _rset_text(r, _status_key(cert_id, "status"), status)
    if attempts is not None:
        _rset_text(r, _status_key(cert_id, "attempts"), attempts)
    if error:
        _rset_text(r, _status_key(cert_id, "last_error"), error[:1000])
    elif status == "sent":
        _rset_text(r, _status_key(cert_id, "last_error"), "")
    now = int(time.time())
    if status == "sent":
        _rset_text(r, _status_key(cert_id, "sent_at"), now)
        _rset_text(r, _status_key(cert_id, "accepted_at"), now)
    else:
        _rset_text(r, _status_key(cert_id, "updated_at"), now)
    if media_type:
        _rset_text(r, _status_key(cert_id, "media_type"), media_type)
    if email:
        _rset_text(r, _status_key(cert_id, "email"), email)
    if download_url:
        _rset_text(r, _status_key(cert_id, "download_url"), download_url)
    if artifact_confirmed is not None:
        _rset_text(r, _status_key(cert_id, "artifact_confirmed"), str(bool(artifact_confirmed)).lower())
    if resend_id:
        _rset_text(r, _status_key(cert_id, "resend_id"), resend_id)
    if resend_response is not None:
        try:
            _rset_text(r, _status_key(cert_id, "resend_response"), json.dumps(resend_response, default=str)[:3000])
        except Exception:
            _rset_text(r, _status_key(cert_id, "resend_response"), str(resend_response)[:3000])


def get_certification_email_status(cert_id: str, redis_conn=None) -> dict:
    r = _redis_conn(redis_conn)
    fields = (
        "status", "attempts", "last_error", "sent_at", "accepted_at", "updated_at",
        "delivered_at", "bounce_reason", "media_type", "email", "download_url",
        "artifact_confirmed", "resend_id", "resend_response",
    )
    out = {"cert_id": cert_id}
    for f in fields:
        out[f] = _rget_text(r, _status_key(cert_id, f))
    return out


def _redis_artifact_exists(r, media_type: str, cert_id: str, download_url: str = "") -> bool:
    try:
        if media_type == "audio":
            return bool(r.exists(f"audiocert:{cert_id}"))
        if media_type in ("video", "photo"):
            return bool(r.exists(f"cert:{cert_id}"))
        if media_type == "document":
            if "/download-certified-file/" in str(download_url):
                return bool(r.exists(f"filecert:{cert_id}"))
            return bool(r.exists(f"doccert:{cert_id}")) or bool(r.exists(f"filecert:{cert_id}"))
    except Exception:
        return False
    return False


def _local_video_artifact_exists(cert_id: str) -> bool:
    """Support sync/local fallback where certified MP4 is on local disk."""
    try:
        from config import CERT_DIR
        p = os.path.join(CERT_DIR, f"{cert_id}.mp4")
        return os.path.exists(p) and os.path.getsize(p) > 1000
    except Exception:
        return False


def artifact_exists_for_cert(cert_id: str, media_type: str, download_url: str = "", redis_conn=None) -> bool:
    """Confirm the certified downloadable artifact exists before sending email."""
    media_type = (media_type or "").lower().strip()
    r = _redis_conn(redis_conn)

    try:
        if media_type == "video":
            from storage import r2_available, certified_exists
            if r2_available() and certified_exists(cert_id):
                return True
            if _local_video_artifact_exists(cert_id):
                return True
        elif media_type == "audio":
            from storage import r2_available, certified_audio_exists
            if r2_available() and certified_audio_exists(cert_id):
                return True
        elif media_type == "photo":
            from storage import r2_available, _get_client, BUCKET
            if r2_available():
                client = _get_client()
                for plan in ("pro", "creator", "enterprise", "free"):
                    for ext in (".jpg", ".jpeg", ".png", ".webp"):
                        try:
                            client.head_object(Bucket=BUCKET, Key=f"certified-photos/{plan}/{cert_id}{ext}")
                            return True
                        except Exception:
                            continue
        elif media_type == "document":
            from storage import r2_available, certified_document_exists, certified_file_package_exists
            if r2_available():
                if "/download-certified-file/" in str(download_url):
                    if certified_file_package_exists(cert_id):
                        return True
                elif certified_document_exists(cert_id) or certified_file_package_exists(cert_id):
                    return True
    except Exception as e:
        log.warning("cert-email: artifact R2/local check failed cert_id=%s media_type=%s error=%s", cert_id, media_type, e)

    return _redis_artifact_exists(r, media_type, cert_id, download_url)


def _media_from_flags(is_photo: bool = False, is_document: bool = False, is_audio: bool = False) -> str:
    if is_audio:
        return "audio"
    if is_document:
        return "document"
    if is_photo:
        return "photo"
    return "video"


def _media_flags(media_type: str) -> dict:
    media_type = (media_type or "video").lower().strip()
    return {
        "is_photo": media_type == "photo",
        "is_document": media_type == "document",
        "is_audio": media_type == "audio",
    }


def send_certification_email_safely(
    *,
    cert_id: str,
    email: str,
    authenticity: int,
    filename: str,
    media_type: str,
    download_url: str,
    redis_conn=None,
    require_artifact: bool = True,
) -> bool:
    """Send certification email after artifact confirmation and record outbox status.

    This is intentionally scheduler-free. If sending fails, status is recorded as
    queued_retry so /admin/resend-certification-email/{cid} can resend manually.
    """
    media_type = (media_type or "video").lower().strip()
    r = _redis_conn(redis_conn)
    cert_id = (cert_id or "").strip()
    email = (email or "").strip()
    download_url = (download_url or "").strip()
    filename = filename or "VeriFYD certified file"
    attempts = _get_attempts(r, cert_id) + 1 if cert_id else 1

    log.info(
        "cert-email: preparing cert_id=%s media_type=%s email=%s download_url=%s attempt=%s",
        cert_id, media_type, email, download_url, attempts,
    )

    if not cert_id or not email or "@" not in email or not download_url:
        err = "missing cert_id, email, or download_url"
        log.warning("cert-email: failed cert_id=%s error=%s", cert_id, err)
        if cert_id:
            _set_email_status(r, cert_id, status="failed", attempts=attempts, error=err,
                              email=email, media_type=media_type, download_url=download_url,
                              artifact_confirmed=False)
        return False

    artifact_confirmed = artifact_exists_for_cert(cert_id, media_type, download_url, r)
    if require_artifact and not artifact_confirmed:
        err = f"certified artifact not confirmed for {media_type}"
        log.warning("cert-email: artifact not confirmed cert_id=%s media_type=%s", cert_id, media_type)
        _set_email_status(r, cert_id, status="queued_retry", attempts=attempts, error=err,
                          email=email, media_type=media_type, download_url=download_url,
                          artifact_confirmed=False)
        return False

    log.info("cert-email: artifact confirmed=%s cert_id=%s media_type=%s", artifact_confirmed, cert_id, media_type)

    try:
        import emailer as _emailer
        flags = _media_flags(media_type)
        log.info("cert-email: sending via Resend cert_id=%s media_type=%s", cert_id, media_type)
        sent = bool(_emailer.send_certification_email(
            email,
            cert_id,
            int(authenticity or 0),
            filename,
            download_url,
            **flags,
        ))
        resend_result = getattr(_emailer, "get_last_resend_result", lambda: {})() or {}
        resend_id = ""
        if isinstance(resend_result, dict):
            resend_id = str(resend_result.get("id") or resend_result.get("email_id") or "")
        log.info("cert-email: sent=%s cert_id=%s media_type=%s resend_id=%s", sent, cert_id, media_type, resend_id)
        if sent:
            _set_email_status(r, cert_id, status="sent", attempts=attempts, error="",
                              email=email, media_type=media_type, download_url=download_url,
                              artifact_confirmed=artifact_confirmed, resend_id=resend_id,
                              resend_response=resend_result)
            return True
        err = "send_certification_email returned False"
    except Exception as e:
        log.exception("cert-email: failed cert_id=%s error=%s", cert_id, e)
        err = str(e)[:1000]
        resend_result = {}

    status = "queued_retry" if attempts < 3 else "failed"
    _set_email_status(r, cert_id, status=status, attempts=attempts, error=err,
                      email=email, media_type=media_type, download_url=download_url,
                      artifact_confirmed=artifact_confirmed, resend_response=resend_result)
    return False


def send_certification_email_outbox_compat(
    to_email: str,
    certificate_id: str,
    authenticity: int,
    original_filename: str,
    download_url: str,
    is_photo: bool = False,
    is_document: bool = False,
    is_audio: bool = False,
) -> bool:
    """Compatibility replacement for emailer.send_certification_email.

    Existing worker/main call sites can keep calling send_certification_email(...).
    The import is rerouted to this function, which preserves the same signature.
    """
    media_type = _media_from_flags(is_photo=is_photo, is_document=is_document, is_audio=is_audio)
    return send_certification_email_safely(
        cert_id=certificate_id,
        email=to_email,
        authenticity=authenticity,
        filename=original_filename,
        media_type=media_type,
        download_url=download_url,
    )


def _infer_media_type(cert_id: str, status: dict | None = None) -> str:
    status = status or {}
    mt = (status.get("media_type") or "").lower().strip()
    if mt:
        return mt
    # Prefer artifact checks that also define the route.
    for candidate in ("audio", "video", "photo", "document"):
        try:
            if artifact_exists_for_cert(cert_id, candidate, ""):
                return candidate
        except Exception:
            continue
    return "video"


def _download_url_for(cert_id: str, media_type: str) -> str:
    from config import BASE_URL
    media_type = (media_type or "video").lower().strip()
    if media_type == "audio":
        return f"{BASE_URL}/download-audio/{cert_id}"
    if media_type == "photo":
        return f"{BASE_URL}/download-photo/{cert_id}"
    if media_type == "document":
        try:
            from storage import certified_file_package_exists, r2_available
            if r2_available() and certified_file_package_exists(cert_id):
                return f"{BASE_URL}/download-certified-file/{cert_id}"
        except Exception:
            pass
        return f"{BASE_URL}/download-document/{cert_id}"
    return f"{BASE_URL}/download/{cert_id}"


def retry_certification_email_job(cert_id: str) -> dict:
    """Manual/admin resend job. Scheduler-free by design."""
    r = _redis_conn()
    cert_id = (cert_id or "").strip()
    if not cert_id:
        return {"sent": False, "status": "failed", "last_error": "missing cert_id"}

    try:
        from database import get_certificate
        cert = get_certificate(cert_id) or {}
    except Exception as e:
        log.exception("cert-email: certificate lookup failed cert_id=%s", cert_id)
        cert = {}
        lookup_error = str(e)[:500]
    else:
        lookup_error = ""

    status = get_certification_email_status(cert_id, r)
    email = (cert.get("email") or status.get("email") or "").strip()
    filename = cert.get("original_file") or status.get("filename") or "VeriFYD certified file"
    try:
        authenticity = int(cert.get("authenticity") or 0)
    except Exception:
        authenticity = 0
    media_type = _infer_media_type(cert_id, status)
    download_url = status.get("download_url") or _download_url_for(cert_id, media_type)

    if not email:
        err = lookup_error or "certificate email not found"
        _set_email_status(r, cert_id, status="failed", attempts=_get_attempts(r, cert_id), error=err,
                          media_type=media_type, download_url=download_url)
        return {"sent": False, "status": "failed", "last_error": err, "cert_id": cert_id,
                "media_type": media_type, "download_url": download_url}

    sent = send_certification_email_safely(
        cert_id=cert_id,
        email=email,
        authenticity=authenticity,
        filename=filename,
        media_type=media_type,
        download_url=download_url,
        redis_conn=r,
    )
    out = get_certification_email_status(cert_id, r)
    return {
        "sent": bool(sent),
        "status": out.get("status"),
        "cert_id": cert_id,
        "email": email,
        "media_type": media_type,
        "download_url": download_url,
        "attempts": out.get("attempts"),
        "last_error": out.get("last_error"),
        "artifact_confirmed": out.get("artifact_confirmed"),
        "resend_id": out.get("resend_id"),
        "resend_response": out.get("resend_response"),
    }
'''


def read(name: str) -> str:
    p = ROOT / name
    if not p.exists():
        raise FileNotFoundError(f"Missing required file: {name}")
    return p.read_text(encoding="utf-8", errors="replace")


def write(name: str, text: str) -> None:
    (ROOT / name).write_text(text, encoding="utf-8", newline="\n")


def patch_emailer() -> None:
    s = read("emailer.py")
    if "LAST_RESEND_RESULT" not in s:
        s = re.sub(r'(BACKEND_URL\s*=\s*"[^"]+"\s*\n)', r'\1\nLAST_RESEND_RESULT = {}\n', s, count=1)
    if "def get_last_resend_result" not in s:
        marker = "\ndef _header_html() -> str:"
        if marker in s:
            s = s.replace(marker, '\n\ndef get_last_resend_result() -> dict:\n    """Return last Resend SDK response for diagnostics/outbox status."""\n    return dict(LAST_RESEND_RESULT or {})\n\n\ndef _header_html() -> str:', 1)
    if "global LAST_RESEND_RESULT" not in s:
        s = s.replace("    try:\n        import resend\n", "    try:\n        global LAST_RESEND_RESULT\n        import resend\n", 1)
    if "LAST_RESEND_RESULT = dict(result or {})" not in s:
        s = s.replace(
            "        result = resend.Emails.send(payload)\n",
            "        result = resend.Emails.send(payload)\n        try:\n            LAST_RESEND_RESULT = dict(result or {})\n        except Exception:\n            LAST_RESEND_RESULT = {\"raw\": str(result)}\n",
            1,
        )
    if "LAST_RESEND_RESULT = {\"error\": str(e)}" not in s:
        s = s.replace(
            "    except Exception as e:\n        log.error(\"Failed to send email: %s\", e)\n        return False\n",
            "    except Exception as e:\n        try:\n            LAST_RESEND_RESULT = {\"error\": str(e)}\n        except Exception:\n            pass\n        log.exception(\"Failed to send email: %s\", e)\n        return False\n",
            1,
        )
    write("emailer.py", s)


def patch_worker() -> None:
    s = read("worker.py")
    s = re.sub(
        r"from\s+emailer\s+import\s+send_certification_email",
        "from notification_helper import send_certification_email_outbox_compat as send_certification_email",
        s,
    )
    write("worker.py", s)


def patch_main() -> None:
    s = read("main.py")
    # Route send_certification_email through compatibility helper while keeping OTP/enterprise from emailer.
    s = re.sub(
        r"from\s+emailer\s+import\s*\(\s*send_otp_email,\s*send_certification_email,\s*send_enterprise_welcome_email\s*\)",
        "from emailer import send_otp_email, send_enterprise_welcome_email\nfrom notification_helper import send_certification_email_outbox_compat as send_certification_email",
        s,
        count=1,
    )
    s = re.sub(
        r"from\s+emailer\s+import\s+send_otp_email\s*,\s*send_certification_email\s*,\s*send_enterprise_welcome_email",
        "from emailer import send_otp_email, send_enterprise_welcome_email\nfrom notification_helper import send_certification_email_outbox_compat as send_certification_email",
        s,
        count=1,
    )
    s = re.sub(
        r"from\s+emailer\s+import\s+send_otp_email\s*,\s*send_enterprise_welcome_email\s*,\s*send_certification_email",
        "from emailer import send_otp_email, send_enterprise_welcome_email\nfrom notification_helper import send_certification_email_outbox_compat as send_certification_email",
        s,
        count=1,
    )
    # If no top import matched but send_certification_email is imported on its own, route it.
    s = re.sub(
        r"from\s+emailer\s+import\s+send_certification_email",
        "from notification_helper import send_certification_email_outbox_compat as send_certification_email",
        s,
    )

    endpoints = r'''

@app.post("/admin/resend-certification-email/{cid}")
def admin_resend_certification_email(cid: str, key: str = ""):
    """Admin: manually resend a certification email with artifact verification."""
    if not _is_admin(key):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    try:
        from notification_helper import retry_certification_email_job, get_certification_email_status
        result = retry_certification_email_job(cid)
        status = get_certification_email_status(cid)
        return JSONResponse({
            "sent": bool(result.get("sent")),
            "cert_id": cid,
            "email": result.get("email") or status.get("email"),
            "media_type": result.get("media_type") or status.get("media_type"),
            "download_url": result.get("download_url") or status.get("download_url"),
            "attempts": result.get("attempts") or status.get("attempts"),
            "last_error": result.get("last_error") or status.get("last_error"),
            "artifact_confirmed": result.get("artifact_confirmed") or status.get("artifact_confirmed"),
            "resend_id": result.get("resend_id") or status.get("resend_id"),
            "resend_response": result.get("resend_response") or status.get("resend_response"),
            "status": result.get("status") or status.get("status"),
        })
    except Exception as e:
        log.exception("admin resend certification email failed for %s", cid)
        return JSONResponse({"sent": False, "cert_id": cid, "error": str(e)[:500]}, status_code=500)


@app.get("/admin/cert-email-status/{cid}")
def admin_cert_email_status(cid: str, key: str = ""):
    """Admin: inspect Redis certification-email outbox status."""
    if not _is_admin(key):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    try:
        from notification_helper import get_certification_email_status
        status = get_certification_email_status(cid)
        cert = None
        try:
            cert = get_certificate(cid)
        except Exception:
            cert = None
        status["certificate_record_exists"] = bool(cert)
        if cert:
            status["certificate_email"] = cert.get("email", "")
            status["certificate_label"] = cert.get("label", "")
            status["certificate_authenticity"] = cert.get("authenticity", "")
            status["certificate_original_file"] = cert.get("original_file", "")
        return JSONResponse(status)
    except Exception as e:
        log.exception("admin cert email status failed for %s", cid)
        return JSONResponse({"cert_id": cid, "error": str(e)[:500]}, status_code=500)
'''
    if "/admin/resend-certification-email/{cid}" not in s:
        markers = ["\n@app.get(\"/upload-limits/\")", "\n@app.get('/upload-limits/')", "\n@app.get(\"/test-email/\")"]
        for marker in markers:
            if marker in s:
                s = s.replace(marker, endpoints + marker, 1)
                break
        else:
            s += endpoints
    write("main.py", s)


def main() -> None:
    write("notification_helper.py", NOTIFICATION_HELPER)
    patch_emailer()
    patch_worker()
    patch_main()
    print("Applied VeriFYD certification email outbox compatibility patch.")
    print("Next run: py -m py_compile notification_helper.py worker.py main.py emailer.py storage.py")


if __name__ == "__main__":
    main()
