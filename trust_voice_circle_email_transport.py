"""Private VeriFYD Circle mail attachment transport (external and internal).

Never exposes R2 URLs, trusts a file name without byte validation, or proxies a
provider-supplied URL without constraining its HTTPS host and exact path.
External mail attachments are NOT malware-scanned. Require explicit user
acknowledgement before downloading them; do not inline or preview.
"""
from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import re
import tempfile
import urllib.error
import urllib.parse
import urllib.request
import uuid

from fastapi import HTTPException

from trust_voice_attachment_validation import validate_attachment_content, AttachmentContentError
from trust_voice_attachments import _get_attachment_client, TRUST_VOICE_BUCKET

log = logging.getLogger("verifyd.circle_email.attachments")
MAX_FILE_BYTES = 8 * 1024 * 1024
MAX_TOTAL_BYTES = 16 * 1024 * 1024
MAX_FILES = 3
ALLOWED_EXTENSIONS = frozenset(
    {".pdf", ".txt", ".docx", ".xlsx", ".png", ".jpg", ".jpeg", ".webp", ".csv", ".pptx", ".mp3", ".wav"}
)
_RESEND_ID = re.compile(r"^[A-Za-z0-9-]{15,90}$")
# Resend's documented receiving attachment CDN and fixed per-message path.
# Never expand this allowlist to arbitrary subdomains of third-party CDNs.
_CDN_HOST = "inbound-cdn.resend.com"


def _safe_name(name: str) -> str:
    # Prevent path traversal, header injection, ambiguous control chars.
    value = str(name or "").replace("\\", "/").split("/")[-1].strip()
    value = re.sub(r"[^A-Za-z0-9_. -]", "_", value).strip(" .")
    return value[:140] or "attachment"


def _valid_bytes(data: bytes, name: str, claimed_type: str) -> str:
    extension = os.path.splitext(name)[1].lower()
    if extension not in ALLOWED_EXTENSIONS or not data or len(data) > MAX_FILE_BYTES:
        raise ValueError("unsupported_file")
    with tempfile.NamedTemporaryFile(suffix=extension) as tmp:
        tmp.write(data)
        tmp.flush()
        try:
            meta = validate_attachment_content(
                tmp.name, extension=extension, supplied_content_type=claimed_type or ""
            )
        except AttachmentContentError as exc:
            raise ValueError("file_content_mismatch") from exc
    if not isinstance(meta, dict):
        raise ValueError("validation_unavailable")
    return str(meta.get("content_type") or "application/octet-stream")


def _load_r2_bytes(client, record: dict) -> bytes:
    size = int(record["byte_size"])
    if size < 1 or size > MAX_FILE_BYTES:
        raise ValueError("file_size_limit")
    obj = client.get_object(Bucket=TRUST_VOICE_BUCKET, Key=record["storage_key"])
    try:
        raw = obj["Body"].read(MAX_FILE_BYTES + 1)
    finally:
        obj["Body"].close()
    if len(raw) != size or hashlib.sha256(raw).hexdigest() != record["sha256"]:
        raise ValueError("attachment_integrity_failure")
    _valid_bytes(raw, str(record["filename"]), str(record.get("content_type") or ""))
    return raw


def prepare_outbound_attachments(rows: list[dict]) -> list[dict]:
    """Convert authenticated, reserved R2 files to Resend Base64 attachment objects."""
    if not rows:
        return []
    if len(rows) > MAX_FILES or sum(int(r["byte_size"]) for r in rows) > MAX_TOTAL_BYTES:
        raise HTTPException(413, detail={"error": "attachment_total_limit"})
    client = _get_attachment_client()
    result = []
    try:
        for record in rows:
            raw = _load_r2_bytes(client, record)
            result.append({
                "filename": _safe_name(record["filename"]),
                "content": base64.b64encode(raw).decode("ascii"),
            })
    except Exception as exc:
        # Do not disclose storage details, names, content or credentials.
        log.warning("Circle mail outbound attachment failed: %s", type(exc).__name__)
        raise HTTPException(503, detail={"error": "attachment_unavailable"}) from None
    return result


def _received_list(provider_id: str) -> list[dict]:
    """List attachment metadata using a full-access receiving key, not webhook JSON."""
    from trust_voice_circle_email_inbound import _receiving_api_key
    if not _RESEND_ID.fullmatch(provider_id):
        raise HTTPException(400, detail={"error": "invalid_provider_id"})
    key = _receiving_api_key()
    if not key:
        raise HTTPException(503, detail={"error": "inbound_not_configured"})
    req = urllib.request.Request(
        "https://api.resend.com/emails/receiving/" + provider_id + "/attachments",
        headers={"Authorization": "Bearer " + key, "Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=12) as response:
            raw = response.read(128_001)
        if len(raw) > 128_000:
            raise ValueError("oversized_metadata")
        envelope = json.loads(raw)
        if not isinstance(envelope, dict) or not isinstance(envelope.get("data"), list):
            raise ValueError("bad_metadata")
        if envelope.get("has_more"):
            # A fourth-plus attachment is intentionally omitted under our
            # three-file limit; never retry the webhook forever for this.
            envelope["data"].append({"id": "", "filename": "", "size": 0})
        return envelope["data"]
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError,
            ValueError, TypeError):
        log.warning("Circle mail inbound attachment metadata could not be fetched")
        raise HTTPException(503, detail={"error": "inbound_attachment_retry"}) from None


def _verify_cdn_url(url: str, provider_id: str, attachment_id: str) -> None:
    if not isinstance(url, str) or len(url) > 4096:
        raise ValueError("unexpected_attachment_url")
    parsed = urllib.parse.urlsplit(url)
    expected_path = "/" + provider_id + "/attachments/" + attachment_id
    # Resend may use a signed CDN distribution whose path is opaque. Authorize
    # that only when an operator has confirmed the *exact* hostname from a real
    # Resend Receiving API response and set it in Render. Never accept wildcard
    # hostnames or URLs from the browser.
    additional = os.getenv("VERIFYD_CIRCLE_EMAIL_RECEIVED_ATTACHMENT_CDN_HOST", "").strip().lower()
    approved_additional = (bool(re.fullmatch(r"[a-z0-9](?:[a-z0-9.-]{1,240})[a-z0-9]",additional))
                           and "." in additional
                           and not additional.endswith((".local",".localhost",".internal"))
                           and not re.fullmatch(r"(?:\\d+\\.){3}\\d+",additional))
    host=parsed.hostname
    host_ok = (host == _CDN_HOST or (approved_additional and host == additional))
    path_ok = (parsed.path == expected_path if host == _CDN_HOST else
               bool(parsed.path.startswith("/") and parsed.path != "/" and
                    not any(piece in (".","..") for piece in parsed.path.split("/"))))
    if (parsed.scheme != "https" or not host_ok or not path_ok or
        parsed.port not in (None, 443) or
        parsed.username is not None or parsed.password is not None or
        parsed.fragment or not parsed.query):
        raise ValueError("unexpected_attachment_url")


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _download_received_file(url: str, provider_id: str, attachment_id: str,
                            claimed_size: int) -> bytes:
    try:
        _verify_cdn_url(url, provider_id, attachment_id)
        req = urllib.request.Request(url, headers={"User-Agent": "VeriFYD-Mail-Receive/1"})
        opener = urllib.request.build_opener(_NoRedirect())
        with opener.open(req, timeout=12) as response:
            raw = response.read(MAX_FILE_BYTES + 1)
        if not raw or len(raw) > MAX_FILE_BYTES or len(raw) != claimed_size:
            raise ValueError("invalid_attachment_size")
        return raw
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError,
            OSError, ValueError):
        raise HTTPException(503, detail={"error": "inbound_attachment_retry"}) from None


def fetch_inbound_attachment_files(provider_id: str, advertised: list) -> tuple[list[dict], int]:
    """Return validated files plus count omitted due to policy/unsupported formats.

    Network/provider failures fail the webhook so Resend can retry; intentionally
    unsupported formats or oversized files are marked omitted, never silently
    offered to the user as safe downloads.
    """
    if not advertised:
        return [], 0
    records = _received_list(provider_id)
    expected_ids = {
        str(a.get("id")) for a in advertised if isinstance(a, dict) and a.get("id")
    }
    if expected_ids and not expected_ids.issubset({
        str(a.get("id")) for a in records if isinstance(a, dict)
    }) and len(records) <= MAX_FILES:
        # A paginated response can omit excess attachments by design.
        raise HTTPException(503, detail={"error": "inbound_attachment_retry"})
    files, omitted, used = [], 0, 0
    for index, item in enumerate(records):
        if not isinstance(item, dict):
            omitted += 1
            continue
        name = _safe_name(item.get("filename"))
        try:
            size = int(item.get("size", 0))
        except (TypeError, ValueError):
            omitted += 1
            continue
        ext = os.path.splitext(name)[1].lower()
        aid = str(item.get("id") or "")
        if (index >= MAX_FILES or ext not in ALLOWED_EXTENSIONS or
            size < 1 or size > MAX_FILE_BYTES or size + used > MAX_TOTAL_BYTES or
            not _RESEND_ID.fullmatch(aid)):
            omitted += 1
            continue
        raw = _download_received_file(str(item.get("download_url") or ""),
                                      provider_id, aid, size)
        try:
            ctype = _valid_bytes(raw, name, str(item.get("content_type") or ""))
        except ValueError:
            omitted += 1
            continue
        files.append({"filename": name, "bytes": raw, "content_type": ctype})
        used += len(raw)
    return files, omitted


def persist_inbound_attachments(db, *, files: list[dict], owner_id: str,
                                mail_id: str, stamp: str) -> list[str]:
    """Save validated external bytes privately and associate them with one inbox mail.

    If R2 or DB upload fails, caller must roll back DB transaction. Partial
    storage objects are cleaned here before the exception reaches webhook.
    """
    if not files:
        return []
    client = _get_attachment_client()
    created = []
    try:
        for file in files:
            name = _safe_name(file["filename"])
            extension = os.path.splitext(name)[1].lower()
            fid = "tvcfile_" + uuid.uuid4().hex
            storage_key = f"circle-email/{owner_id}/incoming/{fid}{extension}"
            raw = file["bytes"]
            client.put_object(
                Bucket=TRUST_VOICE_BUCKET, Key=storage_key, Body=raw,
                ContentType="application/octet-stream",
                ContentDisposition="attachment",
            )
            created.append(storage_key)
            db.cursor().execute(
                """INSERT INTO verifyd_circle_email_files
                (file_id,owner_id,mail_id,filename,content_type,storage_key,
                 byte_size,sha256,created_at,source,scan_status)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,'external','not_scanned')""",
                (fid,owner_id,mail_id,name,file["content_type"],storage_key,
                 len(raw),hashlib.sha256(raw).hexdigest(),stamp),
            )
    except Exception:
        for key in created:
            try:
                client.delete_object(Bucket=TRUST_VOICE_BUCKET, Key=key)
            except Exception:
                log.warning("Circle mail orphaned inbound R2 object cleanup failed")
        raise
    return created
