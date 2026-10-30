"""Hostinger catch-all gateway for all registered VeriFYD handles.

INERT unless BOTH global public-address routing and gateway flags are enabled.
Run as an isolated, single-instance Render cron job. Uses IMAPS on Hostinger,
requires the *first* (gateway-inserted) Received hop to disclose the original
envelope recipient, and ignores untrusted To/X-Original-To/Delivered-To headers.

Do not enable until a real catch-all .eml proves the trusted Received format;
otherwise messages are quarantined without delivery. Never alter apex MX.
"""
from __future__ import annotations

import email
import hashlib
import imaplib
import logging
import os
import re
import ssl
import sys
from email import policy
from email.parser import BytesParser
from email.utils import parseaddr

log = logging.getLogger("verifyd.circle_email.apex_gateway")
_APEX = "vfvid.com"
_IMAP_HOST = "imap.hostinger.com"
_MAX_RAW_BYTES = 18 * 1024 * 1024
_LOCALPART = re.compile(r"^[a-z][a-z0-9_]{2,23}$")
_TRUSTED_BY = re.compile(r"\bby\s+([a-z0-9][a-z0-9.-]+)", re.I)
# Restricted Hostinger inbound MX families observed in actual Received hops.
# Opt-in only: verify the outermost server, MX label, and ESMTPS transaction.
_HOSTINGER_MX_BY = re.compile(r"^(?:de-fra|fr-int)-smtpin[1-9][0-9]{0,2}\.hostinger\.io$", re.I)
_HOSTINGER_MX_SUFFIX = re.compile(r"^\s+\(mx\.hostinger\.com\)\s+with\s+ESMTPS\s+id\s+[a-z0-9]+\b", re.I)
_FOR_ADDRESS = re.compile(r"\bfor\s+<?([a-z][a-z0-9_]{2,23}@vfvid\.com)>?(?=[\s;,]|$)", re.I)
_UID_SIZE = re.compile(rb"RFC822\.SIZE\s+(\d+)")
_MAX_BATCH = 30
_REPLAY_WINDOW_UIDS = 256
_REPLAY_BATCH = 12

# Single-run, non-sensitive diagnostic stage. Never log messages, addresses,
# server responses or IMAP provider exception texts in production logs.
_STAGE = "startup"


def _stage(name: str) -> None:
    global _STAGE
    _STAGE = name



def _on(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in ("1", "true", "yes", "on")


def configured() -> bool:
    """Feature is never enabled by checking only a generic Resend key."""
    from trust_voice_circle_email_public import public_handles_enabled
    return bool(
        _on("VERIFYD_CIRCLE_EMAIL_APEX_GATEWAY_ENABLED")
        and public_handles_enabled()
        and os.getenv("VERIFYD_CIRCLE_EMAIL_HOSTINGER_IMAP_USER", "").lower().endswith("@vfvid.com")
        and os.getenv("VERIFYD_CIRCLE_EMAIL_HOSTINGER_IMAP_PASSWORD", "")
        and _trusted_hosts()
    )


def _trusted_hosts() -> frozenset[str]:
    # EXACT mail server names from a real Hostinger-delivered sample .eml.
    # Do not set wildcard .hostinger.com, do not trust user-supplied headers.
    values = os.getenv("VERIFYD_CIRCLE_EMAIL_APEX_TRUSTED_RECEIVED_BY", "")
    return frozenset(
        v for value in values.split(",")
        if (v := value.strip().lower()) and
        re.fullmatch(r"[a-z0-9][a-z0-9.-]{2,125}", v)
    )


def _replay_policy_fingerprint(trusted_hosts: frozenset[str]) -> str:
    """Replay quarantined mail when the trusted MX verification policy changes.

    Match the legacy fingerprint while rotating-MX support is disabled so the
    existing replay watermark remains stable during ordinary deployments.
    """
    policy_key = ",".join(sorted(trusted_hosts))
    if _on("VERIFYD_CIRCLE_EMAIL_APEX_ALLOW_ROTATING_MX"):
        policy_key += "|allow_rotating_mx=1"
    return hashlib.sha256(policy_key.encode("ascii")).hexdigest()


def original_envelope_recipient(message: email.message.Message,
                                trusted_hosts: frozenset[str]) -> str | None:
    """Fail closed unless Hostinger's actual *outermost* Received says for <user>.

    Visible To/Cc and X-Original-To/Delivered-To are not proof of SMTP routing.
    A hostile sender can forge those fields and fabricate older Received hops.
    """
    if not trusted_hosts:
        return None
    received = message.get_all("Received", [])
    if not received or len(received) > 60:
        return None
    top = str(received[0])
    server = _TRUSTED_BY.search(top)
    if not server:
        return None
    hostname = server.group(1).rstrip(".").lower()
    if hostname not in trusted_hosts:
        # Opt-in fallback for Hostinger's rotating numbered MX servers.
        # Keep rejecting unexpected hosts and headers lacking the MX markers.
        if not (
            _on("VERIFYD_CIRCLE_EMAIL_APEX_ALLOW_ROTATING_MX")
            and _HOSTINGER_MX_BY.fullmatch(hostname)
            and _HOSTINGER_MX_SUFFIX.match(top[server.end():])
        ):
            return None
    match = _FOR_ADDRESS.search(top)
    if not match:
        return None
    recipient = match.group(1).lower()
    if not _LOCALPART.fullmatch(recipient.split("@", 1)[0]):
        return None
    # Multiple "for" recipients or duplicate ambiguous envelope traces fail.
    if len(_FOR_ADDRESS.findall(top)) != 1:
        return None
    return recipient


def parse_catchall(raw: bytes, trusted_hosts: frozenset[str]):
    """Return a recipient handle and sanitized text email, or (None, None)."""
    if not raw or len(raw) > _MAX_RAW_BYTES:
        return None, None
    msg = BytesParser(policy=policy.default).parsebytes(raw)
    if msg.defects:
        return None, None
    recipient = original_envelope_recipient(msg, trusted_hosts)
    if not recipient:
        return None, None
    handle = recipient.split("@", 1)[0]
    sender = parseaddr(str(msg.get("From") or ""))[1].strip().lower()
    if not sender:
        return None, None
    subject = str(msg.get("Subject") or "").strip()[:200]
    try:
        plain = msg.get_body(preferencelist=("plain",))
        html = msg.get_body(preferencelist=("html",))
        text = (plain.get_content() if plain else "") if plain else ""
        html_text = (html.get_content() if html else "") if html else ""
    except (LookupError, ValueError, UnicodeError, AssertionError):
        return None, None
    files = sum(1 for part in msg.walk() if part.get_content_disposition() == "attachment")
    received = {
        "from": sender,
        "to": [recipient],
        "subject": subject,
        "text": str(text or "")[:12000],
        "html": str(html_text or "")[:200000],
        # Attachments deliberately withheld for unsolicited public email.
        "attachments": [{"withheld": True}] if files else [],
    }
    return handle, received


def _schema():
    from database import get_db
    from trust_voice_circle_email_inbound import ensure_inbound_schema
    ensure_inbound_schema()
    with get_db() as db:
        c = db.cursor()
        c.execute("""CREATE TABLE IF NOT EXISTS verifyd_circle_email_apex_cursor (
            mailbox_hash TEXT PRIMARY KEY, uid_validity TEXT NOT NULL,
            last_uid BIGINT NOT NULL DEFAULT 0
        )""")
        c.execute("""CREATE TABLE IF NOT EXISTS verifyd_circle_email_apex_quarantine (
            provider_id TEXT PRIMARY KEY, reason TEXT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )""")
        c.execute("""ALTER TABLE verifyd_circle_email_apex_quarantine
            ADD COLUMN IF NOT EXISTS last_replay_trust_hash TEXT""")
        c.execute("""ALTER TABLE verifyd_circle_email_apex_quarantine
            ADD COLUMN IF NOT EXISTS recovered_at TIMESTAMPTZ""")


def _check_cursor(mailbox_hash: str, uidvalidity: str) -> int:
    from database import get_db
    with get_db() as db:
        c = db.cursor()
        c.execute("SELECT uid_validity,last_uid FROM verifyd_circle_email_apex_cursor WHERE mailbox_hash=%s", (mailbox_hash,))
        x = c.fetchone()
        if not x or str(x["uid_validity"]) != uidvalidity:
            c.execute("""INSERT INTO verifyd_circle_email_apex_cursor
                (mailbox_hash,uid_validity,last_uid) VALUES (%s,%s,0)
                ON CONFLICT (mailbox_hash) DO UPDATE SET uid_validity=EXCLUDED.uid_validity,last_uid=0""",
                (mailbox_hash, uidvalidity))
            return 0
        return int(x["last_uid"])


def _checkpoint(mailbox_hash: str, uidvalidity: str, uid: int, reason: str | None,
                provider_id: str):
    from database import get_db
    with get_db() as db:
        c = db.cursor()
        if reason:
            c.execute("""INSERT INTO verifyd_circle_email_apex_quarantine(provider_id,reason)
                VALUES (%s,%s) ON CONFLICT (provider_id) DO NOTHING""", (provider_id,reason))
        c.execute("""UPDATE verifyd_circle_email_apex_cursor
            SET last_uid=GREATEST(last_uid,%s) WHERE mailbox_hash=%s AND uid_validity=%s""",
            (uid,mailbox_hash,uidvalidity))


def _search_new_uids(conn, last: int, max_count: int = _MAX_BATCH) -> list[int]:
    """Decode IMAP UID SEARCH responses without assuming text rather than bytes.

    imaplib returns a list containing raw bytes, e.g. [b'1 2 17'].
    bytes support isdigit(), not isdecimal(): the latter raised AttributeError
    and prevented the Hostinger gateway from importing any email.
    """
    status, rows = conn.uid("search", None, "UID", f"{last + 1}:*")
    if status != "OK" or not isinstance(rows, (list, tuple)):
        raise RuntimeError("Hostinger IMAP UID SEARCH failed")
    if not rows or rows[0] is None:
        return []
    data = rows[0]
    if not isinstance(data, (bytes, str)):
        raise RuntimeError("Hostinger IMAP UID SEARCH returned unexpected data")
    tokens = data.split()
    ids = set()
    for token in tokens:
        # Both str.isdigit and bytes.isdigit are available. A strict ASCII
        # token check rejects anything other than decimal digits.
        if isinstance(token, bytes):
            if not token.isdigit():
                raise RuntimeError("Hostinger IMAP UID SEARCH returned invalid UID")
            number = int(token.decode("ascii"))
        elif isinstance(token, str):
            if not token.isascii() or not token.isdecimal():
                raise RuntimeError("Hostinger IMAP UID SEARCH returned invalid UID")
            number = int(token)
        else:
            raise RuntimeError("Hostinger IMAP UID SEARCH returned invalid UID")
        if 0 < number <= 0xFFFFFFFF and number > last:
            ids.add(number)
        elif number < 1 or number > 0xFFFFFFFF:
            raise RuntimeError("Hostinger IMAP UID SEARCH returned invalid UID")
    return sorted(ids)[:min(_MAX_BATCH,max(1,max_count))]


def _uidvalidity(conn) -> str:
    """Handle imaplib UIDVALIDITY returned as bytes or str safely."""
    status, data = conn.response("UIDVALIDITY")
    if status != "UIDVALIDITY" or not data or len(data) != 1:
        raise RuntimeError("Hostinger IMAP UIDVALIDITY unavailable")
    item=data[0]
    if isinstance(item, bytes):
        item=item.decode("ascii")
    if not isinstance(item, str) or not item.isascii() or not item.isdecimal():
        raise RuntimeError("Hostinger IMAP UIDVALIDITY invalid")
    return item


def _fetch_size(conn, uid: int) -> int:
    status, data = conn.uid("fetch", str(uid), "(RFC822.SIZE)")
    if status != "OK":
        raise RuntimeError("IMAP size fetch failed")
    for item in data or []:
        if isinstance(item, bytes):
            match = _UID_SIZE.search(item)
            if match:
                return int(match.group(1))
    raise RuntimeError("IMAP message size unknown")


def _fetch_raw(conn, uid: int) -> bytes:
    status, data = conn.uid("fetch", str(uid), "(BODY.PEEK[])")
    if status != "OK":
        raise RuntimeError("IMAP body fetch failed")
    for item in data or []:
        if isinstance(item, tuple) and isinstance(item[1], bytes):
            return item[1]
    raise RuntimeError("IMAP message body unavailable")



def _recent_processed_uids(conn, last_uid: int,
                           window: int = _REPLAY_WINDOW_UIDS) -> list[int]:
    """Only list existing, already checkpointed IMAP UIDs in a bounded window."""
    if last_uid < 1:
        return []
    first = max(1, last_uid - window + 1)
    status, rows = conn.uid("search", None, "UID", f"{first}:{last_uid}")
    if status != "OK" or not isinstance(rows, (list, tuple)):
        raise RuntimeError("Hostinger replay UID SEARCH failed")
    if not rows or rows[0] is None:
        return []
    response = rows[0]
    if not isinstance(response, (bytes, str)):
        raise RuntimeError("Hostinger replay UID SEARCH malformed")
    found = set()
    for token in response.split():
        if isinstance(token, bytes):
            if not token.isdigit():
                raise RuntimeError("Hostinger replay UID malformed")
            number = int(token.decode("ascii"))
        else:
            if not token.isascii() or not token.isdecimal():
                raise RuntimeError("Hostinger replay UID malformed")
            number = int(token)
        if not 0 < number <= 0xFFFFFFFF:
            raise RuntimeError("Hostinger replay UID invalid")
        if first <= number <= last_uid:
            found.add(number)
    return sorted(found, reverse=True)


def _replay_recent_quarantine(conn, mailbox_key: str, valid: str,
                              last_uid: int, trusted_hosts: frozenset[str],
                              max_count: int = _REPLAY_BATCH) -> tuple[int, int]:
    """Recover only verified recipients from previously quarantined mail.

    Never rewind the live cursor, trust To headers, delete provider mail or
    deliver to a handle not validated by store_public_received(). Each
    quarantined provider ID is checked once for each exact trusted-host set.
    """
    if not trusted_hosts or not 0 < max_count <= _REPLAY_BATCH:
        return 0, 0
    from database import get_db
    from trust_voice_circle_email_public import store_public_received

    hosts_hash = _replay_policy_fingerprint(trusted_hosts)
    recent = _recent_processed_uids(conn, last_uid)
    provider_for_uid = {
        uid: "hostinger-" + hashlib.sha256(
            f"{mailbox_key}:{valid}:{uid}".encode("ascii")).hexdigest()
        for uid in recent
    }
    if not provider_for_uid:
        return 0, 0
    with get_db() as db:
        cur = db.cursor()
        cur.execute("""SELECT provider_id
            FROM verifyd_circle_email_apex_quarantine
            WHERE provider_id = ANY(%s)
              AND reason = 'recipient_unverified'
              AND recovered_at IS NULL
              AND last_replay_trust_hash IS DISTINCT FROM %s""",
                    (list(provider_for_uid.values()), hosts_hash))
        eligible = {row["provider_id"] for row in cur.fetchall()}

    checked = recovered = 0
    for uid in recent:
        provider_id = provider_for_uid[uid]
        if provider_id not in eligible:
            continue
        _stage("replay_fetch_size")
        size = _fetch_size(conn, uid)
        delivered = False
        if 0 < size <= _MAX_RAW_BYTES:
            _stage("replay_fetch_raw")
            raw = _fetch_raw(conn, uid)
            _stage("replay_parse")
            handle, received = parse_catchall(raw, trusted_hosts)
            if received is not None:
                _stage("replay_store")
                owner, mail_id = store_public_received(
                    received, provider_id, handle=handle)
                delivered = bool(owner and mail_id)
        # The DB insert above commits before replay state is recorded.
        # A retry is safe: store_public_received is idempotent by provider_id.
        _stage("replay_mark")
        with get_db() as db:
            db.cursor().execute(
                """UPDATE verifyd_circle_email_apex_quarantine
                   SET last_replay_trust_hash=%s,
                       recovered_at=CASE WHEN %s THEN NOW() ELSE recovered_at END
                   WHERE provider_id=%s""",
                (hosts_hash, delivered, provider_id))
        checked += 1
        if delivered:
            recovered += 1
        if checked >= max_count:
            break
    return recovered, checked


def ingest_batch(max_count: int = _MAX_BATCH) -> dict:
    """Read one bounded IMAP batch; checkpoint only after each message commits."""
    from database import get_db
    from trust_voice_circle_email_public import store_public_received
    if not configured():
        return {"status": "disabled", "processed": 0, "quarantined": 0}
    imap_user = os.environ["VERIFYD_CIRCLE_EMAIL_HOSTINGER_IMAP_USER"].strip().lower()
    password = os.environ["VERIFYD_CIRCLE_EMAIL_HOSTINGER_IMAP_PASSWORD"]
    mailbox_key = hashlib.sha256(imap_user.encode()).hexdigest()
    _stage("schema")
    _schema()
    _stage("advisory_lock")
    handled = quarantined = skipped = 0
    # Transaction-scoped lock keeps two concurrent cron invocations from
    # advancing the watermark past one another or skipping any message.
    with get_db() as lockdb:
        lock = lockdb.cursor()
        lock.execute("SELECT pg_try_advisory_xact_lock(hashtext(%s)) AS ok",
                     ("verifyd-apex:" + mailbox_key,))
        if not lock.fetchone()["ok"]:
            return {"status": "busy", "processed": 0, "quarantined": 0}
        _stage("imap_connect")
        connection = imaplib.IMAP4_SSL(_IMAP_HOST, 993, ssl_context=ssl.create_default_context(),timeout=20)
        try:
            _stage("imap_login")
            connection.login(imap_user, password)
            _stage("imap_select")
            status, _ = connection.select("INBOX", readonly=True)
            if status != "OK":
                raise RuntimeError("Could not select Hostinger catch-all INBOX")
            _stage("uid_validity")
            valid = _uidvalidity(connection)
            _stage("cursor")
            last = _check_cursor(mailbox_key,valid)
            _stage("uid_search")
            uids = _search_new_uids(connection, last, max_count)
            for uid in uids:
                provider_id = "hostinger-" + hashlib.sha256(
                    f"{mailbox_key}:{valid}:{uid}".encode("ascii")
                ).hexdigest()
                _stage("fetch_size")
                size = _fetch_size(connection,uid)
                if size < 1 or size > _MAX_RAW_BYTES:
                    _checkpoint(mailbox_key,valid,uid,"size_limit",provider_id)
                    quarantined += 1
                    continue
                _stage("fetch_raw")
                raw = _fetch_raw(connection,uid)
                _stage("parse_catchall")
                handle, received = parse_catchall(raw,_trusted_hosts())
                if received is None:
                    _stage("quarantine")
                    _checkpoint(mailbox_key,valid,uid,"recipient_unverified",provider_id)
                    quarantined += 1
                    continue
                # store_public_received is idempotent on provider_id and
                # resolves active account records in database, not headers.
                _stage("store_received")
                delivered_owner, delivered_id = store_public_received(received,provider_id,handle=handle)
                _stage("checkpoint")
                _checkpoint(mailbox_key,valid,uid,None,provider_id)
                if delivered_owner and delivered_id:
                    handled += 1
                else:
                    # Reserved aliases, unregistered accounts, duplicates and
                    # sender limits are not actual VeriFYD Inbox deliveries.
                    skipped += 1
            _stage("replay_quarantine")
            recovered, rechecked = _replay_recent_quarantine(
                connection, mailbox_key, valid, last, _trusted_hosts())
            handled += recovered
        finally:
            try:
                connection.logout()
            except (imaplib.IMAP4.error,OSError):
                pass
    return {"status": "ok", "processed": handled, "quarantined": quarantined,
            "skipped": skipped, "recovered": recovered, "rechecked": rechecked}


def main() -> int:
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper())
    try:
        result = ingest_batch()
        # No addresses, bodies, authentication secrets or message headers in logs.
        log.info("Apex mail batch status=%s imported=%d quarantined=%d skipped=%d",
                 result["status"],result["processed"],result["quarantined"],result.get("skipped",0))
        log.info("Apex quarantine replay recovered=%d rechecked=%d",
                 result.get("recovered", 0), result.get("rechecked", 0))
        return 0
    except Exception as exc:
        # Never log provider exception messages: they may contain account names
        # or other mailbox data. Logs contain a generic type only.
        log.error("Apex mail batch failed stage=%s error=%s", _STAGE, type(exc).__name__)
        return 1


if __name__ == "__main__":
    sys.exit(main())
