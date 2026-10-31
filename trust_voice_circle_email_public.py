"""All-member VeriFYD email routing: one dynamic account registry, no mailbox setup.

Any active, verified VeriFYD identity has a deterministic address derived from
its unique registered handle. Routing is disabled globally until receiving
infrastructure, abuse controls and user announcements are validated.
There are NO per-member Hostinger mailboxes, forwarders, or allowlist entries.
"""
from __future__ import annotations

import os
import re
import uuid
from email.utils import parseaddr, getaddresses

from database import get_db, is_valid_email

_HANDLE = re.compile(r'^[a-z][a-z0-9_]{2,23}$')
# Keep the total per-account flood cap at 60 external emails per rolling day.
# The former 8-per-sender threshold unexpectedly rejected ordinary threads
# and test messages, even after they were moved to Trash.
_MAX_OWNER_DAY = 60
_MAX_SENDER_DAY = 30


def public_handles_enabled() -> bool:
    return os.getenv('VERIFYD_CIRCLE_EMAIL_PUBLIC_HANDLE_ENABLED', '').strip().lower() in (
        '1', 'true', 'yes', 'on'
    )


_RESERVED_EMAIL_LOCALPARTS = frozenset({
    # No registered member may impersonate an operational email role.
    'reply', 'admin', 'administrator', 'system', 'postmaster', 'abuse',
    'support', 'security', 'billing', 'sales', 'help', 'info', 'contact',
    'noreply', 'no-reply', 'mail', 'team', 'staff', 'hostmaster', 'webmaster',
    'privacy', 'legal', 'dmarc', 'dkim', 'spf', 'bounce', 'bounces',
})


def _reserved_aliases() -> frozenset[str]:
    # Reserved infrastructure mailboxes, never a manually maintained user list.
    # Include any existing Hostinger account or email alias before DNS cutover.
    raw = os.getenv('VERIFYD_CIRCLE_EMAIL_RESERVED_LOCALPARTS', '')
    return _RESERVED_EMAIL_LOCALPARTS | frozenset(
        name for item in raw.split(',')
        if (name := item.strip().lower()) and _HANDLE.fullmatch(name)
    )


def public_address_for_handle(handle: str, domain: str) -> str | None:
    """A canonical address for ANY eligible handle; no per-user provision call."""
    name = str(handle or '').strip().lower().lstrip('@')
    if (public_handles_enabled() and _HANDLE.fullmatch(name)
            and name not in _reserved_aliases()
            and domain == 'reply.vfvid.com'):
        return f'{name}@{domain}'
    return None


def public_apex_address_for_handle(handle: str) -> str | None:
    """Only advertise @vfvid.com after the catch-all gateway was validated.

    A separate flag keeps outgoing Reply-To addresses usable throughout the
    infrastructure rollout, and lets operators stop creating new apex replies
    without immediately disabling receiving old apex messages.
    """
    name = str(handle or '').strip().lower().lstrip('@')
    if (public_handles_enabled()
            and os.getenv('VERIFYD_CIRCLE_EMAIL_APEX_ADVERTISE_ENABLED','').strip().lower()
                  in ('1','true','yes','on')
            and _HANDLE.fullmatch(name)
            and name not in _reserved_aliases()):
        return f'{name}@vfvid.com'
    return None


def public_recipient_from_to(to_list, domain: str) -> str | None:
    """Resolve one exact public address; reject multiple recipients or token aliases."""
    if not public_handles_enabled():
        return None
    addresses = to_list if isinstance(to_list, list) else [to_list]
    if len(addresses) != 1:
        return None
    parsed = getaddresses([str(addresses[0] or '')])
    if len(parsed) != 1:
        return None
    address = parsed[0][1].strip().lower()
    suffix = '@' + domain
    if not address.endswith(suffix):
        return None
    handle = address[:-len(suffix)]
    if public_address_for_handle(handle, domain) != address:
        return None
    return handle


def _public_spam_limits(cur, identity_id: str, sender: str) -> bool:
    """Hard bounds against unsolicited mail floods; never reveal account presence."""
    cur.execute("""SELECT COUNT(*) AS n FROM verifyd_circle_email_messages
        WHERE owner_id=%s AND delivery='received_external'
        AND created_at::timestamptz > NOW() - INTERVAL '24 hours'""",
        (identity_id,))
    if int(cur.fetchone()['n']) >= _MAX_OWNER_DAY:
        return False
    cur.execute("""SELECT COUNT(*) AS n FROM verifyd_circle_email_messages
        WHERE owner_id=%s AND sender_id=%s AND delivery='received_external'
        AND created_at::timestamptz > NOW() - INTERVAL '24 hours'""",
        (identity_id, 'external:' + sender))
    return int(cur.fetchone()['n']) < _MAX_SENDER_DAY


def store_public_received(received: dict, provider_id: str, *, handle: str) -> tuple[str | None, str | None]:
    """Import external mail for any active registered handle, under global gate."""
    from trust_voice_circle_email import risk, utc
    from trust_voice_circle_email_inbound import _plain_body

    # Defense in depth: only eligible registered handle names may be routed.
    # Actual account status and email verification are checked in the DB.
    from trust_voice_circle_email_inbound import reply_domain
    if public_address_for_handle(handle, reply_domain()) is None:
        return None, None
    sender = parseaddr(str(received.get('from') or ''))[1].strip().lower()
    if not sender or not is_valid_email(sender) or len(sender) > 254:
        return None, None
    subject = str(received.get('subject') or '').strip()[:200] or '(no subject)'
    body = _plain_body(received)
    verdict, notes = risk(False, subject, body)
    auth = received.get('authentication') or {}
    if isinstance(auth, dict) and str(auth.get('dmarc') or '').lower() == 'fail':
        verdict = 'suspicious'
        notes = 'Provider reports failed DMARC; ' + notes
    notes = 'Unsolicited public-handle email; sender unverified. ' + notes
    advertised = received.get('attachments') or []
    files = []
    if isinstance(advertised, list) and advertised:
        # Sender-supplied JSON and headers never authorize attachment capture.
        # Only the trusted Hostinger IMAP parser can supply this private MIME
        # object after validating the original envelope recipient.
        from trust_voice_attachments import _storage_configured
        capture = (provider_id.startswith('hostinger-')
                   and received.get('_hostinger_verified_mime') is not None
                   and os.getenv('VERIFYD_CIRCLE_EMAIL_APEX_ATTACHMENT_CAPTURE_ENABLED','')
                       .strip().lower() in ('1','true','yes','on')
                   and _storage_configured())
        if capture:
            from trust_voice_circle_email_transport import extract_hostinger_mime_files
            files, omitted = extract_hostinger_mime_files(
                received['_hostinger_verified_mime'])
            if files:
                notice = ('[External attachments have not been malware scanned. '
                          'Confirm before downloading; do not open unknown files.]')
                if omitted:
                    notice += f' [{omitted} attachment(s) withheld by file policy.]'
            else:
                notice = '[External attachments withheld: unsupported or unsafe files.]'
        else:
            notice = '[External attachments withheld: public mailbox malware scanning is not enabled.]'
        body = (body[:12000-len(notice)-2] + '\n\n' + notice)[:12000]
    now = utc()
    created_keys = []
    try:
        with get_db() as db:
            cur = db.cursor()
            cur.execute("""SELECT id, handle_lower FROM verifyd_identities
                WHERE handle_lower=%s AND email_verified=TRUE AND status='active'
                LIMIT 1""", (handle,))
            owner = cur.fetchone()
            if not owner:
                return None, None
            uid = str(owner['id'])
            # Serialize public mail intake per owner to reduce rate-limit races.
            cur.execute('SELECT pg_advisory_xact_lock(hashtext(%s))', ('circle-public-mail:' + uid,))
            cur.execute('SELECT 1 FROM verifyd_circle_email_received_events WHERE provider_email_id=%s',
                        (provider_id,))
            if cur.fetchone():
                return None, None
            if not _public_spam_limits(cur, uid, sender):
                return None, None
            mail_id = 'tvcmail_' + uuid.uuid4().hex
            thread_id = 'tvcemail_' + uuid.uuid4().hex
            folder = 'spam' if verdict == 'suspicious' else 'inbox'
            cur.execute("""INSERT INTO verifyd_circle_email_messages
                (mail_id,thread_id,owner_id,sender_id,sender_handle,recipient,
                 subject,body,folder,is_read,spam_flag,delivery,verdict,risk_notes,created_at)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,FALSE,%s,'received_external',%s,%s,%s)""",
                (mail_id,thread_id,uid,'external:'+sender,sender,'@'+handle,
                 subject,body,folder,verdict=='suspicious',verdict,notes,now))
            if files:
                from trust_voice_circle_email_transport import persist_inbound_attachments
                created_keys = persist_inbound_attachments(
                    db, files=files, owner_id=uid, mail_id=mail_id, stamp=now)
            cur.execute("""INSERT INTO verifyd_circle_email_received_events
                (provider_email_id,mail_id,created_at) VALUES (%s,%s,%s)""",
                (provider_id,mail_id,now))
    except Exception:
        # If the DB transaction fails after an R2 upload, do not leave
        # orphaned private files. No unverified mail is ever sent to another
        # Circle account through this path.
        if created_keys:
            try:
                from trust_voice_attachments import _get_attachment_client, TRUST_VOICE_BUCKET
                client = _get_attachment_client()
                for key in created_keys:
                    try:
                        client.delete_object(Bucket=TRUST_VOICE_BUCKET, Key=key)
                    except Exception:
                        pass
            except Exception:
                pass
        raise
    return uid,mail_id
