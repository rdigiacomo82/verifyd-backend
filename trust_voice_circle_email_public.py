"""Gated public VeriFYD handle mailboxes on the existing reply subdomain.

This feature is disabled by default and ONLY opted-in handles in a strict
allowlist can receive unsolicited external emails. The short address is public;
it is NOT equivalent to our existing single-recipient private reply tokens.
"""
from __future__ import annotations

import os
import re
import uuid
from email.utils import parseaddr, getaddresses

from database import get_db, is_valid_email

_HANDLE = re.compile(r'^[a-z][a-z0-9_]{2,23}$')
_MAX_OWNER_DAY = 60
_MAX_SENDER_DAY = 8


def public_handles_enabled() -> bool:
    return os.getenv('VERIFYD_CIRCLE_EMAIL_PUBLIC_HANDLE_ENABLED', '').strip().lower() in (
        '1', 'true', 'yes', 'on'
    )


def _allowed_handles() -> frozenset[str]:
    # Never use a wildcard. Each user must affirmatively opt in.
    values = os.getenv('VERIFYD_CIRCLE_EMAIL_PUBLIC_HANDLE_ALLOWLIST', '')
    return frozenset(
        h for value in values.split(',')
        if (h := value.strip().lower().lstrip('@')) and _HANDLE.fullmatch(h)
        and h not in {'reply', 'admin', 'system', 'postmaster', 'abuse', 'support'}
    )


def public_address_for_handle(handle: str, domain: str) -> str | None:
    name = str(handle or '').strip().lower().lstrip('@')
    if public_handles_enabled() and name in _allowed_handles():
        return f'{name}@{domain}'
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
    """Store a brand-new external message to an explicitly opted-in handle."""
    from trust_voice_circle_email import risk, utc
    from trust_voice_circle_email_inbound import _plain_body

    # Defense in depth: the handler only operates for explicitly opted-in
    # handles, even if invoked by a future code path.
    if not public_handles_enabled() or handle not in _allowed_handles():
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
    if isinstance(advertised, list) and advertised:
        # Public address attachments require a separate malware scanning
        # rollout. No provider/CDN file is downloaded in this stage.
        notice = '[External attachments withheld: public mailbox malware scanning is not enabled.]'
        body = (body[:12000-len(notice)-2] + '\n\n' + notice)[:12000]
    now = utc()
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
        cur.execute("""INSERT INTO verifyd_circle_email_received_events
            (provider_email_id,mail_id,created_at) VALUES (%s,%s,%s)""",
            (provider_id,mail_id,now))
    return uid,mail_id
