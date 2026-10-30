"""VeriFYD Circle Email Phase 2: verified inbound replies from Resend.

Incoming external messages are never accepted from unauthenticated browsers.
All routing uses unpredictable per-message Reply-To tokens; arbitrary custom
email headers, display names, subject lines, and user-provided handles cannot
select a VeriFYD recipient. Disabled until receiving-domain DNS + Resend webhook
are configured and validated.
"""
from __future__ import annotations

import asyncio
import base64
import binascii
import hashlib
import hmac
import json
import logging
import os
import re
import secrets
import time
import urllib.error
import urllib.request
import uuid
from email.utils import parseaddr
from html.parser import HTMLParser
from typing import Optional

from fastapi import APIRouter, HTTPException, Request

from database import get_db, is_valid_email
from trust_voice_signaling import _send_to_identity

log = logging.getLogger('verifyd.circle_email.inbound')
router = APIRouter(tags=['VeriFYD Circle Email Inbound'])
INBOUND_VERSION = '0.3.2'
_HEX_TOKEN = re.compile(r'^reply\+([a-f0-9]{40})@([a-z0-9.-]+)$', re.I)
_HANDLE_TOKEN = re.compile(r'^([a-z][a-z0-9_]{2,23})\+([a-f0-9]{32})@([a-z0-9.-]+)$', re.I)
_PROVIDER_ID = re.compile(r'^[a-zA-Z0-9-]{15,90}$')
_INBOUND_SCHEMA_READY = False


def inbound_enabled() -> bool:
    return os.getenv('VERIFYD_CIRCLE_EMAIL_INBOUND_ENABLED', '').strip().lower() in ('1', 'true', 'yes', 'on')


def inbound_attachments_enabled() -> bool:
    # Deliberately opt-in. Do not put working text-email receiving at risk
    # until real Resend/R2 attachment retrieval has been verified.
    return os.getenv('VERIFYD_CIRCLE_EMAIL_INBOUND_ATTACHMENTS_ENABLED', '').strip().lower() in ('1', 'true', 'yes', 'on')


def reply_domain() -> str:
    # Dedicated subdomain: never alter apex vfvid.com MX records.
    raw = (os.getenv('VERIFYD_CIRCLE_EMAIL_REPLY_DOMAIN', '') or 'reply.vfvid.com').strip().lower()
    if not re.fullmatch(r'[a-z0-9-]+(?:\.[a-z0-9-]+)+', raw) or len(raw) > 190:
        raise RuntimeError('Invalid Circle reply domain configuration')
    return raw


def _inbound_secret() -> str:
    return os.getenv('VERIFYD_CIRCLE_EMAIL_INBOUND_WEBHOOK_SECRET', '').strip()


def _receiving_api_key() -> str:
    """Prefer a separately permissioned Resend key for inbound message reads.

    The fallback preserves existing deployments while the new receiving key is
    being provisioned. A send-only fallback key cannot retrieve inbound emails.
    """
    return (os.getenv('VERIFYD_CIRCLE_EMAIL_RECEIVING_API_KEY', '').strip()
            or os.getenv('RESEND_API_KEY', '').strip())


def inbound_configured() -> bool:
    try:
        return bool(inbound_enabled() and _inbound_secret().startswith('whsec_')
                    and _receiving_api_key() and reply_domain())
    except (ValueError, RuntimeError):
        return False


def ensure_inbound_schema() -> None:
    global _INBOUND_SCHEMA_READY
    if _INBOUND_SCHEMA_READY:
        return
    # Import at call time to avoid circular dependency during router registration.
    from trust_voice_circle_email import schema
    schema()
    with get_db() as db:
        cur = db.cursor()
        cur.execute('''CREATE TABLE IF NOT EXISTS verifyd_circle_email_reply_routes (
            token_hash TEXT PRIMARY KEY, owner_id TEXT NOT NULL,
            sender_mail_id TEXT NOT NULL, thread_id TEXT NOT NULL,
            external_recipient TEXT NOT NULL, created_at TEXT NOT NULL,
            revoked_at TEXT
        )''')
        cur.execute('''CREATE INDEX IF NOT EXISTS idx_circle_reply_routes_owner
          ON verifyd_circle_email_reply_routes(owner_id,created_at DESC)''')
        cur.execute('''CREATE TABLE IF NOT EXISTS verifyd_circle_email_received_events (
            provider_email_id TEXT PRIMARY KEY, mail_id TEXT NOT NULL,
            created_at TEXT NOT NULL
        )''')
    _INBOUND_SCHEMA_READY = True


def create_reply_route(db, *, owner_id: str, sent_mail_id: str,
                       thread_id: str, external_recipient: str, stamp: str,
                       sender_handle: str = '') -> str:
    """Store route in the caller's *existing outbound transaction*."""
    if not inbound_configured():
        raise HTTPException(status_code=503, detail={'error':'inbound_not_configured',
            'message':'VeriFYD reply receiving is still being configured.'})
    # Handle is decorative; a random 128-bit token is the ONLY routing key.
    # The 24-character handle limit keeps the SMTP local part <=64 chars.
    display_handle = str(sender_handle or '').strip().lstrip('@').lower()
    if not re.fullmatch(r'[a-z][a-z0-9_]{2,23}', display_handle):
        raise HTTPException(status_code=503, detail={'error': 'invalid_reply_sender_handle'})
    token = secrets.token_hex(16)
    digest = hashlib.sha256(token.encode('ascii')).hexdigest()
    cur = db.cursor()
    cur.execute('''INSERT INTO verifyd_circle_email_reply_routes
      (token_hash,owner_id,sender_mail_id,thread_id,external_recipient,created_at)
      VALUES (%s,%s,%s,%s,%s,%s)''',
      (digest,owner_id,sent_mail_id,thread_id,external_recipient.lower(),stamp))
    return display_handle + '+' + token + '@' + reply_domain()


def verify_resend_webhook(raw: bytes, headers, *, secret: Optional[str] = None,
                          clock: Optional[int] = None) -> None:
    """Validate Resend/Svix HMAC over untouched body with replay window."""
    secret = _inbound_secret() if secret is None else secret
    if not secret or not secret.startswith('whsec_'):
        raise HTTPException(status_code=503, detail={'error':'inbound_not_configured'})
    sid = str(headers.get('svix-id') or '')
    stamp = str(headers.get('svix-timestamp') or '')
    signatures = str(headers.get('svix-signature') or '').split(' ')
    if not sid or len(sid) > 200 or not stamp.isascii() or not stamp.isdecimal():
        raise HTTPException(status_code=401, detail={'error':'invalid_webhook_signature'})
    now = int(time.time()) if clock is None else clock
    if abs(now - int(stamp)) > 300:
        raise HTTPException(status_code=401, detail={'error':'webhook_timestamp_expired'})
    try:
        key = base64.b64decode(secret[6:] + '=' * (-len(secret[6:]) % 4), altchars=b'-_', validate=True)
    except (ValueError, binascii.Error):
        raise HTTPException(status_code=503, detail={'error':'inbound_secret_invalid'})
    if len(key) < 24:
        raise HTTPException(status_code=503, detail={'error':'inbound_secret_invalid'})
    signed_bytes = (sid + '.' + stamp + '.').encode('utf-8') + raw
    wanted = hmac.new(key, signed_bytes, hashlib.sha256).digest()
    valid = False
    for entry in signatures:
        if not entry.startswith('v1,'):
            continue
        try:
            candidate = base64.b64decode(entry[3:], validate=True)
            valid = hmac.compare_digest(wanted, candidate) or valid
        except (ValueError, binascii.Error):
            continue
    if not valid:
        raise HTTPException(status_code=401, detail={'error':'invalid_webhook_signature'})


def _fetch_received_email(provider_id: str) -> dict:
    """Fetch authoritative content; do not trust webhook's unauthenticated metadata."""
    if not _PROVIDER_ID.fullmatch(provider_id):
        raise HTTPException(status_code=400, detail={'error':'invalid_provider_id'})
    key = _receiving_api_key()
    if not key:
        raise HTTPException(status_code=503, detail={'error':'mail_provider_not_configured'})
    req = urllib.request.Request(
        'https://api.resend.com/emails/receiving/' + provider_id,
        headers={
            'Authorization': 'Bearer ' + key,
            'Accept': 'application/json',
            'User-Agent': 'VeriFYD-Circle-Inbound/0.2.1',
        },
        method='GET')
    try:
        with urllib.request.urlopen(req, timeout=12) as response:
            raw = response.read(2_000_001)
    except urllib.error.HTTPError as exc:
        # Diagnostic status only: never log auth headers, API keys, email bodies,
        # reply tokens, provider response bodies, or raw webhook payloads.
        log.warning('Resend received-email retrieval HTTP status: %s', exc.code)
        raise HTTPException(status_code=503, detail={'error':'inbound_provider_retry'}) from None
    except (urllib.error.URLError, TimeoutError):
        log.warning('Resend received-email retrieval network or timeout failure')
        raise HTTPException(status_code=503, detail={'error':'inbound_provider_retry'}) from None
    if len(raw) > 2_000_000:
        raise HTTPException(status_code=413, detail={'error':'inbound_message_too_large'})
    try:
        data = json.loads(raw)
    except (ValueError, UnicodeError):
        raise HTTPException(status_code=503, detail={'error':'invalid_provider_response'})
    if not isinstance(data, dict) or str(data.get('id') or '') != provider_id:
        raise HTTPException(status_code=503, detail={'error':'invalid_provider_response'})
    return data


class _PlainText(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.skip = 0
    def handle_starttag(self, tag, attrs):
        if tag.lower() in ('script', 'style', 'svg'):
            self.skip += 1
        if tag.lower() in ('p', 'br', 'div', 'li'):
            self.parts.append('\n')
    def handle_endtag(self, tag):
        if tag.lower() in ('script', 'style', 'svg') and self.skip:
            self.skip -= 1
    def handle_data(self, data):
        if not self.skip:
            self.parts.append(data)


def _plain_body(received: dict) -> str:
    text = received.get('text')
    if isinstance(text, str) and text.strip():
        return text.strip()[:12000]
    html = received.get('html')
    if not isinstance(html, str) or not html:
        return '[This email contains no plain-text body.]'
    parser = _PlainText()
    parser.feed(html[:200000])
    return ''.join(parser.parts).strip()[:12000] or '[This email contains no plain-text body.]'


def _candidate_routes(to_list, domain: str):
    tokens=[]
    for val in (to_list if isinstance(to_list,list) else [to_list]):
        address=parseaddr(str(val or ''))[1].strip().lower()
        legacy = _HEX_TOKEN.fullmatch(address)
        named = _HANDLE_TOKEN.fullmatch(address)
        if legacy and legacy.group(2)==domain:
            tokens.append(legacy.group(1).lower())
        elif named and named.group(3)==domain and named.group(1).lower()!='reply':
            # "reply" is reserved for legacy addresses; a handle is never
            # trusted for recipient selection, only the stored secret token.
            tokens.append(named.group(2).lower())
    return list(dict.fromkeys(tokens))


def _store_received(received: dict, provider_id: str) -> tuple[Optional[str],Optional[str]]:
    from trust_voice_circle_email import risk, utc
    ensure_inbound_schema()
    domain=reply_domain()
    # Direct public mail is allowed for ANY active, email-verified handle
    # after the one global rollout gate is enabled; no per-user allowlist.
    from trust_voice_circle_email_public import (
        public_recipient_from_to, store_public_received
    )
    public_handle=public_recipient_from_to(received.get('to') or [], domain)
    if public_handle is not None:
        return store_public_received(received,provider_id,handle=public_handle)
    candidates=_candidate_routes(received.get('to') or [], domain)
    if len(candidates)!=1:
        return None,None
    sender=parseaddr(str(received.get('from') or ''))[1].strip().lower()
    if not sender or not is_valid_email(sender) or len(sender)>254:
        return None,None
    digest=hashlib.sha256(candidates[0].encode('ascii')).hexdigest()
    subject=str(received.get('subject') or '').strip()[:200] or '(no subject)'
    body=_plain_body(received)
    authentic=received.get('authentication') or {}
    verdict, notes = risk(False, subject, body)
    if isinstance(authentic,dict) and str(authentic.get('dmarc') or '').lower()=='fail':
        verdict='suspicious';notes='Provider reports failed DMARC authentication; '+notes
    advertised=received.get('attachments') or []
    if not isinstance(advertised,list):
        advertised=[]
    timestamp=utc()
    created_keys = []
    try:
        with get_db() as db:
            cur=db.cursor()
            cur.execute('''SELECT r.owner_id,r.thread_id,r.external_recipient,i.handle_lower
              FROM verifyd_circle_email_reply_routes r
              JOIN verifyd_identities i ON i.id=r.owner_id
              WHERE r.token_hash=%s AND r.revoked_at IS NULL AND i.status='active'
              FOR UPDATE OF r''',(digest,))
            route=cur.fetchone()
            if not route:
                return None,None
            route=dict(route)
            # A reply-to address is private to one outbound recipient. Prevent the
            # address from being forwarded to create a cross-account inbox channel.
            if sender!=str(route['external_recipient']).strip().lower():
                return None,None
            cur.execute('''SELECT 1 FROM verifyd_circle_email_received_events
              WHERE provider_email_id=%s''',(provider_id,))
            if cur.fetchone():
                return None,None  # idempotent Resend redelivery/replay
            uid=str(route['owner_id'])
            # Avoid cross-network delivery after a bilateral block is configured.
            # External addresses are not VeriFYD identities, so do not equate them
            # with a Trust Circle contact; classify them as unverified/uncertain.
            mail_id='tvcmail_'+uuid.uuid4().hex
            folder='spam' if verdict=='suspicious' else 'inbox'
            # Fetch attachment bytes only after verifying the stored reply token and
            # the exact external sender. Provider/network errors remain retryable.
            from trust_voice_circle_email_transport import (
                fetch_inbound_attachment_files, persist_inbound_attachments
            )
            if advertised and not inbound_attachments_enabled():
                # Metadata-only mode preserves working replies until the
                # attachment pipeline has passed a live provider/R2 test.
                attached, omitted = [], len(advertised)
                notice = (f'[{omitted} external attachment(s) withheld: '
                          'incoming file downloads have not yet been activated.]')
                body = (body[:max(0,12000-len(notice)-2)] + '\n\n' + notice)[:12000]
            else:
                attached, omitted = fetch_inbound_attachment_files(provider_id, advertised)
                if attached or omitted:
                    notice = ('[External attachments are not malware scanned. '
                              'Download only after reviewing the warning.]')
                    if omitted:
                        notice += f' [{omitted} attachment(s) withheld due to file restrictions.]'
                    body = (body[:max(0,12000-len(notice)-2)] + '\n\n' + notice)[:12000]
            cur.execute('''INSERT INTO verifyd_circle_email_messages
              (mail_id,thread_id,owner_id,sender_id,sender_handle,recipient,
               subject,body,folder,is_read,spam_flag,delivery,verdict,risk_notes,created_at)
              VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,FALSE,%s,'received_external',%s,%s,%s)''',
              (mail_id,route['thread_id'],uid,'external:'+sender,sender,
               '@'+str(route['handle_lower']),subject,body,folder,
               verdict=='suspicious',verdict,notes,timestamp))
            created_keys = persist_inbound_attachments(db,files=attached,owner_id=uid,
                                        mail_id=mail_id,stamp=timestamp)
            cur.execute('''INSERT INTO verifyd_circle_email_received_events
              (provider_email_id,mail_id,created_at) VALUES (%s,%s,%s)''',
              (provider_id,mail_id,timestamp))
    except Exception:
        # Restore storage/DB consistency if the DB commit or event ledger fails
        # AFTER objects were successfully placed in private R2.
        if created_keys:
            try:
                from trust_voice_attachments import _get_attachment_client, TRUST_VOICE_BUCKET
                client = _get_attachment_client()
                for key in created_keys:
                    try:
                        client.delete_object(Bucket=TRUST_VOICE_BUCKET, Key=key)
                    except Exception:
                        log.warning('Inbound Circle attachment cleanup failed')
            except Exception:
                log.warning('Inbound Circle R2 cleanup client unavailable')
        raise
    return uid,mail_id

@router.get('/circle-email/inbound/health')
def inbound_health():
    return {'status':'ok','version':INBOUND_VERSION,
            'enabled':inbound_enabled(),
            'configuration':'ready' if inbound_configured() else 'not_configured',
            'receiving_key_mode':('dedicated' if os.getenv('VERIFYD_CIRCLE_EMAIL_RECEIVING_API_KEY','').strip() else 'shared'),
            'provider_read_permission':'not_verified_by_health',
            'routing':'handle_prefixed_opaque_reply_alias_legacy_compatible',
            'public_handles':'all_active_verified_accounts_when_enabled',
            'public_handle_enabled':os.getenv('VERIFYD_CIRCLE_EMAIL_PUBLIC_HANDLE_ENABLED','').strip().lower() in ('1','true','yes','on'),
            'external_attachments':('private_r2_download_with_unscanned_ack_required'
                if inbound_attachments_enabled() else 'withheld_until_attachment_validation'),
            'external_attachment_max_files':3,
            'external_attachment_max_size_mb':8}


@router.post('/circle-email/webhooks/resend-inbound')
async def resend_inbound_webhook(request:Request):
    if not inbound_enabled():
        raise HTTPException(status_code=503,detail={'error':'inbound_not_enabled'})
    if not inbound_configured():
        raise HTTPException(status_code=503,detail={'error':'inbound_not_configured'})
    raw=await request.body()
    if len(raw)>65536:
        raise HTTPException(status_code=413,detail={'error':'webhook_too_large'})
    verify_resend_webhook(raw, request.headers)
    try:
        event=json.loads(raw)
    except (ValueError, UnicodeError):
        raise HTTPException(status_code=400,detail={'error':'invalid_event'})
    if not isinstance(event,dict) or event.get('type')!='email.received':
        return {'ok':True,'ignored':True}
    meta=event.get('data') or {}
    if not isinstance(meta,dict):
        raise HTTPException(status_code=400,detail={'error':'invalid_event'})
    provider_id=str(meta.get('email_id') or '').strip()
    if not _PROVIDER_ID.fullmatch(provider_id):
        raise HTTPException(status_code=400,detail={'error':'invalid_event'})
    received=await asyncio.to_thread(_fetch_received_email,provider_id)
    user_id,mail_id=await asyncio.to_thread(_store_received,received,provider_id)
    if user_id and mail_id:
        try:
            await _send_to_identity(user_id,{'type':'circle_mail_received','mail_id':mail_id})
        except Exception:
            log.warning('Inbound mail realtime notification unavailable',exc_info=True)
    # Never leak whether a reply token, recipient, or user exists.
    return {'ok':True}
