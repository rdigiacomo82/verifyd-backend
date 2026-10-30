"""VeriFYD Circle Email Phase 1: private @handle mailbox and gated external outbound.
No unsolicited/inbound external-email claims, no fake AI certification. The separate
attachment API uses existing private Trust Voice R2 storage. Feature default off.
"""
from __future__ import annotations
import base64
import hashlib
import io
import os
import re
import tempfile
import uuid
from datetime import datetime, timezone
from html import escape
from email.utils import parseaddr
from typing import Literal, Optional

from fastapi import APIRouter, BackgroundTasks, File, Header, HTTPException, Query, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from database import get_db, is_valid_email
from trust_voice_messages import _identity_from_bearer, _message_permission_decision
from trust_voice_signaling import _identity_by_handle, _normalize_handle, _send_to_identity
from trust_voice_blocking import identities_blocked
from trust_voice_attachment_validation import validate_attachment_content, AttachmentContentError
from trust_voice_attachments import _get_attachment_client, _storage_configured, TRUST_VOICE_BUCKET

router = APIRouter(tags=['VeriFYD Circle Email'])
VERSION='0.3.2'
MAX_FILE_BYTES=8*1024*1024
ALLOWED_FILES={'.pdf','.txt','.docx','.xlsx','.png','.jpg','.jpeg','.webp','.csv','.pptx','.mp3','.wav'}
# Resend has a higher provider maximum, but limit raw Circle attachments to 16 MiB total.
MAX_EMAIL_TOTAL_BYTES=16*1024*1024
_schema_ready=False

def enabled(name): return os.getenv(name,'').strip().lower() in ('1','true','on','yes')
def feature():
    if not enabled('VERIFYD_CIRCLE_EMAIL_ENABLED'): raise HTTPException(503,detail={'error':'email_not_enabled','message':'Circle Email is not enabled yet.'})
def utc(): return datetime.now(timezone.utc).isoformat()
def fail(code,msg,status=400): raise HTTPException(status,detail={'error':code,'message':msg})

def _has_mail_content(subject:str, body:str, file_ids:list[str]) -> bool:
    """A subject, message body or one attached file makes a valid email."""
    return bool(str(subject or '').strip() or str(body or '').strip() or file_ids)

def _personal_from_header(actor:dict, inbound_reply_active:bool) -> str:
    """Show the verified Circle handle, not the generic application name.

    Keep the verified mailbox address from existing Render configuration; the
    display name is from an authenticated identity, never untrusted compose text.
    """
    handle = str(actor.get('handle') or actor.get('handle_lower') or '').strip().lstrip('@')
    if not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]{2,23}', handle):
        fail('invalid_sender_handle','The sender handle is not valid.',503)
    configured = os.getenv('VERIFYD_CIRCLE_EMAIL_FROM','').strip()
    from_address = (parseaddr(configured)[1].strip() if configured else
                    ('mail@vfvid.com' if inbound_reply_active else 'noreply@vfvid.com'))
    if not is_valid_email(from_address) or any(c in from_address for c in '\r\n'):
        fail('invalid_sender_address','The verified sending address is not configured.',503)
    return f'"@{handle}" <{from_address}>'

def schema():
    global _schema_ready
    if _schema_ready: return
    with get_db() as db:
        c=db.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS verifyd_circle_email_messages(
         mail_id TEXT PRIMARY KEY, thread_id TEXT NOT NULL, owner_id TEXT NOT NULL,
         sender_id TEXT NOT NULL, sender_handle TEXT NOT NULL, recipient TEXT NOT NULL,
         subject TEXT NOT NULL, body TEXT NOT NULL, folder TEXT NOT NULL,
         is_read BOOLEAN NOT NULL DEFAULT FALSE, starred BOOLEAN NOT NULL DEFAULT FALSE,
         spam_flag BOOLEAN NOT NULL DEFAULT FALSE, deleted_at TEXT,
         delivery TEXT NOT NULL DEFAULT 'delivered', verdict TEXT NOT NULL,
         risk_notes TEXT NOT NULL DEFAULT '', created_at TEXT NOT NULL)
        ''')
        c.execute('''CREATE INDEX IF NOT EXISTS idx_circle_email_owner_folder
          ON verifyd_circle_email_messages(owner_id,folder,created_at DESC)''')
        c.execute('''CREATE INDEX IF NOT EXISTS idx_circle_email_thread
          ON verifyd_circle_email_messages(owner_id,thread_id,created_at DESC)''')
        c.execute('''CREATE TABLE IF NOT EXISTS verifyd_circle_email_files (
          file_id TEXT PRIMARY KEY, owner_id TEXT NOT NULL, mail_id TEXT,
          filename TEXT NOT NULL, content_type TEXT NOT NULL, storage_key TEXT NOT NULL,
          byte_size BIGINT NOT NULL, sha256 TEXT NOT NULL, created_at TEXT NOT NULL,
          source TEXT NOT NULL DEFAULT 'internal',
          scan_status TEXT NOT NULL DEFAULT 'not_scanned')
        ''')
        c.execute("ALTER TABLE verifyd_circle_email_files ADD COLUMN IF NOT EXISTS source TEXT NOT NULL DEFAULT 'internal'")
        c.execute("ALTER TABLE verifyd_circle_email_files ADD COLUMN IF NOT EXISTS scan_status TEXT NOT NULL DEFAULT 'not_scanned'")
        c.execute('''CREATE INDEX IF NOT EXISTS idx_circle_email_files_mail
          ON verifyd_circle_email_files(mail_id)''')
    _schema_ready=True

class MailSend(BaseModel):
    to: str=Field(min_length=3,max_length=254)
    subject: str=Field(default='',max_length=200)
    body: str=Field(default='',max_length=12000)
    thread_id: Optional[str]=Field(default=None,max_length=100)
    file_ids: list[str]=Field(default_factory=list,max_length=3)
class MailUpdate(BaseModel):
    action: Literal['read','unread','star','unstar','spam','not_spam','trash','restore']


def risk(sender_internal:bool,subject:str,body:str):
    text=(subject+' '+body).lower()
    suspicious=[]
    if re.search(r'\b(?:urgent(?:ly)?|act now|verify your account|account suspended|wire transfer|gift card|password reset|crypto wallet)\b',text): suspicious.append('Pressure or credential/payment-related language')
    if re.search(r'https?://(?:\d{1,3}\.){3}\d{1,3}(?:[:/]|$)',text): suspicious.append('Link uses a numeric IP address')
    if re.search(r'\b(?:bit\.ly|tinyurl\.com|t\.co)/',text): suspicious.append('Shortened link: verify destination')
    if re.search(r'(?i)\b(?:password|one.time code|verification code|social security number)\b',text) and re.search(r'\b(?:send|share|reply|provide)\b',text): suspicious.append('Requests sensitive information')
    if suspicious: return ('suspicious','; '.join(suspicious))
    if sender_internal: return ('lower_risk','Authenticated VeriFYD identity; content was not malware scanned or AI-certified')
    return ('uncertain','External identity not independently verified; content not analyzed')

def public_mail(row:dict,with_body=False):
    d={k:row[k] for k in ('mail_id','thread_id','sender_handle','recipient','subject','folder','is_read','starred','spam_flag','delivery','verdict','risk_notes','created_at')}
    if with_body: d['body']=row['body']
    return d

def files_for(db,mail_ids:list[str]):
    if not mail_ids:return {}
    c=db.cursor()
    # Legacy files remain attached to the recipient copy. Expose the same
    # authenticated sender-owned file in the matching Sent copy, never in
    # an unrelated conversation that happens to share a thread identifier.
    c.execute('''SELECT m.mail_id AS visible_mail_id, f.file_id,f.filename,
        f.content_type,f.byte_size,f.sha256,f.source,f.scan_status
      FROM verifyd_circle_email_messages m
      JOIN verifyd_circle_email_files f ON
        f.mail_id=m.mail_id
        OR (m.folder='sent' AND f.owner_id=m.owner_id
            AND EXISTS (SELECT 1 FROM verifyd_circle_email_messages recipient
              WHERE recipient.mail_id=f.mail_id AND recipient.thread_id=m.thread_id
                AND recipient.sender_id=m.sender_id
                AND recipient.created_at=m.created_at))
      WHERE m.mail_id=ANY(%s)''',(mail_ids,))
    out={}
    for row in c.fetchall() or []:
        item=dict(row)
        target=item.pop('visible_mail_id')
        item['malware_scanned']=item.pop('scan_status','not_scanned')=='clean'
        item['security_notice']=('External attachment has not been malware scanned. Download only if you trust the sender.'
                                 if item.get('source')=='external' and not item['malware_scanned'] else '')
        out.setdefault(target,[]).append(item)
    return out

@router.get('/circle-email/health')
def email_health():
    return {'status':'ok','version':VERSION,'enabled':enabled('VERIFYD_CIRCLE_EMAIL_ENABLED'),
      'internal_handle_mail':'supported','external_outbound':enabled('VERIFYD_CIRCLE_EMAIL_EXTERNAL_ENABLED'),
      'external_inbound':'enabled' if enabled('VERIFYD_CIRCLE_EMAIL_INBOUND_ENABLED') else 'not_configured','private_attachments':_storage_configured(),
      'external_attachment_send':_storage_configured() and enabled('VERIFYD_CIRCLE_EMAIL_EXTERNAL_ENABLED'),
      'external_attachment_receive':_storage_configured() and enabled('VERIFYD_CIRCLE_EMAIL_INBOUND_ENABLED') and enabled('VERIFYD_CIRCLE_EMAIL_INBOUND_ATTACHMENTS_ENABLED'),
      'public_handle_email':'dynamic_all_verified_accounts_gated',
      'risk_classification':'rule_signals_not_ai_certification','attachment_malware_scanning':'not_available'}

@router.get('/circle-email/my-address')
def my_address(authorization:Optional[str]=Header(default=None)):
    """Return the current account's computed short receiving address."""
    feature();actor=_identity_from_bearer(authorization)
    handle=str(actor.get('handle_lower') or actor.get('handle') or '').strip().lower()
    from trust_voice_circle_email_inbound import reply_domain
    from trust_voice_circle_email_public import public_address_for_handle
    available=(public_address_for_handle(handle,reply_domain())
               if enabled('VERIFYD_CIRCLE_EMAIL_INBOUND_ENABLED')
               and actor.get('email_verified') and actor.get('status')=='active'
               else None)
    return {'internal_handle':'@'+handle,'public_email_address':available,
            'public_email_enabled':bool(available)}


@router.get('/circle-email/messages')
def inbox(folder:Literal['inbox','sent','spam','trash','starred']='inbox',q:str=Query(default='',max_length=100),
          authorization:Optional[str]=Header(default=None)):
    feature();identity=_identity_from_bearer(authorization);schema()
    uid=str(identity['id']);like=f'%{q}%' if q else '%'
    with get_db() as db:
        c=db.cursor()
        if folder=='starred':
            c.execute('''SELECT * FROM verifyd_circle_email_messages WHERE owner_id=%s AND starred=TRUE
                AND deleted_at IS NULL AND (subject ILIKE %s OR sender_handle ILIKE %s OR body ILIKE %s)
                ORDER BY created_at DESC LIMIT 150''',(uid,like,like,like))
        else:
            c.execute('''SELECT * FROM verifyd_circle_email_messages WHERE owner_id=%s AND folder=%s
                AND (subject ILIKE %s OR sender_handle ILIKE %s OR body ILIKE %s)
                ORDER BY created_at DESC LIMIT 150''',(uid,folder,like,like,like))
        rows=[dict(x) for x in c.fetchall() or []]; a=files_for(db,[x['mail_id'] for x in rows]);
        c.execute("SELECT COUNT(*) AS n FROM verifyd_circle_email_messages WHERE owner_id=%s AND folder='inbox' AND is_read=FALSE",(uid,));unread=int(c.fetchone()['n'])
    return {'messages':[{**public_mail(x), 'preview':x['body'][:180], 'attachments':a.get(x['mail_id'],[])} for x in rows], 'unread':unread,'folder':folder}

@router.get('/circle-email/messages/{mail_id}')
def read(mail_id:str,authorization:Optional[str]=Header(default=None)):
    feature();identity=_identity_from_bearer(authorization);schema();uid=str(identity['id'])
    with get_db() as db:
        c=db.cursor();c.execute('''SELECT * FROM verifyd_circle_email_messages WHERE mail_id=%s AND owner_id=%s''',(mail_id,uid)); row=c.fetchone()
        if not row:fail('not_found','Email not found.',404)
        row=dict(row)
        if row['folder']=='inbox' and not row['is_read']:
            c.execute('UPDATE verifyd_circle_email_messages SET is_read=TRUE WHERE mail_id=%s',(mail_id,));row['is_read']=True
        attachments=files_for(db,[mail_id]).get(mail_id,[])
    return {'message':{**public_mail(row,True),'attachments':attachments}}

@router.post('/circle-email/messages/{mail_id}/action')
def action(mail_id:str,payload:MailUpdate,authorization:Optional[str]=Header(default=None)):
    feature();identity=_identity_from_bearer(authorization);schema();uid=str(identity['id'])
    with get_db() as db:
        c=db.cursor();c.execute('SELECT * FROM verifyd_circle_email_messages WHERE mail_id=%s AND owner_id=%s FOR UPDATE',(mail_id,uid));row=c.fetchone()
        if not row:fail('not_found','Email not found.',404)
        row=dict(row);a=payload.action
        if a in ('read','unread'):c.execute('UPDATE verifyd_circle_email_messages SET is_read=%s WHERE mail_id=%s',(a=='read',mail_id))
        elif a in ('star','unstar'):c.execute('UPDATE verifyd_circle_email_messages SET starred=%s WHERE mail_id=%s',(a=='star',mail_id))
        elif a=='spam': c.execute("UPDATE verifyd_circle_email_messages SET folder='spam',spam_flag=TRUE WHERE mail_id=%s",(mail_id,))
        elif a=='not_spam':c.execute("UPDATE verifyd_circle_email_messages SET folder='inbox',spam_flag=FALSE WHERE mail_id=%s",(mail_id,))
        elif a=='trash':c.execute("UPDATE verifyd_circle_email_messages SET folder='trash',deleted_at=%s WHERE mail_id=%s",(utc(),mail_id))
        elif a=='restore': c.execute("UPDATE verifyd_circle_email_messages SET folder=%s,deleted_at=NULL WHERE mail_id=%s",('sent' if row['folder']=='sent' or row['owner_id']==row['sender_id'] else 'inbox',mail_id))
    return {'ok':True}

@router.post('/circle-email/files')
async def upload_file(file:UploadFile=File(...),authorization:Optional[str]=Header(default=None)):
    feature();identity=_identity_from_bearer(authorization);schema()
    if not _storage_configured():fail('storage_unavailable','Private file storage is not configured.',503)
    name=os.path.basename((file.filename or '').strip())[:140];ext=os.path.splitext(name)[1].lower()
    if ext not in ALLOWED_FILES or not name:fail('file_type_not_allowed','This file type is not supported.')
    payload=await file.read(MAX_FILE_BYTES+1)
    if not payload or len(payload)>MAX_FILE_BYTES:fail('file_size','Attach a file smaller than 8 MB.')
    with tempfile.NamedTemporaryFile(suffix=ext) as tmp:
        tmp.write(payload);tmp.flush()
        try:metadata=validate_attachment_content(tmp.name,extension=ext,supplied_content_type=file.content_type or '')
        except AttachmentContentError:fail('file_content_invalid','File type and contents did not match.')
    user_id=str(identity['id']);fid='tvcfile_'+uuid.uuid4().hex
    key=f'circle-email/{user_id}/{fid}{ext}'
    ctype=(metadata.get('content_type') or file.content_type or 'application/octet-stream') if isinstance(metadata,dict) else 'application/octet-stream'
    try:_get_attachment_client().put_object(Bucket=TRUST_VOICE_BUCKET,Key=key,Body=payload,ContentType=ctype)
    except Exception:fail('file_storage_error','Could not store attachment.',503)
    try:
        with get_db() as db:
            c=db.cursor();c.execute('''INSERT INTO verifyd_circle_email_files
                 (file_id,owner_id,mail_id,filename,content_type,storage_key,byte_size,sha256,created_at)
                 VALUES (%s,%s,NULL,%s,%s,%s,%s,%s,%s)''',(fid,user_id,name,ctype,key,len(payload),hashlib.sha256(payload).hexdigest(),utc()))
    except Exception:
        try:_get_attachment_client().delete_object(Bucket=TRUST_VOICE_BUCKET,Key=key)
        except Exception:pass
        raise
    return {'file_id':fid,'filename':name,'size':len(payload)}

@router.get('/circle-email/files/{file_id}/download')
def download_file(file_id:str,authorization:Optional[str]=Header(default=None),
                  acknowledge_unscanned:bool=Query(default=False)):
    feature();identity=_identity_from_bearer(authorization);schema();uid=str(identity['id'])
    with get_db() as db:
        c=db.cursor();c.execute('''SELECT f.* FROM verifyd_circle_email_files f
          LEFT JOIN verifyd_circle_email_messages m ON f.mail_id=m.mail_id
          WHERE f.file_id=%s AND (f.owner_id=%s OR m.owner_id=%s) LIMIT 1''',(file_id,uid,uid))
        r=c.fetchone()
    if not r:fail('file_not_found','Attachment not found.',404)
    r=dict(r)
    if r['mail_id'] is None and r['owner_id']!=uid:fail('file_not_found','Attachment not found.',404)
    if r.get('source')=='external' and r.get('scan_status')!='clean' and not acknowledge_unscanned:
        fail('attachment_confirmation_required',
             'External attachment is not malware scanned. Confirm before downloading.',409)
    try:resp=_get_attachment_client().get_object(Bucket=TRUST_VOICE_BUCKET,Key=r['storage_key'])
    except Exception:fail('file_unavailable','Attachment unavailable.',503)
    safe=re.sub(r'[^A-Za-z0-9_. -]','_',r['filename']).replace('"','_')
    return StreamingResponse(resp['Body'].iter_chunks(chunk_size=65536),media_type='application/octet-stream',headers={
       'Content-Disposition':f'attachment; filename="{safe}"','Cache-Control':'private, no-store',
       'X-Content-Type-Options':'nosniff','X-VeriFYD-Malware-Scanned':('true' if r.get('scan_status')=='clean' else 'false')})

@router.post('/circle-email/send')
async def send(payload:MailSend,tasks:BackgroundTasks,authorization:Optional[str]=Header(default=None)):
    feature();actor=_identity_from_bearer(authorization);schema();uid=str(actor['id'])
    to=payload.to.strip()
    supplied_subject=payload.subject.strip()
    subject=supplied_subject or '(no subject)'
    body=payload.body.strip()
    if not _has_mail_content(supplied_subject,body,payload.file_ids):
        fail('empty_message','Enter a subject, write a message, or attach a file.')
    internal=to.startswith('@') or ('@' not in to)
    if internal:
        target=_identity_by_handle(_normalize_handle(to))
        if not target:fail('recipient_not_found','That VeriFYD handle was not found.',404)
        tid=str(target['id']);
        if tid==uid:fail('recipient_self','Choose another VeriFYD user.')
        if identities_blocked(uid,tid):fail('recipient_unavailable','Cannot send to this user.',403)
        allowed,_reason=_message_permission_decision(actor,target)
        if not allowed:fail('recipient_unavailable','This user is not accepting messages from your account.',403)
        recipient='@'+str(target.get('handle_lower') or '').strip()
    else:
        if not enabled('VERIFYD_CIRCLE_EMAIL_EXTERNAL_ENABLED'):
            fail('external_not_ready','External email delivery is not enabled yet. You can send to VeriFYD @handles now.',503)
        # Use the same persisted address verified during identity registration.
        sender_email=str(actor.get('email_lower') or actor.get('email') or '').strip().lower()
        if not actor.get('email_verified') or not is_valid_email(sender_email):
            fail('verified_email_required','Verify your account email before sending to an external address.',403)
        # Never enable inbound aliases before receiving DNS and signed webhooks
        # are configured; otherwise external replies would disappear.
        inbound_reply_active=enabled('VERIFYD_CIRCLE_EMAIL_INBOUND_ENABLED')
        if inbound_reply_active:
            from trust_voice_circle_email_inbound import inbound_configured,ensure_inbound_schema
            if not inbound_configured():
                fail('inbound_not_configured','The VeriFYD reply mailbox is still being configured.',503)
            ensure_inbound_schema()
        if not re.fullmatch(r'[^\s@<>]+@[^\s@<>]+\.[^\s@<>]+',to):fail('recipient_invalid','Enter a valid email address.')
        recipient=to.lower();tid=None
    if len(set(payload.file_ids))!=len(payload.file_ids):fail('duplicate_file','An attachment was selected more than once.')
    if len(payload.file_ids)>3:fail('too_many_files','Attach no more than three files.')
    with get_db() as db:
        c=db.cursor();c.execute('''SELECT COUNT(*) AS n FROM verifyd_circle_email_messages WHERE sender_id=%s AND folder='sent'
          AND created_at::timestamptz > NOW() - INTERVAL '24 hours' ''',(uid,))
        if int(c.fetchone()['n']) >= (10 if not internal else 60):fail('mail_rate_limit','Daily email limit reached.',429)
        rows=[]
        if payload.file_ids:
            c.execute('''SELECT * FROM verifyd_circle_email_files WHERE file_id=ANY(%s)
             AND owner_id=%s AND mail_id IS NULL FOR UPDATE''',(payload.file_ids,uid))
            rows=[dict(x) for x in c.fetchall() or []]
            if len(rows)!=len(payload.file_ids):fail('file_missing','Upload or remove an expired attachment.')
            if sum(int(x['byte_size']) for x in rows)>MAX_EMAIL_TOTAL_BYTES:
                fail('attachment_total_limit','Attachments exceed the 16 MB combined limit.',413)
        thread=payload.thread_id if payload.thread_id and re.fullmatch(r'tvcemail_[a-f0-9]{32}',payload.thread_id) else 'tvcemail_'+uuid.uuid4().hex
        sent_id='tvcmail_'+uuid.uuid4().hex;now=utc();v,risk_note=risk(True,subject,body)
        if internal:
            recv_id='tvcmail_'+uuid.uuid4().hex;incoming_verdict,reason=risk(True,subject,body)
            for mid,owner,folder,read_flag,verdict,notes in [(sent_id,uid,'sent',True,v,risk_note),(recv_id,tid,'inbox',False,incoming_verdict,reason)]:
                c.execute('''INSERT INTO verifyd_circle_email_messages(mail_id,thread_id,owner_id,sender_id,sender_handle,recipient,
                    subject,body,folder,is_read,verdict,risk_notes,created_at)
                    VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)''',(mid,thread,owner,uid,'@'+str(actor.get('handle_lower') or actor.get('handle') or ''),recipient,subject,body,folder,read_flag,verdict,notes,now))
            for item in rows:
                # One mail attachment is linked by the recipient copy. Sender can still access as owner.
                c.execute('UPDATE verifyd_circle_email_files SET mail_id=%s WHERE file_id=%s',(recv_id,item['file_id']))
        else:
            c.execute('''INSERT INTO verifyd_circle_email_messages(mail_id,thread_id,owner_id,sender_id,sender_handle,recipient,
                subject,body,folder,is_read,delivery,verdict,risk_notes,created_at)
                VALUES(%s,%s,%s,%s,%s,%s,%s,%s,'sent',TRUE,'sending',%s,%s,%s)''',
                (sent_id,thread,uid,uid,'@'+str(actor.get('handle_lower') or actor.get('handle') or ''),recipient,subject,body,v,risk_note,now))
            for item in rows:
                c.execute('UPDATE verifyd_circle_email_files SET mail_id=%s WHERE file_id=%s',
                          (sent_id,item['file_id']))
            if inbound_reply_active:
                from trust_voice_circle_email_inbound import create_reply_route
                reply_address=create_reply_route(db,owner_id=uid,sent_mail_id=sent_id,
                       thread_id=thread,external_recipient=recipient,stamp=now,
                       sender_handle=str(actor.get('handle_lower') or actor.get('handle') or ''))
    if internal:
        try: tasks.add_task(_notify_recipient,tid,recv_id)
        except Exception:pass
        return {'ok':True,'mail_id':sent_id,'delivery':'delivered','recipient':recipient}
    # Outbound external mail goes through existing Resend account, fixed verified FROM,
    # never dynamically from a user-controlled address. Replies go to the verified email.
    try:
        import resend
        api_key=os.getenv('RESEND_API_KEY','')
        if not api_key:raise RuntimeError('Resend key unavailable')
        resend.api_key=api_key
        # User-facing From display is the authenticated handle, while the
        # actual sender remains our configured, verified Resend email address.
        from_address=_personal_from_header(actor,inbound_reply_active)
        if inbound_reply_active:
            # One global feature gate: every eligible registered handle
            # automatically gets a stable address, with no manual forwards.
            from trust_voice_circle_email_public import public_address_for_handle
            from trust_voice_circle_email_inbound import reply_domain
            public_reply=public_address_for_handle(
                str(actor.get('handle_lower') or actor.get('handle') or ''),
                reply_domain())
            reply_to=public_reply or reply_address
        else:
            reply_to=sender_email
        reply_note=('Reply to this message to reach the sender in VeriFYD Circle.'
                    if inbound_reply_active else
                    'Replies go to the sender\'s verified email address until VeriFYD inbound routing is enabled.')
        from trust_voice_circle_email_transport import prepare_outbound_attachments
        attachment_payload=prepare_outbound_attachments(rows)
        send_params={'from':from_address,'to':[recipient],
          'reply_to':reply_to,
          'subject':subject,'text':((body + '\n\n') if body else '') + reply_note}
        if attachment_payload:
            send_params['attachments']=attachment_payload
        result=resend.Emails.send(send_params)
        if not result:raise RuntimeError('No provider acceptance')
        state='accepted_by_provider'
    except Exception:
        state='failed'
    with get_db() as db:
        c=db.cursor();c.execute('UPDATE verifyd_circle_email_messages SET delivery=%s WHERE mail_id=%s AND owner_id=%s',(state,sent_id,uid))
        if state=='failed' and rows:
            # Preserve previously uploaded private files for a retry when the
            # provider rejects the message. The failed Sent entry must not
            # consume these file IDs permanently.
            c.execute('''UPDATE verifyd_circle_email_files SET mail_id=NULL
                  WHERE mail_id=%s AND owner_id=%s''',(sent_id,uid))
    if state=='failed':fail('delivery_failed','External provider could not accept this email. Please try again later.',503)
    return {'ok':True,'mail_id':sent_id,'delivery':state,'recipient':recipient,
            'notice':('External replies route to your VeriFYD Inbox.' if inbound_reply_active else 'Replies go to your verified email until inbound routing is enabled.')}

async def _notify_recipient(identity_id,mail_id):
    try:await _send_to_identity(identity_id,{'type':'circle_mail_received','mail_id':mail_id})
    except Exception:pass

# Independent signed Resend inbound webhook router; no main.py edits needed.
from trust_voice_circle_email_inbound import router as inbound_router
router.include_router(inbound_router)
