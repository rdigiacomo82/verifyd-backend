"""School-scoped, parent-authenticated multi-child handle requests.

IMPORTANT: Intake does NOT provision a child identity, enable /voice calling,
create guardian-child authority, or guarantee handle ownership. A selected
handle remains a proposal until the global identity claim workflow completes.
"""
from __future__ import annotations
import html
import re
import threading
import uuid
from typing import Literal, Optional
from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field
from database import get_db
from emailer import _send, FROM_NAME, FROM_ADDRESS
from trust_voice_identity import _validate_handle
from trust_voice_school_safety_training import _actor, _audit
from trust_voice_school_family_portal import _family_schema
from trust_voice_school_safety_registration import _school_admin

router=APIRouter(prefix="/trust-voice/school-safety",tags=["Family Handle Enrollment"])
_lock=threading.Lock()
_ready=False

def _schema():
    global _ready
    _family_schema()
    if _ready:return
    with _lock:
        if _ready:return
        with get_db() as conn:
            cur=conn.cursor()
            cur.execute("""CREATE TABLE IF NOT EXISTS verifyd_safety_family_intake(
              request_id TEXT PRIMARY KEY,
              school_id TEXT NOT NULL REFERENCES verifyd_safety_training_schools(school_id),
              guardian_id TEXT NOT NULL REFERENCES verifyd_identities(id),
              suggested_handle TEXT NOT NULL,
              handle_lower TEXT NOT NULL,
              child_label TEXT NOT NULL,
              relationship TEXT NOT NULL CHECK(relationship IN ('parent','legal_guardian','authorized_caregiver')),
              status TEXT NOT NULL DEFAULT 'pending'
                CHECK(status IN ('pending','reviewed','cancelled')),
              email_status TEXT NOT NULL DEFAULT 'pending',
              created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
              reviewed_by TEXT REFERENCES verifyd_identities(id),
              reviewed_at TIMESTAMPTZ
            )""")
            cur.execute("""CREATE UNIQUE INDEX IF NOT EXISTS safety_unique_pending_handle_proposal
              ON verifyd_safety_family_intake(handle_lower) WHERE status='pending'""")
            cur.execute("""CREATE INDEX IF NOT EXISTS safety_family_intake_scope
              ON verifyd_safety_family_intake(school_id,status,created_at DESC)""")
        _ready=True

class ChildChoice(BaseModel):
    handle: str=Field(min_length=3,max_length=25)
    display_label: str=Field(min_length=1,max_length=60)

class Intake(BaseModel):
    school_id: str=Field(min_length=8,max_length=100)
    relationship: Literal['parent','legal_guardian','authorized_caregiver']='parent'
    children: list[ChildChoice]=Field(min_length=1,max_length=5)
    parental_authority_acknowledged: bool=False

class Review(BaseModel):
    decision: Literal['reviewed','cancelled']

def _availability(cur,lower):
    cur.execute("SELECT 1 FROM verifyd_identities WHERE handle_lower=%s",(lower,))
    if cur.fetchone(): return False
    cur.execute("SELECT 1 FROM verifyd_safety_family_intake WHERE handle_lower=%s AND status='pending'",(lower,))
    return not bool(cur.fetchone())

@router.get("/family/onboarding/handle-availability/{handle}")
def availability(handle:str,authorization:Optional[str]=Header(default=None)):
    _actor(authorization);_schema()
    _,lower=_validate_handle(handle)
    with get_db() as conn:
        cur=conn.cursor()
        free=_availability(cur,lower)
    return {'ok':True,'handle':'@'+lower,'available':free,
            'note':'Availability is provisional until an independently verified global /voice identity claim.'}

@router.post("/family/onboarding/intake")
def submit(payload:Intake,authorization:Optional[str]=Header(default=None)):
    guardian=_actor(authorization);_schema()
    if not payload.parental_authority_acknowledged:
        raise HTTPException(422,detail={'error':'parental_authority_confirmation_required'})
    gid=str(guardian['id'])
    normalized=[]
    for child in payload.children:
        raw,lower=_validate_handle(child.handle)
        label=' '.join(child.display_label.strip().split())
        if not label:raise HTTPException(422,detail={'error':'child_label_required'})
        normalized.append((raw,lower,label))
    if len({x[1] for x in normalized})!=len(normalized):
        raise HTTPException(422,detail={'error':'duplicate_child_handles'})
    with get_db() as conn:
        cur=conn.cursor()
        cur.execute("""SELECT 1 FROM verifyd_safety_school_applications
             WHERE school_id=%s AND status='approved'""",(payload.school_id,))
        if not cur.fetchone():raise HTTPException(404,detail={'error':'school_not_open_for_registration'})
        cur.execute("SELECT email_lower FROM verifyd_identities WHERE id=%s AND status='active'",(gid,))
        account=cur.fetchone()
        if not account or not account['email_lower']:
            raise HTTPException(403,detail={'error':'verified_guardian_email_required'})
        cur.execute("""SELECT count(*) AS n FROM verifyd_safety_family_intake
           WHERE guardian_id=%s AND created_at>NOW()-INTERVAL '24 hours'""",(gid,))
        if cur.fetchone()['n']+len(normalized)>10:
            raise HTTPException(429,detail={'error':'family_intake_daily_limit'})
        # Transaction-scoped advisory locks protect competing intake claims.
        for _,lower,_ in sorted(normalized,key=lambda v:v[1]):
            cur.execute("SELECT pg_advisory_xact_lock(hashtext(%s))",('safety_handle_'+lower,))
            if not _availability(cur,lower):
                raise HTTPException(409,detail={'error':'handle_unavailable','handle':'@'+lower})
        ids=[]
        for raw,lower,label in normalized:
            rid='ssint_'+uuid.uuid4().hex
            cur.execute("""INSERT INTO verifyd_safety_family_intake
               (request_id,school_id,guardian_id,suggested_handle,handle_lower,child_label,relationship)
               VALUES(%s,%s,%s,%s,%s,%s,%s)""",
               (rid,payload.school_id,gid,raw,lower,label,payload.relationship))
            _audit(cur,payload.school_id,gid,'family_handle_intake_submitted',rid)
            ids.append((rid,raw))
        email=account['email_lower']
    # Each receipt is addressed only to the authenticated guardian's registered email.
    # Do not include children's legal names, addresses, student IDs or other details.
    receipts=[]
    for rid,handle in ids:
        success=_send({
           'from':f'{FROM_NAME} <{FROM_ADDRESS}>',
           'to':[email],
           'subject':'VeriFYD Safety — child registration request received',
           'html':('<p>Your school enrollment request for proposed handle '
              +html.escape('@'+handle)+' was received.</p>'
              '<p>This does not create a child /voice account, grant school access, '
              'or activate calls, messages, or emergency alerts. '
              'The school must independently review family authority.</p>')
        })
        with get_db() as conn:
            conn.cursor().execute("UPDATE verifyd_safety_family_intake SET email_status=%s WHERE request_id=%s",
                                  ('sent' if success else 'failed',rid))
        receipts.append({'request_id':rid,'requested_handle':'@'+handle,
                         'confirmation_email_sent':success,'status':'pending'})
    return {'ok':True,'children':receipts,'guardian_email_used':True,
            'accounts_created':False,'school_access_granted':False}

@router.get("/family/onboarding/mine")
def mine(authorization:Optional[str]=Header(default=None)):
    guardian=_actor(authorization);_schema()
    with get_db() as conn:
        cur=conn.cursor()
        cur.execute("""SELECT r.request_id,r.school_id,s.school_name,
             r.suggested_handle,r.child_label,r.status,r.email_status,r.created_at
             FROM verifyd_safety_family_intake r
             JOIN verifyd_safety_training_schools s USING(school_id)
             WHERE r.guardian_id=%s ORDER BY r.created_at DESC LIMIT 50""",(str(guardian['id']),))
        rows=[dict(x) for x in cur.fetchall()]
    return {'ok':True,'requests':rows}

@router.get("/schools/{school_id}/family/onboarding")
def school_requests(school_id:str,authorization:Optional[str]=Header(default=None)):
    actor=_actor(authorization);_schema()
    with get_db() as conn:
        cur=conn.cursor();_school_admin(cur,school_id,str(actor['id']))
        cur.execute("""SELECT r.request_id,r.suggested_handle,r.child_label,r.relationship,
             r.status,r.created_at,g.handle AS guardian_handle,g.display_name AS guardian_name,
             child.handle AS existing_child_handle
             FROM verifyd_safety_family_intake r
             JOIN verifyd_identities g ON g.id=r.guardian_id
             LEFT JOIN verifyd_identities child ON child.handle_lower=r.handle_lower
             WHERE r.school_id=%s ORDER BY r.created_at DESC LIMIT 150""",(school_id,))
        rows=[dict(x) for x in cur.fetchall()]
    return {'ok':True,'requests':rows,'note':'Proposed handles are not active /voice identities.'}

@router.post("/schools/{school_id}/family/onboarding/{request_id}/review")
def review(school_id:str,request_id:str,payload:Review,authorization:Optional[str]=Header(default=None)):
    actor=_actor(authorization);_schema()
    with get_db() as conn:
        cur=conn.cursor();_school_admin(cur,school_id,str(actor['id']))
        cur.execute("""SELECT status,guardian_id FROM verifyd_safety_family_intake
            WHERE school_id=%s AND request_id=%s FOR UPDATE""",(school_id,request_id))
        row=cur.fetchone()
        if not row:raise HTTPException(404,detail={'error':'intake_not_found'})
        if row['guardian_id']==str(actor['id']):
            raise HTTPException(403,detail={'error':'self_review_disallowed'})
        if row['status']!='pending':
            raise HTTPException(409,detail={'error':'request_already_reviewed'})
        cur.execute("""UPDATE verifyd_safety_family_intake
          SET status=%s,reviewed_by=%s,reviewed_at=NOW()
          WHERE request_id=%s""",(payload.decision,str(actor['id']),request_id))
        _audit(cur,school_id,str(actor['id']),'family_handle_intake_'+payload.decision,request_id)
    return {'ok':True,'status':payload.decision,'child_account_activated':False}
