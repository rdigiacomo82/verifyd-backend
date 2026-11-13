"""VeriFYD Circle Therapy & Adaptive Practice — restricted professional/client pilot.

Isolated schema; no alterations to Classroom, messaging, Trust Circle, or local Studio.
Disabled by default.  This is not an EHR, medical record, clinical assessment,
or HIPAA compliance assertion.  Do not use with PHI until deployment, contracts,
retention, auditing, and health privacy controls have been separately reviewed.

VERIFYD_CIRCLE_THERAPY_PRACTICE_ENABLED=1
VERIFYD_CIRCLE_THERAPY_PROFESSIONAL_IDS=<manually vetted identity ids>
Only adult test accounts may accept pilot invitations.
"""
from __future__ import annotations

import os
import secrets
import threading
import uuid
from datetime import datetime, timezone, timedelta
from typing import Literal, Optional

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field

from database import get_db
from trust_voice_messages import _identity_from_bearer
from trust_voice_permissions import ensure_trust_circle_schema
from trust_voice_blocking import identities_blocked

router = APIRouter(prefix="/trust-voice/therapy-practice", tags=["Circle Therapy Practice Pilot"])
_READY = False
_LOCK = threading.Lock()
_ACTIVITY_IDS = frozenset(("compass", "story", "sequence", "social", "focus", "comfort"))
_TRUE = ("1", "true", "on", "yes")


def enabled():
    return os.getenv("VERIFYD_CIRCLE_THERAPY_PRACTICE_ENABLED", "0").strip().lower() in _TRUE


def _require():
    if not enabled():
        raise HTTPException(404, detail={"error": "therapy_practice_unavailable"})


def _identity(authorization):
    _require()
    account = _identity_from_bearer(authorization)
    if account.get("status") != "active":
        raise HTTPException(401, detail={"error": "session_required"})
    return str(account["id"])


def _is_professional(identity_id):
    return bool(identity_id and identity_id in {
        item.strip() for item in os.getenv("VERIFYD_CIRCLE_THERAPY_PROFESSIONAL_IDS", "").split(",")
        if item.strip()
    })


def _professional(identity_id):
    if not _is_professional(identity_id):
        raise HTTPException(403, detail={"error": "professional_approval_required"})


def _schema():
    global _READY
    if _READY:
        return
    with _LOCK:
        if _READY:
            return
        ensure_trust_circle_schema()
        with get_db() as conn:
            cur = conn.cursor()
            cur.execute("""CREATE TABLE IF NOT EXISTS verifyd_therapy_programs (
                program_id TEXT PRIMARY KEY,
                professional_id TEXT NOT NULL REFERENCES verifyd_identities(id) ON DELETE CASCADE,
                title TEXT NOT NULL, focus TEXT NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )""")
            cur.execute("""CREATE INDEX IF NOT EXISTS idx_therapy_program_owner
                ON verifyd_therapy_programs(professional_id,created_at DESC)""")
            cur.execute("""CREATE TABLE IF NOT EXISTS verifyd_therapy_invitations (
                invitation_id TEXT PRIMARY KEY,
                program_id TEXT NOT NULL REFERENCES verifyd_therapy_programs(program_id) ON DELETE CASCADE,
                professional_id TEXT NOT NULL REFERENCES verifyd_identities(id) ON DELETE CASCADE,
                client_id TEXT NOT NULL REFERENCES verifyd_identities(id) ON DELETE CASCADE,
                status TEXT NOT NULL DEFAULT 'pending' CHECK(status IN ('pending','accepted','declined','revoked')),
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                expires_at TIMESTAMPTZ NOT NULL DEFAULT NOW()+INTERVAL '14 days',
                responded_at TIMESTAMPTZ,
                UNIQUE(program_id,client_id)
            )""")
            cur.execute("""CREATE INDEX IF NOT EXISTS idx_therapy_invitee
                ON verifyd_therapy_invitations(client_id,status,expires_at)""")
            cur.execute("""CREATE TABLE IF NOT EXISTS verifyd_therapy_members (
                program_id TEXT NOT NULL REFERENCES verifyd_therapy_programs(program_id) ON DELETE CASCADE,
                client_id TEXT NOT NULL REFERENCES verifyd_identities(id) ON DELETE CASCADE,
                status TEXT NOT NULL DEFAULT 'active' CHECK(status IN ('active','removed')),
                joined_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                PRIMARY KEY(program_id,client_id)
            )""")
            cur.execute("""CREATE INDEX IF NOT EXISTS idx_therapy_client_program
                ON verifyd_therapy_members(client_id,status)""")
            cur.execute("""CREATE TABLE IF NOT EXISTS verifyd_therapy_tasks (
                task_id TEXT PRIMARY KEY,
                program_id TEXT NOT NULL REFERENCES verifyd_therapy_programs(program_id) ON DELETE CASCADE,
                author_id TEXT NOT NULL REFERENCES verifyd_identities(id) ON DELETE CASCADE,
                title TEXT NOT NULL, instructions TEXT NOT NULL,
                mode TEXT NOT NULL CHECK(mode IN ('studio','reflection','choices')),
                activities TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
                due_at TIMESTAMPTZ,
                status TEXT NOT NULL DEFAULT 'draft' CHECK(status IN ('draft','published')),
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )""")
            cur.execute("""CREATE TABLE IF NOT EXISTS verifyd_therapy_task_recipients (
                task_id TEXT NOT NULL REFERENCES verifyd_therapy_tasks(task_id) ON DELETE CASCADE,
                client_id TEXT NOT NULL REFERENCES verifyd_identities(id) ON DELETE CASCADE,
                PRIMARY KEY(task_id,client_id)
            )""")
            cur.execute("""CREATE INDEX IF NOT EXISTS idx_therapy_task_recipient
                ON verifyd_therapy_task_recipients(client_id,task_id)""")
            cur.execute("""CREATE TABLE IF NOT EXISTS verifyd_therapy_responses (
                task_id TEXT NOT NULL REFERENCES verifyd_therapy_tasks(task_id) ON DELETE CASCADE,
                client_id TEXT NOT NULL REFERENCES verifyd_identities(id) ON DELETE CASCADE,
                response TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL DEFAULT 'draft' CHECK(status IN ('draft','submitted','returned')),
                feedback TEXT NOT NULL DEFAULT '',
                submitted_at TIMESTAMPTZ,
                reviewed_at TIMESTAMPTZ,
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                PRIMARY KEY(task_id,client_id)
            )""")
        _READY = True


def _row(record):
    if record is None:
        return None
    result = dict(record)
    for k,v in result.items():
        if isinstance(v, datetime):
            result[k] = v.isoformat()
    return result


def _owner(cur, program_id, user):
    _professional(user)
    cur.execute("SELECT program_id,professional_id,title,focus FROM verifyd_therapy_programs WHERE program_id=%s AND professional_id=%s",(program_id,user))
    record=cur.fetchone()
    if not record:
        raise HTTPException(404,detail={"error":"practice_group_unavailable"})
    return record


def _membership(cur, program_id, user):
    cur.execute("""SELECT p.program_id,p.professional_id,m.status
        FROM verifyd_therapy_programs p
        LEFT JOIN verifyd_therapy_members m ON m.program_id=p.program_id AND m.client_id=%s
        WHERE p.program_id=%s""", (user,program_id))
    rec=cur.fetchone()
    if not rec or (rec["professional_id"] != user and rec["status"] != 'active'):
        raise HTTPException(404,detail={"error":"practice_group_unavailable"})
    # A deauthorized professional must not leave an accessible client portal.
    if not _is_professional(rec["professional_id"]):
        raise HTTPException(404,detail={"error":"practice_group_unavailable"})
    if rec["professional_id"] == user:
        _professional(user)
    return rec


def _visible_task(cur, program_id, task_id, user):
    cur.execute("""SELECT t.task_id,t.title,t.instructions,t.mode,t.activities,t.due_at,t.status,
                       t.created_at
        FROM verifyd_therapy_tasks t
        JOIN verifyd_therapy_task_recipients r ON r.task_id=t.task_id AND r.client_id=%s
        JOIN verifyd_therapy_members m ON m.program_id=t.program_id AND m.client_id=%s AND m.status='active'
        WHERE t.program_id=%s AND t.task_id=%s AND t.status='published'""",
        (user,user,program_id,task_id))
    record=cur.fetchone()
    if not record:
        raise HTTPException(404,detail={"error":"practice_task_unavailable"})
    return record


class ProgramCreate(BaseModel):
    title: str = Field(min_length=3,max_length=120)
    focus: str = Field(default="Communication & adaptive practice",min_length=3,max_length=120)

class InviteClients(BaseModel):
    client_ids: list[str] = Field(min_length=1,max_length=30)

class AcceptInvite(BaseModel):
    adult_attestation: bool = False

class PracticeTaskCreate(BaseModel):
    title: str = Field(min_length=3,max_length=160)
    instructions: str = Field(min_length=1,max_length=12000)
    mode: Literal['studio','reflection','choices'] = 'studio'
    activities: list[str] = Field(default_factory=lambda: ['compass'],max_length=6)
    recipient_ids: list[str] = Field(min_length=1,max_length=100)
    due_at: Optional[datetime] = None

class PracticeResponse(BaseModel):
    response: str = Field(default="",max_length=8000)
    action: Literal['save','submit'] = 'save'

class PracticeReview(BaseModel):
    feedback: str = Field(min_length=1,max_length=8000)


@router.get('/health')
def health(authorization: Optional[str]=Header(default=None)):
    user=_identity(authorization)
    return {"enabled":True,"service":"therapy_adaptive_practice_pilot",
            "professional_can_manage":_is_professional(user),
            "adult_only":True,"clinical_record_system":False,
            "recordings_uploaded":False,"medical_assessment":False}


@router.get('/contacts')
def professional_contacts(authorization:Optional[str]=Header(default=None)):
    user=_identity(authorization)
    _professional(user)
    _schema()
    with get_db() as conn:
        cur=conn.cursor()
        cur.execute("""SELECT t.member_identity_id AS identity_id,i.handle,t.display_label
            FROM verifyd_trust_circle_members t
            JOIN verifyd_identities i ON i.id=t.member_identity_id AND i.status='active'
            WHERE t.owner_identity_id=%s AND t.contact_type='person'
            ORDER BY t.sort_order,t.created_at LIMIT 200""",(user,))
        rows=[_row(r) for r in cur.fetchall() if r['identity_id'] != user]
    return {"contacts":[x for x in rows if not identities_blocked(user,x['identity_id'])]}


@router.get('/programs')
def programs(authorization:Optional[str]=Header(default=None)):
    user=_identity(authorization)
    _schema()
    with get_db() as conn:
        cur=conn.cursor()
        cur.execute("""SELECT p.program_id,p.title,p.focus,p.created_at,
            CASE WHEN p.professional_id=%s THEN 'professional' ELSE 'client' END AS my_role,
            p.professional_id
            FROM verifyd_therapy_programs p LEFT JOIN verifyd_therapy_members m
            ON m.program_id=p.program_id AND m.client_id=%s AND m.status='active'
            WHERE (p.professional_id=%s AND %s) OR m.client_id=%s
            ORDER BY p.created_at DESC LIMIT 100""",
            (user,user,user,_is_professional(user),user))
        rows=[_row(r) for r in cur.fetchall()]
    visible=[]
    for item in rows:
        if _is_professional(item.pop('professional_id', '')):
            visible.append(item)
    return {"programs":visible}


@router.post('/programs')
def create_program(payload:ProgramCreate,authorization:Optional[str]=Header(default=None)):
    user=_identity(authorization)
    _professional(user)
    _schema()
    program_id=str(uuid.uuid4())
    with get_db() as conn:
        cur=conn.cursor()
        cur.execute("SELECT count(*) AS n FROM verifyd_therapy_programs WHERE professional_id=%s",(user,))
        if cur.fetchone()['n'] >= 25:
            raise HTTPException(429,detail={"error":"practice_group_limit"})
        cur.execute("""INSERT INTO verifyd_therapy_programs(program_id,professional_id,title,focus)
            VALUES (%s,%s,%s,%s)""",(program_id,user,payload.title.strip(),payload.focus.strip()))
    return {"program_id":program_id,"title":payload.title,"focus":payload.focus}


@router.get('/programs/{program_id}/members')
def members(program_id:str,authorization:Optional[str]=Header(default=None)):
    user=_identity(authorization)
    _schema()
    with get_db() as conn:
        cur=conn.cursor()
        _owner(cur,program_id,user)
        cur.execute("""SELECT m.client_id,i.handle,m.status,m.joined_at FROM verifyd_therapy_members m
            JOIN verifyd_identities i ON i.id=m.client_id AND i.status='active'
            WHERE m.program_id=%s AND m.status='active' ORDER BY m.joined_at LIMIT 300""",(program_id,))
        rows=[_row(r) for r in cur.fetchall()]
    return {"members":rows}


@router.post('/programs/{program_id}/invitations')
def invite(program_id:str,payload:InviteClients,authorization:Optional[str]=Header(default=None)):
    user=_identity(authorization)
    _professional(user)
    _schema()
    if len(set(payload.client_ids)) != len(payload.client_ids) or any(not isinstance(x,str) or not (1<=len(x)<=128) for x in payload.client_ids):
        raise HTTPException(422,detail={"error":"invalid_invite_recipients"})
    if user in payload.client_ids:
        raise HTTPException(422,detail={"error":"cannot_invite_self"})
    for other in payload.client_ids:
        if identities_blocked(user,other):
            raise HTTPException(403,detail={"error":"recipient_blocked"})
    ids=[]
    with get_db() as conn:
        cur=conn.cursor()
        _owner(cur,program_id,user)
        cur.execute("""SELECT t.member_identity_id FROM verifyd_trust_circle_members t
            JOIN verifyd_identities i ON i.id=t.member_identity_id AND i.status='active'
            WHERE t.owner_identity_id=%s AND t.contact_type='person'
            AND t.member_identity_id=ANY(%s)""",(user,payload.client_ids))
        permitted={r['member_identity_id'] for r in cur.fetchall()}
        if len(permitted)!=len(payload.client_ids):
            raise HTTPException(403,detail={"error":"client_not_in_trust_circle"})
        cur.execute("""SELECT client_id FROM verifyd_therapy_members
            WHERE program_id=%s AND client_id=ANY(%s) AND status='active'""",(program_id,payload.client_ids))
        enrolled={r['client_id'] for r in cur.fetchall()}
        if enrolled:
            raise HTTPException(409,detail={"error":"client_already_enrolled"})
        for client_id in payload.client_ids:
            invite_id=str(uuid.uuid4())
            cur.execute("""INSERT INTO verifyd_therapy_invitations
                (invitation_id,program_id,professional_id,client_id,status,expires_at)
                VALUES (%s,%s,%s,%s,'pending',NOW()+INTERVAL '14 days')
                ON CONFLICT(program_id,client_id) DO UPDATE SET
                    invitation_id=EXCLUDED.invitation_id,status='pending',created_at=NOW(),
                    expires_at=EXCLUDED.expires_at,responded_at=NULL
                RETURNING invitation_id""",(invite_id,program_id,user,client_id))
            ids.append(cur.fetchone()['invitation_id'])
    return {"sent":len(ids),"invitation_ids":ids,"delivery":"private_in_app_inbox"}


@router.get('/invitations/mine')
def inbox(authorization:Optional[str]=Header(default=None)):
    user=_identity(authorization)
    _schema()
    with get_db() as conn:
        cur=conn.cursor()
        cur.execute("""SELECT v.invitation_id,v.program_id,p.title AS program_title,p.focus,
                       v.professional_id,i.handle AS professional_handle,v.expires_at
            FROM verifyd_therapy_invitations v
            JOIN verifyd_therapy_programs p ON p.program_id=v.program_id
            JOIN verifyd_identities i ON i.id=v.professional_id AND i.status='active'
            WHERE v.client_id=%s AND v.status='pending' AND v.expires_at>NOW()
            ORDER BY v.created_at DESC LIMIT 50""",(user,))
        rows=[_row(r) for r in cur.fetchall()]
    # A pending invitation from a professional whose pilot permission was
    # revoked is not actionable and must not remain in a client's inbox.
    visible=[]
    for item in rows:
        if _is_professional(item.pop('professional_id','')):
            visible.append(item)
    return {"invitations":visible}


@router.post('/invitations/{invitation_id}/accept')
def accept(invitation_id:str,payload:AcceptInvite,authorization:Optional[str]=Header(default=None)):
    user=_identity(authorization)
    if not payload.adult_attestation:
        raise HTTPException(422,detail={"error":"adult_test_consent_required"})
    _schema()
    with get_db() as conn:
        cur=conn.cursor()
        cur.execute("""SELECT v.program_id,v.professional_id,v.client_id,v.status,v.expires_at
            FROM verifyd_therapy_invitations v WHERE v.invitation_id=%s FOR UPDATE""",(invitation_id,))
        item=cur.fetchone()
        if not item or item['client_id'] != user:
            raise HTTPException(404,detail={"error":"invitation_unavailable"})
        if item['status'] != 'pending' or item['expires_at'] <= datetime.now(timezone.utc):
            raise HTTPException(409,detail={"error":"invitation_expired"})
        if not _is_professional(item['professional_id']) or identities_blocked(item['professional_id'],user):
            raise HTTPException(403,detail={"error":"professional_not_approved"})
        cur.execute("""INSERT INTO verifyd_therapy_members(program_id,client_id,status)
            VALUES (%s,%s,'active') ON CONFLICT(program_id,client_id)
            DO UPDATE SET status='active',joined_at=NOW()""",(item['program_id'],user))
        cur.execute("""UPDATE verifyd_therapy_invitations SET status='accepted',responded_at=NOW()
            WHERE invitation_id=%s""",(invitation_id,))
    return {"joined":True,"program_id":item['program_id'],"requires_second_approval":False}


@router.post('/invitations/{invitation_id}/decline')
def decline(invitation_id:str,authorization:Optional[str]=Header(default=None)):
    user=_identity(authorization)
    _schema()
    with get_db() as conn:
        cur=conn.cursor()
        cur.execute("""UPDATE verifyd_therapy_invitations SET status='declined',responded_at=NOW()
            WHERE invitation_id=%s AND client_id=%s AND status='pending'
            RETURNING invitation_id""",(invitation_id,user))
        if not cur.fetchone():
            raise HTTPException(404,detail={"error":"invitation_unavailable"})
    return {"declined":True}


@router.delete('/programs/{program_id}/members/{client_id}')
def remove_member(program_id:str,client_id:str,authorization:Optional[str]=Header(default=None)):
    user=_identity(authorization)
    _schema()
    with get_db() as conn:
        cur=conn.cursor()
        _owner(cur,program_id,user)
        cur.execute("""UPDATE verifyd_therapy_members SET status='removed'
            WHERE program_id=%s AND client_id=%s AND status='active' RETURNING client_id""",(program_id,client_id))
        if not cur.fetchone():
            raise HTTPException(404,detail={"error":"client_not_enrolled"})
        cur.execute("""UPDATE verifyd_therapy_invitations SET status='revoked',responded_at=NOW()
            WHERE program_id=%s AND client_id=%s AND status='pending'""",(program_id,client_id))
        # Preserve historical responses for the existing retention workflow, but
        # never silently re-grant access to old practice tasks if re-invited later.
        cur.execute("""DELETE FROM verifyd_therapy_task_recipients
            WHERE client_id=%s AND task_id IN
            (SELECT task_id FROM verifyd_therapy_tasks WHERE program_id=%s)""",
            (client_id,program_id))
    return {"removed":True,"note":"Client access revoked; retained submissions require a separate retention/deletion workflow."}


@router.post('/programs/{program_id}/tasks')
def create_task(program_id:str,payload:PracticeTaskCreate,authorization:Optional[str]=Header(default=None)):
    user=_identity(authorization)
    _professional(user)
    _schema()
    if len(set(payload.recipient_ids))!=len(payload.recipient_ids):
        raise HTTPException(422,detail={"error":"duplicate_recipients"})
    if len(set(payload.activities))!=len(payload.activities) or any(id not in _ACTIVITY_IDS for id in payload.activities):
        raise HTTPException(422,detail={"error":"invalid_studio_activity"})
    if payload.mode=='studio' and not payload.activities:
        raise HTTPException(422,detail={"error":"studio_activity_required"})
    task_id=str(uuid.uuid4())
    with get_db() as conn:
        cur=conn.cursor()
        _owner(cur,program_id,user)
        cur.execute("""SELECT client_id FROM verifyd_therapy_members WHERE program_id=%s
            AND client_id=ANY(%s) AND status='active'""",(program_id,payload.recipient_ids))
        active={r['client_id'] for r in cur.fetchall()}
        if len(active)!=len(payload.recipient_ids):
            raise HTTPException(422,detail={"error":"recipients_not_enrolled"})
        cur.execute("SELECT count(*) AS n FROM verifyd_therapy_tasks WHERE program_id=%s",(program_id,))
        if cur.fetchone()['n'] >= 500:
            raise HTTPException(429,detail={"error":"practice_task_limit"})
        cur.execute("""INSERT INTO verifyd_therapy_tasks
            (task_id,program_id,author_id,title,instructions,mode,activities,due_at)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s)""",
            (task_id,program_id,user,payload.title.strip(),payload.instructions.strip(),payload.mode,payload.activities,payload.due_at))
        for client_id in payload.recipient_ids:
            cur.execute("INSERT INTO verifyd_therapy_task_recipients(task_id,client_id) VALUES (%s,%s)",(task_id,client_id))
    return {"task_id":task_id,"status":"draft","recipient_count":len(payload.recipient_ids)}


@router.post('/programs/{program_id}/tasks/{task_id}/publish')
def publish_task(program_id:str,task_id:str,authorization:Optional[str]=Header(default=None)):
    user=_identity(authorization)
    _schema()
    with get_db() as conn:
        cur=conn.cursor()
        _owner(cur,program_id,user)
        cur.execute("""SELECT count(*) AS n FROM verifyd_therapy_task_recipients r
            JOIN verifyd_therapy_members m ON m.program_id=%s AND m.client_id=r.client_id AND m.status='active'
            WHERE r.task_id=%s""",(program_id,task_id))
        if cur.fetchone()['n'] < 1:
            raise HTTPException(409,detail={"error":"no_active_recipients"})
        cur.execute("""UPDATE verifyd_therapy_tasks SET status='published'
            WHERE program_id=%s AND task_id=%s AND status='draft' RETURNING task_id""",(program_id,task_id))
        if not cur.fetchone():
            raise HTTPException(409,detail={"error":"task_not_draft"})
    return {"status":"published","task_id":task_id}


@router.get('/programs/{program_id}/tasks')
def list_tasks(program_id:str,authorization:Optional[str]=Header(default=None)):
    user=_identity(authorization)
    _schema()
    with get_db() as conn:
        cur=conn.cursor()
        member=_membership(cur,program_id,user)
        if member['professional_id']==user:
            cur.execute("""SELECT task_id,title,instructions,mode,activities,due_at,status,created_at
                FROM verifyd_therapy_tasks WHERE program_id=%s ORDER BY created_at DESC LIMIT 200""",(program_id,))
        else:
            cur.execute("""SELECT t.task_id,t.title,t.instructions,t.mode,t.activities,t.due_at,t.status,t.created_at
                FROM verifyd_therapy_tasks t JOIN verifyd_therapy_task_recipients r ON r.task_id=t.task_id
                WHERE t.program_id=%s AND t.status='published' AND r.client_id=%s
                ORDER BY t.created_at DESC LIMIT 200""",(program_id,user))
        rows=[_row(r) for r in cur.fetchall()]
    return {"tasks":rows}


@router.get('/programs/{program_id}/tasks/{task_id}/my-response')
def my_response(program_id:str,task_id:str,authorization:Optional[str]=Header(default=None)):
    user=_identity(authorization)
    _schema()
    with get_db() as conn:
        cur=conn.cursor()
        _membership(cur,program_id,user)
        _visible_task(cur,program_id,task_id,user)
        cur.execute("""SELECT response,status,feedback,submitted_at,reviewed_at,updated_at
            FROM verifyd_therapy_responses WHERE task_id=%s AND client_id=%s""",(task_id,user))
        rec=cur.fetchone()
    return {"work":_row(rec) if rec else None}


@router.put('/programs/{program_id}/tasks/{task_id}/my-response')
def save_response(program_id:str,task_id:str,payload:PracticeResponse,
                  authorization:Optional[str]=Header(default=None)):
    user=_identity(authorization)
    _schema()
    if payload.action=='submit' and not payload.response.strip():
        raise HTTPException(422,detail={"error":"response_required"})
    with get_db() as conn:
        cur=conn.cursor()
        _membership(cur,program_id,user)
        _visible_task(cur,program_id,task_id,user)
        cur.execute("""INSERT INTO verifyd_therapy_responses(task_id,client_id)
            VALUES (%s,%s) ON CONFLICT DO NOTHING""",(task_id,user))
        cur.execute("""SELECT status FROM verifyd_therapy_responses WHERE task_id=%s AND client_id=%s FOR UPDATE""",(task_id,user))
        state=cur.fetchone()['status']
        if state in ('submitted','returned'):
            raise HTTPException(409,detail={"error":"response_locked"})
        status='submitted' if payload.action=='submit' else 'draft'
        cur.execute("""UPDATE verifyd_therapy_responses SET response=%s,status=%s,
            submitted_at=CASE WHEN %s='submitted' THEN NOW() ELSE NULL END,updated_at=NOW()
            WHERE task_id=%s AND client_id=%s RETURNING response,status,submitted_at,updated_at""",
            (payload.response.strip(),status,status,task_id,user))
        rec=cur.fetchone()
    return {"work":_row(rec)}


@router.get('/programs/{program_id}/tasks/{task_id}/responses')
def submissions(program_id:str,task_id:str,authorization:Optional[str]=Header(default=None)):
    user=_identity(authorization)
    _schema()
    with get_db() as conn:
        cur=conn.cursor()
        _owner(cur,program_id,user)
        cur.execute("SELECT 1 FROM verifyd_therapy_tasks WHERE program_id=%s AND task_id=%s",(program_id,task_id))
        if not cur.fetchone():
            raise HTTPException(404,detail={"error":"practice_task_unavailable"})
        cur.execute("""SELECT s.client_id,i.handle AS client_handle,s.response,s.status,s.feedback,
            s.submitted_at,s.reviewed_at
            FROM verifyd_therapy_responses s
            JOIN verifyd_identities i ON i.id=s.client_id
            WHERE s.task_id=%s AND s.status IN ('submitted','returned')
            ORDER BY s.submitted_at DESC LIMIT 300""",(task_id,))
        rows=[_row(r) for r in cur.fetchall()]
    return {"submissions":rows}


@router.post('/programs/{program_id}/tasks/{task_id}/responses/{client_id}/review')
def review(program_id:str,task_id:str,client_id:str,payload:PracticeReview,
           authorization:Optional[str]=Header(default=None)):
    user=_identity(authorization)
    _schema()
    with get_db() as conn:
        cur=conn.cursor()
        _owner(cur,program_id,user)
        cur.execute("""UPDATE verifyd_therapy_responses SET status='returned',feedback=%s,
            reviewed_at=NOW(),updated_at=NOW()
            WHERE task_id=%s AND client_id=%s AND status='submitted'
              AND EXISTS (SELECT 1 FROM verifyd_therapy_tasks WHERE task_id=%s AND program_id=%s)
            RETURNING status,feedback,reviewed_at""",
            (payload.feedback.strip(),task_id,client_id,task_id,program_id))
        rec=cur.fetchone()
        if not rec:
            raise HTTPException(409,detail={"error":"submitted_response_not_found"})
    return {"work":_row(rec),"clinical_grade":False,"trust_rank_awarded":False}