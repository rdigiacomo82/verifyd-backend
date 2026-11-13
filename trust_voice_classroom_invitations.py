"""Classroom invitations from teacher-owned Trust Circle contacts.

Optional, fail-closed pilot. Only the invited adult account may accept. Invitations
are NOT Circle membership, and Circle membership alone never enrolls anyone.
Legacy class codes remain on the server for clients not using this UI.

Feature flag: VERIFYD_CIRCLE_CLASSROOM_INVITES_ENABLED=1 (default OFF).
"""
from __future__ import annotations

import os
import threading
import uuid
from typing import Optional, Literal
from datetime import datetime, timezone

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field

from database import get_db
from trust_voice_classrooms import _identity, _schema as classroom_schema, _teacher, _approved_teacher, _teacher_allowed
from trust_voice_permissions import ensure_trust_circle_schema
from trust_voice_blocking import identities_blocked

router = APIRouter(prefix='/trust-voice/classroom', tags=['Circle Classroom Invitations'])
_READY = False
_LOCK = threading.Lock()


def invites_enabled() -> bool:
    return os.getenv('VERIFYD_CIRCLE_CLASSROOM_INVITES_ENABLED', '0').strip().lower() in ('1', 'true', 'yes', 'on')


def _require_invites():
    if not invites_enabled():
        raise HTTPException(404, detail={'error': 'classroom_invitations_unavailable'})


def prepare_invites():
    if invites_enabled():
        _schema()


def _schema():
    global _READY
    if _READY:
        return
    with _LOCK:
        if _READY:
            return
        classroom_schema()
        ensure_trust_circle_schema()
        with get_db() as conn:
            cur = conn.cursor()
            cur.execute('''CREATE TABLE IF NOT EXISTS verifyd_circle_class_invitations (
                invitation_id TEXT PRIMARY KEY,
                class_id TEXT NOT NULL REFERENCES verifyd_circle_classes(class_id) ON DELETE CASCADE,
                teacher_id TEXT NOT NULL REFERENCES verifyd_identities(id) ON DELETE CASCADE,
                student_id TEXT NOT NULL REFERENCES verifyd_identities(id) ON DELETE CASCADE,
                status TEXT NOT NULL DEFAULT 'pending' CHECK(status IN ('pending','accepted','declined','revoked')),
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                expires_at TIMESTAMPTZ NOT NULL DEFAULT NOW() + INTERVAL '14 days',
                responded_at TIMESTAMPTZ,
                UNIQUE(class_id,student_id)
            )''')
            cur.execute('''CREATE INDEX IF NOT EXISTS idx_circle_invites_recipient
                ON verifyd_circle_class_invitations(student_id,status,expires_at)''')
            cur.execute('''CREATE TABLE IF NOT EXISTS verifyd_circle_assignment_audiences (
                assignment_id TEXT PRIMARY KEY REFERENCES verifyd_circle_class_assignments(assignment_id) ON DELETE CASCADE,
                mode TEXT NOT NULL CHECK(mode IN ('all','selected'))
            )''')
            cur.execute('''CREATE TABLE IF NOT EXISTS verifyd_circle_assignment_audience_students (
                assignment_id TEXT NOT NULL REFERENCES verifyd_circle_class_assignments(assignment_id) ON DELETE CASCADE,
                student_id TEXT NOT NULL REFERENCES verifyd_identities(id) ON DELETE CASCADE,
                PRIMARY KEY (assignment_id,student_id)
            )''')
            cur.execute('''CREATE INDEX IF NOT EXISTS idx_class_audience_student
                ON verifyd_circle_assignment_audience_students(student_id,assignment_id)''')
        _READY = True


class InviteSelection(BaseModel):
    student_ids: list[str] = Field(min_length=1, max_length=30)

    @classmethod
    def allowed_count(cls, ids: list[str]) -> bool:
        return all(isinstance(value, str) and 1 <= len(value) <= 128 for value in ids)


class AcceptInvitation(BaseModel):
    adult_attestation: bool = False


@router.get('/invites/contacts')
def trusted_educator_contacts(authorization: Optional[str] = Header(default=None)):
    _require_invites()
    teacher_id = _identity(authorization)
    _approved_teacher(teacher_id)
    _schema()
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute('''SELECT t.member_identity_id AS identity_id,
               i.handle, t.display_label
            FROM verifyd_trust_circle_members t
            JOIN verifyd_identities i ON i.id=t.member_identity_id AND i.status='active'
            WHERE t.owner_identity_id=%s AND t.contact_type='person'
              AND t.member_identity_id<>%s
            ORDER BY t.sort_order,t.created_at LIMIT 200''', (teacher_id, teacher_id))
        rows = list(cur.fetchall())
    contacts = [dict(row) for row in rows if not identities_blocked(teacher_id, row['identity_id'])]
    return {'enabled': True, 'contacts': contacts}


@router.post('/classes/{class_id}/invitations')
def send_class_invitations(class_id: str, payload: InviteSelection,
                           authorization: Optional[str] = Header(default=None)):
    _require_invites()
    teacher_id = _identity(authorization)
    _approved_teacher(teacher_id)
    _schema()
    if not InviteSelection.allowed_count(payload.student_ids):
        raise HTTPException(422, detail={'error': 'invalid_invite_target'})
    if len(set(payload.student_ids)) != len(payload.student_ids):
        raise HTTPException(422, detail={'error': 'duplicate_invite_targets'})
    # Check blocking outside the classroom SQL transaction. This avoids
    # acquiring a second database connection while the first is held open.
    if any(identities_blocked(teacher_id, student_id) for student_id in payload.student_ids):
        raise HTTPException(403, detail={'error': 'student_not_in_trust_circle'})
    sent = []
    with get_db() as conn:
        cur = conn.cursor()
        _teacher(cur, class_id, teacher_id)
        cur.execute('''SELECT count(*) AS n FROM verifyd_circle_class_invitations
            WHERE teacher_id=%s AND created_at > NOW() - INTERVAL '24 hours' ''', (teacher_id,))
        if cur.fetchone()['n'] + len(payload.student_ids) > 50:
            raise HTTPException(429, detail={'error': 'invite_daily_limit'})
        for student_id in payload.student_ids:
            if student_id == teacher_id:
                raise HTTPException(422, detail={'error': 'cannot_invite_self'})
            cur.execute('''SELECT 1 FROM verifyd_trust_circle_members t
                JOIN verifyd_identities i ON i.id=t.member_identity_id AND i.status='active'
                WHERE t.owner_identity_id=%s AND t.member_identity_id=%s AND t.contact_type='person' ''',
                (teacher_id, student_id))
            if not cur.fetchone():
                raise HTTPException(403, detail={'error': 'student_not_in_trust_circle'})
            cur.execute('''SELECT status FROM verifyd_circle_class_members
                WHERE class_id=%s AND student_id=%s''', (class_id, student_id))
            existing = cur.fetchone()
            if existing and existing['status'] == 'approved':
                continue
            cur.execute('''INSERT INTO verifyd_circle_class_invitations
                (invitation_id,class_id,teacher_id,student_id)
                VALUES (%s,%s,%s,%s)
                ON CONFLICT (class_id,student_id) DO UPDATE SET
                    invitation_id=EXCLUDED.invitation_id, status='pending',
                    created_at=NOW(), expires_at=NOW() + INTERVAL '14 days', responded_at=NULL
                RETURNING invitation_id''', (f'tvinv_{uuid.uuid4().hex}',class_id,teacher_id,student_id))
            row = cur.fetchone()
            if row:
                sent.append({'student_id': student_id, 'invitation_id': row['invitation_id']})
    return {'sent': len(sent), 'invitations': sent,
            'note': 'Only selected recipients see these invitations in their private Classroom. No class code is needed.'}


@router.get('/classes/{class_id}/invitations')
def sent_class_invitations(class_id: str, authorization: Optional[str] = Header(default=None)):
    _require_invites()
    teacher_id = _identity(authorization)
    _schema()
    with get_db() as conn:
        cur = conn.cursor()
        _teacher(cur,class_id,teacher_id)
        cur.execute('''SELECT inv.invitation_id,inv.status,inv.student_id,i.handle,
                  inv.created_at,inv.expires_at
            FROM verifyd_circle_class_invitations inv
            JOIN verifyd_identities i ON i.id=inv.student_id
            WHERE inv.class_id=%s ORDER BY inv.created_at DESC LIMIT 200''', (class_id,))
        result = [dict(row) for row in cur.fetchall()]
    return {'invitations': result}


@router.get('/invitations/mine')
def incoming_class_invitations(authorization: Optional[str] = Header(default=None)):
    _require_invites()
    student_id = _identity(authorization)
    _schema()
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute('''SELECT inv.invitation_id, inv.created_at, inv.expires_at,
                c.class_id,c.title,c.subject,c.grade_band,
                i.handle AS teacher_handle
            FROM verifyd_circle_class_invitations inv
            JOIN verifyd_circle_classes c ON c.class_id=inv.class_id
            JOIN verifyd_identities i ON i.id=inv.teacher_id AND i.status='active'
            WHERE inv.student_id=%s AND inv.status='pending' AND inv.expires_at > NOW()
            ORDER BY inv.created_at DESC LIMIT 50''', (student_id,))
        rows = [dict(row) for row in cur.fetchall()]
    return {'enabled': True, 'invitations': rows}


@router.post('/invitations/{invitation_id}/accept')
def accept_class_invitation(invitation_id: str, payload: AcceptInvitation,
                            authorization: Optional[str] = Header(default=None)):
    _require_invites()
    student_id = _identity(authorization)
    if not payload.adult_attestation:
        raise HTTPException(422, detail={'error': 'adult_join_attestation_required'})
    _schema()
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute('''SELECT inv.class_id,inv.teacher_id,inv.status,inv.expires_at,
                c.grade_band FROM verifyd_circle_class_invitations inv
            JOIN verifyd_circle_classes c ON c.class_id=inv.class_id
            WHERE inv.invitation_id=%s AND inv.student_id=%s FOR UPDATE OF inv''', (invitation_id,student_id))
        row = cur.fetchone()
        if not row:
            raise HTTPException(404, detail={'error': 'invitation_not_found'})
        if row['status'] != 'pending' or row['expires_at'] <= datetime.now(timezone.utc):
            raise HTTPException(409, detail={'error': 'invitation_no_longer_available'})
        if not _teacher_allowed(row['teacher_id'],cur):
            raise HTTPException(403, detail={'error': 'teacher_not_active'})
        # Do not auto-enroll a formerly trusted contact after removal from the Circle.
        cur.execute('''SELECT 1 FROM verifyd_trust_circle_members WHERE
            owner_identity_id=%s AND member_identity_id=%s AND contact_type='person' ''',
            (row['teacher_id'], student_id))
        if not cur.fetchone():
            raise HTTPException(403, detail={'error': 'student_not_in_trust_circle'})
        cur.execute('''INSERT INTO verifyd_circle_class_members(class_id,student_id,status,approved_at)
            VALUES (%s,%s,'approved',NOW())
            ON CONFLICT(class_id,student_id) DO UPDATE SET status='approved',approved_at=NOW()''',
            (row['class_id'],student_id))
        cur.execute('''UPDATE verifyd_circle_class_invitations
            SET status='accepted',responded_at=NOW() WHERE invitation_id=%s''', (invitation_id,))
    return {'joined': True, 'class_id':row['class_id'], 'requires_second_approval': False}


@router.post('/invitations/{invitation_id}/decline')
def decline_class_invitation(invitation_id: str, authorization: Optional[str] = Header(default=None)):
    _require_invites()
    student_id = _identity(authorization)
    _schema()
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute('''UPDATE verifyd_circle_class_invitations SET status='declined',responded_at=NOW()
            WHERE invitation_id=%s AND student_id=%s AND status='pending' RETURNING invitation_id''',
            (invitation_id,student_id))
        if not cur.fetchone():
            raise HTTPException(404, detail={'error': 'invitation_not_found'})
    return {'declined': True}


@router.delete('/classes/{class_id}/invitations/{invitation_id}')
def revoke_class_invitation(class_id: str, invitation_id: str,
                            authorization: Optional[str] = Header(default=None)):
    _require_invites()
    teacher_id = _identity(authorization)
    _schema()
    with get_db() as conn:
        cur = conn.cursor()
        _teacher(cur,class_id,teacher_id)
        cur.execute('''UPDATE verifyd_circle_class_invitations
            SET status='revoked',responded_at=NOW()
            WHERE invitation_id=%s AND class_id=%s AND status='pending' RETURNING invitation_id''',
            (invitation_id,class_id))
        if not cur.fetchone():
            raise HTTPException(404, detail={'error': 'invitation_not_found'})
    return {'revoked':True}


# -- Per-assignment audience; independent of personal Trust Circle membership --
def audience_enabled() -> bool:
    return (os.getenv('VERIFYD_CIRCLE_CLASSROOM_AUDIENCE_ENABLED', '0').strip().lower()
            in ('1', 'true', 'yes', 'on')) and (
            os.getenv('VERIFYD_CIRCLE_CLASSROOM_INVITES_ENABLED', '0').strip().lower()
            in ('1', 'true', 'yes', 'on'))


def prepare_audience():
    """Call before opening a member's classwork transaction, never from a locked cursor."""
    if audience_enabled():
        _schema()


def audience_allows(cur, assignment_id: str, student_id: str) -> bool:
    """Used by all student-facing assignment queries. Teacher bypass is at call site."""
    if not audience_enabled():
        return True
    cur.execute("SELECT mode FROM verifyd_circle_assignment_audiences WHERE assignment_id=%s", (assignment_id,))
    rec = cur.fetchone()
    if not rec or rec['mode'] == 'all':
        return True
    cur.execute("""SELECT 1 FROM verifyd_circle_assignment_audience_students
        WHERE assignment_id=%s AND student_id=%s""", (assignment_id, student_id))
    return bool(cur.fetchone())


class AudienceSelection(BaseModel):
    mode: Literal['all','selected']
    student_ids: list[str] = Field(default_factory=list, max_length=300)


@router.get('/classes/{class_id}/assignments/{assignment_id}/audience')
def get_assignment_audience(class_id: str, assignment_id: str,
                            authorization: Optional[str] = Header(default=None)):
    _require_invites()
    if not audience_enabled():
        raise HTTPException(404, detail={'error': 'classroom_assignment_audience_unavailable'})
    teacher_id = _identity(authorization)
    _approved_teacher(teacher_id)
    _schema()
    with get_db() as conn:
        cur = conn.cursor()
        _teacher(cur,class_id,teacher_id)
        cur.execute('SELECT status FROM verifyd_circle_class_assignments WHERE assignment_id=%s AND class_id=%s',
            (assignment_id,class_id))
        record = cur.fetchone()
        if not record:
            raise HTTPException(404, detail={'error':'assignment_not_found'})
        cur.execute('SELECT mode FROM verifyd_circle_assignment_audiences WHERE assignment_id=%s', (assignment_id,))
        mode = cur.fetchone()
        cur.execute('''SELECT student_id FROM verifyd_circle_assignment_audience_students WHERE assignment_id=%s''',(assignment_id,))
        selected = [x['student_id'] for x in cur.fetchall()]
        cur.execute('''SELECT student_id FROM verifyd_circle_class_members
             WHERE class_id=%s AND status='approved' ORDER BY approved_at LIMIT 300''',(class_id,))
        available = [x['student_id'] for x in cur.fetchall()]
    return {'mode':mode['mode'] if mode else 'all', 'student_ids':selected,
            'enrolled_students':available,'editable':record['status']=='draft'}


@router.put('/classes/{class_id}/assignments/{assignment_id}/audience')
def update_assignment_audience(class_id: str, assignment_id: str, payload: AudienceSelection,
                               authorization: Optional[str] = Header(default=None)):
    _require_invites()
    if not audience_enabled():
        raise HTTPException(404, detail={'error': 'classroom_assignment_audience_unavailable'})
    teacher_id = _identity(authorization)
    _approved_teacher(teacher_id)
    if len(set(payload.student_ids)) != len(payload.student_ids) or any(
        not isinstance(x,str) or len(x)>128 or not x for x in payload.student_ids):
        raise HTTPException(422, detail={'error':'invalid_assignment_audience'})
    if payload.mode=='selected' and not payload.student_ids:
        raise HTTPException(422, detail={'error':'select_at_least_one_student'})
    _schema()
    with get_db() as conn:
        cur = conn.cursor()
        _teacher(cur,class_id,teacher_id)
        cur.execute('''SELECT status FROM verifyd_circle_class_assignments
            WHERE assignment_id=%s AND class_id=%s FOR UPDATE''',(assignment_id,class_id))
        rec = cur.fetchone()
        if not rec:
            raise HTTPException(404, detail={'error':'assignment_not_found'})
        if rec['status'] != 'draft':
            raise HTTPException(409, detail={'error':'assignment_audience_locked'})
        cur.execute('''SELECT student_id FROM verifyd_circle_class_members
            WHERE class_id=%s AND status='approved' ''',(class_id,))
        eligible={row['student_id'] for row in cur.fetchall()}
        if any(student not in eligible for student in payload.student_ids):
            raise HTTPException(403, detail={'error':'student_not_enrolled'})
        cur.execute('''INSERT INTO verifyd_circle_assignment_audiences(assignment_id,mode)
            VALUES (%s,%s) ON CONFLICT(assignment_id) DO UPDATE SET mode=EXCLUDED.mode''',
            (assignment_id,payload.mode))
        cur.execute('DELETE FROM verifyd_circle_assignment_audience_students WHERE assignment_id=%s', (assignment_id,))
        if payload.mode=='selected':
            for student in payload.student_ids:
                cur.execute('''INSERT INTO verifyd_circle_assignment_audience_students(assignment_id,student_id)
                    VALUES (%s,%s)''',(assignment_id,student))
    return {'updated':True,'mode':payload.mode,'selected_count':len(payload.student_ids) if payload.mode=='selected' else 0}