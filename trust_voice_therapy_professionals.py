"""VeriFYD Circle Therapy professional registration and independent approval.

Separate from the therapy PRACTICE feature flag.  Receiving an application or
approving a credential does not enable medical use, clinical access, PHI storage,
or HIPAA-compliant processing.  No client records, documents or images accepted.

Flags: VERIFYD_CIRCLE_THERAPY_REGISTRATION_ENABLED (default 0)
       VERIFYD_CIRCLE_THERAPY_ADMIN_IDS (exact vetted VeriFYD identity IDs)
This module does not use an administrator API key in a web client.
"""
from __future__ import annotations

import os
import re
import threading
import uuid
from datetime import datetime
from typing import Literal, Optional

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field

from database import get_db
from trust_voice_messages import _identity_from_bearer

router = APIRouter(prefix='/trust-voice/therapy-professionals', tags=['Therapy Professional Registration'])
_LOCK = threading.Lock()
_READY = False
_YES = ('1', 'true', 'yes', 'on')
_ROLES = (
    'Speech-Language Pathologist', 'Occupational Therapist', 'Physical Therapist',
    'Psychologist', 'Licensed Mental Health Professional', 'Physician',
    'Registered Nurse', 'Other Licensed Professional',
)


def enabled():
    return os.getenv('VERIFYD_CIRCLE_THERAPY_REGISTRATION_ENABLED', '0').strip().lower() in _YES


def _require_enabled():
    if not enabled():
        raise HTTPException(404, detail={'error': 'professional_registration_unavailable'})


def _admin_ids():
    # A blank list grants nobody review permission.  Never use client-provided
    # roles, an email domain, a claimed job title, or the Classroom teacher list.
    return {i.strip() for i in os.getenv('VERIFYD_CIRCLE_THERAPY_ADMIN_IDS', '').split(',') if i.strip()}


def _auth(authorization):
    _require_enabled()
    person = _identity_from_bearer(authorization)
    if person.get('status') != 'active' or not person.get('id'):
        raise HTTPException(401, detail={'error': 'session_required'})
    return str(person['id'])


def _admin(user):
    if user not in _admin_ids():
        raise HTTPException(403, detail={'error': 'reviewer_approval_required'})


def _schema():
    global _READY
    if _READY:
        return
    with _LOCK:
        if _READY:
            return
        with get_db() as conn:
            cur = conn.cursor()
            cur.execute('''CREATE TABLE IF NOT EXISTS verifyd_therapy_professional_applications (
                identity_id TEXT PRIMARY KEY REFERENCES verifyd_identities(id) ON DELETE CASCADE,
                display_name TEXT NOT NULL,
                professional_role TEXT NOT NULL,
                specialty TEXT NOT NULL DEFAULT '',
                organization TEXT NOT NULL DEFAULT '',
                license_jurisdiction TEXT NOT NULL,
                licensing_board TEXT NOT NULL,
                license_number TEXT NOT NULL,
                npi TEXT,
                status TEXT NOT NULL DEFAULT 'pending'
                    CHECK(status IN ('pending','approved','rejected','revoked')),
                applicant_adult_attested_at TIMESTAMPTZ NOT NULL,
                applicant_no_phi_ack_at TIMESTAMPTZ NOT NULL,
                submitted_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                reviewed_by TEXT REFERENCES verifyd_identities(id) ON DELETE SET NULL,
                reviewed_at TIMESTAMPTZ,
                review_note TEXT NOT NULL DEFAULT ''
            )''')
            cur.execute('''CREATE INDEX IF NOT EXISTS idx_therapy_pro_applications_pending
                ON verifyd_therapy_professional_applications(status,submitted_at DESC)''')
            cur.execute('''CREATE TABLE IF NOT EXISTS verifyd_therapy_professional_decisions (
                event_id TEXT PRIMARY KEY,
                applicant_id TEXT NOT NULL REFERENCES verifyd_identities(id) ON DELETE CASCADE,
                reviewer_id TEXT NOT NULL REFERENCES verifyd_identities(id) ON DELETE CASCADE,
                previous_status TEXT NOT NULL,
                next_status TEXT NOT NULL,
                reason TEXT NOT NULL DEFAULT '',
                credential_check_attested BOOLEAN NOT NULL DEFAULT FALSE,
                recorded_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )''')
            cur.execute('''CREATE INDEX IF NOT EXISTS idx_therapy_pro_decisions_applicant
                ON verifyd_therapy_professional_decisions(applicant_id,recorded_at DESC)''')
            cur.execute('''CREATE TABLE IF NOT EXISTS verifyd_therapy_professional_notices (
                notice_id TEXT PRIMARY KEY,
                identity_id TEXT NOT NULL REFERENCES verifyd_identities(id) ON DELETE CASCADE,
                kind TEXT NOT NULL CHECK(kind IN ('submitted','approved','rejected','revoked')),
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                read_at TIMESTAMPTZ
            )''')
            cur.execute('''CREATE INDEX IF NOT EXISTS idx_therapy_pro_notices_unread
                ON verifyd_therapy_professional_notices(identity_id,created_at DESC)''')
        _READY = True


def _display(row):
    if row is None:
        return None
    out = dict(row)
    for key, value in out.items():
        if isinstance(value, datetime):
            out[key] = value.isoformat()
    return out


def professional_status(identity_id):
    """For server-side authorization only.  Always read current DB status.

    Explicit rejections/revocations supersede legacy manual allowlists.  Do not
    grant permission from a display handle, claim, or cached frontend state.
    """
    if not enabled() or not identity_id:
        return None
    _schema()
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute('''SELECT status FROM verifyd_therapy_professional_applications
            WHERE identity_id=%s''', (identity_id,))
        result = cur.fetchone()
    return result['status'] if result else None


class Application(BaseModel):
    display_name: str = Field(min_length=2, max_length=100)
    professional_role: Literal[
        'Speech-Language Pathologist', 'Occupational Therapist', 'Physical Therapist',
        'Psychologist', 'Licensed Mental Health Professional', 'Physician',
        'Registered Nurse', 'Other Licensed Professional'
    ]
    specialty: str = Field(default='', max_length=120)
    organization: str = Field(default='', max_length=120)
    license_jurisdiction: str = Field(min_length=2, max_length=80)
    licensing_board: str = Field(min_length=3, max_length=120)
    license_number: str = Field(min_length=3, max_length=80)
    npi: Optional[str] = Field(default=None, max_length=10)
    adult_attestation: bool = False
    no_patient_data_acknowledgment: bool = False
    truthful_information_acknowledgment: bool = False


class Decision(BaseModel):
    action: Literal['approve', 'reject', 'revoke']
    credential_verified: bool = False
    adult_test_only_confirmed: bool = False
    reason: str = Field(default='', max_length=500)


@router.get('/me')
def my_application(authorization: Optional[str] = Header(default=None)):
    user = _auth(authorization)
    _schema()
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute('''SELECT display_name,professional_role,specialty,organization,
            license_jurisdiction,licensing_board,license_number,npi,status,
            submitted_at,updated_at,reviewed_at,review_note
            FROM verifyd_therapy_professional_applications WHERE identity_id=%s''',(user,))
        application = _display(cur.fetchone())
        cur.execute('''SELECT COUNT(*) AS n FROM verifyd_therapy_professional_notices
            WHERE identity_id=%s AND read_at IS NULL''', (user,))
        unread = int(cur.fetchone()['n'])
    return {'application':application,'is_reviewer':user in _admin_ids(),
            'registration_open':True,'unread_notifications':unread,
            'practice_enabled':os.getenv('VERIFYD_CIRCLE_THERAPY_PRACTICE_ENABLED','0').strip().lower() in _YES,
            'policy':'adult_test_only_no_patient_data'}


@router.post('/apply', status_code=202)
def apply(payload: Application, authorization: Optional[str] = Header(default=None)):
    user = _auth(authorization)
    if not (payload.adult_attestation and payload.no_patient_data_acknowledgment
            and payload.truthful_information_acknowledgment):
        raise HTTPException(422,detail={'error':'application_attestations_required'})
    if payload.npi and not re.fullmatch(r'\d{10}',payload.npi):
        raise HTTPException(422,detail={'error':'invalid_npi_format'})
    license_number=payload.license_number.strip()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9.\- /#]{2,79}', license_number):
        raise HTTPException(422,detail={'error':'invalid_license_number_format'})
    _schema()
    with get_db() as conn:
        cur=conn.cursor()
        cur.execute('''SELECT status,email_verified FROM verifyd_identities
            WHERE id=%s FOR UPDATE''',(user,))
        account=cur.fetchone()
        if not account or account['status']!='active':
            raise HTTPException(401,detail={'error':'session_required'})
        if not account['email_verified']:
            raise HTTPException(403,detail={'error':'email_verification_required'})
        cur.execute('''SELECT status FROM verifyd_therapy_professional_applications
            WHERE identity_id=%s FOR UPDATE''',(user,))
        prior=cur.fetchone()
        if prior and prior['status']!='rejected':
            raise HTTPException(409,detail={'error':'application_already_exists','status':prior['status']})
        values=(payload.display_name.strip(),payload.professional_role,payload.specialty.strip(),
                payload.organization.strip(),payload.license_jurisdiction.strip(),
                payload.licensing_board.strip(),license_number,payload.npi or None,user)
        cur.execute('''INSERT INTO verifyd_therapy_professional_applications
            (display_name,professional_role,specialty,organization,license_jurisdiction,
             licensing_board,license_number,npi,identity_id,
             applicant_adult_attested_at,applicant_no_phi_ack_at,status)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,NOW(),NOW(),'pending')
            ON CONFLICT(identity_id) DO UPDATE SET
                display_name=EXCLUDED.display_name,
                professional_role=EXCLUDED.professional_role,
                specialty=EXCLUDED.specialty,organization=EXCLUDED.organization,
                license_jurisdiction=EXCLUDED.license_jurisdiction,
                licensing_board=EXCLUDED.licensing_board,
                license_number=EXCLUDED.license_number,npi=EXCLUDED.npi,
                applicant_adult_attested_at=NOW(),applicant_no_phi_ack_at=NOW(),
                status='pending',submitted_at=NOW(),updated_at=NOW(),
                reviewed_by=NULL,reviewed_at=NULL,review_note=''
            WHERE verifyd_therapy_professional_applications.status='rejected'
            RETURNING identity_id''',values)
        if not cur.fetchone():
            raise HTTPException(409,detail={'error':'application_already_exists'})
        cur.execute('''INSERT INTO verifyd_therapy_professional_notices(notice_id,identity_id,kind)
            VALUES (%s,%s,'submitted')''',(str(uuid.uuid4()),user))
    return {'status':'pending','message':'Submitted for manual professional review. No patient data may be entered.'}


@router.get('/notifications')
def notifications(authorization: Optional[str] = Header(default=None)):
    user=_auth(authorization)
    _schema()
    with get_db() as conn:
        cur=conn.cursor()
        cur.execute('''SELECT notice_id,kind,created_at,read_at
            FROM verifyd_therapy_professional_notices WHERE identity_id=%s
            ORDER BY created_at DESC LIMIT 30''',(user,))
        rows=[_display(item) for item in cur.fetchall()]
    return {'notifications':rows}


@router.post('/notifications/{notice_id}/read')
def read_notification(notice_id:str,authorization:Optional[str]=Header(default=None)):
    user=_auth(authorization)
    _schema()
    with get_db() as conn:
        cur=conn.cursor()
        cur.execute('''UPDATE verifyd_therapy_professional_notices SET read_at=NOW()
            WHERE notice_id=%s AND identity_id=%s AND read_at IS NULL
            RETURNING notice_id''',(notice_id,user))
        changed=bool(cur.fetchone())
    return {'read':changed}


@router.get('/admin/applications')
def admin_applications(status:Literal['pending','approved','rejected','revoked','all']='pending',
                       authorization:Optional[str]=Header(default=None)):
    user=_auth(authorization)
    _admin(user)
    _schema()
    with get_db() as conn:
        cur=conn.cursor()
        cur.execute('''SELECT a.identity_id,i.handle,i.display_name AS account_name,
            i.email_verified,a.display_name,a.professional_role,a.specialty,
            a.organization,a.license_jurisdiction,a.licensing_board,
            a.license_number,a.npi,a.status,a.submitted_at,a.reviewed_at,
            a.review_note
            FROM verifyd_therapy_professional_applications a
            JOIN verifyd_identities i ON i.id=a.identity_id
            WHERE (%s='all' OR a.status=%s)
            ORDER BY a.submitted_at DESC LIMIT 100''',(status,status))
        rows=[_display(r) for r in cur.fetchall()]
    return {'applications':rows,'reviewer':True,'filter':status}


@router.post('/admin/applications/{applicant_id}/decision')
def decide(applicant_id:str,payload:Decision,authorization:Optional[str]=Header(default=None)):
    user=_auth(authorization)
    _admin(user)
    if applicant_id==user:
        raise HTTPException(403,detail={'error':'self_approval_prohibited'})
    if payload.action=='approve' and not (payload.credential_verified and payload.adult_test_only_confirmed):
        raise HTTPException(422,detail={'error':'independent_verification_required'})
    if payload.action in ('reject','revoke') and len(payload.reason.strip())<3:
        raise HTTPException(422,detail={'error':'reason_required'})
    _schema()
    with get_db() as conn:
        cur=conn.cursor()
        cur.execute('''SELECT a.status,i.email_verified,i.status AS account_status
            FROM verifyd_therapy_professional_applications a
            JOIN verifyd_identities i ON i.id=a.identity_id
            WHERE a.identity_id=%s FOR UPDATE''',(applicant_id,))
        prior=cur.fetchone()
        if not prior:
            raise HTTPException(404,detail={'error':'application_not_found'})
        state=prior['status']
        if payload.action=='approve':
            if state!='pending' or not prior['email_verified'] or prior['account_status']!='active':
                raise HTTPException(409,detail={'error':'application_not_approvable'})
            new='approved'
        elif payload.action=='reject':
            if state!='pending':raise HTTPException(409,detail={'error':'application_not_pending'})
            new='rejected'
        else:
            if state!='approved':raise HTTPException(409,detail={'error':'application_not_approved'})
            new='revoked'
        reason=payload.reason.strip()
        cur.execute('''UPDATE verifyd_therapy_professional_applications
            SET status=%s,reviewed_by=%s,reviewed_at=NOW(),review_note=%s,updated_at=NOW()
            WHERE identity_id=%s''',(new,user,reason,applicant_id))
        cur.execute('''INSERT INTO verifyd_therapy_professional_decisions
            (event_id,applicant_id,reviewer_id,previous_status,next_status,
             reason,credential_check_attested)
            VALUES(%s,%s,%s,%s,%s,%s,%s)''',
            (str(uuid.uuid4()),applicant_id,user,state,new,reason,payload.credential_verified))
        cur.execute('''INSERT INTO verifyd_therapy_professional_notices(notice_id,identity_id,kind)
            VALUES (%s,%s,%s)''',(str(uuid.uuid4()),applicant_id,new))
    return {'status':new,'applicant_id':applicant_id,'notification_created':True,
            'practice_enabled':os.getenv('VERIFYD_CIRCLE_THERAPY_PRACTICE_ENABLED','0').strip().lower() in _YES}