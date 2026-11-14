"""VeriFYD Circle: voluntary occupational profiles and moderated text-only resources.

This is NOT practitioner verification and never unlocks private Therapy Practice.
Authenticated adults may post original general-purpose materials for staff review.
No files, patient information, or child accounts are accepted by this module.
Enable explicitly with VERIFYD_CIRCLE_THERAPY_LIBRARY_ENABLED=1.
"""
from __future__ import annotations

import os
import re
import threading
import uuid
from datetime import datetime, timezone
from typing import Optional, Literal

from fastapi import APIRouter, Header, HTTPException, Query
from pydantic import BaseModel, Field

from database import get_db
from trust_voice_messages import _identity_from_bearer

router = APIRouter(prefix='/trust-voice/therapy-library', tags=['Circle Therapy Resources'])
_LOCK = threading.Lock()
_READY = False
_TRUE = ('1', 'true', 'yes', 'on')
TOPICS = {
    'Speech & Language': ('Articulation & Sounds', 'Language & Vocabulary', 'Social Communication', 'Fluency', 'Voice', 'AAC & Communication Boards', 'Listening & Comprehension', 'Other'),
    'Occupational Therapy': ('Fine Motor', 'Handwriting', 'Sensory Processing', 'Visual Motor', 'Daily Living', 'Executive Function', 'Adaptive Equipment', 'Other'),
    'Physical Therapy': ('Balance & Coordination', 'Strength & Mobility', 'Range of Motion', 'Posture', 'Gait & Walking', 'Movement Games', 'Motor Planning', 'Other'),
    'Behavioral Support': ('Self-Regulation', 'Daily Routines', 'Positive Reinforcement', 'Social Skills', 'Emotion Recognition', 'Transitions', 'Choice Making', 'Other'),
    'Special Education': ('Literacy', 'Numeracy', 'Learning Strategies', 'Executive Function', 'Communication', 'Independent Living', 'Sensory-Friendly Learning', 'Other'),
    'Mental Wellness': ('Mindfulness', 'Breathing Exercises', 'Stress Management', 'Emotion Literacy', 'Positive Habits', 'Self-Advocacy', 'Other'),
    'Recreation & Adaptive Activities': ('Accessible Games', 'Creative Arts', 'Music & Rhythm', 'Movement & Recreation', 'Life Skills', 'Social Interaction', 'Other'),
    'Other Professional': ('Communication', 'Movement', 'Learning', 'Daily Living', 'Wellbeing', 'Accessibility', 'Other'),
}
AGE_BANDS = ('All ages', 'Early childhood (3–5)', 'School age (6–12)', 'Teens (13–17)', 'Adults (18+)', 'Older adults')
LEVELS = ('Introductory', 'Developing', 'Advanced', 'Adaptable')
FORMATS = ('Activity', 'Step-by-step exercise', 'Practice worksheet', 'Discussion prompts', 'Visual choice board', 'Reference guide')
_PRIVATE = re.compile(r'(?i)(?:[\w.+-]+@[\w.-]+\.[a-z]{2,}|(?<!\d)(?:\+?1[-. ]?)?\(?\d{3}\)?[-. ]?\d{3}[-. ]?\d{4}(?!\d)|\b(?:patient|student|client)\s*(?:id|name|dob)\s*[:#]|\b(?:diagnosis|medical record|date of birth|ssn|social security number|home address)\s*[:#])')


def _enabled():
    return os.getenv('VERIFYD_CIRCLE_THERAPY_LIBRARY_ENABLED', '0').lower().strip() in _TRUE


def _require():
    if not _enabled():
        raise HTTPException(503, detail={'error': 'therapy_library_not_enabled'})


def _auth(authorization):
    _require()
    identity = _identity_from_bearer(authorization)
    if identity.get('status') != 'active' or not identity.get('id'):
        raise HTTPException(401, detail={'error': 'session_required'})
    return str(identity['id'])


def _is_moderator(identity_id):
    ids = {v.strip() for v in os.getenv('VERIFYD_CIRCLE_THERAPY_LIBRARY_MODERATOR_IDS', '').split(',') if v.strip()}
    return identity_id in ids


def _moderator(identity_id):
    if not _is_moderator(identity_id):
        raise HTTPException(403, detail={'error': 'moderation_not_authorized'})


def _schema():
    global _READY
    if _READY:
        return
    with _LOCK:
        if _READY:
            return
        with get_db() as conn:
            cur = conn.cursor()
            cur.execute('''CREATE TABLE IF NOT EXISTS verifyd_therapy_library_profiles (
                identity_id TEXT PRIMARY KEY REFERENCES verifyd_identities(id) ON DELETE CASCADE,
                occupation TEXT NOT NULL,
                focus TEXT NOT NULL DEFAULT '',
                self_reported BOOLEAN NOT NULL DEFAULT TRUE,
                adult_attested_at TIMESTAMPTZ NOT NULL,
                privacy_ack_at TIMESTAMPTZ NOT NULL,
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )''')
            cur.execute('''CREATE TABLE IF NOT EXISTS verifyd_therapy_library_resources (
                resource_id TEXT PRIMARY KEY,
                author_id TEXT NOT NULL REFERENCES verifyd_identities(id) ON DELETE CASCADE,
                author_occupation TEXT NOT NULL,
                title TEXT NOT NULL,
                instructions TEXT NOT NULL,
                topic TEXT NOT NULL,
                age_band TEXT NOT NULL,
                difficulty TEXT NOT NULL,
                format TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending' CHECK(status IN ('pending','approved','rejected','withdrawn')),
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                moderated_at TIMESTAMPTZ,
                moderated_by TEXT REFERENCES verifyd_identities(id) ON DELETE SET NULL
            )''')
            cur.execute('''CREATE INDEX IF NOT EXISTS idx_therapy_library_catalog
                ON verifyd_therapy_library_resources(status,author_occupation,created_at DESC)''')
            cur.execute('''CREATE INDEX IF NOT EXISTS idx_therapy_library_author
                ON verifyd_therapy_library_resources(author_id,created_at DESC)''')
        _READY = True


def _clean(row):
    if row is None:
        return None
    data = dict(row)
    for key, val in data.items():
        if isinstance(val, datetime):
            data[key] = val.isoformat()
    data.pop('author_id', None)
    data.pop('moderated_by', None)
    data['author_verification'] = 'Occupation self-reported; credentials not verified'
    return data


class ProfileIn(BaseModel):
    occupation: str = Field(min_length=1, max_length=60)
    focus: str = Field(default='', max_length=100)
    adult_attestation: bool = False
    general_materials_only: bool = False


class ResourceIn(BaseModel):
    title: str = Field(min_length=3, max_length=160)
    instructions: str = Field(min_length=30, max_length=12000)
    topic: str = Field(min_length=2, max_length=80)
    age_band: str = Field(min_length=1, max_length=40)
    difficulty: str = Field(min_length=1, max_length=40)
    format: str = Field(min_length=1, max_length=50)
    rights_and_privacy_confirmed: bool = False


class ReviewIn(BaseModel):
    decision: Literal['approved', 'rejected']


@router.get('/health')
def health():
    # Do not expose identities, user counts or moderation details.
    return {'enabled': _enabled(), 'type': 'general_educational_materials',
            'moderated': True, 'credentials_verified': False, 'private_practice_unlocked': False}


@router.get('/taxonomy')
def taxonomy():
    return {'occupations': list(TOPICS), 'topics': TOPICS,
            'age_bands': AGE_BANDS, 'levels': LEVELS, 'formats': FORMATS,
            'note': 'General educational materials only. Adult accounts in the pilot.'}


@router.get('/profile')
def profile(authorization: Optional[str] = Header(default=None)):
    user = _auth(authorization)
    _schema()
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute('SELECT occupation,focus,self_reported,updated_at FROM verifyd_therapy_library_profiles WHERE identity_id=%s', (user,))
        item = cur.fetchone()
    return {'profile': _clean(item), 'verified': False, 'is_moderator': _is_moderator(user), 'can_share': bool(item),
            'private_practice_unlocked': False}


@router.put('/profile')
def save_profile(payload: ProfileIn, authorization: Optional[str] = Header(default=None)):
    user = _auth(authorization)
    if payload.occupation not in TOPICS or not payload.adult_attestation or not payload.general_materials_only:
        raise HTTPException(422, detail={'error': 'occupation_and_adult_privacy_attestation_required'})
    _schema()
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute('''INSERT INTO verifyd_therapy_library_profiles
             (identity_id,occupation,focus,self_reported,adult_attested_at,privacy_ack_at)
             VALUES (%s,%s,%s,TRUE,NOW(),NOW())
             ON CONFLICT(identity_id) DO UPDATE SET occupation=EXCLUDED.occupation,
             focus=EXCLUDED.focus,self_reported=TRUE,updated_at=NOW()''',
             (user,payload.occupation,payload.focus.strip()))
    return {'saved': True, 'occupation': payload.occupation, 'verified': False,
            'private_practice_unlocked': False}


@router.get('/catalog')
def catalog(occupation: str = '', topic: str = '', age_band: str = '',
            search: str = Query(default='', max_length=80),
            authorization: Optional[str] = Header(default=None)):
    _auth(authorization)
    _schema()
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute('''SELECT resource_id,author_occupation,title,instructions,topic,age_band,
            difficulty,format,status,created_at FROM verifyd_therapy_library_resources
            WHERE status='approved' AND (%s='' OR author_occupation=%s)
             AND (%s='' OR topic=%s) AND (%s='' OR age_band=%s)
             AND (%s='' OR title ILIKE %s OR instructions ILIKE %s)
            ORDER BY created_at DESC LIMIT 100''',
            (occupation,occupation,topic,topic,age_band,age_band,search,f'%{search}%',f'%{search}%'))
        found = [_clean(row) for row in cur.fetchall()]
    return {'resources': found, 'total_shown': len(found), 'moderation': 'approved_only'}


@router.get('/mine')
def mine(authorization: Optional[str] = Header(default=None)):
    user = _auth(authorization)
    _schema()
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute('''SELECT resource_id,author_occupation,title,instructions,topic,age_band,
            difficulty,format,status,created_at FROM verifyd_therapy_library_resources
            WHERE author_id=%s ORDER BY created_at DESC LIMIT 100''',(user,))
        rows=[_clean(row) for row in cur.fetchall()]
    return {'resources': rows}


@router.post('/resources', status_code=201)
def submit(payload: ResourceIn, authorization: Optional[str] = Header(default=None)):
    user = _auth(authorization)
    if not payload.rights_and_privacy_confirmed:
        raise HTTPException(422, detail={'error': 'sharing_consent_required'})
    content = f'{payload.title}\n{payload.instructions}'
    if _PRIVATE.search(content):
        raise HTTPException(422, detail={'error': 'possible_identifying_information'})
    if payload.age_band not in AGE_BANDS or payload.difficulty not in LEVELS or payload.format not in FORMATS:
        raise HTTPException(422, detail={'error': 'invalid_resource_category'})
    _schema()
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute('SELECT occupation FROM verifyd_therapy_library_profiles WHERE identity_id=%s', (user,))
        item = cur.fetchone()
        if not item:
            raise HTTPException(403, detail={'error': 'occupation_profile_required'})
        occupation = item['occupation']
        if payload.topic not in TOPICS.get(occupation, ()):
            raise HTTPException(422, detail={'error': 'topic_not_in_profession'})
        # Avoid mass posting; moderation is required even for valid text.
        cur.execute('''SELECT COUNT(*) AS n FROM verifyd_therapy_library_resources
            WHERE author_id=%s AND created_at > NOW() - INTERVAL '1 day' ''', (user,))
        if int(cur.fetchone()['n']) >= 10:
            raise HTTPException(429, detail={'error': 'daily_resource_limit'})
        resource_id = str(uuid.uuid4())
        cur.execute('''INSERT INTO verifyd_therapy_library_resources
            (resource_id,author_id,author_occupation,title,instructions,topic,age_band,difficulty,format)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)''',
            (resource_id,user,occupation,payload.title.strip(),payload.instructions.strip(),
             payload.topic,payload.age_band,payload.difficulty,payload.format))

    return {'resource_id': resource_id,'status':'pending','message':'Submitted for moderator review; not public yet.'}


@router.post('/resources/{resource_id}/withdraw')
def withdraw(resource_id: str, authorization: Optional[str] = Header(default=None)):
    user = _auth(authorization)
    _schema()
    with get_db() as conn:
        cur=conn.cursor()
        cur.execute('''UPDATE verifyd_therapy_library_resources SET status='withdrawn'
            WHERE resource_id=%s AND author_id=%s AND status IN ('pending','approved')
            RETURNING resource_id''',(resource_id,user))
        if not cur.fetchone():
            raise HTTPException(404, detail={'error':'resource_unavailable'})
    return {'status':'withdrawn'}


@router.get('/moderation/queue')
def queue(authorization: Optional[str] = Header(default=None)):
    user=_auth(authorization); _moderator(user); _schema()
    with get_db() as conn:
        cur=conn.cursor()
        cur.execute('''SELECT resource_id,author_occupation,title,instructions,topic,age_band,
            difficulty,format,status,created_at FROM verifyd_therapy_library_resources
            WHERE status='pending' ORDER BY created_at ASC LIMIT 100''')
        rows=[_clean(row) for row in cur.fetchall()]
    return {'resources':rows}


@router.post('/moderation/{resource_id}')
def review(resource_id: str, payload: ReviewIn, authorization: Optional[str] = Header(default=None)):
    user=_auth(authorization);_moderator(user);_schema()
    with get_db() as conn:
        cur=conn.cursor()
        cur.execute('''UPDATE verifyd_therapy_library_resources SET status=%s,moderated_at=NOW(),moderated_by=%s
            WHERE resource_id=%s AND status='pending' RETURNING resource_id''',
            (payload.decision,user,resource_id))
        if not cur.fetchone():
            raise HTTPException(404,detail={'error':'pending_resource_unavailable'})
    return {'status':payload.decision}
