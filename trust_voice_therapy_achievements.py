"""VeriFYD Circle — server-tracked Therapy Studio participation achievements.

Participation credits are NOT clinical scores, patient records, medical advice,
professional credentials, or evidence of improvement. Never persist responses,
imported documents, content, patient names, input device, or clinical context.

Feature gates:
  VERIFYD_CIRCLE_RANKING_ENABLED=1
  VERIFYD_CIRCLE_THERAPY_RANK_ENABLED=1

A server-issued per-step token and an ordered, server-persisted practice plan
provide basic participation evidence. They do not certify that therapy was
performed or prevent all automation. Credits are one-time per activity + level.
"""
from __future__ import annotations

import os
import secrets
import threading
import uuid
from collections import Counter
from typing import Literal, Optional

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field

from database import get_db
from trust_voice_messages import _identity_from_bearer
from trust_voice_ranks import _schema as rank_schema, enabled as rank_enabled

router = APIRouter(prefix='/trust-voice/therapy-achievements', tags=['Therapy Studio Achievements'])
_LOCK = threading.Lock()
_READY = False
_TRUE = {'1', 'true', 'yes', 'on'}

# Fixed, reviewed, server-authoritative values. No client-provided point counts.
ACTIVITIES = {
    'compass': {'name': 'Communication Compass', 'points': {'gentle': 10, 'explore': 25, 'challenge': 40}},
    'story': {'name': 'Story Builder', 'points': {'gentle': 15, 'explore': 30, 'challenge': 50}},
    'sequence': {'name': 'Everyday Sequencing', 'points': {'gentle': 15, 'explore': 30, 'challenge': 50}},
    'social': {'name': 'Social Scenarios', 'points': {'gentle': 15, 'explore': 30, 'challenge': 50}},
    'focus': {'name': 'Focus & Choices', 'points': {'gentle': 10, 'explore': 25, 'challenge': 40}},
    'comfort': {'name': 'Comfort & Check-In', 'points': {'gentle': 10, 'explore': 20, 'challenge': 35}},
    'everyday-choices-v1': {'name': 'Everyday Choices', 'points': {'gentle': 15, 'explore': 30}},
    'conversation-practice-v1': {'name': 'Conversation Practice', 'points': {'gentle': 20, 'explore': 35}},
    'gaze-target-v1': {'name': 'Gaze Target Practice', 'points': {'gentle': 20}},
}
GUIDED_STEPS = {'everyday-choices-v1': 6, 'conversation-practice-v1': 8, 'gaze-target-v1': 12}
GAZE_TARGETS = (0, 3, 6, 8, 2, 7, 4, 1, 5, 0, 8, 6)
STUDIO_IDS = tuple(key for key in ACTIVITIES if key not in GUIDED_STEPS)
LEVELS = ('gentle', 'explore', 'challenge')


def enabled():
    return (os.getenv('VERIFYD_CIRCLE_THERAPY_RANK_ENABLED', '0').strip().lower() in _TRUE
            and rank_enabled())


def _require():
    if not enabled():
        raise HTTPException(404, detail={'error': 'therapy_rank_not_enabled'})


def _identity(auth):
    _require()
    item = _identity_from_bearer(auth)
    if item.get('status') != 'active' or not item.get('id'):
        raise HTTPException(401, detail={'error': 'session_required'})
    return str(item['id'])


def _schema():
    global _READY
    if _READY:
        return
    with _LOCK:
        if _READY:
            return
        rank_schema()   # Must happen before acquiring individual session row locks.
        with get_db() as conn:
            cur = conn.cursor()
            cur.execute('''CREATE TABLE IF NOT EXISTS verifyd_therapy_rank_sessions (
                session_id TEXT PRIMARY KEY,
                identity_id TEXT NOT NULL REFERENCES verifyd_identities(id) ON DELETE CASCADE,
                level TEXT NOT NULL CHECK(level IN ('gentle','explore','challenge')),
                activity_ids TEXT[] NOT NULL,
                rounds INTEGER NOT NULL CHECK(rounds BETWEEN 1 AND 12),
                plan TEXT[] NOT NULL,
                step_actions TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
                current_index INTEGER NOT NULL DEFAULT 0,
                next_token TEXT NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                completed_at TIMESTAMPTZ,
                expires_at TIMESTAMPTZ NOT NULL DEFAULT NOW() + INTERVAL '30 days'
            )''')
            cur.execute('''CREATE INDEX IF NOT EXISTS idx_therapy_rank_sessions_user
                ON verifyd_therapy_rank_sessions(identity_id,created_at DESC)''')
        _READY = True


class StartSession(BaseModel):
    activity_ids: list[Literal['compass', 'story', 'sequence', 'social', 'focus', 'comfort', 'everyday-choices-v1', 'conversation-practice-v1', 'gaze-target-v1']] = Field(min_length=1, max_length=6)
    level: Literal['gentle', 'explore', 'challenge']
    rounds: int = Field(ge=1, le=12)


class CompleteStep(BaseModel):
    step_index: int = Field(ge=0)
    activity_id: Literal['compass', 'story', 'sequence', 'social', 'focus', 'comfort', 'everyday-choices-v1', 'conversation-practice-v1', 'gaze-target-v1']
    action: Literal['completed', 'skipped']
    step_token: str = Field(min_length=16, max_length=128)
    target_index: Optional[int] = Field(default=None, ge=0, le=8)


def _awards_for(plan, actions, rounds, level):
    """Only full completion of every step for an activity is eligible."""
    if len(plan) != len(actions):
        raise ValueError('Incomplete practice evidence')
    complete = Counter(activity for activity, action in zip(plan, actions) if action == 'completed')
    ids = list(dict.fromkeys(plan))
    return [{'activity_id': activity, 'points': ACTIVITIES[activity]['points'][level],
             'challenge_id': f'therapy:studio:{activity}:{level}' if activity in STUDIO_IDS else f'therapy:guided:{activity}:{level}'}
            for activity in ids if complete[activity] == rounds]


@router.get('/catalog')
def catalog():
    return {'enabled': enabled(), 'levels': list(LEVELS),
            'activities': [{'id': key, 'name': item['name'], 'points': item['points']}
                           for key, item in ACTIVITIES.items()],
            'once_per_activity_and_level': True,
            'guided_steps': GUIDED_STEPS,
            'mode': 'participation_not_clinical_grade',
            'note': 'Only completed, non-skipped practice counts. Imported/private prompts are never sent to the server.'}


@router.get('/me')
def my_achievements(authorization: Optional[str] = Header(default=None)):
    user = _identity(authorization)
    _schema()
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute('''SELECT challenge_id,points,created_at FROM verifyd_circle_rank_awards
            WHERE identity_id=%s AND challenge_id LIKE 'therapy:%%'
            ORDER BY created_at DESC LIMIT 100''', (user,))
        items = [{'challenge_id': r['challenge_id'], 'points': int(r['points']),
                  'created_at': r['created_at'].isoformat() if r['created_at'] else None}
                 for r in cur.fetchall()]
    return {'earned': items, 'total_therapy_points': sum(x['points'] for x in items)}


@router.post('/sessions', status_code=201)
def start(payload: StartSession, authorization: Optional[str] = Header(default=None)):
    user = _identity(authorization)
    if len(set(payload.activity_ids)) != len(payload.activity_ids):
        raise HTTPException(422, detail={'error': 'duplicate_activity'})
    guided = [aid for aid in payload.activity_ids if aid in GUIDED_STEPS]
    if guided and (len(payload.activity_ids) != 1
                   or payload.rounds != GUIDED_STEPS[guided[0]]
                   or payload.level not in ACTIVITIES[guided[0]]['points']):
        raise HTTPException(422, detail={'error': 'invalid_guided_activity_plan'})
    if not guided and payload.rounds > 5:
        raise HTTPException(422, detail={'error': 'invalid_studio_rounds'})
    _schema()
    plan = [activity for activity in payload.activity_ids for _ in range(payload.rounds)]
    sid = str(uuid.uuid4())
    token = secrets.token_urlsafe(32)
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute('''SELECT COUNT(*) AS n FROM verifyd_therapy_rank_sessions
            WHERE identity_id=%s AND created_at > NOW()-INTERVAL '1 day' ''', (user,))
        if int(cur.fetchone()['n']) >= 60:
            raise HTTPException(429, detail={'error': 'too_many_practice_sessions'})
        cur.execute('''INSERT INTO verifyd_therapy_rank_sessions
            (session_id,identity_id,level,activity_ids,rounds,plan,next_token)
            VALUES (%s,%s,%s,%s,%s,%s,%s)''',
            (sid, user, payload.level, payload.activity_ids, payload.rounds, plan, token))
    return {'session_id': sid, 'step_index': 0, 'step_token': token,
            'total_steps': len(plan), 'points_if_first_completion':
            sum(ACTIVITIES[x]['points'][payload.level] for x in payload.activity_ids),
            'once_per_activity_and_level': True}


@router.post('/sessions/{session_id}/step')
def step(session_id: str, payload: CompleteStep,
         authorization: Optional[str] = Header(default=None)):
    user = _identity(authorization)
    _schema()
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute('''SELECT level,activity_ids,rounds,plan,step_actions,current_index,
            next_token,completed_at,expires_at
            FROM verifyd_therapy_rank_sessions
            WHERE session_id=%s AND identity_id=%s FOR UPDATE''', (session_id, user))
        row = cur.fetchone()
        if row is None:
            raise HTTPException(404, detail={'error': 'practice_session_unavailable'})
        if row['completed_at'] is not None:
            # Session is terminal; cannot be replayed for more points.
            raise HTTPException(409, detail={'error': 'practice_session_completed'})
        # No timed tests: 30-day token lifetime supports patients' breaks.
        from datetime import datetime, timezone
        if row['expires_at'] <= datetime.now(timezone.utc):
            raise HTTPException(410, detail={'error': 'practice_session_expired'})
        index = int(row['current_index'])
        plan = list(row['plan'])
        if (payload.step_index != index or index >= len(plan)
                or payload.activity_id != plan[index]
                or not secrets.compare_digest(payload.step_token, row['next_token'])):
            raise HTTPException(409, detail={'error': 'practice_step_mismatch'})
        if (payload.activity_id == 'gaze-target-v1' and payload.action == 'completed'
                and payload.target_index != GAZE_TARGETS[index]):
            raise HTTPException(422, detail={'error': 'incorrect_target'})
        if payload.activity_id in GUIDED_STEPS and payload.action == 'skipped':
            raise HTTPException(422, detail={'error': 'guided_practice_skip_unavailable'})
        actions = list(row['step_actions']) + [payload.action]
        final = index + 1 == len(plan)
        next_token = secrets.token_urlsafe(32)
        cur.execute('''UPDATE verifyd_therapy_rank_sessions SET
            step_actions=%s, current_index=%s, next_token=%s,
            completed_at=CASE WHEN %s THEN NOW() ELSE NULL END
            WHERE session_id=%s''', (actions, index + 1, next_token, final, session_id))
        awarded = []
        if final:
            for item in _awards_for(plan, actions, int(row['rounds']), row['level']):
                cur.execute('''INSERT INTO verifyd_circle_rank_awards
                    (award_id,identity_id,challenge_id,kind,points,evidence_ref)
                    VALUES (%s,%s,%s,'challenge',%s,%s)
                    ON CONFLICT DO NOTHING RETURNING award_id''',
                    (str(uuid.uuid4()), user, item['challenge_id'], item['points'],
                     f'therapy-session:{session_id}:{item["activity_id"]}'))
                awarded.append({**item, 'points_added': item['points'] if cur.fetchone() else 0})
    return {'completed': final, 'next_step_index': index + 1,
            'next_step_token': None if final else next_token,
            'awards': awarded,
            'points_added': sum(item['points_added'] for item in awarded),
            'already_earned': bool(final and awarded and not any(item['points_added'] for item in awarded)),
            'note': 'Achievement points reflect participation, not clinical progress.' if final else None}
