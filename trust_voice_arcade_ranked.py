"""VeriFYD Circle ranked Challenge Arcade + optional teacher-directed game assignments.

Feature-gated, OFF by default. Five existing accessible browser puzzles are verified
against a server-issued seed and round sequence. The client cannot award points.
Never logs puzzle answers, grades disability, or uploads local practice history.
"""
from __future__ import annotations
import os
import secrets
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Literal, Optional
from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field
from database import get_db
from trust_voice_messages import _identity_from_bearer

router = APIRouter(prefix='/trust-voice/arcade', tags=['Circle Ranked Games'])
GAMES = ('pattern', 'logic', 'word', 'route', 'mosaic')
LEVEL_POINTS = (10, 25, 50)
LEVEL_NAMES = ('Beginner', 'Explorer', 'Expert')
ROUND_COUNTS = (4, 5, 6)
SYMBOLS = ('🌟', '💜', '🌈', '🍀', '🎵')
_READY = False
_LOCK = threading.Lock()
_YES = ('1', 'true', 'yes', 'on')


def enabled():
    from trust_voice_ranks import enabled as ranks_enabled
    return os.getenv('VERIFYD_CIRCLE_ARCADE_RANK_ENABLED', '0').lower().strip() in _YES and ranks_enabled()


def classroom_enabled():
    return enabled() and os.getenv('VERIFYD_CIRCLE_CLASSROOM_GAME_ASSIGNMENTS_ENABLED', '0').lower().strip() in _YES


def require():
    if not enabled():
        raise HTTPException(404, detail={'error': 'ranked_arcade_unavailable'})


def identity(authorization):
    require()
    value = _identity_from_bearer(authorization)
    if value.get('status') != 'active':
        raise HTTPException(401, detail={'error': 'session_required'})
    return str(value['id'])


def schema():
    global _READY
    if _READY:
        return
    with _LOCK:
        if _READY:
            return
        from trust_voice_ranks import _schema as rank_schema
        rank_schema()
        from trust_voice_classrooms import _schema as classroom_schema
        classroom_schema()
        with get_db() as conn:
            cur = conn.cursor()
            cur.execute('''CREATE TABLE IF NOT EXISTS verifyd_arcade_sessions (
                session_id TEXT PRIMARY KEY, identity_id TEXT NOT NULL REFERENCES verifyd_identities(id) ON DELETE CASCADE,
                game TEXT NOT NULL, level INTEGER NOT NULL CHECK(level BETWEEN 0 AND 2),
                seed TEXT NOT NULL, target_language TEXT NOT NULL DEFAULT 'en-US',
                current_round INTEGER NOT NULL DEFAULT 0,
                assignment_id TEXT,
                completed_at TIMESTAMPTZ,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                expires_at TIMESTAMPTZ NOT NULL DEFAULT NOW() + INTERVAL '24 hours'
            )''')
            cur.execute('''CREATE INDEX IF NOT EXISTS idx_arcade_session_identity
                ON verifyd_arcade_sessions(identity_id,created_at DESC)''')
            cur.execute('''CREATE TABLE IF NOT EXISTS verifyd_class_game_assignments (
                assignment_id TEXT PRIMARY KEY,
                class_id TEXT NOT NULL REFERENCES verifyd_circle_classes(class_id) ON DELETE CASCADE,
                teacher_id TEXT NOT NULL REFERENCES verifyd_identities(id) ON DELETE CASCADE,
                title TEXT NOT NULL, instructions TEXT NOT NULL DEFAULT '',
                game TEXT NOT NULL, level INTEGER NOT NULL CHECK(level BETWEEN 0 AND 2),
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )''')
            cur.execute('''CREATE TABLE IF NOT EXISTS verifyd_class_game_recipients (
                assignment_id TEXT NOT NULL REFERENCES verifyd_class_game_assignments(assignment_id) ON DELETE CASCADE,
                student_id TEXT NOT NULL REFERENCES verifyd_identities(id) ON DELETE CASCADE,
                PRIMARY KEY(assignment_id,student_id)
            )''')
            cur.execute('''CREATE TABLE IF NOT EXISTS verifyd_class_game_completions (
                assignment_id TEXT NOT NULL REFERENCES verifyd_class_game_assignments(assignment_id) ON DELETE CASCADE,
                student_id TEXT NOT NULL REFERENCES verifyd_identities(id) ON DELETE CASCADE,
                completed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                evidence_session_id TEXT NOT NULL UNIQUE,
                PRIMARY KEY(assignment_id,student_id)
            )''')
        _READY = True


def hash_seed(value):
    # FNV1a 32-bit parity with trustVoiceChallengeEngine.js hashSeed (ASCII seeds).
    result = 2166136261
    raw = str(value).encode('utf-16-le')
    for i in range(0, len(raw), 2):
        unit = raw[i] | (raw[i+1] << 8)
        result = ((result ^ unit) * 16777619) & 0xffffffff
    return result


def shuffled(items, seed):
    values = list(items)
    x = hash_seed(seed) or 1
    for i in range(len(values)-1,0,-1):
        x = (x*1664525 + 1013904223) & 0xffffffff
        j = x % (i+1)
        values[i], values[j] = values[j], values[i]
    return values


def expected(game, level, round_index, seed):
    if game == 'pattern':
        icons = SYMBOLS[:3+level]
        return [icons[hash_seed(f'{seed}:pattern:{round_index}:{i}') % len(icons)]
                for i in range(min(3+level+round_index,10))]
    if game == 'logic':
        hashed = hash_seed(f'{seed}:logic:{round_index}')
        base = 2 + hashed % 6
        step = 1 + (hashed >> 4) % (2 + level * 2)
        if level == 0 or (level == 1 and round_index % 2 == 0):
            return base + 4*step
        if level == 1 or round_index % 3 == 0:
            return base + 2*step
        if round_index % 3 == 1:
            return base + 16*step
        return base + 7*step
    if game == 'word':
        return hash_seed(f'{seed}:word:{round_index}:{level}') % 12
    if game == 'route':
        size = level + 3
        guaranteed = list(range(size)) + [(i+1)*size+size-1 for i in range(size-1)]
        safe = set(guaranteed)
        candidates = shuffled([i for i in range(size*size) if i not in safe], f'{seed}:blocked:{round_index}')
        walls = set(candidates[:min(len(candidates),1 + level*2 + (round_index%2))])
        return (size,walls)
    if game == 'mosaic':
        size = level + 3
        target = []
        for row in range(size):
            for col in range((size+1)//2):
                if hash_seed(f'{seed}:mosaic:{round_index}:{row}:{col}') % 3 != 0:
                    target.append(row*size+col)
                    if col != size-1-col:
                        target.append(row*size+size-1-col)
        if not target:
            target.append(size*size//2)
        if len(target) == size*size:
            target.pop()
        return sorted(target)
    raise ValueError('Invalid game')


def verify_round(game,level,round_index,seed,proof):
    answer = expected(game,level,round_index,seed)
    if game == 'route':
        size,walls = answer
        if not isinstance(proof,list) or not (size*2-1 <= len(proof) <= size*size+10):
            return False
        if proof[0] != 0 or proof[-1] != size*size-1 or any(type(i) is not int or i < 0 or i >= size*size or i in walls for i in proof):
            return False
        return all(abs(a//size-b//size)+abs(a%size-b%size) == 1 for a,b in zip(proof,proof[1:]))
    if game in ('mosaic','pattern'):
        if not isinstance(proof,list): return False
        if game == 'mosaic':
            return all(type(x) is int for x in proof) and len(proof)==len(answer) and len(set(proof))==len(answer) and sorted(proof)==answer
        return proof == answer
    return type(proof) is int and proof == answer


def challenge_id(game,level):
    return f'arcade-{game}-{LEVEL_NAMES[level].lower()}'


class Begin(BaseModel):
    game: Literal['pattern','logic','word','route','mosaic']
    level: int = Field(ge=0,le=2)
    target_language: str = Field(default='en-US',max_length=12)
    assignment_id: Optional[str] = Field(default=None,max_length=128)


class RoundProof(BaseModel):
    round: int = Field(ge=0,le=5)
    proof: Any


class TeacherGameAssignment(BaseModel):
    title: str = Field(min_length=3,max_length=160)
    instructions: str = Field(default='',max_length=2000)
    game: Literal['pattern','logic','word','route','mosaic']
    level: int = Field(ge=0,le=2)
    recipient_ids: list[str] = Field(min_length=1,max_length=100)


@router.get('/catalog')
def catalog(authorization:Optional[str]=Header(default=None)):
    user = identity(authorization)
    schema()
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute('''SELECT challenge_id FROM verifyd_circle_rank_awards WHERE identity_id=%s
            AND challenge_id LIKE 'arcade-%%' ''',(user,))
        earned = {r['challenge_id'] for r in cur.fetchall()}
    return {'games':list(GAMES),'levels':[{'level':i,'label':LEVEL_NAMES[i],'points':LEVEL_POINTS[i]} for i in range(3)],
            'earned_challenges':sorted(earned),'once_per_game_and_level':True,
            'warning':'Activity points are not identity verification or a clinical score.'}


@router.post('/sessions')
def begin(payload:Begin,authorization:Optional[str]=Header(default=None)):
    user = identity(authorization)
    schema()
    if payload.target_language not in ('en-US','es-MX','zh-CN','zh-TW','fr-FR','pt-BR','ar','hi-IN','ja-JP'):
        raise HTTPException(422,detail={'error':'unsupported_language'})
    linked = payload.assignment_id or None
    if linked and not classroom_enabled():
        raise HTTPException(404,detail={'error':'classroom_game_unavailable'})
    session_id = str(uuid.uuid4())
    seed = secrets.token_urlsafe(18)
    with get_db() as conn:
        cur = conn.cursor()
        if linked:
            from trust_voice_classrooms import _member
            cur.execute('''SELECT g.class_id,g.game,g.level FROM verifyd_class_game_assignments g
                JOIN verifyd_class_game_recipients r ON r.assignment_id=g.assignment_id AND r.student_id=%s
                WHERE g.assignment_id=%s''',(user,linked))
            assignment = cur.fetchone()
            if not assignment or assignment['game']!=payload.game or assignment['level']!=payload.level:
                raise HTTPException(404,detail={'error':'game_assignment_unavailable'})
            member = _member(cur,assignment['class_id'],user)
            if member['teacher_id']==user:
                raise HTTPException(403,detail={'error':'students_only'})
        cur.execute('''SELECT COUNT(*) AS n FROM verifyd_arcade_sessions WHERE identity_id=%s
            AND created_at > NOW()-INTERVAL '1 day' AND completed_at IS NULL''',(user,))
        if cur.fetchone()['n'] >= 60:
            raise HTTPException(429,detail={'error':'too_many_game_sessions'})
        cur.execute('''INSERT INTO verifyd_arcade_sessions
            (session_id,identity_id,game,level,seed,target_language,assignment_id)
            VALUES (%s,%s,%s,%s,%s,%s,%s)''',
            (session_id,user,payload.game,payload.level,seed,payload.target_language,linked))
    return {'session_id':session_id,'seed':seed,'game':payload.game,'level':payload.level,
            'target_language':payload.target_language,'round':0,'total_rounds':ROUND_COUNTS[payload.level]}


@router.post('/sessions/{session_id}/round')
def round_complete(session_id:str,payload:RoundProof,authorization:Optional[str]=Header(default=None)):
    user = identity(authorization)
    schema()
    added=0
    verified=False
    already_earned=False
    with get_db() as conn:
        cur=conn.cursor()
        cur.execute('''SELECT game,level,seed,current_round,assignment_id,completed_at,expires_at
            FROM verifyd_arcade_sessions WHERE session_id=%s AND identity_id=%s FOR UPDATE''',(session_id,user))
        row=cur.fetchone()
        if not row:
            raise HTTPException(404,detail={'error':'game_session_unavailable'})
        if row['completed_at']:
            raise HTTPException(409,detail={'error':'game_session_completed'})
        if row['expires_at']<datetime.now(timezone.utc):
            raise HTTPException(410,detail={'error':'game_session_expired'})
        if row['current_round']!=payload.round:
            raise HTTPException(409,detail={'error':'round_out_of_sequence'})
        # Bound input before any potentially expensive puzzle validation.
        if isinstance(payload.proof,list) and (len(payload.proof)>64 or any(
            not isinstance(x,(int,str)) or (isinstance(x,str) and len(x)>12) for x in payload.proof)):
            raise HTTPException(422,detail={'error':'invalid_round_proof'})
        if row['assignment_id'] and payload.round+1 == ROUND_COUNTS[row['level']]:
            from trust_voice_classrooms import _member
            cur.execute('''SELECT class_id FROM verifyd_class_game_assignments a
                JOIN verifyd_class_game_recipients r ON r.assignment_id=a.assignment_id AND r.student_id=%s
                WHERE a.assignment_id=%s''',(user,row['assignment_id']))
            assigned=cur.fetchone()
            if not assigned: raise HTTPException(404,detail={'error':'game_assignment_unavailable'})
            _member(cur,assigned['class_id'],user)
        verified=verify_round(row['game'],row['level'],payload.round,row['seed'],payload.proof)
        if verified:
            next_round=payload.round+1
            done=next_round>=ROUND_COUNTS[row['level']]
            cur.execute('''UPDATE verifyd_arcade_sessions SET current_round=%s,
                completed_at=CASE WHEN %s THEN NOW() ELSE NULL END
                WHERE session_id=%s''',(next_round,done,session_id))
            if done:
                key=challenge_id(row['game'],row['level'])
                cur.execute('''INSERT INTO verifyd_circle_rank_awards
                    (award_id,identity_id,challenge_id,kind,points,evidence_ref)
                    VALUES (%s,%s,%s,'game',%s,%s) ON CONFLICT DO NOTHING RETURNING award_id''',
                    (str(uuid.uuid4()),user,key,LEVEL_POINTS[row['level']],f'arcade-session:{session_id}'))
                added=LEVEL_POINTS[row['level']] if cur.fetchone() else 0
                already_earned=not bool(added)
                if row['assignment_id']:
                    cur.execute('''INSERT INTO verifyd_class_game_completions
                        (assignment_id,student_id,evidence_session_id) VALUES (%s,%s,%s)
                        ON CONFLICT DO NOTHING''',(row['assignment_id'],user,session_id))
    return {'correct':verified,'round':payload.round+1 if verified else payload.round,
            'completed':bool(verified and payload.round+1>=ROUND_COUNTS[row['level']]),
            'points_added':added,'already_earned':already_earned,
            'rank_updated':bool(added),'next_step':'continue' if verified else 'try_again'}


@router.get('/classes/{class_id}/assignments')
def game_assignments(class_id:str,authorization:Optional[str]=Header(default=None)):
    user=identity(authorization)
    if not classroom_enabled(): raise HTTPException(404,detail={'error':'classroom_game_unavailable'})
    schema()
    from trust_voice_classrooms import _member
    with get_db() as conn:
        cur=conn.cursor()
        member=_member(cur,class_id,user)
        teacher=(member['teacher_id']==user)
        if teacher:
            cur.execute('''SELECT g.assignment_id,g.title,g.instructions,g.game,g.level,g.created_at,
                (SELECT COUNT(*) FROM verifyd_class_game_recipients r WHERE r.assignment_id=g.assignment_id) AS recipients,
                (SELECT COUNT(*) FROM verifyd_class_game_completions c WHERE c.assignment_id=g.assignment_id) AS completions
                FROM verifyd_class_game_assignments g WHERE g.class_id=%s
                ORDER BY g.created_at DESC LIMIT 100''',(class_id,))
        else:
            cur.execute('''SELECT g.assignment_id,g.title,g.instructions,g.game,g.level,g.created_at,
                (c.completed_at IS NOT NULL) AS completed
                FROM verifyd_class_game_assignments g
                JOIN verifyd_class_game_recipients r ON r.assignment_id=g.assignment_id AND r.student_id=%s
                LEFT JOIN verifyd_class_game_completions c ON c.assignment_id=g.assignment_id AND c.student_id=%s
                WHERE g.class_id=%s ORDER BY g.created_at DESC LIMIT 100''',(user,user,class_id))
        rows=[dict(item) for item in cur.fetchall()]
    for item in rows:
        if isinstance(item.get('created_at'),datetime):item['created_at']=item['created_at'].isoformat()
    return {'assignments':rows,'my_role':'teacher' if teacher else 'student'}


@router.post('/classes/{class_id}/assignments')
def assign_to_class(class_id:str,payload:TeacherGameAssignment,authorization:Optional[str]=Header(default=None)):
    user=identity(authorization)
    if not classroom_enabled(): raise HTTPException(404,detail={'error':'classroom_game_unavailable'})
    if len(set(payload.recipient_ids))!=len(payload.recipient_ids):
        raise HTTPException(422,detail={'error':'duplicate_recipients'})
    if any(not isinstance(i,str) or not (1<=len(i)<=128) for i in payload.recipient_ids):
        raise HTTPException(422,detail={'error':'invalid_recipient'})
    schema()
    from trust_voice_classrooms import _teacher
    assignment_id=str(uuid.uuid4())
    with get_db() as conn:
        cur=conn.cursor()
        _teacher(cur,class_id,user)
        cur.execute('''SELECT student_id FROM verifyd_circle_class_members
            WHERE class_id=%s AND status='approved' AND student_id=ANY(%s)''',(class_id,payload.recipient_ids))
        if len(cur.fetchall())!=len(payload.recipient_ids):
            raise HTTPException(422,detail={'error':'recipient_not_enrolled'})
        cur.execute('''INSERT INTO verifyd_class_game_assignments
            (assignment_id,class_id,teacher_id,title,instructions,game,level)
            VALUES(%s,%s,%s,%s,%s,%s,%s)''',
            (assignment_id,class_id,user,payload.title.strip(),payload.instructions.strip(),payload.game,payload.level))
        for student_id in payload.recipient_ids:
            cur.execute('''INSERT INTO verifyd_class_game_recipients(assignment_id,student_id)
                VALUES (%s,%s)''',(assignment_id,student_id))
    return {'assignment_id':assignment_id,'assigned':len(payload.recipient_ids),
            'game':payload.game,'level':payload.level,'rank_points_if_first_completion':LEVEL_POINTS[payload.level]}


@router.get('/classes/{class_id}/assignments/{assignment_id}/results')
def assignment_results(class_id:str,assignment_id:str,authorization:Optional[str]=Header(default=None)):
    user=identity(authorization)
    if not classroom_enabled(): raise HTTPException(404,detail={'error':'classroom_game_unavailable'})
    schema()
    from trust_voice_classrooms import _teacher
    with get_db() as conn:
        cur=conn.cursor()
        _teacher(cur,class_id,user)
        cur.execute('''SELECT r.student_id,i.handle,c.completed_at
            FROM verifyd_class_game_assignments a
            JOIN verifyd_class_game_recipients r ON r.assignment_id=a.assignment_id
            JOIN verifyd_identities i ON i.id=r.student_id
            LEFT JOIN verifyd_class_game_completions c ON c.assignment_id=r.assignment_id AND c.student_id=r.student_id
            WHERE a.class_id=%s AND a.assignment_id=%s ORDER BY i.handle LIMIT 100''',(class_id,assignment_id))
        rows=[dict(row) for row in cur.fetchall()]
    for row in rows:
        if isinstance(row.get('completed_at'),datetime): row['completed_at']=row['completed_at'].isoformat()
    return {'recipients':rows,'teacher_only':True}