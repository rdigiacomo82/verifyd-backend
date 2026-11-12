"""Server-controlled Circle Match and Star Quest achievement sessions.

Opt-in only: VERIFYD_CIRCLE_VERIFIED_GAMES_ENABLED=1, plus existing rank flag.
All gameplay state is stored server-side. No endpoint accepts a client's claimed
score or completed activity. Inaccessible/offline devices keep existing games.
"""
from __future__ import annotations

import json
import os
import secrets
import threading
import uuid
from datetime import datetime, timezone
from typing import Literal, Optional

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel

from database import get_db
from trust_voice_ranks import _owner, _schema as rank_schema

router = APIRouter(prefix='/trust-voice/games', tags=['Circle Verified Games'])
LOCK = threading.Lock()
READY = False
MATCH_SYMBOLS = ['★', '♥', '☀', '☾', '◆', '♫']
STAR_SYMBOLS = ['🐶', '🎨', '⭐', '⚽', '🌻', '🎵', '🦋', '🚀']
STAR_CHOICES = (3, 4, 5)
MAX_MOVES = 300


def enabled():
    return os.getenv('VERIFYD_CIRCLE_VERIFIED_GAMES_ENABLED', '0').lower() in ('1', 'yes', 'true', 'on')


def require_enabled():
    if not enabled():
        raise HTTPException(404, detail={'error': 'verified_games_unavailable'})


def schema():
    global READY
    if READY:
        return
    with LOCK:
        if READY:
            return
        with get_db() as conn:
            cur = conn.cursor()
            cur.execute('''CREATE TABLE IF NOT EXISTS verifyd_circle_game_sessions (
                session_id TEXT PRIMARY KEY,
                identity_id TEXT NOT NULL REFERENCES verifyd_identities(id) ON DELETE CASCADE,
                game_key TEXT NOT NULL CHECK (game_key IN ('circle-match','star-quest')),
                variant SMALLINT NOT NULL,
                state JSONB NOT NULL,
                status TEXT NOT NULL DEFAULT 'in_progress'
                    CHECK (status IN ('in_progress','completed','abandoned')),
                started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                expires_at TIMESTAMPTZ NOT NULL DEFAULT (NOW() + INTERVAL '24 hours'),
                completed_at TIMESTAMPTZ
            )''')
            cur.execute('''CREATE INDEX IF NOT EXISTS idx_circle_game_sessions_owner
                ON verifyd_circle_game_sessions(identity_id, status, started_at DESC)''')
        READY = True


def star_round(tier, round_num):
    target_index = (tier * 2 + round_num) % len(STAR_SYMBOLS)
    target = STAR_SYMBOLS[target_index]
    count = STAR_CHOICES[tier]
    placement = (round_num + tier) % count
    options = []
    for position in range(count):
        if position == placement:
            options.append(target)
        else:
            offset = position + 1 if position < placement else position
            options.append(STAR_SYMBOLS[(target_index + offset + 1) % len(STAR_SYMBOLS)])
    return target, options


def create_initial_state(game, variant):
    if game == 'circle-match':
        if variant not in (2, 4, 6):
            raise HTTPException(400, detail={'error': 'invalid_difficulty'})
        symbols = MATCH_SYMBOLS[:variant] * 2
        secrets.SystemRandom().shuffle(symbols)
        return {'deck': symbols, 'matched': [], 'selected': [], 'attempts': 0, 'moves': 0, 'mismatch': False}
    if game == 'star-quest':
        if variant not in (0, 1, 2):
            raise HTTPException(400, detail={'error': 'invalid_level'})
        return {'round': 0, 'attempts': 0, 'moves': 0}
    raise HTTPException(400, detail={'error': 'invalid_game'})


def public_state(game, variant, state, finished=False):
    if game == 'circle-match':
        visible = set(state['matched']) | set(state['selected'])
        return {
            'game': game, 'variant': variant, 'finished': finished,
            'cards': [state['deck'][i] if i in visible else None for i in range(len(state['deck']))],
            'matched': state['matched'], 'revealed': state['selected'],
            'mismatch': state['mismatch'], 'attempts': state['attempts'],
        }
    target, choices = star_round(variant, min(state['round'], 4))
    return {'game': game, 'variant': variant, 'finished': finished,
            'round': state['round'], 'target': target, 'choices': choices,
            'attempts': state['attempts']}


def apply_choice(game, variant, state, choice):
    if state['moves'] >= MAX_MOVES:
        raise HTTPException(429, detail={'error': 'too_many_moves'})
    state['moves'] += 1
    if game == 'circle-match':
        if not isinstance(choice, int) or isinstance(choice, bool) or not 0 <= choice < len(state['deck']):
            raise HTTPException(400, detail={'error': 'invalid_card'})
        if choice in state['matched']:
            raise HTTPException(409, detail={'error': 'already_matched'})
        if state['mismatch']:
            state['mismatch'] = False
            state['selected'] = []
        if choice in state['selected']:
            raise HTTPException(409, detail={'error': 'card_already_selected'})
        if not state['selected']:
            state['selected'] = [choice]
        else:
            first = state['selected'][0]
            state['attempts'] += 1
            if state['deck'][first] == state['deck'][choice]:
                state['matched'].extend((first, choice))
                state['selected'] = []
            else:
                state['selected'] = [first, choice]
                state['mismatch'] = True
        return len(state['matched']) == len(state['deck'])
    target, options = star_round(variant, state['round'])
    if not isinstance(choice, str) or choice not in options:
        raise HTTPException(400, detail={'error': 'invalid_choice'})
    state['attempts'] += 1
    if choice == target:
        state['round'] += 1
    return state['round'] >= 5


class Start(BaseModel):
    game: Literal['circle-match', 'star-quest']
    variant: int


class Move(BaseModel):
    choice: int | str


@router.get('/catalog')
def catalog():
    require_enabled()
    return {'version': '0.1.0', 'games': [
        {'id': 'circle-match', 'variants': [2, 4, 6]},
        {'id': 'star-quest', 'variants': [0, 1, 2]},
    ], 'points_per_new_daily_variant': 100,
    'note': 'One award per activity variation per UTC day; therapy activities do not earn rank points.'}


@router.post('/me/start')
def start(payload: Start, authorization: Optional[str] = Header(default=None)):
    require_enabled()
    identity = _owner(authorization)
    state = create_initial_state(payload.game, payload.variant)
    schema()
    rank_schema()
    identity_id = identity['id']
    session_id = 'tvgame_' + secrets.token_hex(20)
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute('''SELECT COUNT(*) AS active_count FROM verifyd_circle_game_sessions
            WHERE identity_id=%s AND status='in_progress' AND expires_at>NOW()''', (identity_id,))
        if int(cur.fetchone()['active_count']) >= 12:
            raise HTTPException(429, detail={'error': 'too_many_active_games'})
        if payload.game == 'star-quest' and payload.variant > 0:
            cur.execute('''SELECT EXISTS(SELECT 1 FROM verifyd_circle_rank_awards
                WHERE identity_id=%s AND challenge_id LIKE %s) AS unlocked''',
                (identity_id, f'game:star-quest:{payload.variant - 1}:%'))
            if not cur.fetchone()['unlocked']:
                raise HTTPException(403, detail={'error': 'verify_previous_level_first'})
        cur.execute('''INSERT INTO verifyd_circle_game_sessions
            (session_id, identity_id, game_key, variant, state)
            VALUES (%s,%s,%s,%s,%s::jsonb)''',
            (session_id, identity_id, payload.game, payload.variant, json.dumps(state)))
    return {'session_id': session_id, 'state': public_state(payload.game, payload.variant, state),
            'server_verified': True, 'points_added': 0}


@router.post('/me/{session_id}/move')
def move(session_id: str, payload: Move, authorization: Optional[str] = Header(default=None)):
    require_enabled()
    identity_id = _owner(authorization)['id']
    schema()
    rank_schema()
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute('''SELECT game_key,variant,state,status,expires_at
            FROM verifyd_circle_game_sessions WHERE session_id=%s AND identity_id=%s FOR UPDATE''',
            (session_id, identity_id))
        record = cur.fetchone()
        if not record or record['status'] != 'in_progress' or record['expires_at'] <= datetime.now(timezone.utc):
            raise HTTPException(409, detail={'error': 'game_session_not_active'})
        state = dict(record['state'])
        game, variant = record['game_key'], int(record['variant'])
        finished = apply_choice(game, variant, state, payload.choice)
        cur.execute('''UPDATE verifyd_circle_game_sessions
            SET state=%s::jsonb, status=%s, completed_at=CASE WHEN %s THEN NOW() ELSE NULL END
            WHERE session_id=%s''', (json.dumps(state), 'completed' if finished else 'in_progress', finished, session_id))
        points = 0
        if finished:
            day = datetime.now(timezone.utc).date().isoformat()
            challenge_id = f'game:{game}:{variant}:{day}'
            cur.execute('''INSERT INTO verifyd_circle_rank_awards
                (award_id,identity_id,challenge_id,kind,points,evidence_ref)
                VALUES (%s,%s,%s,'game',100,%s)
                ON CONFLICT DO NOTHING RETURNING award_id''',
                (str(uuid.uuid4()), identity_id, challenge_id, f'verified-session:{session_id}'))
            points = 100 if cur.fetchone() else 0
    return {'session_id': session_id, 'state': public_state(game, variant, state, finished),
            'server_verified': True, 'points_added': points,
            'rank_credit': 'awarded_once' if points else 'none'}
