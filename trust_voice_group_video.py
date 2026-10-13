# ============================================================
# VeriFYD Trust Voice — Group Video (LiveKit SFU)
# VERIFYD_TRUST_VOICE_GROUP_VIDEO_V1
# ============================================================

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Header, HTTPException
from livekit import api
from livekit.api import CreateRoomRequest, DeleteRoomRequest, ListRoomsRequest

from database import get_db
from trust_voice_groups import (
    MAX_GROUP_MEMBERS,
    _group_member_ids,
    _group_member_rows,
    _public_group_member,
    _require_group_member,
    ensure_group_schema,
)
from trust_voice_messages import _identity_from_bearer, _public_contact
from trust_voice_signaling import _identity_by_id, _send_to_identity
from trust_voice_group_video_recording import enforce_recording_join_policy

log = logging.getLogger('verifyd.trust_voice.group_video')
router = APIRouter(prefix='/trust-voice', tags=['Trust Voice Group Video'])

FEATURE_VERSION = '0.1.3'
DEFAULT_SESSION_MINUTES = 120
DEFAULT_TOKEN_MINUTES = 60
DEFAULT_EMPTY_TIMEOUT_SECONDS = 300
_schema_ready = False


def _enabled() -> bool:
    return (os.environ.get('VERIFYD_TRUST_VOICE_GROUP_VIDEO_ENABLED', '') or '').strip().lower() in {'1','true','yes','on'}


def _livekit_url() -> str:
    return (os.environ.get('LIVEKIT_URL', '') or '').strip()


def _livekit_api_url() -> str:
    # Browsers connect to LiveKit over ws:// or wss://.
    # The backend RoomService API is HTTP/Twirp.
    url = _livekit_url()
    if url.startswith('wss://'):
        return 'https://' + url[len('wss://'):]
    if url.startswith('ws://'):
        return 'http://' + url[len('ws://'):]
    return url


def _livekit_key() -> str:
    return (os.environ.get('LIVEKIT_API_KEY', '') or '').strip()


def _livekit_secret() -> str:
    return (os.environ.get('LIVEKIT_API_SECRET', '') or '').strip()


def _configured() -> bool:
    return bool(_livekit_url() and _livekit_key() and _livekit_secret())


def _require_available() -> None:
    if not _enabled():
        raise HTTPException(status_code=404, detail={'error':'not_found'})
    if not _configured():
        raise HTTPException(status_code=503, detail={'error':'group_video_not_configured','message':'Group video is not configured yet.'})


def _env_int(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int((os.environ.get(name, str(default)) or '').strip())
    except Exception:
        value = default
    return max(minimum, min(value, maximum))


def _session_minutes() -> int:
    return _env_int('VERIFYD_TRUST_VOICE_GROUP_VIDEO_SESSION_MINUTES', DEFAULT_SESSION_MINUTES, 15, 720)


def _token_minutes() -> int:
    return _env_int('VERIFYD_TRUST_VOICE_GROUP_VIDEO_TOKEN_MINUTES', DEFAULT_TOKEN_MINUTES, 5, 120)


def _empty_timeout_seconds() -> int:
    return _env_int('VERIFYD_TRUST_VOICE_GROUP_VIDEO_EMPTY_TIMEOUT_SECONDS', DEFAULT_EMPTY_TIMEOUT_SECONDS, 60, 1800)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _participant_identity(identity_id: str) -> str:
    # Opaque identity: do not put a name, handle, email, or phone number into LiveKit identity.
    return 'tvp_' + hashlib.sha256(str(identity_id or '').encode('utf-8')).hexdigest()[:32]


def _room_name() -> str:
    # Opaque room name: never use the human-readable Trust Voice group name.
    return 'tvgv_' + uuid.uuid4().hex


def ensure_schema() -> None:
    global _schema_ready
    if _schema_ready:
        return
    ensure_group_schema()
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute('''
            CREATE TABLE IF NOT EXISTS verifyd_voice_group_video_sessions (
                session_id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                room_name TEXT UNIQUE NOT NULL,
                started_by_identity_id TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'active',
                started_at TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                ended_at TEXT,
                ended_by_identity_id TEXT,
                ended_reason TEXT
            )
        ''')
        cur.execute('''
            CREATE INDEX IF NOT EXISTS idx_verifyd_voice_group_video_lookup
            ON verifyd_voice_group_video_sessions (conversation_id, status, started_at DESC)
        ''')
        cur.execute('''
            CREATE UNIQUE INDEX IF NOT EXISTS uq_verifyd_voice_group_video_active
            ON verifyd_voice_group_video_sessions (conversation_id)
            WHERE status = 'active'
        ''')
    _schema_ready = True


def _expire_old(conversation_id: str) -> None:
    ensure_schema()
    now = _now().isoformat()
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute('''
            UPDATE verifyd_voice_group_video_sessions
            SET status='expired', ended_at=%s, ended_reason='session_timeout'
            WHERE conversation_id=%s AND status='active'
              AND expires_at::timestamptz <= %s::timestamptz
        ''', (now, conversation_id, now))


def _active_session(conversation_id: str) -> Optional[dict]:
    _expire_old(conversation_id)
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute('''
            SELECT * FROM verifyd_voice_group_video_sessions
            WHERE conversation_id=%s AND status='active'
            ORDER BY started_at DESC LIMIT 1
        ''', (conversation_id,))
        row = cur.fetchone()
    return dict(row) if row else None


def _mark_ended(session_id: str, identity_id: Optional[str], reason: str) -> None:
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute('''
            UPDATE verifyd_voice_group_video_sessions
            SET status='ended', ended_at=%s, ended_by_identity_id=%s, ended_reason=%s
            WHERE session_id=%s AND status='active'
        ''', (_now().isoformat(), identity_id, reason, session_id))


def _public_session(session: dict) -> dict:
    starter_id = str(session.get('started_by_identity_id') or '')
    starter = _identity_by_id(starter_id) if starter_id else None
    return {
        'session_id': session.get('session_id') or '',
        'group_id': session.get('conversation_id') or '',
        'status': session.get('status') or 'active',
        'started_by': _public_contact(starter) if starter else None,
        'started_at': session.get('started_at') or '',
        'expires_at': session.get('expires_at') or '',
    }


def _directory(conversation_id: str, owner_identity_id: str) -> list[dict]:
    result = []
    for row in _group_member_rows(conversation_id):
        public = _public_group_member(row, owner_identity_id)
        identity_id = str(public.get('identity_id') or row.get('id') or row.get('identity_id') or '')
        result.append({'participant_identity': _participant_identity(identity_id), 'identity': public})
    return result


def _connection(identity: dict, session: dict, owner_identity_id: str) -> dict:
    identity_id = str(identity.get('id') or '')
    participant_identity = _participant_identity(identity_id)
    token = (
        api.AccessToken(_livekit_key(), _livekit_secret())
        .with_identity(participant_identity)
        .with_grants(api.VideoGrants(
            room_join=True,
            room=str(session.get('room_name') or ''),
            can_publish=True,
            can_subscribe=True,
            can_publish_data=True,
        ))
        .with_ttl(timedelta(minutes=_token_minutes()))
        .to_jwt()
    )
    return {
        'server_url': _livekit_url(),
        'participant_token': token,
        'participant_identity': participant_identity,
        'token_expires_in_seconds': _token_minutes() * 60,
        'participants': _directory(str(session.get('conversation_id') or ''), owner_identity_id),
    }


async def _room_exists(room_name: str) -> bool:
    try:
        async with api.LiveKitAPI(
            _livekit_api_url(),
            api_key=_livekit_key(),
            api_secret=_livekit_secret(),
        ) as lkapi:
            response = await lkapi.room.list_rooms(ListRoomsRequest(names=[room_name]))
        return any(str(room.name) == room_name for room in response.rooms)
    except Exception:
        log.exception('LiveKit room lookup failed')
        raise HTTPException(status_code=503, detail={'error':'group_video_service_unavailable','message':'Group video is temporarily unavailable.'})


async def _create_room(room_name: str) -> None:
    try:
        async with api.LiveKitAPI(
            _livekit_api_url(),
            api_key=_livekit_key(),
            api_secret=_livekit_secret(),
        ) as lkapi:
            await lkapi.room.create_room(CreateRoomRequest(
                name=room_name,
                empty_timeout=_empty_timeout_seconds(),
                max_participants=MAX_GROUP_MEMBERS,
            ))
    except Exception:
        log.exception('LiveKit room create failed')
        raise HTTPException(status_code=503, detail={'error':'group_video_service_unavailable','message':'Unable to start group video right now.'})


async def _delete_room_quietly(room_name: str) -> None:
    if not room_name:
        return
    try:
        async with api.LiveKitAPI(
            _livekit_api_url(),
            api_key=_livekit_key(),
            api_secret=_livekit_secret(),
        ) as lkapi:
            await lkapi.room.delete_room(DeleteRoomRequest(room=room_name))
    except Exception:
        log.warning('LiveKit room delete failed', exc_info=True)


async def _notify_group(conversation_id: str, payload: dict, exclude_identity_id: Optional[str] = None) -> None:
    for recipient_id in _group_member_ids(conversation_id):
        if exclude_identity_id and recipient_id == exclude_identity_id:
            continue
        await _send_to_identity(recipient_id, payload)

        try:
            from trust_voice_push import dispatch_push_for_event

            await asyncio.to_thread(
                dispatch_push_for_event,
                recipient_id,
                payload,
            )
        except Exception:
            log.warning(
                "Trust Voice group-video Web Push dispatch failed",
                exc_info=True,
            )


@router.get('/group-video/health')
def group_video_health():
    ensure_schema()
    return {
        'status':'ok',
        'feature':'trust_voice_group_video',
        'version':FEATURE_VERSION,
        'enabled':_enabled(),
        'configured':_configured(),
        'transport':'livekit_sfu_v1',
        'server_api_transport':'https_normalized_v1',
        'membership_authorization':'trust_voice_group_member_v1',
        'session_store':'postgresql_v1',
        'room_identity':'opaque_v1',
        'participant_identity':'opaque_hash_v1',
        'max_group_members':MAX_GROUP_MEMBERS,
        'session_minutes':_session_minutes(),
        'token_minutes':_token_minutes(),
        'end_behavior':'mark_ended_then_background_room_delete_v1',
    }


@router.get('/groups/{conversation_id}/video/status')
async def video_status(conversation_id: str, authorization: Optional[str] = Header(default=None)):
    _require_available()
    identity = _identity_from_bearer(authorization)
    group = _require_group_member(conversation_id, str(identity['id']))
    session = _active_session(conversation_id)
    if not session:
        return {'active':False, 'session':None}
    if not await _room_exists(str(session.get('room_name') or '')):
        _mark_ended(str(session.get('session_id') or ''), None, 'room_closed')
        return {'active':False, 'session':None}
    return {
        'active':True,
        'session':_public_session(session),
        'is_owner':str(group.get('owner_identity_id') or '') == str(identity['id']),
    }


@router.post('/groups/{conversation_id}/video/start')
async def video_start(conversation_id: str, background_tasks: BackgroundTasks, authorization: Optional[str] = Header(default=None)):
    _require_available()
    identity = _identity_from_bearer(authorization)
    identity_id = str(identity['id'])
    group = _require_group_member(conversation_id, identity_id)

    existing = _active_session(conversation_id)
    if existing and await _room_exists(str(existing.get('room_name') or '')):
        enforce_recording_join_policy(
            conversation_id,
            str(existing.get('session_id') or ''),
            identity_id,
        )
        return {
            'ok':True,
            'created':False,
            'session':_public_session(existing),
            'connection':_connection(identity, existing, str(group.get('owner_identity_id') or '')),
        }
    if existing:
        _mark_ended(str(existing.get('session_id') or ''), None, 'room_closed')

    session_id = 'tvgvs_' + uuid.uuid4().hex
    room_name = _room_name()
    started_at = _now()
    expires_at = started_at + timedelta(minutes=_session_minutes())
    await _create_room(room_name)

    try:
        with get_db() as conn:
            cur = conn.cursor()
            cur.execute('''
                INSERT INTO verifyd_voice_group_video_sessions (
                    session_id, conversation_id, room_name, started_by_identity_id,
                    status, started_at, expires_at, ended_at,
                    ended_by_identity_id, ended_reason
                ) VALUES (%s,%s,%s,%s,'active',%s,%s,NULL,NULL,NULL)
            ''', (session_id, conversation_id, room_name, identity_id, started_at.isoformat(), expires_at.isoformat()))
    except Exception:
        await _delete_room_quietly(room_name)
        raise

    session = _active_session(conversation_id)
    if not session:
        await _delete_room_quietly(room_name)
        raise HTTPException(status_code=500, detail={'error':'group_video_session_create_failed'})

    public_session = _public_session(session)
    background_tasks.add_task(
        _notify_group,
        conversation_id,
        {'type':'group_video_started','group_id':conversation_id,'session':public_session},
        identity_id,
    )

    return {
        'ok':True,
        'created':True,
        'session':public_session,
        'connection':_connection(identity, session, str(group.get('owner_identity_id') or '')),
    }


@router.post('/groups/{conversation_id}/video/join')
async def video_join(conversation_id: str, authorization: Optional[str] = Header(default=None)):
    _require_available()
    identity = _identity_from_bearer(authorization)
    group = _require_group_member(conversation_id, str(identity['id']))
    session = _active_session(conversation_id)
    if not session:
        raise HTTPException(status_code=409, detail={'error':'no_active_group_video','message':'There is no active group video call to join.'})
    if not await _room_exists(str(session.get('room_name') or '')):
        _mark_ended(str(session.get('session_id') or ''), None, 'room_closed')
        raise HTTPException(status_code=409, detail={'error':'no_active_group_video','message':'This group video call has ended.'})

    enforce_recording_join_policy(
        conversation_id,
        str(session.get('session_id') or ''),
        str(identity['id']),
    )

    return {
        'ok':True,
        'session':_public_session(session),
        'connection':_connection(identity, session, str(group.get('owner_identity_id') or '')),
    }


@router.post('/groups/{conversation_id}/video/end')
async def video_end(conversation_id: str, background_tasks: BackgroundTasks, authorization: Optional[str] = Header(default=None)):
    _require_available()
    identity = _identity_from_bearer(authorization)
    identity_id = str(identity['id'])
    group = _require_group_member(conversation_id, identity_id)
    session = _active_session(conversation_id)
    if not session:
        return {'ok':True,'ended':False,'reason':'already_inactive'}

    starter_id = str(session.get('started_by_identity_id') or '')
    owner_id = str(group.get('owner_identity_id') or '')
    if identity_id not in {starter_id, owner_id}:
        raise HTTPException(status_code=403, detail={
            'error':'group_video_end_not_allowed',
            'message':'Only the person who started the video or the group owner can end it for everyone.'
        })

    room_name = str(session.get('room_name') or '')

    # End the authoritative Trust Voice session immediately. The browser
    # should never wait on LiveKit RoomService deletion before it can leave.
    _mark_ended(
        str(session.get('session_id') or ''),
        identity_id,
        'ended_by_member',
    )

    # Notify peers first so their clients can disconnect promptly, then ask
    # LiveKit to delete the room after this HTTP response has already returned.
    background_tasks.add_task(
        _notify_group,
        conversation_id,
        {
            'type':'group_video_ended',
            'group_id':conversation_id,
            'session_id':session.get('session_id') or '',
            'ended_by':_public_contact(identity),
            'ended_at':_now().isoformat(),
        },
        identity_id,
    )
    background_tasks.add_task(
        _delete_room_quietly,
        room_name,
    )
    return {'ok':True,'ended':True,'session_id':session.get('session_id') or ''}
