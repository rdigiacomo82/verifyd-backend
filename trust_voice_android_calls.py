"""VeriFYD Circle Android FCM *pilot* (disabled by default).

FCM acceptance is not proof that a device rang, or that media is connected.
Device registration is bound to an existing verified, revocable Trust Voice session.
No OAuth secrets, tokens, addresses or user PII are included in push payloads.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import re
import threading
import time
from typing import Literal, Optional
from uuid import UUID

import httpx
from cryptography.fernet import Fernet, InvalidToken
from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field

from database import get_db
from native_call_contract import build_invite, fcm_data

log = logging.getLogger('verifyd.android_calls')
router = APIRouter(prefix='/trust-voice', tags=['Android Native Calls'])
_schema_lock = threading.Lock()
_schema_ready = False
PACKAGE = 'com.verifyd.circle'
CIRCLE_PACKAGES = {PACKAGE, f'{PACKAGE}.dev', f'{PACKAGE}.pilot', f'{PACKAGE}.pilot.dev'}
TOKEN_PATTERN = r'^[a-zA-Z0-9_:\-]{80,4096}$'


class AndroidDeviceRegistration(BaseModel):
    installation_id: UUID
    fcm_token: str = Field(min_length=80, max_length=4096, pattern=TOKEN_PATTERN)
    package_name: str = Field(min_length=3, max_length=255)
    opted_in: Literal[True]


def enabled() -> bool:
    return os.getenv('VERIFYD_ANDROID_CALLS_ENABLED', 'false').lower() in {'true','1','yes'}


def pilot_allowed(identity: dict) -> bool:
    allowlist = {v.strip().lower().lstrip('@') for v in
                 os.getenv('VERIFYD_ANDROID_CALLS_PILOT_HANDLES', '').split(',') if v.strip()}
    handle = str(identity.get('handle_lower') or identity.get('handle') or '').lower().lstrip('@')
    return enabled() and bool(handle and handle in allowlist)


def allowed_packages() -> set[str]:
    raw = os.getenv('VERIFYD_ANDROID_CALLS_ALLOWED_PACKAGES', PACKAGE)
    return CIRCLE_PACKAGES.intersection(p.strip() for p in raw.split(','))


def _require_enabled() -> None:
    from trust_voice_signaling import _require_enabled as signaling_required
    signaling_required()
    if not enabled():
        raise HTTPException(404, detail={'error': 'not_found'})


def _auth(authorization: Optional[str], *, pilot: bool=True) -> tuple[dict, dict]:
    from trust_voice_signaling import _identity_by_id, _token_from_authorization, _verify_token
    payload = _verify_token(_token_from_authorization(authorization))
    session_id = str(payload.get('session_id') or '')
    if not session_id:
        raise HTTPException(401, detail={'error':'persistent_session_required'})
    identity = _identity_by_id(str(payload.get('identity_id') or ''))
    if not identity:
        raise HTTPException(401, detail={'error':'invalid_session'})
    if pilot and not pilot_allowed(identity):
        raise HTTPException(403, detail={'error':'android_call_pilot_required'})
    return identity, payload


def _cipher() -> Fernet:
    try:
        return Fernet(os.environ['VERIFYD_NATIVE_CALL_TOKEN_ENCRYPTION_KEY'].encode())
    except (KeyError, ValueError):
        raise HTTPException(503, detail={'error':'token_encryption_unavailable'}) from None


def ensure_schema() -> None:
    global _schema_ready
    if _schema_ready: return
    from trust_voice_signaling import ensure_persistent_session_schema
    ensure_persistent_session_schema()
    with _schema_lock:
        if _schema_ready: return
        with get_db() as conn:
            conn.cursor().execute('''CREATE TABLE IF NOT EXISTS verifyd_android_call_devices (
              installation_id TEXT NOT NULL, package_name TEXT NOT NULL,
              identity_id TEXT NOT NULL, session_id TEXT NOT NULL,
              token_hash TEXT NOT NULL, token_ciphertext TEXT NOT NULL,
              updated_at DOUBLE PRECISION NOT NULL, disabled_at DOUBLE PRECISION,
              PRIMARY KEY(installation_id,package_name), UNIQUE(token_hash,package_name))''')
            conn.cursor().execute('''CREATE INDEX IF NOT EXISTS idx_android_call_identity
              ON verifyd_android_call_devices(identity_id,disabled_at)''')
        _schema_ready = True


@router.post('/android-call-devices')
def register(body: AndroidDeviceRegistration, authorization: Optional[str]=Header(default=None)):
    _require_enabled()
    identity, session = _auth(authorization)
    if body.package_name not in allowed_packages():
        raise HTTPException(422, detail={'error':'invalid_package'})
    encrypted = _cipher().encrypt(body.fcm_token.encode()).decode()
    digest = hashlib.sha256(body.fcm_token.encode()).hexdigest()
    ensure_schema()
    identity_id = str(identity['id'])
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute('SELECT pg_advisory_xact_lock(hashtext(%s))', ('android-device:'+identity_id,))
        cur.execute('''SELECT COUNT(*) AS n FROM verifyd_android_call_devices d
          JOIN verifyd_voice_sessions s ON s.session_id=d.session_id
            AND s.identity_id=d.identity_id AND s.revoked_at IS NULL
          WHERE d.identity_id=%s AND d.disabled_at IS NULL AND d.installation_id<>%s''',
          (identity_id,str(body.installation_id)))
        if int(cur.fetchone()['n']) >= 8:
            raise HTTPException(409,detail={'error':'native_device_limit'})
        cur.execute('SELECT pg_advisory_xact_lock(hashtext(%s))', ('android-fcm:'+digest,))
        cur.execute('''SELECT installation_id, identity_id FROM verifyd_android_call_devices
          WHERE token_hash=%s AND package_name=%s''', (digest, body.package_name))
        token_owner = cur.fetchone()
        if token_owner and (
            token_owner['identity_id'] != identity_id
            or token_owner['installation_id'] != str(body.installation_id)
        ):
            raise HTTPException(409, detail={'error': 'native_device_conflict'})
        cur.execute('''INSERT INTO verifyd_android_call_devices
          (installation_id,package_name,identity_id,session_id,token_hash,token_ciphertext,updated_at,disabled_at)
          VALUES (%s,%s,%s,%s,%s,%s,%s,NULL)
          ON CONFLICT(installation_id,package_name) DO UPDATE SET
          session_id=EXCLUDED.session_id,
          token_hash=EXCLUDED.token_hash,token_ciphertext=EXCLUDED.token_ciphertext,
          updated_at=EXCLUDED.updated_at,disabled_at=NULL
          WHERE verifyd_android_call_devices.identity_id=EXCLUDED.identity_id
          RETURNING installation_id''',
          (str(body.installation_id),body.package_name,identity_id,str(session['session_id']),digest,encrypted,time.time()))
        if cur.fetchone() is None:
            raise HTTPException(409, detail={'error': 'native_device_conflict'})
    return {'ok':True,'installation_id':str(body.installation_id)}


@router.delete('/android-call-devices/{installation_id}')
def unregister(installation_id:UUID, authorization: Optional[str]=Header(default=None)):
    identity, session = _auth(authorization,pilot=False)
    with get_db() as conn:
        cursor = conn.cursor()
        # Unregister is still available after disabling the pilot.
        cursor.execute("SELECT to_regclass('verifyd_android_call_devices') AS registry")
        if cursor.fetchone()['registry'] is None:
            return {'ok': True}
        cursor.execute('''DELETE FROM verifyd_android_call_devices
          WHERE installation_id=%s AND identity_id=%s AND session_id=%s''',
          (str(installation_id),str(identity['id']),str(session['session_id'])))
    return {'ok':True}


def active_devices(identity_id:str) -> list[dict]:
    ensure_schema()
    with get_db() as conn:
        cur=conn.cursor()
        cur.execute('''SELECT d.* FROM verifyd_android_call_devices d
          JOIN verifyd_voice_sessions s ON s.session_id=d.session_id
            AND s.identity_id=d.identity_id AND s.revoked_at IS NULL
          WHERE d.identity_id=%s AND d.disabled_at IS NULL ORDER BY d.updated_at DESC LIMIT 8''',(identity_id,))
        rows=[dict(r) for r in cur.fetchall()]
    result=[]
    cipher=_cipher()
    for r in rows:
        if r['package_name'] not in allowed_packages():continue
        try:r['fcm_token']=cipher.decrypt(r['token_ciphertext'].encode()).decode();result.append(r)
        except (InvalidToken,UnicodeError):log.warning('Android token decryption failed')
    return result


def configured() -> bool:
    return bool(os.getenv('VERIFYD_ANDROID_FIREBASE_PROJECT_ID')
                and os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
                and os.getenv('VERIFYD_NATIVE_CALL_TOKEN_ENCRYPTION_KEY'))


def reachable(identity:dict) -> bool:
    if not pilot_allowed(identity) or not configured():return False
    try:return bool(active_devices(str(identity['id'])))
    except Exception:
        log.warning('Android push registry unavailable')
        return False


def _access_token() -> str:
    # Google service-account credentials remain server-side. Do not log access tokens.
    from google.auth import default
    from google.auth.transport.requests import Request
    credentials, _ = default(scopes=['https://www.googleapis.com/auth/firebase.messaging'])
    credentials.refresh(Request())
    if not credentials.token:
        raise RuntimeError('FCM authorization unavailable')
    return str(credentials.token)


async def dispatch_invite(identity:dict,call:dict,is_current) -> bool:
    """Send only an existing live call with bounded original TTL (no retry)."""
    if not pilot_allowed(identity) or not configured():return False
    try:
        invite=build_invite(call['call_id'],call['call_type'],call['created_at'])
        devices=await asyncio.to_thread(active_devices,str(identity['id']))
        if not devices or not await is_current():return False
        bearer=await asyncio.to_thread(_access_token)
        project=os.environ['VERIFYD_ANDROID_FIREBASE_PROJECT_ID']
        if not re.fullmatch(r'[a-z0-9][a-z0-9-]{5,62}',project):return False
        url=f'https://fcm.googleapis.com/v1/projects/{project}/messages:send'
        async with httpx.AsyncClient(timeout=2.0,follow_redirects=False) as client:
            any_ok=False
            for d in devices:
                if not await is_current():
                    break
                time_left = invite.expires_at - time.time()
                if time_left <= 0:
                    break
                remaining = min(30, int(time_left))
                request={'message':{'token':d['fcm_token'],'data':fcm_data(invite),
                    'android':{'priority':'HIGH','ttl':f'{remaining}s',
                               'restricted_package_name':d['package_name']}}}
                try:
                    r=await client.post(url,json=request,headers={'Authorization':f'Bearer {bearer}'})
                    if r.status_code==200:any_ok=True
                    else:log.warning('Android FCM refused incoming invite, status=%d',r.status_code)
                except httpx.HTTPError:
                    log.warning('Android FCM request failed')
            return any_ok
    except Exception:
        log.warning('Android native push dispatch unavailable',exc_info=False)
        return False


@router.get('/android-calls/{call_id}')
async def call_status(call_id:str,authorization:Optional[str]=Header(default=None)):
    _require_enabled()
    identity, _ = await asyncio.to_thread(_auth, authorization)
    from trust_voice_signaling import native_call_snapshot
    return await native_call_snapshot(call_id,str(identity['id']))


@router.post('/android-calls/{call_id}/{action}')
async def call_action(call_id:str,action:Literal['answer','decline','end'],
                      authorization:Optional[str]=Header(default=None)):
    _require_enabled()
    identity, _ = await asyncio.to_thread(_auth, authorization)
    from trust_voice_signaling import apply_call_action
    return await apply_call_action(call_id,str(identity['id']),'call_'+action)
