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
import secrets
import threading
import time
from typing import Literal, Optional
from uuid import UUID

import httpx
from cryptography.fernet import Fernet, InvalidToken
from fastapi import APIRouter, Header, HTTPException, Response
from pydantic import BaseModel, Field

from database import get_db
from native_call_contract import build_invite, fcm_data

log = logging.getLogger('verifyd.android_calls')
router = APIRouter(prefix='/trust-voice', tags=['Android Native Calls'])
_schema_lock = threading.Lock()
_schema_ready = False
PAIRING_TTL_SECONDS = 120
PAIR_CODE_PATTERN = re.compile(r'ap_[A-Za-z0-9_-]{43}')
DEVICE_TOKEN_PATTERN = re.compile(r'ac_[A-Za-z0-9_-]{43}')
PACKAGE = 'com.verifyd.circle'
CIRCLE_PACKAGES = {PACKAGE, f'{PACKAGE}.dev', f'{PACKAGE}.pilot', f'{PACKAGE}.pilot.dev'}
TOKEN_PATTERN = r'^[a-zA-Z0-9_:\-]{80,4096}$'


class AndroidDeviceRegistration(BaseModel):
    installation_id: UUID
    fcm_token: str = Field(min_length=80, max_length=4096, pattern=TOKEN_PATTERN)
    package_name: str = Field(min_length=3, max_length=255)
    opted_in: Literal[True]


class AndroidDeviceTokenUpdate(BaseModel):
    fcm_token: str = Field(min_length=80, max_length=4096, pattern=TOKEN_PATTERN)


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
            conn.cursor().execute('''ALTER TABLE verifyd_android_call_devices
              ADD COLUMN IF NOT EXISTS control_token_hash TEXT''')
            conn.cursor().execute('''CREATE UNIQUE INDEX IF NOT EXISTS idx_android_call_control_token
              ON verifyd_android_call_devices(control_token_hash) WHERE control_token_hash IS NOT NULL''')
            conn.cursor().execute('''CREATE TABLE IF NOT EXISTS verifyd_android_call_pairings (
              session_id TEXT PRIMARY KEY, identity_id TEXT NOT NULL,
              code_hash TEXT NOT NULL UNIQUE, expires_at DOUBLE PRECISION NOT NULL)''')
        _schema_ready = True


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _write_device(cur, body: AndroidDeviceRegistration, identity_id: str, session_id: str,
                  control_token_hash: str) -> None:
    if body.package_name not in allowed_packages():
        raise HTTPException(422, detail={'error': 'invalid_package'})
    encrypted = _cipher().encrypt(body.fcm_token.encode()).decode()
    digest = _digest(body.fcm_token)
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
      (installation_id,package_name,identity_id,session_id,token_hash,token_ciphertext,updated_at,disabled_at,control_token_hash)
      VALUES (%s,%s,%s,%s,%s,%s,%s,NULL,%s)
      ON CONFLICT(installation_id,package_name) DO UPDATE SET
      session_id=EXCLUDED.session_id,
      token_hash=EXCLUDED.token_hash,token_ciphertext=EXCLUDED.token_ciphertext,
      updated_at=EXCLUDED.updated_at,disabled_at=NULL,
      control_token_hash=EXCLUDED.control_token_hash
      WHERE verifyd_android_call_devices.identity_id=EXCLUDED.identity_id
      RETURNING installation_id''',
      (str(body.installation_id),body.package_name,identity_id,session_id,digest,encrypted,time.time(),control_token_hash))
    if cur.fetchone() is None:
        raise HTTPException(409, detail={'error': 'native_device_conflict'})


@router.get('/android-call-pairings')
def pairing_available(authorization: Optional[str] = Header(default=None)):
    _require_enabled()
    _auth(authorization)
    return {'available': True}


@router.post('/android-call-pairings')
def issue_pairing(response: Response, authorization: Optional[str] = Header(default=None)):
    _require_enabled()
    identity, session = _auth(authorization)
    ensure_schema()
    code = 'ap_' + secrets.token_urlsafe(32)
    with get_db() as conn:
        cur = conn.cursor()
        now = time.time()
        cur.execute('DELETE FROM verifyd_android_call_pairings WHERE expires_at <= %s', (now,))
        cur.execute('''INSERT INTO verifyd_android_call_pairings (session_id, identity_id, code_hash, expires_at)
          VALUES (%s,%s,%s,%s) ON CONFLICT (session_id) DO UPDATE SET
          identity_id=EXCLUDED.identity_id, code_hash=EXCLUDED.code_hash, expires_at=EXCLUDED.expires_at''',
          (str(session['session_id']), str(identity['id']), _digest(code), now + PAIRING_TTL_SECONDS))
    response.headers['Cache-Control'] = 'no-store'
    return {'code': code, 'expires_in_seconds': PAIRING_TTL_SECONDS}


@router.post('/android-call-pairings/redeem')
def redeem_pairing(body: AndroidDeviceRegistration, response: Response,
                   authorization: Optional[str] = Header(default=None)):
    from trust_voice_signaling import _token_from_authorization
    _require_enabled()
    code = _token_from_authorization(authorization)
    if not PAIR_CODE_PATTERN.fullmatch(code):
        raise HTTPException(401, detail={'error': 'invalid_pairing'})
    ensure_schema()
    device_token = 'ac_' + secrets.token_urlsafe(32)
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute('''DELETE FROM verifyd_android_call_pairings p USING verifyd_voice_sessions s
          WHERE p.code_hash=%s AND p.expires_at > EXTRACT(EPOCH FROM clock_timestamp())
            AND s.session_id=p.session_id
            AND s.identity_id=p.identity_id AND s.revoked_at IS NULL
          RETURNING p.identity_id, p.session_id''', (_digest(code),))
        pairing = cur.fetchone()
        if not pairing:
            raise HTTPException(401, detail={'error': 'invalid_pairing'})
        cur.execute("SELECT handle_lower FROM verifyd_identities WHERE id=%s AND status='active'",
                    (pairing['identity_id'],))
        identity = cur.fetchone()
        if not identity or not pilot_allowed(identity):
            raise HTTPException(403, detail={'error': 'android_call_pilot_required'})
        _write_device(cur, body, pairing['identity_id'], pairing['session_id'], _digest(device_token))
    response.headers['Cache-Control'] = 'no-store'
    return {'device_token': device_token}


def _control_token_hash(authorization: Optional[str]) -> str:
    from trust_voice_signaling import _token_from_authorization
    token = _token_from_authorization(authorization)
    if not DEVICE_TOKEN_PATTERN.fullmatch(token):
        raise HTTPException(401, detail={'error': 'invalid_android_device'})
    return _digest(token)


def _paired_device(cur, control_token_hash: str) -> dict:
    cur.execute('''SELECT d.identity_id, d.package_name, d.control_token_hash
      FROM verifyd_android_call_devices d JOIN verifyd_voice_sessions s
        ON s.session_id=d.session_id AND s.identity_id=d.identity_id AND s.revoked_at IS NULL
      WHERE d.control_token_hash=%s AND d.disabled_at IS NULL''', (control_token_hash,))
    device = cur.fetchone()
    if not device:
        raise HTTPException(401, detail={'error': 'invalid_android_device'})
    return device


def _paired_identity(authorization: Optional[str]) -> str:
    from trust_voice_signaling import _identity_by_id
    control_token_hash = _control_token_hash(authorization)
    ensure_schema()
    with get_db() as conn:
        device = _paired_device(conn.cursor(), control_token_hash)
    identity = _identity_by_id(device['identity_id'])
    if not identity or not pilot_allowed(identity) or device['package_name'] not in allowed_packages():
        raise HTTPException(403, detail={'error': 'android_call_pilot_required'})
    return device['identity_id']


@router.put('/android-call-devices/current/token')
def refresh_device_token(body: AndroidDeviceTokenUpdate,
                         authorization: Optional[str] = Header(default=None)):
    from psycopg2.errors import UniqueViolation
    _require_enabled()
    _paired_identity(authorization)
    try:
        with get_db() as conn:
            cur = conn.cursor()
            device = _paired_device(cur, _control_token_hash(authorization))
            cur.execute('''UPDATE verifyd_android_call_devices d
              SET token_hash=%s, token_ciphertext=%s, updated_at=%s
              FROM verifyd_voice_sessions s WHERE d.control_token_hash=%s
                AND s.session_id=d.session_id AND s.identity_id=d.identity_id AND s.revoked_at IS NULL
                AND d.disabled_at IS NULL''', (_digest(body.fcm_token),
                    _cipher().encrypt(body.fcm_token.encode()).decode(), time.time(),
                    device['control_token_hash']))
            if cur.rowcount != 1:
                raise HTTPException(401, detail={'error': 'invalid_android_device'})
    except UniqueViolation:
        raise HTTPException(409, detail={'error': 'native_device_conflict'}) from None
    return {'ok': True}


@router.delete('/android-call-devices/current')
def unlink_paired_device(authorization: Optional[str] = Header(default=None)):
    control_token_hash = _control_token_hash(authorization)
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("SELECT to_regclass('verifyd_android_call_devices') AS registry")
        if cur.fetchone()['registry'] is None:
            raise HTTPException(401, detail={'error': 'invalid_android_device'})
        device = _paired_device(cur, control_token_hash)
        cur.execute('DELETE FROM verifyd_android_call_devices WHERE control_token_hash=%s',
                    (device['control_token_hash'],))
    return {'ok': True}


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
          WHERE d.identity_id=%s AND d.disabled_at IS NULL AND d.control_token_hash IS NOT NULL
          ORDER BY d.updated_at DESC LIMIT 8''',(identity_id,))
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
    identity_id = await asyncio.to_thread(_paired_identity, authorization)
    from trust_voice_signaling import native_call_snapshot
    snapshot = await native_call_snapshot(call_id, identity_id)
    if not snapshot['is_callee']:
        raise HTTPException(404, detail={'error': 'call_not_found'})
    return snapshot


@router.post('/android-calls/{call_id}/{action}')
async def call_action(call_id:str,action:Literal['answer','decline','end'],
                      authorization:Optional[str]=Header(default=None)):
    _require_enabled()
    identity_id = await asyncio.to_thread(_paired_identity, authorization)
    from trust_voice_signaling import apply_call_action, native_call_snapshot
    snapshot = await native_call_snapshot(call_id, identity_id)
    if not snapshot['is_callee']:
        raise HTTPException(404, detail={'error': 'call_not_found'})
    if action != 'decline':
        raise HTTPException(409, detail={'error': 'native_media_not_ready'})
    return await apply_call_action(call_id, identity_id, 'call_decline')
