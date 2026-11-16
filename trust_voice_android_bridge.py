"""Stage 5 Android TWA native-calling enrollment bridge (PILOT ONLY, OFF BY DEFAULT).

The authenticated webpage creates an intent; the device redeems it with an
Android Keystore P-256 signature, without receiving the web session token.

Important: this does NOT yet verify Play Integrity or a production app-signing
certificate. It is deliberately restricted to the isolated Circle debug pilot.
Never expand PACKAGE to release builds until app attestation is enforced.
"""
from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import os
import re
import secrets
import threading
import time
from typing import Literal, Optional
from uuid import UUID

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from fastapi import APIRouter, Header, HTTPException, Response
from pydantic import BaseModel, Field

from database import get_db

router = APIRouter(prefix='/trust-voice/native-bridge/android', tags=['Android Native Bridge Pilot'])
PACKAGE = 'com.verifyd.circle.pilot.dev'
TTL_SECONDS = 60
MAX_INTENTS_PER_HOUR = 15
_ID_PATTERN = re.compile(r'^nbi_[A-Za-z0-9_-]{32}$')
_NONCE_PATTERN = re.compile(r'^[A-Za-z0-9_-]{43}$')
_B64_PATTERN = re.compile(r'^[A-Za-z0-9_-]+$')
_lock = threading.Lock()
_schema_ready = False


class IntentRequest(BaseModel):
    installation_id: UUID
    package_name: Literal['com.verifyd.circle.pilot.dev']
    public_key_spki_b64: str = Field(min_length=100, max_length=256)
    opted_in: Literal[True]


class RedeemRequest(IntentRequest):
    intent_id: str = Field(min_length=36, max_length=36)
    challenge: str = Field(min_length=43, max_length=43)
    signature_b64: str = Field(min_length=40, max_length=256)
    fcm_token: str = Field(min_length=80, max_length=4096, pattern=r'^[A-Za-z0-9_:\-]{80,4096}$')


def enabled() -> bool:
    return os.getenv('VERIFYD_ANDROID_NATIVE_BRIDGE_ENABLED', '0').lower().strip() in ('1','true','yes','on')


def _require_enabled() -> None:
    # Do not expose the enrollment API while the pilot is not explicitly on.
    if not enabled():
        raise HTTPException(404, detail={'error': 'not_found'})
    from trust_voice_android_calls import _require_enabled as calls_required
    calls_required()


def _unb64(value: str, *, minimum: int, maximum: int) -> bytes:
    if not _B64_PATTERN.fullmatch(value) or '=' in value:
        raise ValueError('invalid_base64url')
    if len(value) % 4 == 1:
        raise ValueError('invalid_base64url_length')
    try:
        data = base64.urlsafe_b64decode(value + '=' * (-len(value) % 4))
    except (binascii.Error, ValueError) as exc:
        raise ValueError('invalid_base64url') from exc
    if not minimum <= len(data) <= maximum:
        raise ValueError('invalid_length')
    if base64.urlsafe_b64encode(data).decode().rstrip('=') != value:
        raise ValueError('non_canonical_base64url')
    return data


def parse_public_key(encoded: str):
    raw = _unb64(encoded, minimum=85, maximum=140)
    key = serialization.load_der_public_key(raw)
    if not isinstance(key, ec.EllipticCurvePublicKey) or not isinstance(key.curve, ec.SECP256R1):
        raise ValueError('requires_p256_key')
    # Strict canonical SPKI DER, not a permissive BER or alternate format.
    if key.public_bytes(serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo) != raw:
        raise ValueError('non_canonical_public_key')
    return key, hashlib.sha256(raw).hexdigest()


def message_to_sign(intent_id: str, challenge: str, installation_id: str,
                    package_name: str, public_key_hash: str) -> bytes:
    if not _ID_PATTERN.fullmatch(intent_id) or not _NONCE_PATTERN.fullmatch(challenge):
        raise ValueError('invalid_intent')
    UUID(installation_id)
    if package_name != PACKAGE or not re.fullmatch('[0-9a-f]{64}', public_key_hash):
        raise ValueError('invalid_binding')
    return ('verifyd.native-bridge.android.v1\n' + intent_id + '\n' + challenge + '\n'
            + installation_id + '\n' + package_name + '\n' + public_key_hash).encode('ascii')


def verify_proof(*, signature_b64: str, public_key_spki_b64: str, intent_id: str,
                 challenge: str, installation_id: str, package_name: str) -> str:
    key, digest = parse_public_key(public_key_spki_b64)
    signature = _unb64(signature_b64, minimum=64, maximum=88)
    try:
        key.verify(signature, message_to_sign(intent_id, challenge, installation_id,
                                                package_name, digest), ec.ECDSA(hashes.SHA256()))
    except InvalidSignature as exc:
        raise ValueError('invalid_signature') from exc
    return digest


def _schema() -> None:
    global _schema_ready
    if _schema_ready: return
    from trust_voice_signaling import ensure_persistent_session_schema
    ensure_persistent_session_schema()
    with _lock:
        if _schema_ready: return
        with get_db() as conn:
            conn.cursor().execute("""CREATE TABLE IF NOT EXISTS verifyd_android_native_bridge_intents (
                intent_id TEXT PRIMARY KEY,
                identity_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                installation_id TEXT NOT NULL,
                package_name TEXT NOT NULL,
                public_key_hash TEXT NOT NULL,
                challenge_hash TEXT NOT NULL,
                created_at DOUBLE PRECISION NOT NULL,
                expires_at DOUBLE PRECISION NOT NULL,
                redeemed_at DOUBLE PRECISION
            )""")
            conn.cursor().execute("""CREATE INDEX IF NOT EXISTS idx_android_bridge_session_created
                ON verifyd_android_native_bridge_intents(session_id, created_at)""")
        _schema_ready = True


def _web_identity(auth: Optional[str]):
    from trust_voice_android_calls import _auth, allowed_packages
    identity, session = _auth(auth)
    if PACKAGE not in allowed_packages():
        raise HTTPException(403, detail={'error': 'pilot_package_disabled'})
    return identity, session


@router.post('/intents')
def issue_intent(body: IntentRequest, response: Response,
                 authorization: Optional[str] = Header(default=None)):
    _require_enabled()
    identity, session = _web_identity(authorization)
    try:
        _, digest = parse_public_key(body.public_key_spki_b64)
    except (ValueError, TypeError) as exc:
        raise HTTPException(422, detail={'error': 'invalid_device_public_key'}) from exc
    _schema()
    intent_id = 'nbi_' + secrets.token_urlsafe(24)  # 32 chars following prefix
    challenge = secrets.token_urlsafe(32)
    now = time.time()
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute('SELECT pg_advisory_xact_lock(hashtext(%s))', ('circle-bridge:' + str(session['session_id']),))
        cur.execute("""SELECT 1 FROM verifyd_voice_sessions WHERE session_id=%s
            AND identity_id=%s AND revoked_at IS NULL FOR UPDATE""",
                    (str(session['session_id']), str(identity['id'])))
        if not cur.fetchone():
            raise HTTPException(401, detail={'error': 'invalid_session'})
        cur.execute('DELETE FROM verifyd_android_native_bridge_intents WHERE expires_at < %s', (now - 3600,))
        cur.execute("""SELECT COUNT(*) AS n FROM verifyd_android_native_bridge_intents
            WHERE session_id=%s AND created_at>%s""", (str(session['session_id']), now - 3600))
        if int(cur.fetchone()['n']) >= MAX_INTENTS_PER_HOUR:
            raise HTTPException(429, detail={'error': 'bridge_rate_limited'})
        cur.execute("""INSERT INTO verifyd_android_native_bridge_intents
            (intent_id,identity_id,session_id,installation_id,package_name,public_key_hash,
             challenge_hash,created_at,expires_at)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
            (intent_id,str(identity['id']),str(session['session_id']),str(body.installation_id),
             body.package_name,digest,hashlib.sha256(challenge.encode('ascii')).hexdigest(),
             now,now + TTL_SECONDS))
    response.headers['Cache-Control'] = 'no-store'
    return {'schema': 'verifyd.native-bridge.android.v1', 'intent_id': intent_id,
            'challenge': challenge, 'expires_at': int(now + TTL_SECONDS)}


@router.post('/intents/redeem')
def redeem_intent(body: RedeemRequest, response: Response):
    _require_enabled()
    from trust_voice_android_calls import (
        AndroidDeviceRegistration, _write_device, allowed_packages, pilot_allowed,
    )
    from trust_voice_signaling import _identity_by_id
    if body.package_name != PACKAGE or PACKAGE not in allowed_packages():
        raise HTTPException(403, detail={'error': 'pilot_package_disabled'})
    if not _ID_PATTERN.fullmatch(body.intent_id) or not _NONCE_PATTERN.fullmatch(body.challenge):
        raise HTTPException(400, detail={'error': 'invalid_intent'})
    try:
        digest = verify_proof(signature_b64=body.signature_b64,
                              public_key_spki_b64=body.public_key_spki_b64,
                              intent_id=body.intent_id,challenge=body.challenge,
                              installation_id=str(body.installation_id),package_name=body.package_name)
    except (ValueError, TypeError) as exc:
        raise HTTPException(401, detail={'error': 'invalid_device_proof'}) from exc
    _schema()
    # A fresh pilot database may not yet have the Android device registry.
    # Create/upgrade the existing registry before opening the redemption txn.
    # Never attempt the device write against an uninitialized schema.
    from trust_voice_android_calls import ensure_schema as ensure_android_device_schema
    ensure_android_device_schema()
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("""SELECT i.*,s.revoked_at FROM verifyd_android_native_bridge_intents i
            JOIN verifyd_voice_sessions s ON s.session_id=i.session_id AND s.identity_id=i.identity_id
            WHERE i.intent_id=%s FOR UPDATE OF i""", (body.intent_id,))
        row = cur.fetchone()
        now = time.time()
        if (not row or row['redeemed_at'] is not None or row['expires_at'] <= now
            or row['revoked_at'] is not None or row['installation_id'] != str(body.installation_id)
            or row['package_name'] != body.package_name or row['public_key_hash'] != digest
            or not hmac.compare_digest(row['challenge_hash'], hashlib.sha256(body.challenge.encode('ascii')).hexdigest())):
            raise HTTPException(401, detail={'error': 'invalid_or_expired_intent'})
        identity = _identity_by_id(row['identity_id'])
        if not identity or not pilot_allowed(identity):
            raise HTTPException(403, detail={'error': 'android_call_pilot_required'})
        control_token = 'ac_' + secrets.token_urlsafe(32)
        reg = AndroidDeviceRegistration(installation_id=body.installation_id,
                                        fcm_token=body.fcm_token,package_name=body.package_name,opted_in=True)
        _write_device(cur,reg,row['identity_id'],row['session_id'],hashlib.sha256(control_token.encode()).hexdigest())
        cur.execute("""UPDATE verifyd_android_native_bridge_intents SET redeemed_at=%s
            WHERE intent_id=%s AND redeemed_at IS NULL""", (now,body.intent_id))
        if cur.rowcount != 1:
            raise HTTPException(409, detail={'error': 'intent_already_redeemed'})
    response.headers['Cache-Control'] = 'no-store'
    return {'device_token': control_token, 'schema': 'verifyd.native-bridge.android.v1'}


@router.get('/status/{installation_id}')
def bridge_status(installation_id: UUID, response: Response,
                  authorization: Optional[str] = Header(default=None)):
    """Only the signed-in owner/session can confirm its own native registration."""
    _require_enabled()
    identity, session = _web_identity(authorization)
    _schema()
    # Device table is created by the existing isolated Android registry,
    # not by the bridge intent-table initializer.
    from trust_voice_android_calls import ensure_schema as ensure_android_device_schema
    ensure_android_device_schema()
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("""SELECT 1 FROM verifyd_android_call_devices d
            JOIN verifyd_voice_sessions s ON s.session_id=d.session_id
              AND s.identity_id=d.identity_id AND s.revoked_at IS NULL
            WHERE d.installation_id=%s AND d.package_name=%s
              AND d.identity_id=%s AND d.session_id=%s AND d.disabled_at IS NULL
              AND d.control_token_hash IS NOT NULL AND d.fcm_token_invalidated_at IS NULL""",
            (str(installation_id),PACKAGE,str(identity['id']),str(session['session_id'])))
        active = cur.fetchone() is not None
    response.headers['Cache-Control'] = 'no-store'
    return {'registered': active, 'schema': 'verifyd.native-bridge.android.v1'}
