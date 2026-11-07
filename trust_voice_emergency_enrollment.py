"""VeriFYD Circle emergency enrollment STAGING ONLY.

No carrier, PSAP, 911, 933, phone dial, SIP, or SMS capability is implemented here.
The feature is OFF unless explicitly enabled, with a separately provisioned encryption key.
This module MUST NOT be described as a working E911 service.
"""
from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from threading import Lock
from typing import Literal, Optional

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator

router = APIRouter(prefix='/trust-voice/emergency-services', tags=['Emergency Enrollment — not 911'])

_SCHEMA_LOCK = Lock()
_SCHEMA_READY = False
_CONSENT_VERSION = 'e911-preenrollment-limitations-v1-2026-11-06'
_SUPPORTED_STATES = frozenset(('AL AK AZ AR CA CO CT DE FL GA HI ID IL IN IA KS KY LA ME MD MA MI MN '
                              'MS MO MT NE NV NH NJ NM NY NC ND OH OK OR PA RI SC SD TN TX UT VT VA '
                              'WA WV WI WY DC').split())


def _flag() -> bool:
    return os.environ.get('VERIFYD_E911_ENROLLMENT_STAGING_ENABLED', '').strip().lower() in {'1','true','yes','on'}


def _identity(authorization: Optional[str]) -> str:
    # Central, persistent, server-revocable VeriFYD Circle session authentication.
    # Imported lazily to keep isolated unit tests independent from production DB modules.
    from trust_voice_messages import _identity_from_bearer
    user = _identity_from_bearer(authorization)
    identity_id = str(user.get('id') or '').strip()
    if not identity_id:
        raise HTTPException(status_code=401, detail={'error':'invalid_identity'})
    return identity_id


def _key():
    # Separate from Circle session signing and from LiveKit/API keys.
    from cryptography.fernet import Fernet
    raw = os.environ.get('VERIFYD_E911_PROFILE_FERNET_KEY', '').strip().encode('ascii', errors='ignore')
    if not raw:
        raise HTTPException(status_code=503, detail={'error':'encrypted_storage_unavailable'})
    try:
        return Fernet(raw)
    except Exception:
        raise HTTPException(status_code=503, detail={'error':'encrypted_storage_unavailable'})


def _require_enrollment():
    if not _flag():
        raise HTTPException(status_code=503, detail={'error':'enrollment_not_available',
                                                     'message':'Emergency service enrollment is not yet available.'})
    return _key()


def _db():
    from database import get_db
    return get_db()


def _ensure_schema() -> None:
    global _SCHEMA_READY
    if _SCHEMA_READY:
        return
    with _SCHEMA_LOCK:
        if _SCHEMA_READY:
            return
        with _db() as conn:
            with conn.cursor() as cur:
                cur.execute('''
                    CREATE TABLE IF NOT EXISTS verifyd_e911_enrollment_staging (
                      identity_id TEXT PRIMARY KEY,
                      encrypted_profile BYTEA NOT NULL,
                      mobile_last4 CHAR(4) NOT NULL,
                      consent_version TEXT NOT NULL,
                      consent_at TIMESTAMPTZ NOT NULL,
                      created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                      updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    )
                ''')
        _SCHEMA_READY = True


class EnrollmentRequest(BaseModel):
    model_config = ConfigDict(extra='forbid', str_strip_whitespace=True)
    address1: str = Field(min_length=4, max_length=160)
    address2: str = Field(default='', max_length=120)
    city: str = Field(min_length=2, max_length=100)
    state: str = Field(min_length=2, max_length=2)
    zip: str = Field(min_length=5, max_length=10)
    mobile: str = Field(min_length=10, max_length=25)
    consent_to_store: Literal[True]
    acknowledges_not_911: Literal[True]

    @field_validator('state')
    @classmethod
    def valid_state(cls, v: str) -> str:
        s = v.upper()
        if s not in _SUPPORTED_STATES:
            raise ValueError('Invalid US state or District of Columbia')
        return s

    @field_validator('zip')
    @classmethod
    def valid_zip(cls, v: str) -> str:
        if not re.fullmatch(r'\d{5}(?:-\d{4})?', v):
            raise ValueError('Invalid US ZIP code')
        return v

    @field_validator('mobile')
    @classmethod
    def valid_mobile(cls, v: str) -> str:
        # Normalize +1 (US/Canada). Carrier SMS ownership verification is NOT yet integrated.
        digits = re.sub(r'\D', '', v)
        if len(digits) == 11 and digits.startswith('1'):
            digits = digits[1:]
        if len(digits) != 10 or digits[0] in '01' or digits[3] in '01':
            raise ValueError('A valid ten-digit North American mobile number is required')
        return '+1' + digits


def _not_e911() -> dict:
    return {
        'service': 'emergency_enrollment_preview_only',
        'can_place_911_call': False,
        'provider_connected': False,
        'address_provider_validated': False,
        'mobile_sms_verified': False,
        'callback_number_provisioned': False,
        'emergency_routing_validated': False,
        'real_933_call_placed': False,
        'real_911_call_placed': False,
        'message': 'VeriFYD Circle cannot call 911. For an emergency, use a phone/device with working emergency calling.',
    }


@router.get('/status')
def enrollment_status(authorization: Optional[str] = Header(default=None)):
    identity_id = _identity(authorization)
    result = {**_not_e911(), 'enrollment_available': False, 'enrollment_status': 'not_available',
              'mobile_last4': None}
    if not _flag():
        return result
    # No database reads until encrypted storage is configured and feature explicitly enabled.
    _key()
    _ensure_schema()
    with _db() as conn:
        with conn.cursor() as cur:
            cur.execute('''SELECT mobile_last4 FROM verifyd_e911_enrollment_staging
                           WHERE identity_id=%s''', (identity_id,))
            row = cur.fetchone()
    return {**result, 'enrollment_available': True,
            'enrollment_status': 'verification_pending' if row else 'not_registered',
            'mobile_last4': str(row['mobile_last4']).strip() if row else None}


@router.post('/enrollment', status_code=202)
def submit_enrollment(body: EnrollmentRequest, authorization: Optional[str] = Header(default=None)):
    identity_id = _identity(authorization)
    cipher = _require_enrollment()
    _ensure_schema()
    # No address/mobile/identity printed to logs; only encrypted JSON is stored.
    details = {k: getattr(body, k) for k in ('address1','address2','city','state','zip','mobile')}
    encrypted = cipher.encrypt(json.dumps(details, separators=(',', ':'), ensure_ascii=False).encode('utf-8'))
    with _db() as conn:
        with conn.cursor() as cur:
            # Changing address or mobile invalidates any future verification/provisioning.
            cur.execute('''
                INSERT INTO verifyd_e911_enrollment_staging
                  (identity_id, encrypted_profile, mobile_last4, consent_version, consent_at)
                VALUES (%s,%s,%s,%s,NOW())
                ON CONFLICT(identity_id) DO UPDATE SET
                  encrypted_profile=EXCLUDED.encrypted_profile,
                  mobile_last4=EXCLUDED.mobile_last4,
                  consent_version=EXCLUDED.consent_version,
                  consent_at=NOW(), updated_at=NOW()
            ''', (identity_id, encrypted, body.mobile[-4:], _CONSENT_VERSION))
    return {**_not_e911(), 'enrollment_status':'verification_pending',
            'mobile_last4':body.mobile[-4:], 'mobile_sms_verified':False}


@router.delete('/enrollment')
def delete_enrollment(authorization: Optional[str] = Header(default=None)):
    identity_id = _identity(authorization)
    _require_enrollment()
    _ensure_schema()
    with _db() as conn:
        with conn.cursor() as cur:
            cur.execute('DELETE FROM verifyd_e911_enrollment_staging WHERE identity_id=%s', (identity_id,))
    return {'deleted': True, **_not_e911()}


@router.post('/mobile/request-code')
def request_mobile_code(authorization: Optional[str] = Header(default=None)):
    _identity(authorization)
    raise HTTPException(status_code=503, detail={'error':'sms_provider_not_configured'})


@router.post('/calls/911')
def dial_911_disabled(authorization: Optional[str] = Header(default=None)):
    _identity(authorization)
    # Deliberately no provider/SIP dial code. Unsafe to activate without full integration.
    raise HTTPException(status_code=503, detail={'error':'e911_not_implemented',
        'message':'VeriFYD cannot place 911 calls. Use your device’s native phone service.'})