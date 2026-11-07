"""Optional, OFF-by-default VeriFYD Circle private gateway staging broker.

This router has no SIP call API, no outbound trunk, no telephone-number or
location input, and no emergency-calling functionality. Do not mount in the
public production app. Requires the existing revocable Trust Voice bearer
session validator AND a narrow, configured identity-ID allowlist.

A dedicated STAGING LiveKit deployment (not production LiveKit, not public SIP)
is required before enabling token issuance. The operator makes the only lab
Echo() call, restricted to extension 6000 in the separate private test stack.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import re
import time
from typing import Optional
from urllib.parse import urlsplit

from fastapi import APIRouter, Header, HTTPException, Response
from pydantic import BaseModel, ConfigDict

from trust_voice_messages import _identity_from_bearer

router = APIRouter(prefix='/trust-voice/private-gateway', tags=['Private Gateway STAGING ONLY'])
_FIXED_ROOM = 'verifyd-private-echo'
_TTL_SECONDS = 180
_ID_RE = re.compile(r'^[A-Za-z0-9_-]{1,128}$')


def _flag(name: str) -> bool:
    return str(os.environ.get(name, '')).strip().lower() in {'1', 'true', 'on', 'yes'}


def _guard(authorization: Optional[str]) -> str:
    if not _flag('VERIFYD_PRIVATE_GATEWAY_STAGE_ENABLED'):
        raise HTTPException(status_code=404, detail={'error': 'not_found'})
    # This validates the user's existing, revocable Circle session on the server.
    identity = _identity_from_bearer(authorization)
    identity_id = str(identity.get('id') or '').strip()
    allowed = {v.strip().lower() for v in os.environ.get('VERIFYD_PRIVATE_GATEWAY_STAGE_ALLOW_IDS', '').split(',') if v.strip()}
    if not identity_id or not _ID_RE.fullmatch(identity_id) or identity_id.lower() not in allowed:
        raise HTTPException(status_code=403, detail={'error': 'gateway_lab_not_authorized'})
    return identity_id


def _staging_config() -> tuple[str, str, str] | None:
    url = str(os.environ.get('VERIFYD_PRIVATE_GATEWAY_STAGE_LIVEKIT_URL', '')).strip()
    key = str(os.environ.get('VERIFYD_PRIVATE_GATEWAY_STAGE_LIVEKIT_KEY', '')).strip()
    secret = str(os.environ.get('VERIFYD_PRIVATE_GATEWAY_STAGE_LIVEKIT_SECRET', '')).strip()
    if not url or not key or len(secret) < 32 or not _ID_RE.fullmatch(key):
        return None
    if (key == os.environ.get('LIVEKIT_API_KEY', '') or
            hmac.compare_digest(secret, os.environ.get('LIVEKIT_API_SECRET', ''))):
        # Prevent reusing the existing production group-video credentials.
        return None
    try:
        parsed = urlsplit(url)
    except ValueError:
        return None
    if parsed.username or parsed.password or parsed.query or parsed.fragment or parsed.path not in ('', '/'):
        return None
    if parsed.hostname is None or parsed.port is None:
        return None
    if parsed.scheme == 'ws':
        if not _flag('VERIFYD_PRIVATE_GATEWAY_STAGE_ALLOW_LOCALHOST'):
            return None
        if parsed.hostname not in ('localhost', '127.0.0.1'):
            return None
    elif parsed.scheme == 'wss':
        if parsed.hostname in ('localhost', '127.0.0.1'):
            return None
        if parsed.hostname == urlsplit(os.environ.get('LIVEKIT_URL', 'https://invalid.example')).hostname:
            return None
    else:
        return None
    return url, key, secret


def _jwt_segment(value: dict) -> str:
    return base64.urlsafe_b64encode(json.dumps(value, separators=(',', ':'), sort_keys=True).encode('utf-8')).rstrip(b'=').decode('ascii')


def _staging_jwt(identity_id: str, key: str, secret: str) -> str:
    now = int(time.time())
    # Avoid leaking user handles, emails, or permanent identity IDs to LiveKit.
    opaque = hmac.new(secret.encode(), identity_id.encode(), hashlib.sha256).hexdigest()[:32]
    claims = {
        'iss': key, 'sub': 'circle-stage-' + opaque,
        'iat': now, 'nbf': now-2, 'exp': now + _TTL_SECONDS,
        'video': {
            'room': _FIXED_ROOM, 'roomJoin': True, 'canSubscribe': True,
            'canPublish': True, 'canPublishData': False,
            'canPublishSources': ['microphone'],
        },
    }
    message = _jwt_segment({'alg': 'HS256', 'typ': 'JWT'}) + '.' + _jwt_segment(claims)
    mac = hmac.new(secret.encode(), message.encode('ascii'), hashlib.sha256).digest()
    return message + '.' + base64.urlsafe_b64encode(mac).rstrip(b'=').decode('ascii')


class EmptyStageRequest(BaseModel):
    """Reject destinations, phone numbers, location, arbitrary rooms, and IDs."""
    model_config = ConfigDict(extra='forbid')


@router.get('/staging-status')
def staging_status(response: Response, authorization: Optional[str] = Header(default=None)) -> dict:
    _guard(authorization)
    configured = _staging_config() is not None
    response.headers['Cache-Control'] = 'no-store'
    return {
        'mode': 'PRIVATE_GATEWAY_STAGING',
        'configured': configured,
        'network_verified': False,  # A config read is not a connectivity check.
        'room': _FIXED_ROOM,
        'test_only': True,
        'outbound_calls_enabled': False,
        'carrier_connected': False,
        'emergency_service_available': False,
    }


@router.post('/staging-room-token')
def staging_room_token(
    request: EmptyStageRequest,
    response: Response,
    authorization: Optional[str] = Header(default=None),
) -> dict:
    identity_id = _guard(authorization)
    cfg = _staging_config()
    if cfg is None:
        raise HTTPException(status_code=503, detail={'error': 'private_gateway_staging_not_configured'})
    url, key, secret = cfg
    response.headers['Cache-Control'] = 'no-store'
    return {
        'mode': 'PRIVATE_GATEWAY_STAGING',
        'url': url,
        'token': _staging_jwt(identity_id, key, secret),
        'expires_in_seconds': _TTL_SECONDS,
        'room': _FIXED_ROOM,
        'test_only': True,
        'outbound_calls_enabled': False,
        'emergency_service_available': False,
    }