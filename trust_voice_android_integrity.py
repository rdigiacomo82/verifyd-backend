"""Production Android Circle Play Integrity verifier — fail closed and isolated.

Uses Google's official decodeIntegrityToken endpoint. Each token is bound to a
one-time enrollment intent + P-256 installation key + FCM token hash.
Do NOT enable without a real Play-distributed, signed com.verifyd.circle app.
No credentials or integrity tokens may be logged.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import os
import re
import time

import httpx
from fastapi import HTTPException

PRODUCTION_PACKAGE = 'com.verifyd.circle'
PRODUCTION_ENABLED_VAR = 'VERIFYD_ANDROID_NATIVE_BRIDGE_PRODUCTION_ENABLED'
CERT_FINGERPRINT_VAR = 'VERIFYD_ANDROID_CIRCLE_PLAY_SIGNING_SHA256'
PLAY_INTEGRITY_PROJECT_VAR = 'VERIFYD_ANDROID_CIRCLE_PLAY_INTEGRITY_PROJECT_NUMBER'


def production_enabled() -> bool:
    return os.getenv(PRODUCTION_ENABLED_VAR, '0').strip().lower() in {'1', 'true', 'yes', 'on'}


def expected_certificate_digest() -> str:
    """Play API returns base64url SHA256; config uses the official colon-hex Play fingerprint."""
    raw = (os.getenv(CERT_FINGERPRINT_VAR) or '').strip().upper()
    if not re.fullmatch(r'(?:[0-9A-F]{2}:){31}[0-9A-F]{2}', raw):
        raise HTTPException(503, detail={'error': 'circle_app_signing_not_configured'})
    digest = bytes.fromhex(raw.replace(':', ''))
    return base64.urlsafe_b64encode(digest).decode('ascii').rstrip('=')


def integrity_request_hash(message_to_sign: bytes, fcm_token: str) -> str:
    """Same non-sensitive digest must be requested inside the signed Android app."""
    token_hash = hashlib.sha256(fcm_token.encode('ascii')).hexdigest().encode('ascii')
    digest = hashlib.sha256(message_to_sign + b'\n' + token_hash).digest()
    return base64.urlsafe_b64encode(digest).decode('ascii').rstrip('=')


def require_production_integrity(*, token: str, request_hash: str) -> None:
    if not production_enabled():
        raise HTTPException(404, detail={'error': 'native_production_disabled'})
    expected = expected_certificate_digest()
    cloud_number = (os.getenv(PLAY_INTEGRITY_PROJECT_VAR) or '').strip()
    if not re.fullmatch(r'\d{6,20}', cloud_number):
        raise HTTPException(503, detail={'error': 'circle_play_integrity_project_not_configured'})
    if not isinstance(token, str) or not (100 <= len(token) <= 8192):
        raise HTTPException(401, detail={'error': 'play_integrity_required'})
    if not isinstance(request_hash, str) or not re.fullmatch(r'[A-Za-z0-9_-]{43}', request_hash):
        raise HTTPException(400, detail={'error': 'invalid_integrity_request_hash'})
    try:
        from google.auth import default
        from google.auth.transport.requests import Request
        credentials, _ = default(scopes=['https://www.googleapis.com/auth/playintegrity'])
        credentials.refresh(Request())
        if not credentials.token:
            raise ValueError('missing_auth')
        response = httpx.post(
            'https://playintegrity.googleapis.com/v1/com.verifyd.circle:decodeIntegrityToken',
            json={'integrity_token': token},
            headers={'Authorization': 'Bearer ' + str(credentials.token)},
            timeout=6.0,
            follow_redirects=False,
        )
        if response.status_code != 200:
            raise ValueError('decode_failed')
        payload = response.json().get('tokenPayloadExternal')
        if not isinstance(payload, dict):
            raise ValueError('missing_payload')
        details = payload.get('requestDetails') or {}
        app = payload.get('appIntegrity') or {}
        device = payload.get('deviceIntegrity') or {}
        account = payload.get('accountDetails') or {}
        timestamp = int(details.get('timestampMillis') or -1)
        now_ms = int(time.time() * 1000)
        if not (now_ms - 120000 <= timestamp <= now_ms + 15000):
            raise ValueError('stale_integrity_verdict')
        if details.get('requestPackageName') != PRODUCTION_PACKAGE:
            raise ValueError('package_mismatch')
        if not hmac.compare_digest(str(details.get('requestHash') or ''), request_hash):
            raise ValueError('intent_mismatch')
        if app.get('packageName') != PRODUCTION_PACKAGE or app.get('appRecognitionVerdict') != 'PLAY_RECOGNIZED':
            raise ValueError('unrecognized_app')
        certs = app.get('certificateSha256Digest') or []
        if not isinstance(certs, list) or not any(
            isinstance(c, str) and hmac.compare_digest(c.rstrip('='), expected) for c in certs
        ):
            raise ValueError('signing_mismatch')
        labels = device.get('deviceRecognitionVerdict') or []
        if not isinstance(labels, list) or 'MEETS_DEVICE_INTEGRITY' not in labels:
            raise ValueError('device_integrity_unavailable')
        if account.get('appLicensingVerdict') != 'LICENSED':
            raise ValueError('play_license_unavailable')
    except HTTPException:
        raise
    except Exception as exc:
        # Never log service-account secrets, provider response bodies or Play tokens.
        raise HTTPException(403, detail={'error': 'android_app_attestation_not_verified'}) from None
