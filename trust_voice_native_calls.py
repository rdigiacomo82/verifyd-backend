"""Session-bound iOS VoIP registration and bounded APNs delivery.

Disabled until an explicit pilot allowlist and APNs signing settings exist.
APNs acceptance is not evidence of ringing or media connection.
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import os
import re
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal, Optional
from uuid import UUID

import httpx
from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric.utils import decode_dss_signature
from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field

from database import get_db
from native_call_contract import apns_voip_payload, build_invite

@asynccontextmanager
async def _readiness_lifespan(_app):
    # Local checks only: no APNs request, database write, or delivery activation.
    logging.getLogger("uvicorn.error").info(
        "VERIFYD_NATIVE_CALL_READINESS %s", json.dumps(readiness_report(), sort_keys=True))
    yield


router = APIRouter(prefix="/trust-voice", tags=["Native Calls"], lifespan=_readiness_lifespan)
log = logging.getLogger("verifyd.native_calls")
BUNDLE_ID = "com.verifyd.videoauth"
_schema_ready = False
_schema_lock = threading.Lock()
_jwt_lock = threading.Lock()
_jwt_cache: tuple | None = None


class DeviceRegistration(BaseModel):
    installation_id: UUID
    token: str = Field(min_length=64, max_length=512, pattern=r"^[0-9a-f]+$")
    environment: Literal["sandbox", "production"]
    bundle_id: Literal["com.verifyd.videoauth"]
    opted_in: Literal[True]


def enabled() -> bool:
    return os.getenv("VERIFYD_NATIVE_CALLS_ENABLED", "0").lower().strip() in {"1", "true", "on", "yes"}


def pilot_allowed(identity: dict) -> bool:
    handles = {h.strip().lower().lstrip("@") for h in
               os.getenv("VERIFYD_NATIVE_CALLS_PILOT_HANDLES", "").split(",") if h.strip()}
    handle = str(identity.get("handle_lower") or identity.get("handle") or "").lower().lstrip("@")
    return enabled() and bool(handle and handle in handles)


def _require_enabled() -> None:
    from trust_voice_signaling import _require_enabled as signaling_required
    signaling_required()
    if not enabled():
        raise HTTPException(404, detail={"error": "not_found"})


def _auth(authorization: Optional[str], *, require_pilot: bool = True) -> tuple[dict, dict]:
    from trust_voice_signaling import _identity_by_id, _token_from_authorization, _verify_token
    payload = _verify_token(_token_from_authorization(authorization))
    if not payload.get("session_id"):
        raise HTTPException(401, detail={"error": "persistent_session_required"})
    identity = _identity_by_id(str(payload.get("identity_id") or ""))
    if not identity:
        raise HTTPException(401, detail={"error": "invalid_session"})
    if require_pilot and not pilot_allowed(identity):
        raise HTTPException(403, detail={"error": "native_call_pilot_required"})
    return identity, payload


def ensure_schema() -> None:
    global _schema_ready
    if _schema_ready:
        return
    from trust_voice_signaling import ensure_persistent_session_schema
    ensure_persistent_session_schema()
    with _schema_lock:
        if _schema_ready:
            return
        with get_db() as conn:
            cur = conn.cursor()
            cur.execute("""CREATE TABLE IF NOT EXISTS verifyd_native_call_devices (
                installation_id TEXT NOT NULL,
                environment TEXT NOT NULL CHECK(environment IN ('sandbox','production')),
                bundle_id TEXT NOT NULL,
                identity_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                token_hash TEXT NOT NULL,
                token_ciphertext TEXT NOT NULL,
                updated_at DOUBLE PRECISION NOT NULL,
                disabled_at DOUBLE PRECISION,
                PRIMARY KEY(installation_id, environment, bundle_id),
                UNIQUE(token_hash, environment, bundle_id)
            )""")
            cur.execute("""CREATE INDEX IF NOT EXISTS idx_native_call_identity
                ON verifyd_native_call_devices(identity_id, disabled_at)""")
        _schema_ready = True


@router.post("/native-call-devices")
def register_device(body: DeviceRegistration, authorization: Optional[str] = Header(default=None)):
    _require_enabled()
    identity, session = _auth(authorization)
    if len(body.token) % 2:
        raise HTTPException(422, detail={"error": "invalid_token"})
    cipher = _token_cipher()
    token_hash = hashlib.sha256(body.token.encode()).hexdigest()
    encrypted = cipher.encrypt(body.token.encode()).decode()
    ensure_schema()
    with get_db() as conn:
        cur = conn.cursor()
        # Serialize registrations for this account to enforce the device cap.
        cur.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (str(identity["id"]),))
        cur.execute("""SELECT COUNT(*) AS n FROM verifyd_native_call_devices d
            JOIN verifyd_voice_sessions s ON s.session_id=d.session_id
                AND s.identity_id=d.identity_id AND s.revoked_at IS NULL
            WHERE d.identity_id=%s AND d.disabled_at IS NULL
              AND d.installation_id<>%s""", (str(identity["id"]), str(body.installation_id)))
        if int(cur.fetchone()["n"]) >= 8:
            raise HTTPException(409, detail={"error": "native_device_limit"})
        cur.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", ("native-token:" + token_hash,))
        # Rotation/account switching: a token has exactly one current owner.
        cur.execute("""DELETE FROM verifyd_native_call_devices
            WHERE token_hash=%s AND environment=%s AND bundle_id=%s
              AND installation_id<>%s""",
                    (token_hash, body.environment, body.bundle_id, str(body.installation_id)))
        cur.execute("""INSERT INTO verifyd_native_call_devices
            (installation_id,environment,bundle_id,identity_id,session_id,token_hash,token_ciphertext,updated_at,disabled_at)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,NULL)
            ON CONFLICT(installation_id,environment,bundle_id) DO UPDATE SET
                identity_id=EXCLUDED.identity_id, session_id=EXCLUDED.session_id,
                token_hash=EXCLUDED.token_hash, token_ciphertext=EXCLUDED.token_ciphertext,
                updated_at=EXCLUDED.updated_at, disabled_at=NULL""",
                    (str(body.installation_id), body.environment, body.bundle_id,
                     str(identity["id"]), session["session_id"], token_hash, encrypted, time.time()))
    return {"ok": True, "installation_id": str(body.installation_id)}


@router.delete("/native-call-devices/{installation_id}")
def unregister_device(installation_id: UUID, authorization: Optional[str] = Header(default=None)):
    # Opt-out must remain possible even if pilot delivery has been switched off.
    identity, session = _auth(authorization, require_pilot=False)
    ensure_schema()
    with get_db() as conn:
        conn.cursor().execute("""DELETE FROM verifyd_native_call_devices
            WHERE installation_id=%s AND identity_id=%s AND session_id=%s""",
                              (str(installation_id), str(identity["id"]), session["session_id"]))
    return {"ok": True}


def active_devices(identity_id: str) -> list[dict]:
    ensure_schema()
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("""SELECT d.* FROM verifyd_native_call_devices d
            JOIN verifyd_voice_sessions s ON s.session_id=d.session_id
              AND s.identity_id=d.identity_id AND s.revoked_at IS NULL
            WHERE d.identity_id=%s AND d.disabled_at IS NULL
            ORDER BY d.updated_at DESC LIMIT 8""", (identity_id,))
        rows = [dict(row) for row in cur.fetchall()]
    cipher = _token_cipher()
    devices = []
    for row in rows:
        try:
            row["token"] = cipher.decrypt(row["token_ciphertext"].encode()).decode()
            devices.append(row)
        except (InvalidToken, UnicodeError):
            log.warning("Native device token decryption unavailable")
    return devices


def _token_cipher() -> Fernet:
    try:
        return Fernet(os.environ["VERIFYD_NATIVE_CALL_TOKEN_ENCRYPTION_KEY"].encode())
    except (KeyError, ValueError):
        raise HTTPException(503, detail={"error": "native_device_encryption_not_configured"}) from None


def configured() -> bool:
    return bool(re.fullmatch(r"[A-Z0-9]{10}", os.getenv("VERIFYD_APNS_TEAM_ID", ""))
                and re.fullmatch(r"[A-Z0-9]{10}", os.getenv("VERIFYD_APNS_KEY_ID", ""))
                and os.getenv("VERIFYD_APNS_PRIVATE_KEY_FILE")
                and os.getenv("VERIFYD_NATIVE_CALL_TOKEN_ENCRYPTION_KEY"))


def readiness_report() -> dict[str, bool]:
    """Safe startup summary. Never serialize credentials or exception details."""
    def passes(check) -> bool:
        try:
            return bool(check())
        except Exception:
            return False

    return {
        "required_settings": passes(configured),
        "apns_key_readable_and_signable": passes(_provider_token),
        "device_token_encryption": passes(_token_cipher),
        "pilot_accounts_configured": bool(os.getenv("VERIFYD_NATIVE_CALLS_PILOT_HANDLES", "").strip()),
        "native_delivery_enabled": enabled(),
    }


def reachable(identity: dict) -> bool:
    if not pilot_allowed(identity) or not configured():
        return False
    try:
        return bool(active_devices(str(identity["id"])))
    except Exception:
        log.warning("Native call device lookup unavailable")
        return False


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _provider_token() -> str:
    # Reuse JWTs for 45 minutes; APNs rejects overly frequent token updates.
    # A changed key file/key ID/team immediately invalidates the cache.
    global _jwt_cache
    path = Path(os.environ["VERIFYD_APNS_PRIVATE_KEY_FILE"])
    version = (str(path), path.stat().st_mtime_ns, os.environ["VERIFYD_APNS_KEY_ID"],
               os.environ["VERIFYD_APNS_TEAM_ID"])
    with _jwt_lock:
        now = time.time()
        if _jwt_cache and _jwt_cache[0] == version and 0 <= now - _jwt_cache[1] < 2700:
            return _jwt_cache[2]
        key = serialization.load_pem_private_key(path.read_bytes(), password=None)
        if not isinstance(key, ec.EllipticCurvePrivateKey) or not isinstance(key.curve, ec.SECP256R1):
            raise ValueError("APNs requires a P-256 signing key")
        header = _b64(json.dumps({"alg": "ES256", "kid": version[2]}).encode())
        claims = _b64(json.dumps({"iss": version[3], "iat": int(now)}).encode())
        signing_input = f"{header}.{claims}"
        r, s = decode_dss_signature(key.sign(signing_input.encode(), ec.ECDSA(hashes.SHA256())))
        token = signing_input + "." + _b64(r.to_bytes(32, "big") + s.to_bytes(32, "big"))
        _jwt_cache = (version, now, token)
        return token


def _disable_invalid_token(device: dict) -> None:
    with get_db() as conn:
        conn.cursor().execute("""UPDATE verifyd_native_call_devices SET disabled_at=%s
            WHERE installation_id=%s AND environment=%s AND bundle_id=%s
              AND token_hash=%s AND updated_at=%s""",
                              (time.time(), device["installation_id"], device["environment"],
                               device["bundle_id"], device["token_hash"], device["updated_at"]))


async def dispatch_invite(identity: dict, call: dict, is_current) -> bool:
    """Only real ringing calls; no retries and no refresh of the original expiry."""
    if not pilot_allowed(identity) or not configured():
        return False
    try:
        invite = build_invite(call["call_id"], call["call_type"], call["created_at"])
        devices = await asyncio.to_thread(active_devices, str(identity["id"]))
        if not devices or not await is_current():
            return False
        jwt = await asyncio.to_thread(_provider_token)
        async with httpx.AsyncClient(http2=True, timeout=2.0, follow_redirects=False) as client:
            async def send(device):
                # Reconcile revoked/rotated devices immediately before dispatch.
                current = await asyncio.to_thread(active_devices, str(identity["id"]))
                if device not in current or not await is_current() or time.time() >= invite.expires_at:
                    return False
                host = ("api.sandbox.push.apple.com" if device["environment"] == "sandbox"
                        else "api.push.apple.com")
                try:
                    response = await client.post(f"https://{host}/3/device/{device['token']}",
                        headers={"authorization": f"bearer {jwt}",
                                 "apns-topic": BUNDLE_ID + ".voip", "apns-push-type": "voip",
                                 "apns-priority": "10", "apns-expiration": str(invite.expires_at)},
                        json=apns_voip_payload(invite))
                    if response.status_code == 200:
                        return True
                    try:
                        reason = response.json().get("reason")
                    except (ValueError, AttributeError):
                        reason = None
                    if response.status_code == 410 or (response.status_code == 400 and reason == "BadDeviceToken"):
                        await asyncio.to_thread(_disable_invalid_token, device)
                    log.warning("APNs native call rejected status=%d", response.status_code)
                except httpx.HTTPError:
                    # HTTP exceptions include request URLs containing tokens; do not log them.
                    log.warning("APNs native call request unavailable")
                return False
            results = await asyncio.wait_for(asyncio.gather(*(send(d) for d in devices)), timeout=4.0)
            return any(results)
    except Exception:
        log.warning("Native call dispatch unavailable")
        return False


@router.get("/native-calls/{call_id}")
async def call_status(call_id: str, authorization: Optional[str] = Header(default=None)):
    _require_enabled()
    identity, _ = await asyncio.to_thread(_auth, authorization)
    from trust_voice_signaling import native_call_snapshot
    return await native_call_snapshot(call_id, str(identity["id"]))


@router.post("/native-calls/{call_id}/{action}")
async def call_action(call_id: str, action: Literal["answer", "decline", "end"],
                      authorization: Optional[str] = Header(default=None)):
    _require_enabled()
    identity, _ = await asyncio.to_thread(_auth, authorization)
    from trust_voice_signaling import apply_call_action
    return await apply_call_action(call_id, str(identity["id"]), "call_" + action)
