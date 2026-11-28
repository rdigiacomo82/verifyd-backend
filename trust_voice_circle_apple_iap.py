"""VeriFYD Circle Apple IAP — signed transaction entitlement service.

Disabled by default. Trust only Apple's official server library (>=3.1.2),
Apple roots provisioned by Render as secret files, and an authenticated
Trust Voice persistent session. Neither client-supplied paid flags nor
unverified JWT claims ever create entitlements.

Production rollout must include App Store Server Notifications V2 and periodic
App Store Server API reconciliation; both are essential to refunds/renewals.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field

from database import get_db
from trust_voice_messages import _identity_from_bearer

router = APIRouter(prefix="/trust-voice/apple-iap", tags=["Circle Apple IAP"])
log = logging.getLogger("verifyd.circle.apple_iap")

BUNDLE = "com.verifyd.circle"
PRODUCT = "com.verifyd.circle.monthly"
APPLE_APP_ID = 6826285833
_verifiers = {}

class TransactionClaim(BaseModel):
    transaction_jws: str = Field(min_length=40, max_length=65536)

class NotificationRequest(BaseModel):
    signedPayload: str = Field(min_length=40, max_length=262144)

def _flag(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in ("1", "true", "yes", "on")

def _enabled() -> bool:
    return _flag("VERIFYD_CIRCLE_APPLE_IAP_ENABLED")

def _notifications_on() -> bool:
    return _flag("VERIFYD_CIRCLE_APPLE_IAP_NOTIFICATIONS_ENABLED")

def _roots() -> list[str]:
    return [x.strip() for x in os.getenv("VERIFYD_CIRCLE_APPLE_ROOT_CERT_PATHS", "").split(",") if x.strip()]

def _configured() -> bool:
    """Do not advertise checkout before signed notifications and verifier exist."""
    if not (_enabled() and _notifications_on() and _roots()):
        return False
    if not all(Path(p).is_file() for p in _roots()):
        return False
    try:
        import appstoreserverlibrary  # noqa: F401
    except ImportError:
        return False
    return True

def _require_ready() -> None:
    if not _configured():
        raise HTTPException(503, detail={"error": "apple_iap_not_configured"})

def _user(authorization: Optional[str]) -> dict:
    identity = _identity_from_bearer(authorization)
    if not identity or identity.get("status", "active") != "active":
        raise HTTPException(401, detail={"error": "signed_in_circle_account_required"})
    return identity

def _sandbox_allowed(identity_id: str) -> bool:
    # App Review and test purchases may use a valid signed Sandbox transaction.
    # A sandbox receipt must not unlock ordinary production accounts.
    permitted = {s.strip() for s in os.getenv(
        "VERIFYD_CIRCLE_APPLE_IAP_SANDBOX_ACCOUNT_IDS", "").split(",") if s.strip()}
    return bool(identity_id in permitted)

def _verifier(environment: str):
    if environment not in ("Production", "Sandbox"):
        raise ValueError("invalid_apple_environment")
    _require_ready()
    # No network download or unverified roots; use Apple's official DER certs.
    from appstoreserverlibrary.models.Environment import Environment
    from appstoreserverlibrary.signed_data_verifier import SignedDataVerifier
    key = (environment, tuple(_roots()))
    if key not in _verifiers:
        certificates = [Path(p).read_bytes() for p in _roots()]
        _verifiers[key] = SignedDataVerifier(
            certificates, True,
            Environment.PRODUCTION if environment == "Production" else Environment.SANDBOX,
            BUNDLE, APPLE_APP_ID if environment == "Production" else None,
        )
    return _verifiers[key]

def _verify_signed_transaction(jws: str):
    """Verify cryptographic JWS, chain, bundle, app, environment and dates."""
    for environment in ("Production", "Sandbox"):
        try:
            transaction = _verifier(environment).verify_and_decode_signed_transaction(jws)
        except Exception:
            continue
        if transaction.bundleId != BUNDLE or transaction.productId != PRODUCT:
            raise HTTPException(400, detail={"error": "wrong_apple_product"})
        if not transaction.originalTransactionId or not transaction.transactionId:
            raise HTTPException(400, detail={"error": "invalid_apple_transaction"})
        if transaction.expiresDate is None:
            raise HTTPException(400, detail={"error": "apple_expiration_missing"})
        expires = datetime.fromtimestamp(transaction.expiresDate / 1000, tz=timezone.utc)
        ownership = str(getattr(transaction.inAppOwnershipType, "value",
                                transaction.inAppOwnershipType) or "")
        if ownership not in ("PURCHASED", "FAMILY_SHARED"):
            raise HTTPException(400, detail={"error": "apple_ownership_unrecognized"})
        return {
            "environment": environment,
            "original": str(transaction.originalTransactionId),
            "transaction_id": str(transaction.transactionId),
            "expires": expires,
            "revoked": transaction.revocationDate is not None,
            "ownership": ownership,
        }
    raise HTTPException(400, detail={"error": "apple_signature_verification_failed"})

def _schema():
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS circle_apple_subscription_owners (
                environment TEXT NOT NULL CHECK (environment IN ('Production','Sandbox')),
                original_transaction_id TEXT NOT NULL,
                circle_identity_id TEXT NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                PRIMARY KEY(environment,original_transaction_id)
            )""")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS circle_apple_subscription_transactions (
                environment TEXT NOT NULL CHECK (environment IN ('Production','Sandbox')),
                transaction_id TEXT NOT NULL,
                original_transaction_id TEXT NOT NULL,
                circle_identity_id TEXT NOT NULL,
                product_id TEXT NOT NULL CHECK (product_id = 'com.verifyd.circle.monthly'),
                bundle_id TEXT NOT NULL CHECK (bundle_id = 'com.verifyd.circle'),
                ownership_type TEXT NOT NULL,
                expires_at TIMESTAMPTZ NOT NULL,
                revoked BOOLEAN NOT NULL DEFAULT FALSE,
                recorded_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                PRIMARY KEY(environment,transaction_id)
            )""")
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_circle_iap_by_identity
            ON circle_apple_subscription_transactions
            (circle_identity_id,environment,original_transaction_id,expires_at DESC)""")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS circle_apple_notifications (
                environment TEXT NOT NULL,
                notification_uuid TEXT NOT NULL,
                notification_type TEXT NOT NULL,
                received_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                PRIMARY KEY(environment,notification_uuid)
            )""")

def _record(cur, owner: str, tx: dict, revoked_override: bool = False):
    cur.execute("""
        INSERT INTO circle_apple_subscription_transactions
        (environment,transaction_id,original_transaction_id,circle_identity_id,
         product_id,bundle_id,ownership_type,expires_at,revoked)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
        ON CONFLICT (environment, transaction_id) DO UPDATE SET
            revoked = circle_apple_subscription_transactions.revoked OR EXCLUDED.revoked,
            recorded_at = NOW()
        WHERE circle_apple_subscription_transactions.circle_identity_id = EXCLUDED.circle_identity_id
          AND circle_apple_subscription_transactions.original_transaction_id = EXCLUDED.original_transaction_id
        """, (tx["environment"], tx["transaction_id"], tx["original"],
              owner, PRODUCT, BUNDLE, tx["ownership"], tx["expires"],
              tx["revoked"] or revoked_override))
    if cur.rowcount != 1:
        raise HTTPException(409, detail={"error": "apple_transaction_owner_conflict"})

def _status_for(cur, user_id: str):
    # Latest verified renewal per original transaction; no falling back to an
    # older, unrevoked transaction if a newer renewal was refunded.
    cur.execute("""
        WITH latest AS (
            SELECT DISTINCT ON (environment,original_transaction_id)
                environment,original_transaction_id,expires_at,revoked
            FROM circle_apple_subscription_transactions
            WHERE circle_identity_id=%s
            ORDER BY environment,original_transaction_id,expires_at DESC,recorded_at DESC
        )
        SELECT environment,expires_at FROM latest
        WHERE expires_at>NOW() AND revoked=FALSE
          AND (environment='Production' OR %s=TRUE)
        ORDER BY expires_at DESC LIMIT 1
        """, (user_id, _sandbox_allowed(user_id)))
    return cur.fetchone()

@router.get("/config")
def config(authorization: Optional[str] = Header(None)):
    # Authentication is required, even for readiness, to avoid confusing
    # public /voice browsing with an active iOS native subscription.
    _user(authorization)
    return {"product_id": PRODUCT, "bundle_id": BUNDLE,
            "verification_ready": _configured(),
            "purchase_method": "apple_storekit_only"}

@router.get("/status")
def status(authorization: Optional[str] = Header(None)):
    actor = _user(authorization)
    _require_ready()
    _schema()
    with get_db() as conn:
        row = _status_for(conn.cursor(), str(actor["id"]))
    return {"product_id": PRODUCT, "entitlement_active": bool(row),
            "environment": row["environment"] if row else None,
            "expires_at": row["expires_at"].isoformat() if row else None}

@router.post("/verify")
def verify(payload: TransactionClaim, authorization: Optional[str] = Header(None)):
    actor = _user(authorization)
    _require_ready()
    tx = _verify_signed_transaction(payload.transaction_jws)
    if tx["environment"] == "Sandbox" and not _sandbox_allowed(str(actor["id"])):
        raise HTTPException(403, detail={"error": "sandbox_account_not_authorized"})
    if tx["expires"] <= datetime.now(timezone.utc) or tx["revoked"]:
        raise HTTPException(403, detail={"error": "apple_subscription_inactive"})
    _schema()
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("SELECT pg_advisory_xact_lock(hashtext(%s))",
                    ("apple-iap:"+tx["environment"]+":"+tx["original"],))
        cur.execute("""
            INSERT INTO circle_apple_subscription_owners
            (environment,original_transaction_id,circle_identity_id)
            VALUES (%s,%s,%s) ON CONFLICT DO NOTHING
            """, (tx["environment"], tx["original"], str(actor["id"])))
        cur.execute("""
            SELECT circle_identity_id FROM circle_apple_subscription_owners
            WHERE environment=%s AND original_transaction_id=%s FOR UPDATE
            """, (tx["environment"], tx["original"]))
        record = cur.fetchone()
        if not record or str(record["circle_identity_id"]) != str(actor["id"]):
            raise HTTPException(409, detail={"error": "apple_purchase_claimed_by_other_account"})
        _record(cur, str(actor["id"]), tx)
        row = _status_for(cur, str(actor["id"]))
    return {"entitlement_active": bool(row), "product_id": PRODUCT,
            "environment": tx["environment"],
            "expires_at": row["expires_at"].isoformat() if row else None}

@router.post("/notifications")
def signed_notification(payload: NotificationRequest):
    """Only Apple's cryptographically verified Notification V2 is accepted."""
    _require_ready()
    matched = None
    environment = None
    for name in ("Production", "Sandbox"):
        try:
            matched = _verifier(name).verify_and_decode_notification(payload.signedPayload)
            environment = name
            break
        except Exception:
            continue
    if not matched or not getattr(matched, "notificationUUID", None):
        raise HTTPException(400, detail={"error": "invalid_apple_notification"})
    data = getattr(matched, "data", None)
    signed_tx = getattr(data, "signedTransactionInfo", None) if data else None
    signed_renewal = getattr(data, "signedRenewalInfo", None) if data else None
    if signed_renewal:
        try:
            _verifier(environment).verify_and_decode_renewal_info(signed_renewal)
        except Exception:
            raise HTTPException(400, detail={"error": "invalid_apple_renewal"})
    tx = None
    if signed_tx:
        tx = _verify_signed_transaction(signed_tx)
        if tx["environment"] != environment:
            raise HTTPException(400, detail={"error": "apple_environment_mismatch"})
    event = str(getattr(matched, "notificationType", "") or "")
    event = event.split(".")[-1].upper()
    _schema()
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO circle_apple_notifications(environment,notification_uuid,notification_type)
            VALUES (%s,%s,%s) ON CONFLICT DO NOTHING RETURNING notification_uuid
            """, (environment, str(matched.notificationUUID), event))
        if not cur.fetchone():
            return {"accepted": True, "duplicate": True}
        if tx:
            cur.execute("""SELECT circle_identity_id
                FROM circle_apple_subscription_owners
                WHERE environment=%s AND original_transaction_id=%s FOR UPDATE
                """, (environment, tx["original"]))
            owner = cur.fetchone()
            if owner:
                _record(cur, str(owner["circle_identity_id"]), tx,
                        revoked_override=(event in ("REFUND","REVOKE")))
    return {"accepted": True, "duplicate": False}
