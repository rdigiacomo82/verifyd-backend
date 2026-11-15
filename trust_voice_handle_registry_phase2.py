"""VeriFYD Universal Handle Registry (Phase 2).

Isolated, opt-in ownership snapshot, privacy preferences and audit trail.
The existing verifyd_identities table remains authoritative: NO name changes,
NO public discovery, NO call/message/email interception, NO transfers.
A current user's row is seeded once from their authenticated identity.
Bulk legacy import is a separately gated, explicit maintenance action.
"""
from __future__ import annotations

import logging
import os
import re
import threading
from typing import Optional

from fastapi import APIRouter, Header, HTTPException

from database import get_db
from trust_voice_messages import _identity_from_bearer

log = logging.getLogger("verifyd.universal_handle_registry")
router = APIRouter(prefix="/trust-voice/registry", tags=["Universal Handle Registry"])
VERSION = "0.2.0-draft"
NAMESPACE = "vfvid.com"
_TRUE = {"1", "true", "yes", "on"}
_HANDLE = re.compile(r"^[a-z][a-z0-9_]{2,23}$")
_schema_lock = threading.Lock()
_schema_ready = False


def enabled() -> bool:
    return os.environ.get("VERIFYD_UNIVERSAL_HANDLE_REGISTRY_ENABLED", "0").strip().lower() in _TRUE


def _require_enabled() -> None:
    if not enabled():
        raise HTTPException(404, detail={"error": "registry_not_enabled"})


def _normalize_existing(identity: dict) -> tuple[str, str]:
    """Reject malformed/unverified/disabled accounts, never change the source identity."""
    if not identity or str(identity.get("status") or "") != "active":
        raise HTTPException(403, detail={"error": "inactive_identity"})
    identity_id = str(identity.get("id") or "").strip()
    handle = str(identity.get("handle_lower") or identity.get("handle") or "").strip().lower().lstrip("@")
    if not identity_id or not _HANDLE.fullmatch(handle):
        raise HTTPException(409, detail={"error": "legacy_identity_requires_review"})
    return identity_id, handle


def ensure_schema() -> None:
    """Idempotent additive DDL; no edits to existing identity or communication tables."""
    global _schema_ready
    _require_enabled()
    if _schema_ready:
        return
    with _schema_lock:
        if _schema_ready:
            return
        with get_db() as conn:
            cur = conn.cursor()
            cur.execute("""CREATE TABLE IF NOT EXISTS verifyd_universal_handle_registry (
                identity_id TEXT PRIMARY KEY REFERENCES verifyd_identities(id) ON DELETE CASCADE,
                namespace TEXT NOT NULL DEFAULT 'vfvid.com' CHECK (namespace = 'vfvid.com'),
                handle_lower TEXT NOT NULL UNIQUE,
                allocation_state TEXT NOT NULL DEFAULT 'existing_account'
                   CHECK (allocation_state IN ('existing_account','held_for_review')),
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                CHECK (handle_lower ~ '^[a-z][a-z0-9_]{2,23}$')
            )""")
            cur.execute("""CREATE TABLE IF NOT EXISTS verifyd_universal_handle_privacy (
                identity_id TEXT PRIMARY KEY REFERENCES verifyd_identities(id) ON DELETE CASCADE,
                public_discovery BOOLEAN NOT NULL DEFAULT FALSE,
                partner_discovery BOOLEAN NOT NULL DEFAULT FALSE,
                public_call_route BOOLEAN NOT NULL DEFAULT FALSE,
                public_message_route BOOLEAN NOT NULL DEFAULT FALSE,
                public_email_route BOOLEAN NOT NULL DEFAULT FALSE,
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )""")
            cur.execute("""CREATE TABLE IF NOT EXISTS verifyd_universal_handle_audit (
                event_id TEXT PRIMARY KEY,
                identity_id TEXT NOT NULL REFERENCES verifyd_identities(id) ON DELETE CASCADE,
                handle_lower TEXT NOT NULL,
                event_type TEXT NOT NULL CHECK
                  (event_type IN ('legacy_allocation_recorded','allocation_review_held')),
                actor_type TEXT NOT NULL CHECK
                  (actor_type IN ('authenticated_owner','authorized_maintenance')),
                policy_version TEXT NOT NULL DEFAULT 'draft-v0.1',
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )""")
            cur.execute("""CREATE INDEX IF NOT EXISTS idx_vuhr_audit_identity_time
                           ON verifyd_universal_handle_audit(identity_id,created_at DESC)""")
        _schema_ready = True


def _seed_identity(cur, identity_id: str, handle: str, actor_type: str) -> None:
    """Never allow another identity to claim a previously seeded registry row."""
    cur.execute("""SELECT identity_id, handle_lower, allocation_state
        FROM verifyd_universal_handle_registry
        WHERE identity_id=%s OR handle_lower=%s FOR UPDATE""", (identity_id, handle))
    existing = cur.fetchall() or []
    if any(str(row["identity_id"]) != identity_id or row["handle_lower"] != handle for row in existing):
        raise HTTPException(409, detail={"error": "registry_ownership_conflict"})
    if not existing:
        cur.execute("""INSERT INTO verifyd_universal_handle_registry
            (identity_id,namespace,handle_lower) VALUES (%s,%s,%s)
            ON CONFLICT DO NOTHING RETURNING identity_id""",
            (identity_id,NAMESPACE,handle))
        if not cur.fetchone():
            raise HTTPException(409, detail={"error": "registry_ownership_conflict"})
        # Fixed event ID means retries cannot add a duplicate event.
        cur.execute("""INSERT INTO verifyd_universal_handle_audit
            (event_id,identity_id,handle_lower,event_type,actor_type)
            VALUES ('legacy:' || md5(%s || ':' || %s),%s,%s,'legacy_allocation_recorded',%s)
            ON CONFLICT(event_id) DO NOTHING""",
            (identity_id,handle,identity_id,handle,actor_type))
    cur.execute("""INSERT INTO verifyd_universal_handle_privacy(identity_id)
                   VALUES (%s) ON CONFLICT(identity_id) DO NOTHING""",(identity_id,))


@router.get("/health")
def health():
    return {"status":"ok", "version":VERSION,"enabled":enabled(),
            "public_discovery_enabled":False,"handle_transfers_enabled":False,
            "existing_identity_authoritative":True}


@router.get("/me")
def my_registry_record(authorization: Optional[str]=Header(default=None)):
    """Authenticated owner-only lookup; no public discovery or private route details."""
    _require_enabled()
    owner = _identity_from_bearer(authorization)
    owner_id, handle = _normalize_existing(owner)
    ensure_schema()
    with get_db() as conn:
        cur=conn.cursor()
        _seed_identity(cur, owner_id, handle, "authenticated_owner")
        cur.execute("""SELECT r.identity_id,r.handle_lower,r.allocation_state,
            p.public_discovery,p.partner_discovery,p.public_call_route,
            p.public_message_route,p.public_email_route
            FROM verifyd_universal_handle_registry r
            JOIN verifyd_universal_handle_privacy p USING(identity_id)
            WHERE r.identity_id=%s""",(owner_id,))
        row=cur.fetchone()
        if not row:
            raise HTTPException(503,detail={"error":"registry_record_unavailable"})
    return {
        "identity_id":row["identity_id"],"handle":"@"+row["handle_lower"],
        "namespace":NAMESPACE,"qualified_address":"acct:"+row["handle_lower"]+"@"+NAMESPACE,
        "allocation_state":row["allocation_state"],
        "discovery":{"public":bool(row["public_discovery"]),
                     "partners":bool(row["partner_discovery"])},
        "routing":{"public_calls":bool(row["public_call_route"]),
                   "public_messages":bool(row["public_message_route"]),
                   "public_email":bool(row["public_email_route"])},
        "permissions_editable":False,"federated_resolution_enabled":False,
        "source":"existing_verifyd_identity"
    }


def _bootstrap_legacy_accounts(*, dry_run: bool=True) -> dict:
    """Explicit maintenance helper; caller must gate access to the operation.

    No route exposes this method. It reads existing active account labels and
    writes only to the isolated registry/permissions/audit tables.
    """
    _require_enabled()
    if os.environ.get("VERIFYD_UNIVERSAL_HANDLE_REGISTRY_MAINTENANCE_ALLOWED", "0").strip().lower() not in _TRUE:
        raise HTTPException(403,detail={"error":"maintenance_not_enabled"})
    ensure_schema()
    with get_db() as conn:
        cur=conn.cursor()
        cur.execute("""SELECT id,handle_lower,status FROM verifyd_identities
                       WHERE status='active' ORDER BY id""")
        identities=cur.fetchall() or []
        valid=[]
        for account in identities:
            valid.append(_normalize_existing(account))
        cur.execute("""SELECT identity_id,handle_lower FROM verifyd_universal_handle_registry""")
        allocated=cur.fetchall() or []
        by_id={str(row["identity_id"]):row["handle_lower"] for row in allocated}
        by_handle={row["handle_lower"]:str(row["identity_id"]) for row in allocated}
        conflicts=[{"identity_id":i,"error":"ownership_conflict"} for i,h in valid
                   if (i in by_id and by_id[i]!=h) or (h in by_handle and by_handle[h]!=i)]
        if conflicts:
            raise HTTPException(409,detail={"error":"registry_ownership_conflict","count":len(conflicts)})
        to_seed=[(i,h) for i,h in valid if i not in by_id]
        if not dry_run:
            # Existing identities only; audit and preferences are written atomically.
            for i,h in to_seed:
                _seed_identity(cur,i,h,"authorized_maintenance")
    return {"dry_run":bool(dry_run),"active_legacy_accounts":len(valid),
            "new_records":len(to_seed),"unchanged_records":len(valid)-len(to_seed),
            "public_discovery_enabled":False,"identity_records_modified":0}
