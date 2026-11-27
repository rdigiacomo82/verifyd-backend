"""Verified VeriFYD Circle account deletion intake (App Review 5.1.1(v)).

Users initiate permanent deletion inside the signed-in app. A durable receipt
is created for a manually executed, verified data-erasure workflow. We NEVER
claim that records were erased when only a deletion request was received.
A request does not bypass school/legal retention requirements or auto-cancel
third-party subscriptions. Operators must follow up with deletion completion.
"""
from __future__ import annotations

import logging
import os
import uuid
from datetime import datetime, timedelta, timezone
from typing import Literal, Optional

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field

from database import get_db
from trust_voice_messages import _identity_from_bearer

router = APIRouter(prefix="/trust-voice", tags=["Account deletion"])
log = logging.getLogger("verifyd.account_deletion")

DAYS_TO_COMPLETE = 30


class DeleteRequest(BaseModel):
    confirm: Literal["DELETE"]
    reason: Optional[str] = Field(default=None, max_length=300)


def _actor(authorization: Optional[str]) -> dict:
    actor = _identity_from_bearer(authorization)
    if str(actor.get("status") or "active") != "active":
        raise HTTPException(401, detail={"error": "active_session_required"})
    return actor


def _schema():
    with get_db() as conn:
        conn.cursor().execute("""CREATE TABLE IF NOT EXISTS verifyd_voice_account_deletion_requests (
            request_id TEXT PRIMARY KEY,
            identity_id TEXT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            target_completion_at TIMESTAMPTZ NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending'
                CHECK(status IN ('pending','completed')),
            completed_at TIMESTAMPTZ,
            operator_note TEXT NOT NULL DEFAULT '',
            UNIQUE(identity_id,status)
        )""")


def _request_public(row):
    return {
        "ok": True,
        "request_id": str(row["request_id"]),
        "status": str(row["status"]),
        "deletion_complete": str(row["status"]) == "completed",
        "requested_at": row["created_at"].isoformat(),
        "target_completion_at": row["target_completion_at"].isoformat(),
        "manual_processing": True,
        "subscription_cancellation_separate": True,
        "message": (
            "Your account-deletion request has been received. We will complete "
            "the deletion process within 30 days and send confirmation when finished. "
            "Any records that must be retained by law will be handled separately. "
            "Deleting your account does not cancel an Apple or PayPal subscription."
            if row["status"] == "pending" else
            "Your account deletion has been confirmed as completed."
        ),
    }


def _notify_operator(receipt: str):
    """Best-effort administrative notification; queue is durable if mail fails.

    Never include account email, messages or school data in the notification.
    """
    address = os.getenv("VERIFYD_ACCOUNT_DELETION_NOTIFY_EMAIL", "support@vfvid.com").strip()
    if not address:
        return
    try:
        from emailer import FROM_NAME, FROM_ADDRESS, _send
        _send({
            "from": f"{FROM_NAME} <{FROM_ADDRESS}>",
            "to": [address],
            "subject": "VeriFYD Circle account deletion request",
            "html": (
                "<p>A signed-in user initiated a VeriFYD Circle account deletion.</p>"
                "<p>Receipt: <code>" + receipt + "</code></p>"
                "<p>Open the authenticated deletion queue and complete required "
                "erasure, legal-retention review, and confirmation.</p>"
            ),
        })
    except Exception:
        # Do not log customer email, identity or request body.
        log.warning("Account deletion operator notification unavailable")


@router.post("/account-deletion",status_code=202)
def create_deletion_request(payload: DeleteRequest,
                            authorization: Optional[str] = Header(default=None)):
    actor = _actor(authorization)
    identity_id = str(actor["id"])
    _schema()
    newly_created = False
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("SELECT pg_advisory_xact_lock(hashtext(%s))",
                    ("account_deletion:" + identity_id,))
        cur.execute("""SELECT request_id,status,created_at,target_completion_at
            FROM verifyd_voice_account_deletion_requests
            WHERE identity_id=%s AND status='pending'
            ORDER BY created_at DESC LIMIT 1""",(identity_id,))
        row = cur.fetchone()
        if not row:
            # Prevent a stale login or forged identity_id from initiating a
            # request for a different account. The supplied identity_id,
            # if any, is ignored; only the verified bearer decides the owner.
            cur.execute("""SELECT status FROM verifyd_identities WHERE id=%s
                FOR UPDATE""",(identity_id,))
            identity = cur.fetchone()
            if not identity or identity["status"] != "active":
                raise HTTPException(401,detail={"error":"active_identity_required"})
            receipt = "tvdel_" + uuid.uuid4().hex
            created = datetime.now(timezone.utc)
            deadline = created + timedelta(days=DAYS_TO_COMPLETE)
            cur.execute("""INSERT INTO verifyd_voice_account_deletion_requests
                (request_id,identity_id,created_at,target_completion_at)
                VALUES(%s,%s,%s,%s)
                RETURNING request_id,status,created_at,target_completion_at""",
                (receipt,identity_id,created,deadline))
            row = cur.fetchone()
            newly_created = True
    if newly_created:
        _notify_operator(str(row["request_id"]))
    return _request_public(row)


@router.get("/account-deletion/status")
def account_deletion_status(authorization: Optional[str] = Header(default=None)):
    actor = _actor(authorization)
    _schema()
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("""SELECT request_id,status,created_at,target_completion_at
             FROM verifyd_voice_account_deletion_requests
             WHERE identity_id=%s ORDER BY created_at DESC LIMIT 1""",
             (str(actor["id"]),))
        row = cur.fetchone()
    if row:
        return _request_public(row)
    return {"ok":True,"status":"not_requested","deletion_complete":False,
            "message":"No account deletion request is on file."}


@router.get("/account-deletion/operator/queue")
def list_account_deletion_queue(authorization: Optional[str] = Header(default=None)):
    actor = _actor(authorization)
    allowed = {s.strip() for s in os.getenv(
        "VERIFYD_ACCOUNT_DELETION_OPERATOR_IDS","").split(",") if s.strip()}
    if not allowed or str(actor["id"]) not in allowed:
        raise HTTPException(403,detail={"error":"account_deletion_operator_required"})
    _schema()
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("""SELECT request_id,identity_id,created_at,target_completion_at,status
            FROM verifyd_voice_account_deletion_requests
            WHERE status='pending'
            ORDER BY created_at LIMIT 250""")
        queue = [dict(row) for row in cur.fetchall()]
    return {"ok":True,"pending_requests":queue,
            "manual_data_deletion_required":True,
            "do_not_mark_complete_before_erasure":True}
