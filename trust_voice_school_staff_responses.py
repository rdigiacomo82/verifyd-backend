"""Separate, audited School Safety TEST response thread.

Staff may respond to their originally assigned TEST incident, but they can
never read the student/guardian private family thread. The reporting student
may answer staff through the restricted supervised-device session. No SMS,
phone calls, automatic emergency dispatch or ordinary /voice access.
"""
from __future__ import annotations

import uuid
from typing import Optional

from fastapi import APIRouter, Header, HTTPException
from database import get_db
from trust_voice_school_safety_training import _actor, _audit, _membership
from trust_voice_school_incident_chat import (
    Reply, _authorize, _schema as _incident_schema, _notifications_allowed,
    _dispatch, _record_event,
)

router = APIRouter(prefix="/trust-voice/school-safety",
                   tags=["School Safety TEST Staff Responses"])


def _schema():
    _incident_schema()
    with get_db() as conn:
        conn.cursor().execute("""CREATE TABLE IF NOT EXISTS verifyd_safety_incident_school_responses (
            message_id TEXT PRIMARY KEY,
            report_id TEXT NOT NULL REFERENCES verifyd_safety_training_reports(report_id),
            sender_id TEXT NOT NULL REFERENCES verifyd_identities(id),
            client_id TEXT NOT NULL,
            body TEXT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            UNIQUE(report_id,sender_id,client_id)
        )""")


def _school_chat_authorize(cur, report_id: str, actor_id: str):
    """Only reporting student + authorized initial-incident school responders.

    _authorize verifies school membership, current incident recipient,
    Principal authorization or designation for responder, and rejects adults
    whose approval has been revoked. A guardian may never enter this thread.
    """
    report = _authorize(cur, report_id, actor_id)
    if actor_id == str(report["student_id"]):
        _membership(cur, report["school_id"], actor_id, {"student"})
        return report
    role = _membership(cur, report["school_id"], actor_id)
    if role not in ("principal", "responder"):
        raise HTTPException(403, detail={"error": "school_response_student_or_approved_staff_only"})
    return report


def _read_school_response(report_id: str, actor_id: str, *, school_id: str | None = None):
    _schema()
    with get_db() as conn:
        cur = conn.cursor()
        report = _school_chat_authorize(cur, report_id, actor_id)
        if school_id is not None and str(report["school_id"]) != school_id:
            raise HTTPException(403, detail={"error": "restricted_student_school_mismatch"})
        cur.execute("""SELECT m.message_id,i.handle AS sender_handle,m.body,m.created_at
            FROM verifyd_safety_incident_school_responses m
            JOIN verifyd_identities i ON i.id=m.sender_id
            WHERE m.report_id=%s
            ORDER BY m.created_at,m.message_id LIMIT 300""", (report_id,))
        messages = [dict(row) for row in cur.fetchall()]
    return {"ok": True, "test_only": True, "school_staff_thread": True,
            "private_guardian_thread_visible": False, "messages": messages}


def _send_school_response(report_id: str, actor_id: str, payload: Reply,
                          *, school_id: str | None = None):
    _schema()
    message_id = "ssschoolmsg_" + uuid.uuid4().hex
    with get_db() as conn:
        cur = conn.cursor()
        report = _school_chat_authorize(cur, report_id, actor_id)
        if school_id is not None and str(report["school_id"]) != school_id:
            raise HTTPException(403, detail={"error": "restricted_student_school_mismatch"})
        if report["closed_at"] or report["incident_status"] != "active":
            raise HTTPException(409, detail={"error": "incident_closed"})
        cur.execute("""INSERT INTO verifyd_safety_incident_school_responses
            (message_id,report_id,sender_id,client_id,body)
            VALUES(%s,%s,%s,%s,%s)
            ON CONFLICT(report_id,sender_id,client_id) DO NOTHING
            RETURNING message_id""",
            (message_id,report_id,actor_id,payload.client_id,payload.body.strip()))
        saved = cur.fetchone()
        if saved:
            _audit(cur,report["school_id"],actor_id,
                   "school_test_response_saved",report_id)
            cur.execute("""SELECT 1 FROM verifyd_safety_incident_lifecycle_events
                WHERE report_id=%s AND event_type='school_response_initiated' LIMIT 1""",(report_id,))
            if not cur.fetchone():
                _record_event(cur,report_id,str(report["school_id"]),"school_response_initiated",actor_id)
            if _notifications_allowed(str(report["school_id"]),str(report.get("report_mode") or "test")):
                # School thread recipient selection stays separate from guardians.
                # Student may not have a push subscription; foreground polling
                # still receives the stored text through its restricted token.
                cur.execute("""SELECT r.student_id AS identity_id
                    FROM verifyd_safety_training_reports r WHERE r.report_id=%s
                    UNION
                    SELECT x.recipient_id AS identity_id
                    FROM verifyd_safety_training_recipients x
                    WHERE x.report_id=%s""",(report_id,report_id))
                for item in cur.fetchall():
                    recipient = str(item["identity_id"])
                    if recipient == actor_id:
                        continue
                    try:
                        _school_chat_authorize(cur,report_id,recipient)
                    except HTTPException:
                        continue
                    cur.execute("""INSERT INTO verifyd_safety_incident_outbox
                        (report_id,recipient_id,event_key)
                        VALUES(%s,%s,%s) ON CONFLICT DO NOTHING""",
                        (report_id,recipient,"schoolreply:"+message_id))
    notification = {"attempted_recipients":0,"push_provider_accepted":0,
                    "device_delivery_confirmed":False}
    if saved and _notifications_allowed(str(report["school_id"]),str(report.get("report_mode") or "test")):
        try:
            notification = _dispatch(report_id,"schoolreply:"+message_id)
        except Exception:
            pass  # Saved message remains available and outbox retry is durable.
    return {"ok":True,"saved":True,"duplicate":not bool(saved),
            "school_staff_thread":True,"private_guardian_thread_visible":False,
            "mode":report.get("report_mode","test"),"notification":notification,
            "emergency_dispatch":False,"device_delivery_confirmed":False}


@router.get("/reports/{report_id}/school-response")
def read_school_response(report_id:str,authorization:Optional[str]=Header(default=None)):
    actor = _actor(authorization)
    return _read_school_response(report_id,str(actor["id"]))


@router.post("/reports/{report_id}/school-response")
def send_school_response(report_id:str,payload:Reply,
                         authorization:Optional[str]=Header(default=None)):
    actor = _actor(authorization)
    return _send_school_response(report_id,str(actor["id"]),payload)
