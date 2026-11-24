"""Restricted student-device activation for supervised TEST-school identities.

Never yields an ordinary VeriFYD login token. No access to /voice, unrelated
schools, staff tools, email, 911/E911, or personal safety records. Disabled by
default and allowlisted to named training schools.
"""
from __future__ import annotations

import hashlib
import os
import secrets
import threading
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field

from database import get_db
from trust_voice_school_safety_training import (
    _actor, _audit, _membership, _schema as _school_schema,
    _enabled as _training_enabled, ReportInput, _create_training_report_for_student,
    _verified_training_contact,
)
from trust_voice_school_safety_registration import _school_admin
from trust_voice_school_incident_chat import Reply
from trust_voice_school_safety_consents import (
    _consent_schema, _guardian_count, _school_accepted, _student_accepted,
    POLICY_VERSION,
)

router = APIRouter(prefix="/trust-voice/school-safety", tags=["TEST Supervised Student Device"])
_ready = False
_lock = threading.Lock()
SESSION_MINUTES = 45
CODE_MINUTES = 10


def _pilot():
    if not _training_enabled() or os.getenv("VERIFYD_SAFETY_STUDENT_DEVICE_ENABLED", "0").lower() not in (
        "1", "yes", "true", "on"
    ):
        raise HTTPException(503, detail={"error": "student_device_pilot_not_enabled"})


def _school_allowed(school_id):
    allowed = {s.strip() for s in os.getenv("VERIFYD_SAFETY_STUDENT_DEVICE_SCHOOL_IDS", "").split(",")
               if s.strip()}
    if school_id not in allowed:
        raise HTTPException(403, detail={"error": "student_device_school_not_authorized"})


def _schema():
    global _ready
    _consent_schema()
    if _ready:
        return
    with _lock:
        if _ready:
            return
        with get_db() as conn:
            cur = conn.cursor()
            cur.execute("""CREATE TABLE IF NOT EXISTS verifyd_safety_student_device_codes (
                school_id TEXT NOT NULL REFERENCES verifyd_safety_training_schools(school_id) ON DELETE CASCADE,
                student_id TEXT NOT NULL REFERENCES verifyd_identities(id),
                code_hash TEXT NOT NULL UNIQUE,
                issued_by TEXT NOT NULL REFERENCES verifyd_identities(id),
                issued_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                expires_at TIMESTAMPTZ NOT NULL,
                attempts INTEGER NOT NULL DEFAULT 0,
                used_at TIMESTAMPTZ,
                PRIMARY KEY(school_id, student_id)
            )""")
            cur.execute("""CREATE TABLE IF NOT EXISTS verifyd_safety_student_device_sessions (
                session_hash TEXT PRIMARY KEY,
                school_id TEXT NOT NULL REFERENCES verifyd_safety_training_schools(school_id) ON DELETE CASCADE,
                student_id TEXT NOT NULL REFERENCES verifyd_identities(id),
                code_issued_by TEXT NOT NULL REFERENCES verifyd_identities(id),
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                expires_at TIMESTAMPTZ NOT NULL,
                revoked_at TIMESTAMPTZ
            )""")
            cur.execute("""CREATE INDEX IF NOT EXISTS safety_student_device_live_session_idx
                ON verifyd_safety_student_device_sessions(school_id,student_id,expires_at)
                WHERE revoked_at IS NULL""")
        _ready = True


def _digest(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _guardian_ready(cur, school_id, student_id):
    if not _school_accepted(cur, school_id):
        raise HTTPException(409, detail={"error": "school_training_signoff_required"})
    if not _guardian_count(cur, school_id, student_id):
        raise HTTPException(409, detail={"error": "verified_guardian_training_acknowledgment_required"})


def _supervised(cur, school_id, student_id):
    _membership(cur, school_id, student_id, {"student"})
    cur.execute("SELECT handle,display_name,status FROM verifyd_identities WHERE id=%s", (student_id,))
    row = cur.fetchone()
    if not row or row["status"] != "supervised_child":
        raise HTTPException(403, detail={"error": "supervised_student_required"})
    return row


def _session(x_safety_device_session: Optional[str]):
    _pilot()
    if not x_safety_device_session or len(x_safety_device_session) > 180:
        raise HTTPException(401, detail={"error": "restricted_student_session_required"})
    _schema()
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("""SELECT school_id,student_id,code_issued_by,expires_at FROM verifyd_safety_student_device_sessions
          WHERE session_hash=%s AND revoked_at IS NULL AND expires_at>NOW()""",
                    (_digest(x_safety_device_session),))
        row = cur.fetchone()
        if not row:
            raise HTTPException(401, detail={"error": "restricted_student_session_expired"})
        school_id = str(row["school_id"])
        student_id = str(row["student_id"])
        _school_allowed(school_id)
        # An owner-revoked backup or removed Principal cannot continue authorizing
        # a child's device merely because its 45-minute token has not expired.
        _school_admin(cur, school_id, str(row["code_issued_by"]))
        student = _supervised(cur, school_id, student_id)
        _guardian_ready(cur, school_id, student_id)
        return {"school_id": school_id, "student_id": student_id,
                "student_handle": student["handle"],
                "expires_at": row["expires_at"].isoformat()}


class IssueCode(BaseModel):
    student_id: str = Field(min_length=5, max_length=100)


class ActivateCode(BaseModel):
    student_handle: str = Field(min_length=3, max_length=40)
    code: str = Field(min_length=16, max_length=24)


class StudentAcknowledgment(BaseModel):
    policy_version: str
    accepted: bool
    understood_test_only: bool


class SessionReport(ReportInput):
    pass


@router.get("/schools/{school_id}/student-devices/roster")
def device_student_roster(school_id: str, authorization: Optional[str] = Header(default=None)):
    _pilot()
    _school_allowed(school_id)
    actor = _actor(authorization)
    _schema()
    with get_db() as conn:
        cur = conn.cursor()
        _school_admin(cur, school_id, str(actor["id"]))
        cur.execute("""SELECT m.identity_id,i.handle,i.display_name FROM verifyd_safety_training_members m
            JOIN verifyd_identities i ON i.id=m.identity_id
            WHERE m.school_id=%s AND m.role='student' AND i.status='supervised_child'
            ORDER BY i.handle""", (school_id,))
        students = []
        for row in cur.fetchall():
            student_id = str(row["identity_id"])
            students.append({
                "student_id": student_id, "handle": row["handle"],
                "display_name": row["display_name"],
                "guardian_acknowledged": bool(_guardian_count(cur, school_id, student_id)),
                "student_acknowledged": bool(_student_accepted(cur, school_id, student_id)),
            })
    return {"ok": True, "test_only": True, "school_id": school_id, "students": students}


@router.post("/schools/{school_id}/student-devices/issue")
def issue_code(school_id: str, payload: IssueCode, authorization: Optional[str] = Header(default=None)):
    """A verified Principal/accepted backup issues a one-time code in person.

    The code appears only once and is NEVER sent via SMS, URL, logs or email.
    """
    _pilot()
    _school_allowed(school_id)
    actor = _actor(authorization)
    _schema()
    code = secrets.token_hex(8).upper()  # 64-bit unpredictable, 16 easy-to-type characters
    with get_db() as conn:
        cur = conn.cursor()
        _school_admin(cur, school_id, str(actor["id"]))
        student = _supervised(cur, school_id, payload.student_id)
        _guardian_ready(cur, school_id, payload.student_id)
        cur.execute("""INSERT INTO verifyd_safety_student_device_codes
            (school_id,student_id,code_hash,issued_by,expires_at)
            VALUES(%s,%s,%s,%s,NOW()+INTERVAL '10 minutes')
            ON CONFLICT(school_id,student_id) DO UPDATE SET
              code_hash=EXCLUDED.code_hash,issued_by=EXCLUDED.issued_by,
              issued_at=NOW(),expires_at=EXCLUDED.expires_at,
              attempts=0,used_at=NULL
            RETURNING expires_at""",
            (school_id, payload.student_id, _digest(code), str(actor["id"])))
        expires_at = cur.fetchone()["expires_at"]
        # A new code invalidates stale device sessions for this student.
        cur.execute("""UPDATE verifyd_safety_student_device_sessions SET revoked_at=NOW()
            WHERE school_id=%s AND student_id=%s AND revoked_at IS NULL""",
            (school_id, payload.student_id))
        _audit(cur, school_id, str(actor["id"]), "supervised_student_device_code_issued", payload.student_id)
    return {"ok": True, "test_only": True, "student_handle": student["handle"],
            "code": code, "expires_at": expires_at.isoformat(), "single_use": True,
            "normal_voice_login_enabled": False}


@router.post("/student-devices/activate")
def activate_device(payload: ActivateCode):
    _pilot()
    _schema()
    code = payload.code.replace("-", "").strip().upper()
    # Uniform error; never disclose whether a child identity exists.
    invalid = HTTPException(403, detail={"error": "invalid_or_expired_student_device_code"})
    if len(code) != 16 or any(c not in "0123456789ABCDEF" for c in code):
        raise invalid
    student_handle = payload.student_handle.lstrip("@").strip().lower()
    session = secrets.token_urlsafe(32)
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("""SELECT codes.school_id,codes.student_id,codes.issued_by,codes.attempts,
                       codes.expires_at,codes.used_at,i.handle,i.status
            FROM verifyd_safety_student_device_codes codes
            JOIN verifyd_identities i ON i.id=codes.student_id
            WHERE codes.code_hash=%s AND i.handle_lower=%s
            FOR UPDATE OF codes""", (_digest(code), student_handle))
        row = cur.fetchone()
        if not row:
            raise invalid
        sid = str(row["school_id"])
        student_id = str(row["student_id"])
        _school_allowed(sid)
        _school_admin(cur, sid, str(row["issued_by"]))
        if row["used_at"] or row["expires_at"] <= datetime.now(timezone.utc) or row["attempts"] >= 5:
            raise invalid
        _supervised(cur, sid, student_id)
        _guardian_ready(cur, sid, student_id)
        cur.execute("""UPDATE verifyd_safety_student_device_codes
            SET used_at=NOW() WHERE school_id=%s AND student_id=%s AND used_at IS NULL
            RETURNING student_id""", (sid, student_id))
        if not cur.fetchone():
            raise invalid
        cur.execute("""INSERT INTO verifyd_safety_student_device_sessions
            (session_hash,school_id,student_id,code_issued_by,expires_at)
            VALUES(%s,%s,%s,%s,NOW()+INTERVAL '45 minutes') RETURNING expires_at""",
            (_digest(session), sid, student_id, str(row["issued_by"])))
        expires_at = cur.fetchone()["expires_at"]
        _audit(cur, sid, student_id, "supervised_student_device_session_activated", student_id)
    return {"ok": True, "test_only": True, "restricted_safety_only": True,
            "student_handle": row["handle"], "school_id": sid,
            "session_token": session, "expires_at": expires_at.isoformat(),
            "normal_voice_login_enabled": False, "emergency_dispatch": False}


@router.get("/student-devices/me")
def student_device_me(x_safety_device_session: Optional[str] = Header(default=None, alias="X-VeriFYD-Safety-Session")):
    ctx = _session(x_safety_device_session)
    with get_db() as conn:
        cur = conn.cursor()
        signed = _student_accepted(cur, ctx["school_id"], ctx["student_id"])
        cur.execute("""SELECT primary_id,backup_id FROM verifyd_safety_training_contacts
            WHERE school_id=%s AND student_id=%s""", (ctx["school_id"], ctx["student_id"]))
        contacts = cur.fetchone()
        recipients_ready = False
        if contacts and contacts["primary_id"] != contacts["backup_id"]:
            try:
                if _verified_training_contact(cur,ctx["school_id"],ctx["student_id"],str(contacts["primary_id"])) == "guardian":
                    _verified_training_contact(cur,ctx["school_id"],ctx["student_id"],str(contacts["backup_id"]))
                    recipients_ready = True
            except HTTPException:
                pass
        cur.execute("SELECT school_name FROM verifyd_safety_training_schools WHERE school_id=%s", (ctx["school_id"],))
        school = cur.fetchone()
    return {"ok": True, "test_only": True, "school_id": ctx["school_id"],
            "school_name": school["school_name"] if school else "TEST School",
            "student_handle": ctx["student_handle"], "expires_at": ctx["expires_at"],
            "policy_version": POLICY_VERSION, "student_acknowledged": bool(signed),
            "recipients_ready": bool(recipients_ready),
            "can_submit_test_report": bool(signed and recipients_ready),
            "emergency_dispatch": False}


@router.get("/student-devices/choices")
def device_school_choices(x_safety_device_session: Optional[str] = Header(default=None, alias="X-VeriFYD-Safety-Session")):
    ctx = _session(x_safety_device_session)
    from trust_voice_school_choice_catalog import _catalog_schema, _read_catalog
    _catalog_schema()
    with get_db() as conn:
        catalog = _read_catalog(conn.cursor(), ctx["school_id"])
    return {"ok": True, "test_only": True, "school_id": ctx["school_id"], **catalog}


@router.post("/student-devices/acknowledgment")
def device_student_acknowledgment(payload: StudentAcknowledgment,
                                  x_safety_device_session: Optional[str] = Header(default=None, alias="X-VeriFYD-Safety-Session")):
    ctx = _session(x_safety_device_session)
    if payload.accepted is not True or payload.understood_test_only is not True or payload.policy_version != POLICY_VERSION:
        raise HTTPException(422, detail={"error": "student_training_acknowledgment_required"})
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("""INSERT INTO verifyd_safety_training_signoffs
            (signoff_id,school_id,actor_id,kind,subject_id,policy_version)
            VALUES(%s,%s,%s,'student',%s,%s)
            ON CONFLICT(school_id,actor_id,kind,subject_id) DO UPDATE
            SET policy_version=EXCLUDED.policy_version,accepted_at=NOW(),revoked_at=NULL""",
            ("sssig_"+secrets.token_hex(16), ctx["school_id"], ctx["student_id"],
             ctx["student_id"], POLICY_VERSION))
        _audit(cur, ctx["school_id"], ctx["student_id"], "supervised_device_student_acknowledged", ctx["student_id"])
    return {"ok": True, "test_only": True, "student_acknowledged": True, "policy_version": POLICY_VERSION}


@router.post("/student-devices/reports")
def device_send_report(payload: SessionReport, x_safety_device_session: Optional[str] = Header(default=None, alias="X-VeriFYD-Safety-Session")):
    ctx = _session(x_safety_device_session)
    if payload.school_id != ctx["school_id"] or not payload.choice_id:
        raise HTTPException(403, detail={"error": "restricted_student_school_or_choice_mismatch"})
    return _create_training_report_for_student(payload, ctx["student_id"])


@router.get("/student-devices/reports")
def device_my_reports(x_safety_device_session: Optional[str] = Header(default=None, alias="X-VeriFYD-Safety-Session")):
    ctx = _session(x_safety_device_session)
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("""SELECT r.report_id,r.category,r.description,r.created_at,
               r.closed_at,i.status AS incident_status
          FROM verifyd_safety_training_reports r
          JOIN verifyd_safety_training_incidents i ON i.incident_id=r.incident_id
          WHERE r.school_id=%s AND r.student_id=%s
          ORDER BY r.created_at DESC LIMIT 25""", (ctx["school_id"], ctx["student_id"]))
        rows = [dict(x) for x in cur.fetchall()]
    return {"ok": True, "test_only": True, "reports": rows}


@router.get("/student-devices/reports/{report_id}/conversation")
def device_conversation(report_id: str, x_safety_device_session: Optional[str] = Header(default=None, alias="X-VeriFYD-Safety-Session")):
    ctx = _session(x_safety_device_session)
    from trust_voice_school_incident_chat import _schema as _chat_schema, _family_chat_authorize
    _chat_schema()
    with get_db() as conn:
        cur = conn.cursor()
        report = _family_chat_authorize(cur, report_id, ctx["student_id"])
        if str(report["school_id"]) != ctx["school_id"]:
            raise HTTPException(403, detail={"error": "restricted_student_school_mismatch"})
        cur.execute("""SELECT m.message_id,m.sender_id,i.handle AS sender_handle,m.body,m.created_at
          FROM verifyd_safety_incident_messages m
          JOIN verifyd_identities i ON i.id=m.sender_id
          WHERE m.report_id=%s ORDER BY m.created_at,m.message_id LIMIT 300""", (report_id,))
        messages = [dict(x) for x in cur.fetchall()]
    return {"ok": True, "test_only": True, "messages": messages}


@router.post("/student-devices/reports/{report_id}/conversation")
def device_conversation_reply(report_id: str, payload: Reply,
                              x_safety_device_session: Optional[str] = Header(default=None, alias="X-VeriFYD-Safety-Session")):
    # Reply schema is loaded at module import below to preserve FastAPI validation.
    ctx = _session(x_safety_device_session)
    from trust_voice_school_incident_chat import (
        _schema as _chat_schema, _family_chat_authorize,
        _allowed, _dispatch,
    )
    _chat_schema()
    message_id = "ssmsg_" + secrets.token_hex(16)
    with get_db() as conn:
        cur = conn.cursor()
        report = _family_chat_authorize(cur, report_id, ctx["student_id"])
        if str(report["school_id"]) != ctx["school_id"]:
            raise HTTPException(403, detail={"error": "restricted_student_school_mismatch"})
        if report["closed_at"] or report["incident_status"] != "active":
            raise HTTPException(409, detail={"error": "incident_closed"})
        cur.execute("""INSERT INTO verifyd_safety_incident_messages
             (message_id,report_id,sender_id,client_id,body) VALUES(%s,%s,%s,%s,%s)
             ON CONFLICT(report_id,sender_id,client_id) DO NOTHING RETURNING message_id""",
            (message_id,report_id,ctx["student_id"],payload.client_id,payload.body.strip()))
        saved = cur.fetchone()
        if saved and _allowed(ctx["school_id"]):
            cur.execute("""SELECT recipient_id FROM verifyd_safety_training_recipients
                WHERE report_id=%s UNION SELECT student_id FROM verifyd_safety_training_reports WHERE report_id=%s""",
                (report_id,report_id))
            for candidate in cur.fetchall():
                rid = str(candidate["recipient_id"])
                if rid == ctx["student_id"]:
                    continue
                try:
                    _family_chat_authorize(cur, report_id, rid)
                except HTTPException:
                    continue
                cur.execute("""INSERT INTO verifyd_safety_incident_outbox
                    (report_id,recipient_id,event_key) VALUES(%s,%s,%s) ON CONFLICT DO NOTHING""",
                    (report_id,rid,"reply:"+message_id))
        if saved:
            _audit(cur, ctx["school_id"], ctx["student_id"], "supervised_device_family_reply_saved", report_id)
    notification = {"attempted_recipients":0,"push_provider_accepted":0}
    if saved and _allowed(ctx["school_id"]):
        try:
            notification = _dispatch(report_id, "reply:"+message_id)
        except Exception:
            pass
    return {"ok": True, "test_only": True, "saved": True, "duplicate": not bool(saved),
            "notification": notification, "emergency_dispatch": False}


@router.post("/student-devices/logout")
def logout_device(x_safety_device_session: Optional[str] = Header(default=None, alias="X-VeriFYD-Safety-Session")):
    ctx = _session(x_safety_device_session)
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("""UPDATE verifyd_safety_student_device_sessions SET revoked_at=NOW()
            WHERE session_hash=%s""", (_digest(x_safety_device_session),))
        _audit(cur, ctx["school_id"], ctx["student_id"], "supervised_device_session_closed", ctx["student_id"])
    return {"ok": True, "test_only": True, "revoked": True}


@router.post("/schools/{school_id}/student-devices/revoke")
def revoke_student_sessions(school_id: str, payload: IssueCode,
                            authorization: Optional[str] = Header(default=None)):
    _pilot()
    _school_allowed(school_id)
    actor = _actor(authorization)
    _schema()
    with get_db() as conn:
        cur = conn.cursor()
        _school_admin(cur, school_id, str(actor["id"]))
        _supervised(cur, school_id, payload.student_id)
        cur.execute("""DELETE FROM verifyd_safety_student_device_codes
                WHERE school_id=%s AND student_id=%s""", (school_id,payload.student_id))
        cur.execute("""UPDATE verifyd_safety_student_device_sessions SET revoked_at=NOW()
                WHERE school_id=%s AND student_id=%s AND revoked_at IS NULL""",
                (school_id,payload.student_id))
        _audit(cur,school_id,str(actor["id"]),"supervised_student_device_access_revoked",payload.student_id)
    return {"ok": True, "test_only": True, "revoked": True}

