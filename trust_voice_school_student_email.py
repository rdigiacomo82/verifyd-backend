"""Email-verified, TEST-only supervised student Safety sign-in.

This is deliberately separate from VeriFYD /voice authentication. A school roster
email is NOT a verified credential until its mailbox code is successfully checked.
Never creates an active identity, changes the child's internal email sentinel,
or grants calling, SMS, 911, or non-school access.
"""
from __future__ import annotations

import hashlib
import hmac
import html
import os
import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field, field_validator

from database import get_db, is_valid_email
from trust_voice_school_safety_training import _actor, _audit
from trust_voice_school_safety_registration import _school_admin
from trust_voice_school_student_device import (
    _pilot, _school_allowed, _schema as _device_schema,
    _guardian_ready, _supervised, _digest, SESSION_MINUTES,
)

router = APIRouter(prefix="/trust-voice/school-safety/student-devices/email",
                   tags=["TEST Supervised Student Email"])
_GENERIC = {"ok": True, "test_only": True,
            "message": "If this is a verified school-approved student email, a sign-in code may be sent.",
            "no_account_information_disclosed": True}
_MAX_DAILY = 5
_COOLDOWN_SECONDS = 120


def _enabled():
    _pilot()  # Existing default-off supervised-device feature gate
    if os.getenv("VERIFYD_SAFETY_STUDENT_EMAIL_ENABLED", "0").strip().lower() not in (
        "1", "true", "yes", "on"
    ):
        raise HTTPException(503, detail={"error": "student_email_test_pilot_not_enabled"})


def _allow_email_school(school_id: str):
    _school_allowed(school_id)
    allowed = {v.strip() for v in
               os.getenv("VERIFYD_SAFETY_STUDENT_EMAIL_SCHOOL_IDS", "").split(",") if v.strip()}
    if school_id not in allowed:
        raise HTTPException(403, detail={"error": "student_email_school_not_authorized"})


def _schema():
    _device_schema()
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("""CREATE TABLE IF NOT EXISTS verifyd_safety_student_email_bindings (
            school_id TEXT NOT NULL REFERENCES verifyd_safety_training_schools(school_id) ON DELETE CASCADE,
            student_id TEXT NOT NULL REFERENCES verifyd_identities(id) ON DELETE CASCADE,
            email_lower TEXT NOT NULL UNIQUE,
            verified_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            revoked_at TIMESTAMPTZ,
            PRIMARY KEY (school_id, student_id)
        )""")
        cur.execute("""CREATE TABLE IF NOT EXISTS verifyd_safety_student_email_challenges (
            email_hash TEXT PRIMARY KEY,
            school_id TEXT NOT NULL REFERENCES verifyd_safety_training_schools(school_id) ON DELETE CASCADE,
            student_id TEXT NOT NULL REFERENCES verifyd_identities(id) ON DELETE CASCADE,
            salt TEXT NOT NULL, code_hash TEXT NOT NULL,
            sent_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            expires_at TIMESTAMPTZ NOT NULL,
            attempts INTEGER NOT NULL DEFAULT 0,
            window_start TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            window_count INTEGER NOT NULL DEFAULT 1,
            used_at TIMESTAMPTZ
        )""")


class EmailRequest(BaseModel):
    email: str = Field(min_length=5, max_length=254)

    @field_validator("email")
    @classmethod
    def normalize(cls, value: str) -> str:
        email = value.strip().lower()
        if not is_valid_email(email):
            raise ValueError("invalid_school_student_email")
        return email


class EmailVerify(EmailRequest):
    code: str = Field(pattern=r"^[0-9]{8}$")


class RevokeEmail(BaseModel):
    student_id: str = Field(min_length=5, max_length=100)


def _candidate(cur, email: str):
    """Return exactly one authorized managed student; never trust client identity.

    Shared addresses, stale students and adult/guardian account emails are not
    eligible for direct student email authentication.
    """
    cur.execute("""SELECT s.school_id,s.local_ref,s.student_identity_id
        FROM verifyd_safety_managed_students s
        JOIN verifyd_safety_managed_student_contacts c
          ON c.school_id=s.school_id AND c.local_ref=s.local_ref
        JOIN verifyd_identities i
          ON i.id=s.student_identity_id AND i.status='supervised_child'
        JOIN verifyd_safety_training_members m
          ON m.school_id=s.school_id AND m.identity_id=s.student_identity_id
         AND m.role='student'
        WHERE lower(c.student_email)=%s AND s.enrollment_status='school_approved'
        LIMIT 2""", (email,))
    rows = cur.fetchall()
    if len(rows) != 1:
        return None
    row = rows[0]
    school_id, student_id = str(row["school_id"]), str(row["student_identity_id"])
    try:
        _allow_email_school(school_id)
        _supervised(cur, school_id, student_id)
        _guardian_ready(cur, school_id, student_id)
    except HTTPException:
        return None
    cur.execute("""SELECT 1 FROM verifyd_identities
        WHERE email_lower=%s LIMIT 1""", (email,))
    if cur.fetchone():
        return None
    cur.execute("""SELECT 1 FROM verifyd_safety_managed_student_contacts
        WHERE primary_guardian_email=%s OR backup_guardian_email=%s LIMIT 1""",
                (email, email))
    if cur.fetchone():
        return None
    cur.execute("""SELECT 1 FROM verifyd_safety_student_email_bindings
        WHERE email_lower=%s AND (school_id<>%s OR student_id<>%s)
        LIMIT 1""", (email, school_id, student_id))
    if cur.fetchone():
        return None
    return {"school_id": school_id, "student_id": student_id}


def _hash_code(salt: str, code: str) -> str:
    return hashlib.sha256((salt + ":" + code).encode("ascii")).hexdigest()


def _send_code(email: str, code: str) -> bool:
    """Send only a generic, non-identifying TEST code; do not log student PII."""
    key = os.getenv("RESEND_API_KEY", "").strip()
    if not key:
        return False
    try:
        import resend
        resend.api_key = key
        msg = ("VeriFYD Safety TEST sign-in code: " + code +
               ". Expires in 10 minutes. Only use this code if you requested it. "
               "This does not contact emergency services.")
        result = resend.Emails.send({
            "from": "VeriFYD Safety <noreply@vfvid.com>",
            "to": [email],
            "subject": "VeriFYD Safety TEST sign-in code",
            "text": msg,
            "html": ("<p>Your <strong>VeriFYD Safety TEST</strong> sign-in code is:</p>"
                     "<p style='font-size:28px;letter-spacing:4px'><strong>" +
                     html.escape(code) + "</strong></p>"
                     "<p>Expires in 10 minutes. This is not a live emergency service. "
                     "Ignore the message if you did not request it.</p>"),
        })
        return bool(result and (result.get("id") if isinstance(result, dict) else
                                getattr(result, "id", None)))
    except Exception:
        return False


@router.post("/request", status_code=202)
def request_email_code(payload: EmailRequest):
    _enabled()
    _schema()
    email = payload.email
    email_hash = _digest(email)
    code = secrets.randbelow(100000000)
    code_str = f"{code:08d}"
    salt = secrets.token_hex(16)
    send = False
    with get_db() as conn:
        cur = conn.cursor()
        child = _candidate(cur, email)
        if child is not None:
            cur.execute("""SELECT sent_at,window_start,window_count
                FROM verifyd_safety_student_email_challenges
                WHERE email_hash=%s FOR UPDATE""", (email_hash,))
            previous = cur.fetchone()
            now = datetime.now(timezone.utc)
            if previous:
                recent = (now - previous["sent_at"]).total_seconds() < _COOLDOWN_SECONDS
                same_window = now - previous["window_start"] < timedelta(days=1)
                count = previous["window_count"] if same_window else 0
            else:
                recent, same_window, count = False, False, 0
            if not recent and count < _MAX_DAILY:
                cur.execute("""INSERT INTO verifyd_safety_student_email_challenges
                    (email_hash,school_id,student_id,salt,code_hash,expires_at,
                     window_start,window_count)
                    VALUES(%s,%s,%s,%s,%s,NOW()+INTERVAL '10 minutes',NOW(),1)
                    ON CONFLICT(email_hash) DO UPDATE SET
                       school_id=EXCLUDED.school_id,student_id=EXCLUDED.student_id,
                       salt=EXCLUDED.salt,code_hash=EXCLUDED.code_hash,
                       sent_at=NOW(),expires_at=EXCLUDED.expires_at,
                       attempts=0,used_at=NULL,
                       window_start=CASE WHEN
                           verifyd_safety_student_email_challenges.window_start
                                < NOW()-INTERVAL '1 day'
                           THEN NOW() ELSE verifyd_safety_student_email_challenges.window_start END,
                       window_count=CASE WHEN
                           verifyd_safety_student_email_challenges.window_start
                                < NOW()-INTERVAL '1 day'
                           THEN 1 ELSE verifyd_safety_student_email_challenges.window_count+1 END""",
                    (email_hash,child["school_id"],child["student_id"],salt,
                     _hash_code(salt,code_str)))
                send = True
    if send and not _send_code(email,code_str):
        # A delivery failure must not leave a redeemable code in the database.
        with get_db() as conn:
            conn.cursor().execute("""UPDATE verifyd_safety_student_email_challenges
                SET used_at=NOW() WHERE email_hash=%s AND code_hash=%s AND used_at IS NULL""",
                (email_hash,_hash_code(salt,code_str)))
    return dict(_GENERIC)


@router.post("/verify")
def verify_email_code(payload: EmailVerify):
    _enabled()
    _schema()
    email_hash = _digest(payload.email)
    result = None
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("""SELECT school_id,student_id,salt,code_hash,attempts,
                       expires_at,used_at
            FROM verifyd_safety_student_email_challenges
            WHERE email_hash=%s FOR UPDATE""", (email_hash,))
        challenge = cur.fetchone()
        if challenge:
            cur.execute("""UPDATE verifyd_safety_student_email_challenges
                SET attempts=attempts+1 WHERE email_hash=%s""", (email_hash,))
            valid = (challenge["used_at"] is None
                     and challenge["expires_at"] > datetime.now(timezone.utc)
                     and challenge["attempts"] < 5
                     and hmac.compare_digest(challenge["code_hash"],
                                             _hash_code(challenge["salt"],payload.code)))
            if valid:
                child = _candidate(cur,payload.email)
                if (child and child["school_id"] == str(challenge["school_id"])
                        and child["student_id"] == str(challenge["student_id"])):
                    cur.execute("""SELECT identity_id FROM verifyd_safety_training_members
                        WHERE school_id=%s AND role='principal'
                        ORDER BY verified_at DESC LIMIT 1""", (child["school_id"],))
                    principal = cur.fetchone()
                    if principal:
                        try:
                            _school_admin(cur,child["school_id"],str(principal["identity_id"]))
                        except HTTPException:
                            principal = None
                    if principal:
                        cur.execute("""INSERT INTO verifyd_safety_student_email_bindings
                            (school_id,student_id,email_lower,verified_at,revoked_at)
                            VALUES(%s,%s,%s,NOW(),NULL)
                            ON CONFLICT(school_id,student_id) DO UPDATE SET
                            email_lower=EXCLUDED.email_lower,verified_at=NOW(),revoked_at=NULL""",
                            (child["school_id"],child["student_id"],payload.email))
                        cur.execute("""UPDATE verifyd_safety_student_email_challenges
                            SET used_at=NOW() WHERE email_hash=%s""",(email_hash,))
                        session = secrets.token_urlsafe(32)
                        cur.execute("""INSERT INTO verifyd_safety_student_device_sessions
                            (session_hash,school_id,student_id,code_issued_by,expires_at,
                             auth_method,bound_email)
                            VALUES(%s,%s,%s,%s,NOW()+INTERVAL '45 minutes',
                             'student_email',%s)
                            RETURNING expires_at""",
                            (_digest(session),child["school_id"],child["student_id"],
                             str(principal["identity_id"]),payload.email))
                        expires = cur.fetchone()["expires_at"]
                        _audit(cur,child["school_id"],child["student_id"],
                               "student_email_test_session_verified",child["student_id"])
                        result = {"ok":True,"test_only":True,"restricted_safety_only":True,
                                  "email_verified_for_safety":True,
                                  "session_token":session,"expires_at":expires.isoformat(),
                                  "school_id":child["school_id"],"normal_voice_login_enabled":False,
                                  "emergency_dispatch":False}
    if result is None:
        raise HTTPException(403,detail={"error":"invalid_or_expired_school_student_email_code"})
    return result


@router.post("/schools/{school_id}/revoke")
def revoke_email_access(school_id: str,payload: RevokeEmail,
                        authorization: Optional[str]=Header(default=None)):
    _enabled()
    _allow_email_school(school_id)
    actor = _actor(authorization)
    _schema()
    with get_db() as conn:
        cur=conn.cursor()
        _school_admin(cur,school_id,str(actor["id"]))
        _supervised(cur,school_id,payload.student_id)
        cur.execute("""UPDATE verifyd_safety_student_email_bindings
            SET revoked_at=NOW() WHERE school_id=%s AND student_id=%s""",
            (school_id,payload.student_id))
        cur.execute("""UPDATE verifyd_safety_student_email_challenges
            SET used_at=NOW() WHERE school_id=%s AND student_id=%s""",
            (school_id,payload.student_id))
        cur.execute("""UPDATE verifyd_safety_student_device_sessions
            SET revoked_at=NOW() WHERE school_id=%s AND student_id=%s
              AND revoked_at IS NULL""",(school_id,payload.student_id))
        _audit(cur,school_id,str(actor["id"]),"student_email_test_access_revoked",payload.student_id)
    return {"ok":True,"test_only":True,"revoked":True}
