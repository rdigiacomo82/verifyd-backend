"""Schoolwide + one-click family electronic acknowledgment, staged students.

A copied generic link or staff-reviewed Resend invitation leads to the SAME
guardian-controlled flow. Uploaded contact details are never enough by themselves:
only staff-confirmed family_contact_drafts can be matched, and matched children
are disclosed only after mailbox OTP verification. No incident or 911 activation.
"""
from __future__ import annotations

import hashlib
import html
import os
import secrets
import threading
import uuid
from datetime import datetime, timezone
from typing import Literal, Optional

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field
from database import get_db, create_otp, verify_otp, is_valid_email
from emailer import _send, _send_with_status, FROM_ADDRESS, FROM_NAME, send_otp_email_with_status
from trust_voice_school_safety_training import _actor, _audit
from trust_voice_school_safety_registration import _school_admin
from trust_voice_school_family_link import _schema as _link_schema
from trust_voice_school_family_invitations import _enabled as _invites_enabled
from trust_voice_school_safety_consents import POLICY_VERSION
from trust_voice_school_data_notice import DATA_NOTICE_VERSION, DATA_NOTICE_STATEMENTS, APP_USE_STATEMENTS

router = APIRouter(prefix="/trust-voice/school-safety", tags=["School Electronic Guardian Authorization"])
_lock = threading.Lock()
_ready = False
_MAX_CAMPAIGN = 40
# Keep the historical 55/day family-mail ceiling until provider capacity is
# confirmed. Split it into independently protected budgets: authentication
# messages keep 40 slots and schoolwide invitations may use 15. Per-school
# purpose limits are also separate, so invitations cannot starve OTP traffic.
_DEFAULT_SCHOOL_INVITE_DAILY = 40
_DEFAULT_SCHOOL_OTP_DAILY = 40
_DEFAULT_GLOBAL_INVITE_DAILY = 15
_DEFAULT_GLOBAL_OTP_DAILY = 40
_MAX_CHILDREN = 12


def _hash(v: str) -> str:
    return hashlib.sha256(v.encode("utf-8")).hexdigest()


def _limit(name: str, default: int, ceiling: int) -> int:
    try:
        return max(1, min(ceiling, int(os.getenv(name, str(default)))))
    except ValueError:
        return default


def _require_electronic_enabled(school_id: str):
    """Independent, default-off and school-allowlisted rollout gate."""
    _invites_enabled(school_id)
    on=os.getenv("VERIFYD_SAFETY_ELECTRONIC_ACK_ENABLED","0").lower() in ("1","true","yes","on")
    allowed={x.strip() for x in os.getenv("VERIFYD_SAFETY_ELECTRONIC_ACK_SCHOOL_IDS","").split(",") if x.strip()}
    if not on or school_id not in allowed:
        raise HTTPException(404,detail={"error":"electronic_school_enrollment_not_enabled_for_school"})


def _schema():
    global _ready
    _link_schema()
    if _ready:
        return
    with _lock:
        if _ready:
            return
        with get_db() as conn:
            cur = conn.cursor()
            cur.execute("""CREATE TABLE IF NOT EXISTS verifyd_safety_electronic_guardian_ack (
                school_id TEXT NOT NULL,
                local_ref TEXT NOT NULL,
                slot INTEGER NOT NULL CHECK(slot IN (1,2)),
                guardian_email TEXT NOT NULL,
                guardian_name TEXT NOT NULL,
                relationship TEXT NOT NULL,
                evidence_ref TEXT NOT NULL UNIQUE,
                policy_version TEXT NOT NULL,
                verified_at TIMESTAMPTZ NOT NULL,
                acknowledged_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                status TEXT NOT NULL DEFAULT 'pending_school_review'
                    CHECK(status IN ('pending_school_review','approved','revoked')),
                approved_at TIMESTAMPTZ,
                approved_by TEXT REFERENCES verifyd_identities(id),
                guardian_identity_id TEXT REFERENCES verifyd_identities(id),
                PRIMARY KEY (school_id,local_ref,slot),
                FOREIGN KEY(school_id,local_ref) REFERENCES
                    verifyd_safety_managed_students(school_id,local_ref)
            )""")
            cur.execute("""ALTER TABLE verifyd_safety_electronic_guardian_ack
                ADD COLUMN IF NOT EXISTS data_notice_version TEXT NOT NULL DEFAULT ''""")
            cur.execute("""ALTER TABLE verifyd_safety_electronic_guardian_ack
                ADD COLUMN IF NOT EXISTS app_use_acknowledged_at TIMESTAMPTZ""")
            cur.execute("""CREATE TABLE IF NOT EXISTS verifyd_safety_family_email_sessions (
                session_hash TEXT PRIMARY KEY,
                school_id TEXT NOT NULL,
                guardian_email TEXT NOT NULL,
                verified_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                expires_at TIMESTAMPTZ NOT NULL DEFAULT NOW()+INTERVAL '15 minutes',
                consumed_at TIMESTAMPTZ
            )""")
            cur.execute("""CREATE TABLE IF NOT EXISTS verifyd_safety_family_mail_ledger (
                id TEXT PRIMARY KEY,
                school_id TEXT NOT NULL,
                recipient_hash TEXT NOT NULL,
                purpose TEXT NOT NULL CHECK(purpose IN ('schoolwide_invite','guardian_otp')),
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                accepted_by_provider BOOLEAN
            )""")
            cur.execute("""ALTER TABLE verifyd_safety_family_mail_ledger
                ADD COLUMN IF NOT EXISTS campaign_id TEXT""")
            cur.execute("""ALTER TABLE verifyd_safety_family_mail_ledger
                ADD COLUMN IF NOT EXISTS student_refs TEXT[] NOT NULL DEFAULT '{}'""")
            cur.execute("""ALTER TABLE verifyd_safety_family_mail_ledger
                ADD COLUMN IF NOT EXISTS sent_by TEXT""")
            cur.execute("""CREATE INDEX IF NOT EXISTS safety_family_mail_limit
                ON verifyd_safety_family_mail_ledger(school_id,created_at)""")
            cur.execute("""CREATE INDEX IF NOT EXISTS safety_family_mail_global_limit
                ON verifyd_safety_family_mail_ledger(purpose,created_at)
                WHERE accepted_by_provider IS DISTINCT FROM FALSE""")
            cur.execute("""CREATE TABLE IF NOT EXISTS verifyd_safety_family_mail_queue (
                job_id TEXT PRIMARY KEY,
                school_id TEXT NOT NULL,
                campaign_id TEXT NOT NULL,
                recipient_hash TEXT NOT NULL,
                recipient_email TEXT,
                student_refs TEXT[] NOT NULL DEFAULT '{}',
                status TEXT NOT NULL DEFAULT 'queued'
                    CHECK(status IN ('queued','sending','accepted','retry','failed')),
                attempt_count INTEGER NOT NULL DEFAULT 0,
                available_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                last_attempt_at TIMESTAMPTZ,
                provider_message_id TEXT,
                last_error_code TEXT,
                mail_ledger_id TEXT,
                created_by TEXT,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                UNIQUE(school_id,recipient_hash,campaign_id)
            )""")
            cur.execute("""CREATE INDEX IF NOT EXISTS safety_family_mail_queue_due
                ON verifyd_safety_family_mail_queue(status,available_at,created_at)""")
            cur.execute("""CREATE TABLE IF NOT EXISTS verifyd_safety_family_mail_events (
                event_id TEXT PRIMARY KEY,
                job_id TEXT NOT NULL REFERENCES verifyd_safety_family_mail_queue(job_id) ON DELETE CASCADE,
                school_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                actor_id TEXT,
                attempt_count INTEGER NOT NULL DEFAULT 0,
                provider_message_id TEXT,
                error_code TEXT,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )""")
            cur.execute("""CREATE INDEX IF NOT EXISTS safety_family_mail_event_lookup
                ON verifyd_safety_family_mail_events(school_id,created_at DESC)""")
            cur.execute("""ALTER TABLE verifyd_safety_family_mail_queue
                ADD COLUMN IF NOT EXISTS mail_ledger_id TEXT""")
            cur.execute("""ALTER TABLE verifyd_safety_family_mail_ledger
                ADD COLUMN IF NOT EXISTS provider_message_id TEXT""")
        _ready = True


def _mail_event(cur,job_id,school_id,event_type,actor_id=None,attempt_count=0,
                provider_message_id=None,error_code=None):
    cur.execute("""INSERT INTO verifyd_safety_family_mail_events
        (event_id,job_id,school_id,event_type,actor_id,attempt_count,
         provider_message_id,error_code)
        VALUES(%s,%s,%s,%s,%s,%s,%s,%s)""",
        ("sfme_"+uuid.uuid4().hex,job_id,school_id,event_type,actor_id,
         int(attempt_count or 0),provider_message_id,error_code))


def _school_for_code(cur, code: str):
    cur.execute("""SELECT school_id,public_code FROM verifyd_safety_school_family_links
        WHERE public_code=%s""", (code.strip(),))
    school = cur.fetchone()
    if not school:
        return None
    try:
        _require_electronic_enabled(str(school["school_id"]))
    except HTTPException:
        return None
    return str(school["school_id"])


def _matches(cur, school_id: str, email: str):
    """Only school-verified assignments; never infer a link from a child's name."""
    cur.execute("""SELECT c.local_ref,c.slot,c.guardian_name,c.relationship,
        s.first_name,s.last_name,s.enrollment_status
        FROM verifyd_safety_family_contact_drafts c
        JOIN verifyd_safety_managed_students s
          ON s.school_id=c.school_id AND s.local_ref=c.local_ref
        WHERE c.school_id=%s AND c.guardian_email=%s
          AND s.enrollment_status='staged'
        ORDER BY s.last_name,s.first_name,c.slot LIMIT 13""",
        (school_id,email))
    rows = [dict(r) for r in cur.fetchall()]
    if len(rows)>_MAX_CHILDREN:
        raise HTTPException(409, detail={"error":"school_family_group_requires_staff_assistance"})
    return rows


def _reserve_mail(cur, school_id: str, email: str, purpose: str, *, campaign_id=None, student_refs=None, sent_by=None):
    if purpose not in ("schoolwide_invite", "guardian_otp"):
        raise ValueError("unsupported school family email purpose")
    cur.execute("SELECT pg_advisory_xact_lock(hashtext(%s))",
                ("school_family_mail_global",))
    prefix = "INVITE" if purpose == "schoolwide_invite" else "OTP"
    school_limit = _limit(
        f"VERIFYD_SAFETY_FAMILY_{prefix}_SCHOOL_DAILY",
        _DEFAULT_SCHOOL_INVITE_DAILY if prefix == "INVITE" else _DEFAULT_SCHOOL_OTP_DAILY,
        10000,
    )
    global_limit = _limit(
        f"VERIFYD_SAFETY_FAMILY_{prefix}_GLOBAL_DAILY",
        _DEFAULT_GLOBAL_INVITE_DAILY if prefix == "INVITE" else _DEFAULT_GLOBAL_OTP_DAILY,
        10000,
    )
    cur.execute("""SELECT count(*) AS n FROM verifyd_safety_family_mail_ledger
         WHERE purpose=%s AND accepted_by_provider IS DISTINCT FROM FALSE
           AND created_at>NOW()-INTERVAL '24 hours'""", (purpose,))
    if int(cur.fetchone()["n"]) >= global_limit:
        return None
    cur.execute("""SELECT count(*) AS n FROM verifyd_safety_family_mail_ledger
         WHERE school_id=%s AND purpose=%s AND accepted_by_provider IS DISTINCT FROM FALSE
           AND created_at>NOW()-INTERVAL '24 hours'""",(school_id,purpose))
    if int(cur.fetchone()["n"]) >= school_limit:
        return None
    fingerprint=_hash(school_id+":"+purpose+":"+email)
    if purpose == "schoolwide_invite":
        # Successful sends remain in the Principal's history, not the active
        # queue. Never resend from a duplicate click or a page refresh.
        cur.execute("""SELECT 1 FROM verifyd_safety_family_mail_ledger
            WHERE school_id=%s AND purpose='schoolwide_invite'
              AND recipient_hash=%s AND accepted_by_provider IS TRUE LIMIT 1""",
            (school_id,fingerprint))
        if cur.fetchone():
            return None
    cur.execute("""SELECT count(*) AS n FROM verifyd_safety_family_mail_ledger
         WHERE school_id=%s AND purpose=%s AND recipient_hash=%s
           AND (purpose='guardian_otp' OR accepted_by_provider IS DISTINCT FROM FALSE)
           AND created_at>NOW()-INTERVAL '24 hours'""",(school_id,purpose,fingerprint))
    if int(cur.fetchone()["n"]) >= (2 if purpose=="guardian_otp" else 1):
        return None
    id="sfmail_"+uuid.uuid4().hex
    cur.execute("""INSERT INTO verifyd_safety_family_mail_ledger
        (id,school_id,recipient_hash,purpose,campaign_id,student_refs,sent_by)
        VALUES(%s,%s,%s,%s,%s,%s,%s)""",
        (id,school_id,fingerprint,purpose,campaign_id,student_refs or [],sent_by))
    return id


def _delivery_result(id: str, accepted: bool, provider_message_id=None):
    with get_db() as conn:
        conn.cursor().execute("""UPDATE verifyd_safety_family_mail_ledger
            SET accepted_by_provider=%s,provider_message_id=%s WHERE id=%s""",
            (bool(accepted),provider_message_id,id))


def _school_mail_limit(school_id: str, purpose: str) -> int:
    prefix = "INVITE" if purpose == "schoolwide_invite" else "OTP"
    return _limit(
        f"VERIFYD_SAFETY_FAMILY_{prefix}_SCHOOL_DAILY",
        _DEFAULT_SCHOOL_INVITE_DAILY if prefix == "INVITE" else _DEFAULT_SCHOOL_OTP_DAILY,
        10000,
    )


def _global_mail_limit(purpose: str) -> int:
    prefix = "INVITE" if purpose == "schoolwide_invite" else "OTP"
    return _limit(
        f"VERIFYD_SAFETY_FAMILY_{prefix}_GLOBAL_DAILY",
        _DEFAULT_GLOBAL_INVITE_DAILY if prefix == "INVITE" else _DEFAULT_GLOBAL_OTP_DAILY,
        10000,
    )


def dispatch_schoolwide_mail_once(school_id: Optional[str] = None, limit: int = 10):
    """Send a bounded slice of due schoolwide mail jobs; safe for parallel workers."""
    _schema()
    batch_limit = max(1, min(20, int(limit)))
    with get_db() as conn:
        cur = conn.cursor()
        # Recover claims left by a process restart. Provider acceptance remains
        # at-least-once in this failure window; the same generic invitation is
        # idempotent for registration and never carries student details.
        cur.execute("""UPDATE verifyd_safety_family_mail_queue
            SET status=CASE WHEN attempt_count>=4 THEN 'failed' ELSE 'retry' END,
                available_at=NOW(),last_error_code='worker_claim_expired',updated_at=NOW()
            WHERE status='sending' AND last_attempt_at<NOW()-INTERVAL '10 minutes'
              AND (%s::text IS NULL OR school_id=%s)
            RETURNING job_id,school_id,attempt_count,status""",(school_id,school_id))
        for stale in cur.fetchall():
            _mail_event(cur,stale["job_id"],stale["school_id"],"worker_claim_expired",
                attempt_count=stale["attempt_count"],error_code="worker_claim_expired")
        cur.execute("""SELECT job_id,school_id,campaign_id,recipient_hash,
                    recipient_email,student_refs,attempt_count,created_by
             FROM verifyd_safety_family_mail_queue
             WHERE status IN ('queued','retry') AND available_at<=NOW()
               AND (%s::text IS NULL OR school_id=%s)
             ORDER BY available_at,created_at
             LIMIT %s FOR UPDATE SKIP LOCKED""", (school_id,school_id,batch_limit))
        jobs = [dict(row) for row in cur.fetchall()]

    attempted = accepted = deferred = failed = 0
    for job in jobs:
        if not job.get("recipient_email"):
            with get_db() as conn:
                cur=conn.cursor()
                cur.execute("""UPDATE verifyd_safety_family_mail_queue
                    SET status='failed',last_error_code='recipient_unavailable',updated_at=NOW()
                    WHERE job_id=%s AND status IN ('queued','retry')""",(job["job_id"],))
                if cur.rowcount:
                    _mail_event(cur,job["job_id"],job["school_id"],"failed",
                        attempt_count=job.get("attempt_count"),error_code="recipient_unavailable")
            failed += 1
            continue
        try:
            _require_electronic_enabled(str(job["school_id"]))
        except HTTPException:
            continue

        with get_db() as conn:
            cur = conn.cursor()
            # Claim the row and reserve quota atomically; no provider call is made
            # while holding a database transaction open.
            cur.execute("""UPDATE verifyd_safety_family_mail_queue
                SET status='sending',
                    last_attempt_at=NOW(),updated_at=NOW()
                WHERE job_id=%s AND status IN ('queued','retry')
                RETURNING attempt_count""",(job["job_id"],))
            claimed = cur.fetchone()
            if not claimed:
                continue
            ledger_id = _reserve_mail(cur,str(job["school_id"]),str(job["recipient_email"]),
                "schoolwide_invite",campaign_id=job["campaign_id"],
                student_refs=job["student_refs"] or [],sent_by=job["created_by"])
            if not ledger_id:
                # A previous provider-accepted invitation is final; do not send
                # it again. Otherwise wait until a rolling-window quota opens.
                cur.execute("""SELECT 1 FROM verifyd_safety_family_mail_ledger
                    WHERE school_id=%s AND purpose='schoolwide_invite'
                      AND recipient_hash=%s AND accepted_by_provider IS TRUE LIMIT 1""",
                    (job["school_id"],job["recipient_hash"]))
                already_accepted = bool(cur.fetchone())
                cur.execute("""UPDATE verifyd_safety_family_mail_queue
                    SET status=%s,recipient_email=CASE WHEN %s THEN NULL ELSE recipient_email END,
                        last_error_code=%s,
                        available_at=CASE WHEN %s THEN available_at ELSE GREATEST(
                          NOW()+INTERVAL '1 minute',
                          COALESCE((SELECT MIN(created_at)+INTERVAL '24 hours'
                            FROM verifyd_safety_family_mail_ledger
                            WHERE purpose='schoolwide_invite'),NOW()+INTERVAL '1 minute'),
                          COALESCE((SELECT MIN(created_at)+INTERVAL '24 hours'
                            FROM verifyd_safety_family_mail_ledger
                            WHERE school_id=%s AND purpose='schoolwide_invite'),NOW()+INTERVAL '1 minute')) END,
                        updated_at=NOW() WHERE job_id=%s""",
                    ("accepted" if already_accepted else "queued",already_accepted,
                     "already_accepted" if already_accepted else "quota_wait",
                     already_accepted,job["school_id"],job["job_id"]))
                _mail_event(cur,job["job_id"],job["school_id"],
                    "already_accepted" if already_accepted else "quota_deferred",
                    actor_id=job.get("created_by"),attempt_count=claimed["attempt_count"],
                    error_code=None if already_accepted else "quota_wait")
                deferred += int(not already_accepted)
                accepted += int(already_accepted)
                continue
            cur.execute("""UPDATE verifyd_safety_family_mail_queue
                SET mail_ledger_id=%s,attempt_count=attempt_count+1
                WHERE job_id=%s RETURNING attempt_count""",(ledger_id,job["job_id"]))
            claimed=cur.fetchone()
            _mail_event(cur,job["job_id"],job["school_id"],"sending",
                actor_id=job.get("created_by"),attempt_count=claimed["attempt_count"])

        with get_db() as conn:
            cur=conn.cursor()
            cur.execute("""SELECT s.school_name,l.public_code
                FROM verifyd_safety_training_schools s
                LEFT JOIN verifyd_safety_school_family_links l ON l.school_id=s.school_id
                WHERE s.school_id=%s""",(job["school_id"],))
            school=cur.fetchone() or {}
        school_name=html.escape(str(school.get("school_name") or "Your school"))
        public_code=str(school.get("public_code") or "")
        url="https://vfvid.com/Safety#school-family="+public_code
        result=_send_with_status({
            "from":f"{FROM_NAME} <{FROM_ADDRESS}>",
            "to":[job["recipient_email"]],
            "subject":"VeriFYD Safety — School family registration",
            "html":("<p>"+school_name+" is inviting families to review their VeriFYD Safety registration.</p>"
                "<p><a href='"+html.escape(url,quote=True)+"'>Open school family registration</a></p>"
                "<p>Use your own email address to verify. No student details are contained in this email. "
                "Ignore the invitation if you are not a guardian at this school.</p>")})
        attempted += 1
        if result.get("accepted"):
            with get_db() as conn:
                cur=conn.cursor()
                cur.execute("""UPDATE verifyd_safety_family_mail_ledger
                    SET accepted_by_provider=TRUE,provider_message_id=%s WHERE id=%s""",
                    (result.get("provider_message_id"),ledger_id))
                cur.execute("""UPDATE verifyd_safety_family_mail_queue SET status='accepted',
                    recipient_email=NULL,provider_message_id=%s,last_error_code=NULL,updated_at=NOW()
                    WHERE job_id=%s""",(result.get("provider_message_id"),job["job_id"]))
                _mail_event(cur,job["job_id"],job["school_id"],"provider_accepted",
                    actor_id=job.get("created_by"),attempt_count=claimed["attempt_count"],
                    provider_message_id=result.get("provider_message_id"))
            accepted += 1
        else:
            code=str(result.get("error_code") or "provider_error")[:80]
            attempt_count=int(claimed["attempt_count"])
            retry=attempt_count<4
            delay=min(3600,60*(2**max(0,attempt_count-1)))
            with get_db() as conn:
                cur=conn.cursor()
                cur.execute("UPDATE verifyd_safety_family_mail_ledger SET accepted_by_provider=FALSE WHERE id=%s",(ledger_id,))
                cur.execute("""UPDATE verifyd_safety_family_mail_queue
                    SET status=%s,last_error_code=%s,
                        available_at=CASE WHEN %s THEN NOW()+(%s * INTERVAL '1 second') ELSE available_at END,
                        updated_at=NOW() WHERE job_id=%s""",
                    ("retry" if retry else "failed",code,retry,delay,job["job_id"]))
                _mail_event(cur,job["job_id"],job["school_id"],"retry_scheduled" if retry else "failed",
                    actor_id=job.get("created_by"),attempt_count=attempt_count,error_code=code)
            failed += 1
    return {"attempted":attempted,"provider_accepted":accepted,"deferred_by_quota":deferred,
            "failed_or_retrying":failed,"batch_limit":batch_limit}


class PublicCodeRequest(BaseModel):
    public_code: str = Field(min_length=16,max_length=96)
    guardian_email: str = Field(min_length=5,max_length=254)


class PublicVerify(PublicCodeRequest):
    email_code: str = Field(min_length=6,max_length=12)


class ElectronicAccept(BaseModel):
    claim_session: str = Field(min_length=32,max_length=180)
    children: list[str] = Field(min_length=1,max_length=_MAX_CHILDREN)
    relationship_confirmed: Literal[True]
    acknowledge_school_safety_policy: Literal[True]
    acknowledge_school_data_use_notice: Literal[True]
    acknowledge_app_use_instructions: Literal[True]
    data_notice_version: str = Field(min_length=8,max_length=100)
    policy_version: str = Field(min_length=5,max_length=100)


class Campaign(BaseModel):
    local_refs: list[str] = Field(min_length=1,max_length=250)
    school_confirmed_guardian_recipient_assignments: Literal[True]
    principal_confirms_send: Literal[True]


@router.get("/family-invitations/data-use-notice")
def family_data_use_notice():
    """Versioned public notice with no guardian or roster information."""
    return {"ok":True,"data_notice_version":DATA_NOTICE_VERSION,
            "data_use_statements":DATA_NOTICE_STATEMENTS,
            "app_use_statements":APP_USE_STATEMENTS,
            "training_only":True,"district_notice_required_for_real_students":True,
            "live_emergency_dispatch":False}


@router.post("/family-invitations/electronic/request-code")
def request_schoolwide_otp(payload:PublicCodeRequest):
    """Never reveal if this mailbox or any child exists at the school."""
    generic={"ok":True,"message":"If the school has confirmed your email, check for a verification code. Otherwise contact the school.","student_information_disclosed":False}
    email=payload.guardian_email.strip().lower()
    if not is_valid_email(email):
        return generic
    _schema()
    with get_db() as conn:
        cur=conn.cursor()
        school_id=_school_for_code(cur,payload.public_code)
        if not school_id or not _matches(cur,school_id,email):
            return generic
        reserved=_reserve_mail(cur,school_id,email,"guardian_otp")
    if reserved:
        result={"accepted":False,"provider_message_id":None}
        try:
            otp=create_otp(email)
            result=send_otp_email_with_status(email,otp)
            # A retry uses the same short-lived one-time code. This avoids
            # invalidating a message that a provider may have accepted before
            # its response was lost. Duplicate email copies carry one code.
            if not result.get("accepted"):
                result=send_otp_email_with_status(email,otp)
        except Exception:
            result={"accepted":False,"provider_message_id":None}
        _delivery_result(reserved,bool(result.get("accepted")),result.get("provider_message_id"))
    return generic


@router.post("/family-invitations/electronic/verify")
def verify_schoolwide_otp(payload:PublicVerify):
    """Return assigned children only after email ownership was independently verified."""
    email=payload.guardian_email.strip().lower()
    if not is_valid_email(email):
        raise HTTPException(403,detail={"error":"invalid_guardian_email_verification"})
    _schema()
    with get_db() as conn:
        cur=conn.cursor()
        school_id=_school_for_code(cur,payload.public_code)
        if not school_id or not _matches(cur,school_id,email):
            raise HTTPException(403,detail={"error":"invalid_guardian_email_verification"})
    passed,_=verify_otp(email,payload.email_code)
    if not passed:
        raise HTTPException(403,detail={"error":"invalid_guardian_email_verification"})
    session=secrets.token_urlsafe(36)
    with get_db() as conn:
        cur=conn.cursor()
        if not _school_for_code(cur,payload.public_code)==school_id:
            raise HTTPException(403,detail={"error":"school_registration_link_rotated"})
        children=_matches(cur,school_id,email)
        if not children:
            raise HTTPException(403,detail={"error":"school_guardian_assignment_no_longer_available"})
        cur.execute("""INSERT INTO verifyd_safety_family_email_sessions
            (session_hash,school_id,guardian_email) VALUES(%s,%s,%s)""",
            (_hash(session),school_id,email))
    return {"ok":True,"claim_session":session,"policy_version":POLICY_VERSION,
        "data_notice_version":DATA_NOTICE_VERSION,
        "children":[{"local_ref":c["local_ref"],"student_name":c["first_name"]+" "+c["last_name"],
                     "guardian_name":c["guardian_name"],"relationship":c["relationship"],
                     "slot":c["slot"],"enrollment_status":c["enrollment_status"]}
                    for c in children],
        "family_access_granted":False,"verification_required_before_children":True}


@router.post("/family-invitations/electronic/acknowledge")
def acknowledge_schoolwide(payload:ElectronicAccept):
    """Record an individual electronic acknowledgment; NOT a family permission."""
    if payload.policy_version!=POLICY_VERSION:
        raise HTTPException(409,detail={"error":"guardian_policy_changed_reload"})
    if payload.data_notice_version!=DATA_NOTICE_VERSION:
        raise HTTPException(409,detail={"error":"school_data_notice_version_changed_reload_registration"})
    refs=[str(s).strip().upper() for s in payload.children]
    if len(refs)!=len(set(refs)):
        raise HTTPException(422,detail={"error":"duplicate_child_in_acknowledgment"})
    _schema()
    with get_db() as conn:
        cur=conn.cursor()
        cur.execute("""SELECT school_id,guardian_email,verified_at,expires_at>NOW() AS active
            FROM verifyd_safety_family_email_sessions
            WHERE session_hash=%s AND consumed_at IS NULL FOR UPDATE""",
            (_hash(payload.claim_session),))
        session=cur.fetchone()
        if not session or not session["active"]:
            raise HTTPException(403,detail={"error":"guardian_email_session_expired"})
        school_id=str(session["school_id"])
        _require_electronic_enabled(school_id)
        email=str(session["guardian_email"])
        matches=_matches(cur,school_id,email)
        chosen=[row for row in matches if row["local_ref"] in refs]
        if len(chosen)!=len(refs):
            raise HTTPException(403,detail={"error":"child_not_in_school_verified_guardian_assignments"})
        cur.execute("""SELECT local_ref,slot,guardian_email,status
           FROM verifyd_safety_electronic_guardian_ack
           WHERE school_id=%s AND local_ref=ANY(%s) FOR UPDATE""",(school_id,refs))
        existing={(r["local_ref"],r["slot"]):dict(r) for r in cur.fetchall()}
        for row in chosen:
            found=existing.get((row["local_ref"],row["slot"]))
            if found and (found["guardian_email"]!=email or found["status"]!="pending_school_review"):
                raise HTTPException(409,detail={"error":"guardian_acknowledgment_already_assigned_contact_school"})
        result=[]
        for row in chosen:
            evidence_ref="EACK-"+uuid.uuid4().hex[:20].upper()
            cur.execute("""INSERT INTO verifyd_safety_electronic_guardian_ack
              (school_id,local_ref,slot,guardian_email,guardian_name,relationship,
               evidence_ref,policy_version,verified_at,data_notice_version,app_use_acknowledged_at)
              VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,NOW())
              ON CONFLICT(school_id,local_ref,slot) DO NOTHING
              RETURNING evidence_ref""",
              (school_id,row["local_ref"],row["slot"],email,row["guardian_name"],
               row["relationship"],evidence_ref,POLICY_VERSION,session["verified_at"],
               DATA_NOTICE_VERSION))
            created=cur.fetchone()
            result.append({"local_ref":row["local_ref"],"status":"pending_school_review",
                           "evidence_ref":created["evidence_ref"] if created else "previously_recorded"})
        cur.execute("""UPDATE verifyd_safety_family_email_sessions SET consumed_at=NOW()
            WHERE session_hash=%s""",(_hash(payload.claim_session),))
    return {"ok":True,"acknowledgments":result,"policy_version":POLICY_VERSION,
            "data_notice_version":DATA_NOTICE_VERSION,
            "family_access_granted":False,"student_identity_activated":False,
            "requires_independent_school_review":True}


@router.get("/schools/{school_id}/family-invitations/electronic-queue")
def electronic_school_queue(school_id:str,authorization:Optional[str]=Header(default=None)):
    _require_electronic_enabled(school_id)
    actor=_actor(authorization)
    _schema()
    with get_db() as conn:
        cur=conn.cursor()
        _school_admin(cur,school_id,str(actor["id"]))
        cur.execute("""SELECT a.local_ref,a.slot,a.guardian_name,a.guardian_email,
             a.relationship,a.policy_version,a.data_notice_version,
             a.app_use_acknowledged_at,a.evidence_ref,a.verified_at,
             a.acknowledged_at,a.status,a.approved_at,
             s.first_name,s.last_name,s.enrollment_status
             FROM verifyd_safety_electronic_guardian_ack a
             JOIN verifyd_safety_managed_students s
               ON s.school_id=a.school_id AND s.local_ref=a.local_ref
             WHERE a.school_id=%s ORDER BY a.acknowledged_at DESC LIMIT 1000""",
             (school_id,))
        rows=[dict(r) for r in cur.fetchall()]
    return {"ok":True,"records":rows,"guardian_relationship_school_approved":False}


@router.get("/schools/{school_id}/family-invitations/schoolwide-history")
def schoolwide_invitation_history(school_id:str,authorization:Optional[str]=Header(default=None)):
    """School-admin-only history; provider acceptance is not inbox delivery."""
    _require_electronic_enabled(school_id)
    actor=_actor(authorization)
    _schema()
    with get_db() as conn:
        cur=conn.cursor()
        _school_admin(cur,school_id,str(actor["id"]))
        cur.execute("""SELECT recipient_hash,MAX(created_at) AS accepted_at
            FROM verifyd_safety_family_mail_ledger
            WHERE school_id=%s AND purpose='schoolwide_invite'
              AND accepted_by_provider IS TRUE
            GROUP BY recipient_hash""",(school_id,))
        accepted_by_hash={str(x["recipient_hash"]):x["accepted_at"] for x in cur.fetchall()}
        cur.execute("""SELECT id,recipient_hash,created_at,accepted_by_provider,
               campaign_id,student_refs
            FROM verifyd_safety_family_mail_ledger
            WHERE school_id=%s AND purpose='schoolwide_invite'
            ORDER BY created_at DESC LIMIT 100""",(school_id,))
        recent=[dict(x) for x in cur.fetchall()]
        cur.execute("""SELECT s.local_ref,s.first_name,s.last_name,c.guardian_email
            FROM verifyd_safety_managed_students s
            JOIN verifyd_safety_family_contact_drafts c
              ON c.school_id=s.school_id AND c.local_ref=s.local_ref
            WHERE s.school_id=%s AND s.enrollment_status='staged'""",(school_id,))
        matches=[dict(x) for x in cur.fetchall()]
    grouped={}
    for item in matches:
        ref=str(item["local_ref"])
        email=str(item["guardian_email"]).strip().lower()
        if not is_valid_email(email):
            continue
        if ref not in grouped:
            grouped[ref]={"local_ref":ref,"first_name":item["first_name"],
                          "last_name":item["last_name"],"recipients":{}}
        key=_hash(school_id+":schoolwide_invite:"+email)
        grouped[ref]["recipients"][email]=accepted_by_hash.get(key)
    students=[]
    for item in grouped.values():
        recipient_states=item.pop("recipients")
        accepted=[email for email,at in recipient_states.items() if at is not None]
        students.append({**item,"total_guardian_emails":len(recipient_states),
            "accepted_guardian_emails":len(accepted),
            "pending_guardian_emails":len(recipient_states)-len(accepted),
            "accepted_recipient_emails":accepted,
            "last_accepted_at":max((at for at in recipient_states.values() if at is not None),default=None)})
    deliveries=[{"delivery_id":r["id"],"campaign_id":r["campaign_id"],
          "student_refs":r["student_refs"] or [],"attempted_at":r["created_at"],
          "status":("accepted_by_sender" if r["accepted_by_provider"] is True
                    else "sender_rejected" if r["accepted_by_provider"] is False
                    else "awaiting_sender_response")}
          for r in recent]
    return {"ok":True,"students":students,"recent_delivery_attempts":deliveries,
            "provider_acceptance_is_not_inbox_delivery":True,"school_only":True}


@router.post("/schools/{school_id}/family-invitations/send-schoolwide")
def send_schoolwide_campaign(school_id:str,payload:Campaign,
    authorization:Optional[str]=Header(default=None)):
    """Queue a bounded, admin-reviewed campaign. No student data or bulk BCC."""
    _require_electronic_enabled(school_id)
    actor=_actor(authorization)
    _schema()
    refs=[r.strip().upper() for r in payload.local_refs]
    if len(refs)!=len(set(refs)):
        raise HTTPException(422,detail={"error":"duplicate_student_reference"})
    with get_db() as conn:
        cur=conn.cursor()
        _school_admin(cur,school_id,str(actor["id"]))
        cur.execute("""SELECT s.local_ref,c.guardian_email
             FROM verifyd_safety_managed_students s
             JOIN verifyd_safety_family_contact_drafts c
               ON c.school_id=s.school_id AND c.local_ref=s.local_ref
             WHERE s.school_id=%s AND s.local_ref=ANY(%s)
               AND s.enrollment_status='staged'
             ORDER BY c.guardian_email""",(school_id,refs))
        records=[dict(row) for row in cur.fetchall()]
        emails=sorted({r["guardian_email"].lower() for r in records
                        if is_valid_email(r["guardian_email"])})
        if not emails:
            raise HTTPException(409,detail={"error":"no_verified_guardian_recipients_for_schoolwide_link"})
        if len(emails)>_MAX_CAMPAIGN:
            raise HTTPException(422,detail={"error":"schoolwide_campaign_batch_maximum_40"})
        cur.execute("""SELECT public_code FROM verifyd_safety_school_family_links
            WHERE school_id=%s""",(school_id,))
        link=cur.fetchone()
        if not link:
            code="SC_"+secrets.token_urlsafe(18)
            cur.execute("""INSERT INTO verifyd_safety_school_family_links
                (school_id,public_code,created_by) VALUES(%s,%s,%s)""",
                (school_id,code,str(actor["id"])))
        else:
            code=link["public_code"]
        cur.execute("""SELECT school_name FROM verifyd_safety_training_schools
            WHERE school_id=%s""",(school_id,))
        school_name=str(cur.fetchone()["school_name"])
        campaign_id="sfmailbatch_"+uuid.uuid4().hex
        queued=duplicate=0
        for email in emails:
            refs_for_email=sorted({str(r["local_ref"]) for r in records
                                   if str(r["guardian_email"]).strip().lower()==email})
            fingerprint=_hash(school_id+":schoolwide_invite:"+email)
            cur.execute("""SELECT EXISTS(SELECT 1 FROM verifyd_safety_family_mail_ledger
                WHERE school_id=%s AND purpose='schoolwide_invite'
                  AND recipient_hash=%s AND accepted_by_provider IS TRUE) AS found""",
                (school_id,fingerprint))
            previously_accepted=bool(cur.fetchone()["found"])
            cur.execute("""SELECT EXISTS(SELECT 1 FROM verifyd_safety_family_mail_queue
                WHERE school_id=%s AND recipient_hash=%s
                  AND status IN ('queued','sending','retry')) AS found""",
                (school_id,fingerprint))
            already_pending=bool(cur.fetchone()["found"])
            if previously_accepted or already_pending:
                duplicate+=1
                continue
            job_id="sfmailq_"+uuid.uuid4().hex
            cur.execute("""INSERT INTO verifyd_safety_family_mail_queue
                (job_id,school_id,campaign_id,recipient_hash,recipient_email,student_refs,created_by)
                VALUES(%s,%s,%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING""",
                (job_id,school_id,campaign_id,fingerprint,email,refs_for_email,str(actor["id"])))
            if cur.rowcount:
                queued+=1
                _mail_event(cur,job_id,school_id,"queued",actor_id=str(actor["id"]))
            else:
                duplicate+=1
    # Start a small first slice immediately; the queue worker drains the rest
    # and handles exponential retries without holding the browser request open.
    first_slice=dispatch_schoolwide_mail_once(school_id=school_id,limit=10) if queued else {
        "attempted":0,"provider_accepted":0,"deferred_by_quota":0,"failed_or_retrying":0}
    url="https://vfvid.com/Safety#school-family="+code
    return {"ok":True,"unique_verified_recipients":len(emails),"campaign_id":campaign_id,
            "queued":queued,"duplicate_or_already_accepted":duplicate,
            "provider_accepted":first_slice["provider_accepted"],
            "attempted":first_slice["attempted"],
            "deferred_by_quota":first_slice["deferred_by_quota"],
            "failed_or_retrying":first_slice["failed_or_retrying"],
            "mail_service":"Resend","school_invitation_daily_limit":_school_mail_limit(school_id,"schoolwide_invite"),
            "provider_acceptance_is_not_inbox_delivery":True,
            "guardian_access_granted":False,"schoolwide_link":url}


@router.get("/schools/{school_id}/family-invitations/delivery-status")
def family_mail_delivery_status(school_id:str,authorization:Optional[str]=Header(default=None)):
    """School-admin view of its own delivery capacity and queue; never exposes addresses."""
    _require_electronic_enabled(school_id)
    actor=_actor(authorization);_schema()
    with get_db() as conn:
        cur=conn.cursor();_school_admin(cur,school_id,str(actor["id"]))
        cur.execute("""SELECT purpose,count(*) AS used FROM verifyd_safety_family_mail_ledger
            WHERE school_id=%s AND accepted_by_provider IS DISTINCT FROM FALSE
              AND created_at>NOW()-INTERVAL '24 hours' GROUP BY purpose""",(school_id,))
        used={str(r["purpose"]):int(r["used"]) for r in cur.fetchall()}
        cur.execute("""SELECT purpose,count(*) AS used FROM verifyd_safety_family_mail_ledger
            WHERE accepted_by_provider IS DISTINCT FROM FALSE
              AND created_at>NOW()-INTERVAL '24 hours' GROUP BY purpose""")
        global_used={str(r["purpose"]):int(r["used"]) for r in cur.fetchall()}
        cur.execute("""SELECT status,count(*) AS n,min(available_at) AS next_at
            FROM verifyd_safety_family_mail_queue WHERE school_id=%s
              AND status IN ('queued','retry','sending','failed') GROUP BY status""",(school_id,))
        queue={str(r["status"]):{"count":int(r["n"]),"next_at":r["next_at"]} for r in cur.fetchall()}
        cur.execute("""SELECT campaign_id,status,count(*) AS n,min(created_at) AS created_at,
                min(available_at) AS next_at
            FROM verifyd_safety_family_mail_queue WHERE school_id=%s
            GROUP BY campaign_id,status ORDER BY min(created_at) DESC LIMIT 80""",(school_id,))
        campaigns=[dict(r) for r in cur.fetchall()]
        cur.execute("""SELECT job_id,campaign_id,student_refs,status,
                available_at,attempt_count,last_error_code,provider_message_id
            FROM verifyd_safety_family_mail_queue WHERE school_id=%s
            ORDER BY created_at DESC LIMIT 200""",(school_id,))
        jobs=[dict(r) for r in cur.fetchall()]
        cur.execute("""SELECT event_type,attempt_count,provider_message_id,error_code,created_at
            FROM verifyd_safety_family_mail_events WHERE school_id=%s
            ORDER BY created_at DESC LIMIT 200""",(school_id,))
        events=[dict(r) for r in cur.fetchall()]
    invite_limit=_school_mail_limit(school_id,"schoolwide_invite")
    otp_limit=_school_mail_limit(school_id,"guardian_otp")
    return {"ok":True,"school_id":school_id,
        "quota_window":"rolling_24_hours",
        "quotas":{"schoolwide_invite":{"daily_limit":invite_limit,
                    "used":used.get("schoolwide_invite",0),
                    "remaining":max(0,invite_limit-used.get("schoolwide_invite",0)),
                    "global_daily_limit":_global_mail_limit("schoolwide_invite"),
                    "global_used":global_used.get("schoolwide_invite",0),
                    "global_remaining":max(0,_global_mail_limit("schoolwide_invite")-global_used.get("schoolwide_invite",0))},
                 "guardian_otp":{"daily_limit":otp_limit,
                    "used":used.get("guardian_otp",0),
                    "remaining":max(0,otp_limit-used.get("guardian_otp",0)),
                    "global_daily_limit":_global_mail_limit("guardian_otp"),
                    "global_used":global_used.get("guardian_otp",0),
                    "global_remaining":max(0,_global_mail_limit("guardian_otp")-global_used.get("guardian_otp",0))}},
        "queue":queue,"campaigns":campaigns,"jobs":jobs,"recent_events":events,
        "campaign_batch_maximum":_MAX_CAMPAIGN,
        "provider_acceptance_is_not_inbox_delivery":True}


@router.post("/schools/{school_id}/family-invitations/delivery-queue/dispatch")
def dispatch_school_mail_queue(school_id:str,authorization:Optional[str]=Header(default=None)):
    _require_electronic_enabled(school_id)
    actor=_actor(authorization);_schema()
    with get_db() as conn:
        _school_admin(conn.cursor(),school_id,str(actor["id"]))
    result=dispatch_schoolwide_mail_once(school_id=school_id,limit=10)
    with get_db() as conn:
        _audit(conn.cursor(),school_id,str(actor["id"]),"family_mail_queue_dispatched",
               str(result.get("attempted",0)))
    return {"ok":True,**result,"school_only":True}


@router.post("/schools/{school_id}/family-invitations/delivery-queue/retry-failed")
def retry_failed_school_mail(school_id:str,authorization:Optional[str]=Header(default=None)):
    _require_electronic_enabled(school_id)
    actor=_actor(authorization);_schema()
    with get_db() as conn:
        cur=conn.cursor();_school_admin(cur,school_id,str(actor["id"]))
        cur.execute("""WITH retryable AS (
            SELECT job_id FROM verifyd_safety_family_mail_queue
            WHERE school_id=%s AND status='failed' AND recipient_email IS NOT NULL
              AND attempt_count<6
            ORDER BY updated_at LIMIT 20 FOR UPDATE SKIP LOCKED
        ) UPDATE verifyd_safety_family_mail_queue q
            SET status='queued',last_error_code=NULL,
                available_at=NOW(),updated_at=NOW()
            FROM retryable r WHERE q.job_id=r.job_id
            RETURNING q.job_id,q.attempt_count""",(school_id,))
        count=int(cur.rowcount or 0)
        for row in cur.fetchall():
            _mail_event(cur,row["job_id"],school_id,"manual_retry_queued",
                actor_id=str(actor["id"]),attempt_count=row["attempt_count"])
        _audit(cur,school_id,str(actor["id"]),"family_mail_failed_jobs_requeued",str(count))
    result=dispatch_schoolwide_mail_once(school_id=school_id,limit=10) if count else {
        "attempted":0,"provider_accepted":0,"deferred_by_quota":0,"failed_or_retrying":0}
    return {"ok":True,"requeued":count,**result,"school_only":True}
