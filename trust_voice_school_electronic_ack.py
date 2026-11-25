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
from emailer import _send, FROM_ADDRESS, FROM_NAME, send_otp_email
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
_DEFAULT_DAILY = 40  # Resend account currently allows 100/day; protect auth OTP capacity.
_DEFAULT_GLOBAL = 55
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
        _ready = True


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
    cur.execute("SELECT pg_advisory_xact_lock(hashtext(%s))",
                ("school_family_mail_global",))
    cur.execute("""SELECT count(*) AS n FROM verifyd_safety_family_mail_ledger
         WHERE created_at>NOW()-INTERVAL '24 hours'""")
    if int(cur.fetchone()["n"]) >= _limit("VERIFYD_SAFETY_FAMILY_MAIL_GLOBAL_DAILY", _DEFAULT_GLOBAL, 10000):
        return None
    cur.execute("""SELECT count(*) AS n FROM verifyd_safety_family_mail_ledger
         WHERE school_id=%s AND created_at>NOW()-INTERVAL '24 hours'""",(school_id,))
    if int(cur.fetchone()["n"]) >= _limit("VERIFYD_SAFETY_FAMILY_MAIL_SCHOOL_DAILY",_DEFAULT_DAILY, 10000):
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
           AND created_at>NOW()-INTERVAL '24 hours'""",(school_id,purpose,fingerprint))
    if int(cur.fetchone()["n"]) >= (2 if purpose=="guardian_otp" else 1):
        return None
    id="sfmail_"+uuid.uuid4().hex
    cur.execute("""INSERT INTO verifyd_safety_family_mail_ledger
        (id,school_id,recipient_hash,purpose,campaign_id,student_refs,sent_by)
        VALUES(%s,%s,%s,%s,%s,%s,%s)""",
        (id,school_id,fingerprint,purpose,campaign_id,student_refs or [],sent_by))
    return id


def _delivery_result(id: str, accepted: bool):
    with get_db() as conn:
        conn.cursor().execute("""UPDATE verifyd_safety_family_mail_ledger
            SET accepted_by_provider=%s WHERE id=%s""",(bool(accepted),id))


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
        accepted=False
        try:
            otp=create_otp(email)
            accepted=bool(send_otp_email(email,otp))
        except Exception:
            accepted=False
        _delivery_result(reserved,accepted)
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
    """Admin-reviewed, quota bounded. NO student data in messages. No bulk BCC."""
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
        planned=[]
        campaign_id="sfmailbatch_"+uuid.uuid4().hex
        for email in emails:
            refs_for_email=sorted({str(r["local_ref"]) for r in records
                                   if str(r["guardian_email"]).strip().lower()==email})
            mail_id=_reserve_mail(cur,school_id,email,"schoolwide_invite",
                  campaign_id=campaign_id,student_refs=refs_for_email,
                  sent_by=str(actor["id"]))
            if mail_id:
                planned.append((email,mail_id))
    url="https://vfvid.com/Safety#school-family="+code
    accepted=failed=0
    for email,mail_id in planned:
        success=False
        try:
            success=bool(_send({"from":f"{FROM_NAME} <{FROM_ADDRESS}>",
                "to":[email],
                "subject":"VeriFYD Safety — School family registration",
                "html":("<p>"+html.escape(school_name)+" is inviting families to "
                        "review their VeriFYD Safety registration.</p>"
                        "<p><a href='"+html.escape(url,quote=True)+"'>Open school family registration</a></p>"
                        "<p>Use your own email address to verify. No student details are contained in this email. "
                        "Ignore the invitation if you are not a guardian at this school.</p>")}))
        except Exception:
            success=False
        _delivery_result(mail_id,success)
        accepted+=int(success)
        failed+=int(not success)
    return {"ok":True,"unique_verified_recipients":len(emails),
            "campaign_id":campaign_id,"provider_accepted":accepted,"provider_failed":failed,
            "skipped_due_to_quota_or_recent_send":len(emails)-len(planned),
            "mail_service":"Resend","daily_school_limit":_limit("VERIFYD_SAFETY_FAMILY_MAIL_SCHOOL_DAILY",_DEFAULT_DAILY,10000),
            "guardian_access_granted":False,"schoolwide_link":url}
