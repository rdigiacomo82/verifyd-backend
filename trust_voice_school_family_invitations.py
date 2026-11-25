"""Verified family invitations for school-managed VeriFYD Safety identities.

A Principal invites a specific parent email to a specific roster student. An
invitation code is NOT a student credential: child details and the preassigned
handle are revealed only after that email receives and passes an OTP. The adult
self-acknowledges the policy, then enters a Principal's final-review queue.
A proposed Backup gets a DIFFERENT invitation and must verify and acknowledge
independently. Nothing here authorizes ordinary student login, live 911, or SMS.
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
from emailer import _send, FROM_NAME, FROM_ADDRESS, send_otp_email
from trust_voice_identity import _ensure_identity_schema, _recent_otp_request
from trust_voice_school_safety_training import _actor, _audit, _membership, _schema as _training_schema
from trust_voice_school_safety_registration import _school_admin
from trust_voice_school_managed_roster import _schema as _roster_schema, _require_enrollment_enabled
from trust_voice_school_safety_consents import POLICY_VERSION, _consent_schema

router = APIRouter(prefix="/trust-voice/school-safety", tags=["School Family Invitations"])
_ready = False
_lock = threading.Lock()
_MAX_SEND = 30
_DAILY_SEND = 300


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _enabled(school_id: str):
    _require_enrollment_enabled(school_id)
    if os.getenv("VERIFYD_SAFETY_FAMILY_INVITES_ENABLED", "0").strip().lower() not in ("1", "true", "yes", "on"):
        raise HTTPException(404, detail={"error": "family_invitations_not_enabled_for_school"})
    schools = {part.strip() for part in os.getenv(
        "VERIFYD_SAFETY_FAMILY_INVITE_SCHOOL_IDS", "").split(",") if part.strip()}
    if school_id not in schools:
        raise HTTPException(404, detail={"error": "family_invitations_not_enabled_for_school"})


def _schema():
    global _ready
    _ensure_identity_schema()
    _roster_schema()
    _training_schema()
    _consent_schema()
    if _ready:
        return
    with _lock:
        if _ready:
            return
        with get_db() as conn:
            cur = conn.cursor()
            cur.execute("""CREATE TABLE IF NOT EXISTS verifyd_safety_family_invitations (
                invite_id TEXT PRIMARY KEY,
                school_id TEXT NOT NULL REFERENCES verifyd_safety_training_schools(school_id),
                local_ref TEXT NOT NULL,
                student_id TEXT NOT NULL REFERENCES verifyd_identities(id),
                slot INTEGER NOT NULL CHECK(slot IN (1,2)),
                recipient_email TEXT NOT NULL,
                guardian_name TEXT NOT NULL,
                relationship TEXT NOT NULL
                  CHECK(relationship IN ('parent','legal_guardian','authorized_caregiver')),
                token_hash TEXT UNIQUE NOT NULL,
                created_by TEXT NOT NULL REFERENCES verifyd_identities(id),
                delivery_status TEXT NOT NULL DEFAULT 'queued',
                otp_requests INTEGER NOT NULL DEFAULT 0,
                verification_attempts INTEGER NOT NULL DEFAULT 0,
                claimed_by TEXT REFERENCES verifyd_identities(id),
                acknowledged_policy TEXT,
                claimed_at TIMESTAMPTZ,
                issued_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                expires_at TIMESTAMPTZ NOT NULL DEFAULT NOW()+INTERVAL '7 days',
                revoked_at TIMESTAMPTZ,
                reviewed_at TIMESTAMPTZ,
                reviewed_by TEXT REFERENCES verifyd_identities(id),
                approval_status TEXT NOT NULL DEFAULT 'awaiting_claim'
                  CHECK(approval_status IN ('awaiting_claim','pending_principal','approved','rejected')),
                review_note TEXT NOT NULL DEFAULT '',
                FOREIGN KEY(school_id,local_ref)
                  REFERENCES verifyd_safety_managed_students(school_id,local_ref)
            )""")
            cur.execute("""ALTER TABLE verifyd_safety_family_invitations
                ADD COLUMN IF NOT EXISTS send_attempts INTEGER NOT NULL DEFAULT 0""")
            cur.execute("""ALTER TABLE verifyd_safety_family_invitations
                ADD COLUMN IF NOT EXISTS last_sent_at TIMESTAMPTZ""")
            cur.execute("""CREATE TABLE IF NOT EXISTS verifyd_safety_family_contact_drafts (
                school_id TEXT NOT NULL,
                local_ref TEXT NOT NULL,
                slot INTEGER NOT NULL CHECK(slot IN (1,2)),
                guardian_name TEXT NOT NULL,
                guardian_email TEXT NOT NULL,
                relationship TEXT NOT NULL CHECK(relationship IN
                    ('parent','legal_guardian','authorized_caregiver')),
                verified_by TEXT NOT NULL REFERENCES verifyd_identities(id),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                PRIMARY KEY(school_id,local_ref,slot),
                FOREIGN KEY(school_id,local_ref) REFERENCES
                    verifyd_safety_managed_students(school_id,local_ref)
            )""")
            cur.execute("""CREATE INDEX IF NOT EXISTS safety_family_invite_school_status
                 ON verifyd_safety_family_invitations(school_id,approval_status,issued_at DESC)""")
            cur.execute("""CREATE TABLE IF NOT EXISTS verifyd_safety_family_invite_claim_sessions (
                session_hash TEXT PRIMARY KEY,
                invite_id TEXT NOT NULL REFERENCES verifyd_safety_family_invitations(invite_id),
                expires_at TIMESTAMPTZ NOT NULL DEFAULT NOW()+INTERVAL '15 minutes',
                consumed_at TIMESTAMPTZ
            )""")
        _ready = True


class InvitationTarget(BaseModel):
    local_ref: str = Field(min_length=3, max_length=64)
    slot: Literal[1,2] = 1
    guardian_name: str = Field(min_length=3, max_length=100)
    guardian_email: str = Field(min_length=5, max_length=254)
    relationship: Literal["parent","legal_guardian","authorized_caregiver"] = "parent"


class InvitationBatch(BaseModel):
    guardians: list[InvitationTarget] = Field(min_length=1, max_length=_MAX_SEND)
    principal_verified_contact_addresses: Literal[True]
    send_individual_invites: Literal[True]


class SavedFamilyContacts(BaseModel):
    contacts: list[InvitationTarget] = Field(min_length=1, max_length=250)
    school_verified_emails_and_relationships: Literal[True]


class SendSavedInvitations(BaseModel):
    local_refs: list[str] = Field(min_length=1, max_length=15)
    principal_confirms_recipient_list: Literal[True]


class ResendInvitation(BaseModel):
    principal_confirms_address: Literal[True]


class InvitationCode(BaseModel):
    invitation_code: str = Field(min_length=32, max_length=128)


class VerifyInvitation(InvitationCode):
    email_code: str = Field(min_length=6, max_length=12)


class ConfirmFamily(BaseModel):
    claim_session: str = Field(min_length=32, max_length=180)
    relationship_confirmed: Literal[True]
    acknowledge_school_safety_policy: Literal[True]
    request_backup_invitation: bool = False
    backup_name: Optional[str] = Field(default=None, max_length=100)
    backup_email: Optional[str] = Field(default=None, max_length=254)


class FinalBatchApproval(BaseModel):
    invite_ids: list[str] = Field(min_length=1, max_length=250)
    independently_verified_family_relationships: Literal[True]
    review_note: str = Field(min_length=15, max_length=700)


def _invite_locked(cur, code: str):
    cur.execute("""SELECT i.*,s.school_name,r.first_name,r.last_name,
          r.proposed_handle,r.enrollment_status
       FROM verifyd_safety_family_invitations i
       JOIN verifyd_safety_training_schools s ON s.school_id=i.school_id
       JOIN verifyd_safety_managed_students r
         ON r.school_id=i.school_id AND r.local_ref=i.local_ref
       WHERE i.token_hash=%s FOR UPDATE OF i""", (_hash(code.strip()),))
    row = cur.fetchone()
    if not row:
        raise HTTPException(403, detail={"error":"invalid_or_expired_invitation"})
    item = dict(row)
    _enabled(item["school_id"])
    if (item["revoked_at"] or item["claimed_at"]
        or item["expires_at"] <= datetime.now(timezone.utc)
        or item["enrollment_status"] != "school_approved"):
        raise HTTPException(403, detail={"error":"invalid_or_expired_invitation"})
    return item


def _mask(email: str):
    local, _, domain = email.partition("@")
    return local[:1] + "***@" + domain


def _send_invitation(email: str, school_name: str, code: str) -> bool:
    # URL fragment stays client-side. No student names/handles or credentials
    # are included in the email subject/body.
    url = "https://vfvid.com/Safety#family-invite=" + code
    message = (
        "A school has invited you to register for VeriFYD Safety. "
        "Use the secure link and verify your email address before any "
        "student information is displayed. This invitation expires in 7 days. "
        "Never forward your invitation. If unexpected, contact the school."
    )
    return bool(_send({
        "from": f"{FROM_NAME} <{FROM_ADDRESS}>",
        "to": [email],
        "subject": "VeriFYD Safety — Verify your school family invitation",
        "html": "<p>" + html.escape(message) + "</p><p><a href=\""
            + html.escape(url, quote=True)
            + "\">Verify my school invitation</a></p>",
    }))


@router.post("/schools/{school_id}/family-invitations/send")
def send_invitation_batch(school_id: str, payload: InvitationBatch,
                          authorization: Optional[str] = Header(default=None)):
    _enabled(school_id)
    actor = _actor(authorization)
    _schema()
    prepared, seen = [], set()
    for child in payload.guardians:
        ref = child.local_ref.strip().upper()
        mail = child.guardian_email.strip().lower()
        if not is_valid_email(mail):
            raise HTTPException(422, detail={"error":"invalid_guardian_email"})
        if (ref,child.slot) in seen:
            raise HTTPException(422, detail={"error":"duplicate_student_guardian_slot"})
        seen.add((ref,child.slot))
        prepared.append((ref,child.slot,child.guardian_name.strip(),mail,child.relationship))
    # A single adult may not be invited as both Primary and Backup for one
    # child, even when both slots appear in the SAME bulk submission.
    grouped={}
    for ref,slot,_,mail,_ in prepared:
        by_slot=grouped.setdefault(ref,{})
        if mail in by_slot.values():
            raise HTTPException(422,detail={"error":"primary_and_backup_need_distinct_emails"})
        by_slot[slot]=mail
    sends = []
    with get_db() as conn:
        cur = conn.cursor()
        _school_admin(cur, school_id, str(actor["id"]))
        cur.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", ("family_invites:"+school_id,))
        cur.execute("""SELECT count(*) AS n FROM verifyd_safety_family_invitations
           WHERE school_id=%s AND issued_at>NOW()-INTERVAL '24 hours'""", (school_id,))
        if int(cur.fetchone()["n"]) + len(prepared) > _DAILY_SEND:
            raise HTTPException(429, detail={"error":"school_daily_invite_limit"})
        refs = sorted({row[0] for row in prepared})
        cur.execute("""SELECT local_ref,student_identity_id FROM verifyd_safety_managed_students
          WHERE school_id=%s AND local_ref=ANY(%s)
            AND enrollment_status='school_approved' AND student_identity_id IS NOT NULL""",
          (school_id,refs))
        approved = {str(row["local_ref"]):str(row["student_identity_id"]) for row in cur.fetchall()}
        if len(approved) != len(refs):
            raise HTTPException(409, detail={"error":"student_must_be_school_approved_before_invitation"})
        cur.execute("SELECT school_name FROM verifyd_safety_training_schools WHERE school_id=%s", (school_id,))
        school = cur.fetchone()
        if not school:
            raise HTTPException(404, detail={"error":"school_not_found"})
        for ref,slot,name,email,relationship in prepared:
            cur.execute("""SELECT slot,recipient_email,approval_status FROM verifyd_safety_family_invitations
                WHERE school_id=%s AND local_ref=%s AND revoked_at IS NULL
                  AND (claimed_at IS NOT NULL OR expires_at>NOW())
                ORDER BY issued_at DESC""", (school_id,ref))
            existing = [dict(row) for row in cur.fetchall()]
            if any(x["slot"]==slot for x in existing):
                raise HTTPException(409, detail={"error":"guardian_slot_already_assigned_review_before_resend"})
            if any(x["recipient_email"]==email for x in existing):
                raise HTTPException(409, detail={"error":"same_guardian_cannot_claim_both_slots"})
        for ref,slot,name,email,relationship in prepared:
            token = secrets.token_urlsafe(32)
            invite_id = "ssinv_"+uuid.uuid4().hex
            cur.execute("""INSERT INTO verifyd_safety_family_invitations
                 (invite_id,school_id,local_ref,student_id,slot,recipient_email,
                 guardian_name,relationship,token_hash,created_by)
                 VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                 (invite_id,school_id,ref,approved[ref],slot,email,name,
                  relationship,_hash(token),str(actor["id"])))
            sends.append((invite_id,email,school["school_name"],token))
        _audit(cur,school_id,str(actor["id"]),"school_family_invitations_prepared",str(len(sends)))
    results = []
    for invite_id,email,school_name,token in sends:
        sent = _send_invitation(email,school_name,token)
        with get_db() as conn:
            conn.cursor().execute("""UPDATE verifyd_safety_family_invitations
                SET delivery_status=%s,last_sent_at=NOW(),send_attempts=send_attempts+1
                WHERE invite_id=%s""",
                ("sent" if sent else "failed",invite_id))
        results.append({"invitation_id":invite_id,"recipient":_mask(email),"email_accepted":sent})
    return {"ok":True,"invitations":results,"individual_email_only":True,
            "sms_sent":False,"guardian_access_granted":False}


@router.post("/family-invitations/request-code")
def request_invitation_email_otp(payload: InvitationCode):
    _schema()
    with get_db() as conn:
        cur = conn.cursor()
        invite = _invite_locked(cur,payload.invitation_code)
        if invite["otp_requests"] >= 6:
            raise HTTPException(429,detail={"error":"invite_verification_limit"})
        cur.execute("""UPDATE verifyd_safety_family_invitations
             SET otp_requests=otp_requests+1 WHERE invite_id=%s""", (invite["invite_id"],))
        email = invite["recipient_email"]
    if _recent_otp_request(email):
        raise HTTPException(429, detail={"error":"verification_code_recently_sent"})
    otp = create_otp(email)
    if not send_otp_email(email, otp):
        raise HTTPException(503, detail={"error":"verification_email_not_sent"})
    return {"ok":True,"verification_code_sent":True,"to":_mask(email),
            "expires_minutes":10,"student_information_disclosed":False}


@router.post("/family-invitations/verify")
def verify_family_invitation(payload: VerifyInvitation):
    _schema()
    with get_db() as conn:
        cur = conn.cursor()
        invite = _invite_locked(cur,payload.invitation_code)
        if invite["verification_attempts"] >= 5:
            raise HTTPException(429, detail={"error":"invitation_verification_locked_contact_school"})
        email = invite["recipient_email"]
        invitation_id = invite["invite_id"]
    passed, _ = verify_otp(email,payload.email_code)
    if not passed:
        # Commit the invalid-attempt count; do not roll it back with HTTP 403.
        with get_db() as conn:
            conn.cursor().execute("""UPDATE verifyd_safety_family_invitations
                SET verification_attempts=verification_attempts+1
                WHERE invite_id=%s AND claimed_at IS NULL""",(invitation_id,))
        raise HTTPException(403,detail={"error":"email_verification_failed"})
    session = secrets.token_urlsafe(32)
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("""SELECT i.*,s.school_name,r.first_name,r.last_name,r.proposed_handle
           FROM verifyd_safety_family_invitations i
           JOIN verifyd_safety_training_schools s ON s.school_id=i.school_id
           JOIN verifyd_safety_managed_students r
            ON r.school_id=i.school_id AND r.local_ref=i.local_ref
           WHERE i.invite_id=%s FOR UPDATE OF i""",(invitation_id,))
        row = cur.fetchone()
        if (not row or row["claimed_at"] or row["revoked_at"]
            or row["expires_at"]<=datetime.now(timezone.utc)
            or row["verification_attempts"]>=5):
            raise HTTPException(403,detail={"error":"invalid_or_expired_invitation"})
        cur.execute("""INSERT INTO verifyd_safety_family_invite_claim_sessions
            (session_hash,invite_id) VALUES (%s,%s)""",
            (_hash(session),invitation_id))
        details = dict(row)
    return {"ok":True,"claim_session":session,"expires_minutes":15,
        "school":details["school_name"],
        "student_name":details["first_name"]+" "+details["last_name"],
        "student_handle":"@"+details["proposed_handle"],
        "guardian_name":details["guardian_name"],"guardian_slot":details["slot"],
        "relationship":details["relationship"],"policy_version":POLICY_VERSION,
        "verified_email":_mask(details["recipient_email"]),
        "school_only":True}


def _new_guardian_identity(cur, email: str, name: str, school_id: str | None = None):
    # Two guardian invitations for siblings may be claimed concurrently.
    # Serialize by verified mailbox before creating a globally unique identity.
    cur.execute("SELECT pg_advisory_xact_lock(hashtext(%s))",
                ("school_guardian_identity:"+email,))
    cur.execute("""SELECT id,status FROM verifyd_identities
       WHERE email_lower=%s FOR UPDATE""", (email,))
    existing = cur.fetchone()
    if existing:
        if existing["status"]!="active":
            raise HTTPException(409,detail={"error":"guardian_account_requires_manual_review"})
        return str(existing["id"]),False
    guardian_id = "vfyd_usr_"+uuid.uuid4().hex
    # New guardian identities use the existing school's prefix, matching the
    # staged student pattern. Existing VeriFYD account handles remain intact.
    prefix="Guardian"
    if school_id:
        # Older TEST schools may not have ever opened the managed-roster
        # module. Do not fail family account creation for missing settings.
        cur.execute("SELECT to_regclass('verifyd_safety_managed_roster_settings') AS name")
        if cur.fetchone()["name"] is not None:
            cur.execute("""SELECT school_code FROM verifyd_safety_managed_roster_settings
                WHERE school_id=%s""",(school_id,))
            settings=cur.fetchone()
            if settings and settings["school_code"]:
                prefix=str(settings["school_code"]).upper()[:16]
    for attempt in range(8):
        handle = (prefix + "_G" + secrets.token_hex(5).upper())
        cur.execute("SELECT 1 FROM verifyd_identities WHERE handle_lower=%s", (handle.lower(),))
        if cur.fetchone():
            continue
        cur.execute("""INSERT INTO verifyd_identities
            (id,email_lower,handle,handle_lower,display_name,display_emoji,
             email_verified,identity_verified,organization_verified,verification_level,
             can_receive_calls,call_privacy,status,created_at,updated_at)
            VALUES(%s,%s,%s,%s,%s,'',TRUE,FALSE,FALSE,'email_verified',
                FALSE,'nobody','active',NOW()::text,NOW()::text)""",
            (guardian_id,email,handle,handle.lower(),name[:80]))
        return guardian_id,True
    raise HTTPException(409,detail={"error":"guardian_handle_allocation_failed"})


@router.post("/family-invitations/confirm")
def confirm_family_claim(payload: ConfirmFamily):
    _schema()
    optional_backup = None
    if payload.request_backup_invitation:
        if not payload.backup_name or len(payload.backup_name.strip())<3 or not payload.backup_email:
            raise HTTPException(422,detail={"error":"backup_name_and_email_required"})
        if not is_valid_email(payload.backup_email.strip().lower()):
            raise HTTPException(422,detail={"error":"invalid_backup_email"})
        optional_backup = (payload.backup_name.strip(),payload.backup_email.strip().lower())
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("""SELECT s.expires_at AS session_expires_at,i.*
          FROM verifyd_safety_family_invite_claim_sessions s
          JOIN verifyd_safety_family_invitations i ON i.invite_id=s.invite_id
          WHERE s.session_hash=%s AND s.consumed_at IS NULL FOR UPDATE OF s,i""",
          (_hash(payload.claim_session),))
        invite = cur.fetchone()
        if (not invite or invite["session_expires_at"]<=datetime.now(timezone.utc)
            or invite["expires_at"]<=datetime.now(timezone.utc)
            or invite["claimed_at"] or invite["revoked_at"]):
            raise HTTPException(403,detail={"error":"family_claim_session_invalid_or_expired"})
        _enabled(invite["school_id"])
        email = invite["recipient_email"]
        if optional_backup and optional_backup[1]==email:
            raise HTTPException(422,detail={"error":"backup_guardian_requires_different_email"})
        gid, created = _new_guardian_identity(cur,email,invite["guardian_name"],invite["school_id"])
        _membership(cur,invite["school_id"],invite["student_id"],{"student"})
        cur.execute("""SELECT role FROM verifyd_safety_training_members
            WHERE school_id=%s AND identity_id=%s""",(invite["school_id"],gid))
        role = cur.fetchone()
        if role and role["role"]!="guardian":
            raise HTTPException(403,detail={"error":"cannot_claim_family_as_school_staff_or_student"})
        cur.execute("""SELECT 1 FROM verifyd_safety_training_guardians
           WHERE school_id=%s AND child_id=%s AND guardian_id=%s""",
           (invite["school_id"],invite["student_id"],gid))
        if cur.fetchone():
            raise HTTPException(409,detail={"error":"existing_guardian_link_use_family_portal"})
        cur.execute("""UPDATE verifyd_safety_family_invitations
            SET claimed_at=NOW(),claimed_by=%s,acknowledged_policy=%s,
                approval_status='pending_principal'
            WHERE invite_id=%s""",(gid,POLICY_VERSION,invite["invite_id"]))
        cur.execute("""UPDATE verifyd_safety_family_invite_claim_sessions
            SET consumed_at=NOW() WHERE session_hash=%s""",(_hash(payload.claim_session),))
        _audit(cur,invite["school_id"],gid,"guardian_self_acknowledged_school_safety_invitation",
               invite["invite_id"])
        new_backup = None
        if optional_backup and invite["slot"]==1:
            cur.execute("""SELECT 1 FROM verifyd_safety_family_invitations
               WHERE school_id=%s AND local_ref=%s AND slot=2
                 AND revoked_at IS NULL AND (claimed_at IS NOT NULL OR expires_at>NOW())
               LIMIT 1""",(invite["school_id"],invite["local_ref"]))
            if not cur.fetchone():
                backup_token = secrets.token_urlsafe(32)
                backup_id = "ssinv_"+uuid.uuid4().hex
                cur.execute("""INSERT INTO verifyd_safety_family_invitations
                    (invite_id,school_id,local_ref,student_id,slot,recipient_email,
                     guardian_name,relationship,token_hash,created_by)
                    VALUES(%s,%s,%s,%s,2,%s,%s,'parent',%s,%s)""",
                    (backup_id,invite["school_id"],invite["local_ref"],
                     invite["student_id"],optional_backup[1],optional_backup[0],
                     _hash(backup_token),gid))
                new_backup = (backup_id,optional_backup[1],backup_token)
                _audit(cur,invite["school_id"],gid,"second_guardian_invitation_requested",
                       backup_id)
        school_id = invite["school_id"]
    backup_sent = None
    if new_backup:
        backup_id, backup_email, backup_code = new_backup
        backup_sent = _send_invitation(backup_email,"Your school",backup_code)
        with get_db() as conn:
            conn.cursor().execute("""UPDATE verifyd_safety_family_invitations
                SET delivery_status=%s WHERE invite_id=%s""",
                ("sent" if backup_sent else "failed",backup_id))
    return {"ok":True,"school_id":school_id,"status":"pending_final_school_authorization",
            "guardian_account_created":created,"guardian_account_handle_auto_generated":created,
            "guardian_acknowledgment_recorded":True,
            "guardian_family_access_granted":False,
            "backup_invitation_sent":backup_sent,"backup_requires_own_verification":True,
            "student_account_handle_was_preassigned":True}


@router.get("/schools/{school_id}/family-invitations")
def school_family_invitation_queue(school_id: str, authorization: Optional[str] = Header(default=None)):
    _enabled(school_id)
    actor = _actor(authorization)
    _schema()
    with get_db() as conn:
        cur = conn.cursor()
        _school_admin(cur,school_id,str(actor["id"]))
        cur.execute("""SELECT i.invite_id,i.local_ref,i.slot,i.guardian_name,i.relationship,
            i.recipient_email,i.delivery_status,i.issued_at,i.claimed_at,
            i.approval_status,i.acknowledged_policy,r.first_name,r.last_name,
            r.proposed_handle
          FROM verifyd_safety_family_invitations i
          JOIN verifyd_safety_managed_students r
          ON r.school_id=i.school_id AND r.local_ref=i.local_ref
          WHERE i.school_id=%s ORDER BY i.issued_at DESC LIMIT 1000""", (school_id,))
        rows = [dict(row) for row in cur.fetchall()]
    for row in rows:
        row["recipient_email"]=_mask(row["recipient_email"])
    return {"ok":True,"invitations":rows,"school_only":True,
            "principal_final_approval_required":True}


@router.post("/schools/{school_id}/family-invitations/approve-batch")
def approve_family_invitation_batch(school_id: str, payload: FinalBatchApproval,
                                     authorization: Optional[str]=Header(default=None)):
    _enabled(school_id)
    actor = _actor(authorization)
    ids = list(payload.invite_ids)
    if len(set(ids))!=len(ids):
        raise HTTPException(422,detail={"error":"duplicate_family_invitation_id"})
    _schema()
    with get_db() as conn:
        cur=conn.cursor()
        _school_admin(cur,school_id,str(actor["id"]))
        cur.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", ("school_family_final_approval:"+school_id,))
        cur.execute("""SELECT * FROM verifyd_safety_family_invitations
           WHERE school_id=%s AND invite_id=ANY(%s) ORDER BY invite_id FOR UPDATE""",
           (school_id,ids))
        claims={row["invite_id"]:dict(row) for row in cur.fetchall()}
        if len(claims)!=len(ids):
            raise HTTPException(404,detail={"error":"one_or_more_invites_missing"})
        for invite_id in ids:
            i=claims[invite_id]
            if (i["approval_status"]!="pending_principal" or not i["claimed_by"]
                or i["acknowledged_policy"]!=POLICY_VERSION or i["revoked_at"]):
                raise HTTPException(409,detail={"error":"only_guardian_verified_claims_can_be_approved"})
            if str(i["claimed_by"])==str(actor["id"]):
                raise HTTPException(403,detail={"error":"cannot_approve_own_family_relationship"})
            _membership(cur,school_id,str(i["student_id"]),{"student"})
            cur.execute("""SELECT role FROM verifyd_safety_training_members
                WHERE school_id=%s AND identity_id=%s""",
                (school_id,str(i["claimed_by"])))
            role=cur.fetchone()
            if role and role["role"]!="guardian":
                raise HTTPException(403,detail={"error":"school_staff_cannot_claim_guardian_role"})
        for invite_id in ids:
            i=claims[invite_id]
            gid=str(i["claimed_by"])
            sid=str(i["student_id"])
            cur.execute("""INSERT INTO verifyd_safety_training_members
                (school_id,identity_id,role,verified_by)
                VALUES(%s,%s,'guardian',%s)
                ON CONFLICT DO NOTHING""", (school_id,gid,str(actor["id"])))
            cur.execute("""INSERT INTO verifyd_safety_training_guardians
                (school_id,child_id,guardian_id,verified_by)
                VALUES(%s,%s,%s,%s) ON CONFLICT DO NOTHING""",
                (school_id,sid,gid,str(actor["id"])))
            # The guardian explicitly acknowledged when claiming the invitation.
            # The Principal finalizes authorization; no forged guardian signature.
            cur.execute("""INSERT INTO verifyd_safety_training_signoffs
                (signoff_id,school_id,actor_id,kind,subject_id,policy_version,accepted_at)
                VALUES(%s,%s,%s,'guardian',%s,%s,%s)
                ON CONFLICT(school_id,actor_id,kind,subject_id)
                DO UPDATE SET policy_version=EXCLUDED.policy_version,
                   accepted_at=EXCLUDED.accepted_at,revoked_at=NULL""",
                ("sssig_"+uuid.uuid4().hex,school_id,gid,sid,POLICY_VERSION,i["claimed_at"]))
            cur.execute("""UPDATE verifyd_safety_family_invitations
                 SET approval_status='approved',reviewed_at=NOW(),
                     reviewed_by=%s,review_note=%s
                 WHERE invite_id=%s""",
                (str(actor["id"]),payload.review_note.strip(),invite_id))
            _audit(cur,school_id,str(actor["id"]),"guardian_invitation_final_approved",invite_id)
        _audit(cur,school_id,str(actor["id"]),
               "family_group_final_authorization",str(len(ids)))
    return {"ok":True,"approved":len(ids),"family_access_authorized":True,
            "live_emergency_dispatch_enabled":False,"school_only":True}


@router.post("/schools/{school_id}/family-invitations/{invite_id}/revoke")
def revoke_school_family_invitation(school_id: str,invite_id: str,
                                    authorization: Optional[str]=Header(default=None)):
    _enabled(school_id)
    actor=_actor(authorization)
    _schema()
    with get_db() as conn:
        cur=conn.cursor()
        _school_admin(cur,school_id,str(actor["id"]))
        cur.execute("""SELECT approval_status FROM verifyd_safety_family_invitations
            WHERE school_id=%s AND invite_id=%s FOR UPDATE""",(school_id,invite_id))
        row=cur.fetchone()
        if not row:
            raise HTTPException(404,detail={"error":"invitation_not_found"})
        if row["approval_status"]=="approved":
            raise HTTPException(409,detail={"error":"approved_family_use_guardian_deauthorization"})
        cur.execute("""UPDATE verifyd_safety_family_invitations
            SET revoked_at=NOW(),approval_status='rejected',
                reviewed_by=%s,reviewed_at=NOW()
            WHERE invite_id=%s""",(str(actor["id"]),invite_id))
        _audit(cur,school_id,str(actor["id"]),"family_invitation_revoked",invite_id)
    return {"ok":True,"revoked":True,"guardian_access_changed":False}



@router.get("/schools/{school_id}/family-invitations/contact-drafts")
def get_school_family_contact_drafts(school_id: str, authorization: Optional[str]=Header(default=None)):
    """School-admin-only record of guardian invitations ready for sending."""
    _enabled(school_id)
    actor=_actor(authorization)
    _schema()
    with get_db() as conn:
        cur=conn.cursor()
        _school_admin(cur,school_id,str(actor["id"]))
        cur.execute("""SELECT local_ref,slot,guardian_name,guardian_email,relationship,updated_at
            FROM verifyd_safety_family_contact_drafts WHERE school_id=%s
            ORDER BY local_ref,slot LIMIT 6000""",(school_id,))
        contacts=[dict(x) for x in cur.fetchall()]
    return {"ok":True,"contacts":contacts,"school_only":True}


@router.put("/schools/{school_id}/family-invitations/contact-drafts")
def save_school_family_contact_drafts(school_id: str,payload: SavedFamilyContacts,
                                      authorization: Optional[str]=Header(default=None)):
    """Persist staff-verified invite recipients without sending an email.

    Never assume parental authority just because email exists in a school CSV.
    """
    _enabled(school_id)
    actor=_actor(authorization)
    seen=set()
    cleaned=[]
    for item in payload.contacts:
        ref=item.local_ref.strip().upper()
        email=item.guardian_email.strip().lower()
        name=item.guardian_name.strip()
        if not is_valid_email(email):
            raise HTTPException(422,detail={"error":"invalid_guardian_email"})
        if (ref,item.slot) in seen:
            raise HTTPException(422,detail={"error":"duplicate_guardian_slot_in_contact_drafts"})
        seen.add((ref,item.slot))
        cleaned.append((ref,item.slot,name,email,item.relationship))
    _schema()
    with get_db() as conn:
        cur=conn.cursor()
        _school_admin(cur,school_id,str(actor["id"]))
        cur.execute("SELECT pg_advisory_xact_lock(hashtext(%s))",
                    ("family_contact_drafts:"+school_id,))
        refs=sorted({x[0] for x in cleaned})
        cur.execute("""SELECT local_ref FROM verifyd_safety_managed_students
            WHERE school_id=%s AND local_ref=ANY(%s)""",(school_id,refs))
        matches={str(x["local_ref"]) for x in cur.fetchall()}
        if matches!=set(refs):
            raise HTTPException(404,detail={"error":"guardian_contact_student_not_in_school_roster"})
        for ref,slot,name,email,relationship in cleaned:
            cur.execute("""SELECT 1 FROM verifyd_safety_family_invitations
                WHERE school_id=%s AND local_ref=%s AND slot=%s
                  AND revoked_at IS NULL AND (claimed_at IS NOT NULL OR expires_at>NOW())
                LIMIT 1""",(school_id,ref,slot))
            if cur.fetchone():
                raise HTTPException(409,detail={"error":"revoke_prior_invitation_before_changing_guardian_contact"})
        cur.execute("""SELECT local_ref,slot,guardian_email
            FROM verifyd_safety_family_contact_drafts
            WHERE school_id=%s AND local_ref=ANY(%s) FOR UPDATE""",
            (school_id,refs))
        group={}
        for saved in cur.fetchall():
            group.setdefault(str(saved["local_ref"]),{})[int(saved["slot"])]=saved["guardian_email"]
        for ref,slot,name,email,relationship in cleaned:
            group.setdefault(ref,{})[slot]=email
        for ref,slots in group.items():
            if 1 in slots and 2 in slots and slots[1]==slots[2]:
                raise HTTPException(422,detail={"error":"primary_and_backup_need_distinct_emails"})
        for ref,slot,name,email,relationship in cleaned:
            cur.execute("""INSERT INTO verifyd_safety_family_contact_drafts
                (school_id,local_ref,slot,guardian_name,guardian_email,relationship,verified_by)
                VALUES(%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT(school_id,local_ref,slot) DO UPDATE
                SET guardian_name=EXCLUDED.guardian_name,
                    guardian_email=EXCLUDED.guardian_email,
                    relationship=EXCLUDED.relationship,
                    verified_by=EXCLUDED.verified_by,updated_at=NOW()""",
                (school_id,ref,slot,name,email,relationship,str(actor["id"])))
        _audit(cur,school_id,str(actor["id"]),"family_invitation_contacts_saved",str(len(cleaned)))
    return {"ok":True,"contacts_saved":len(cleaned),"sent":0,
            "family_access_granted":False}


@router.post("/schools/{school_id}/family-invitations/send-saved")
def send_saved_family_invitations(school_id: str,payload: SendSavedInvitations,
                                  authorization: Optional[str]=Header(default=None)):
    """One action sends prepared individual emails after school approval.

    Repeated clicks never create duplicate pending invitations. The dedicated
    Resend action rotates the one-use secret after a reasonable interval.
    """
    _enabled(school_id)
    actor=_actor(authorization)
    refs=[x.strip().upper() for x in payload.local_refs]
    if len(refs)!=len(set(refs)):
        raise HTTPException(422,detail={"error":"duplicate_student_reference"})
    _schema()
    with get_db() as conn:
        cur=conn.cursor()
        _school_admin(cur,school_id,str(actor["id"]))
        cur.execute("""SELECT local_ref FROM verifyd_safety_managed_students
            WHERE school_id=%s AND local_ref=ANY(%s)
              AND enrollment_status='school_approved' AND student_identity_id IS NOT NULL""",
            (school_id,refs))
        found={str(r["local_ref"]) for r in cur.fetchall()}
        if found!=set(refs):
            raise HTTPException(409,detail={"error":"student_not_yet_approved_for_family_invitation"})
        cur.execute("""SELECT local_ref,slot,guardian_name,guardian_email,relationship
            FROM verifyd_safety_family_contact_drafts
            WHERE school_id=%s AND local_ref=ANY(%s)
            ORDER BY local_ref,slot""",(school_id,refs))
        contacts=[dict(x) for x in cur.fetchall()]
        cur.execute("""SELECT local_ref,slot,approval_status,revoked_at,expires_at,claimed_at
            FROM verifyd_safety_family_invitations
            WHERE school_id=%s AND local_ref=ANY(%s)
              AND revoked_at IS NULL AND (claimed_at IS NOT NULL OR expires_at>NOW())""",
            (school_id,refs))
        occupied={(str(x["local_ref"]),int(x["slot"])) for x in cur.fetchall()}
    prepared=[InvitationTarget(**x) for x in contacts
              if (str(x["local_ref"]),int(x["slot"])) not in occupied]
    skipped=len(contacts)-len(prepared)
    if not prepared:
        return {"ok":True,"sent":0,"existing_invitations":skipped,
                "missing_contact_records":len(refs)-len({r["local_ref"] for r in contacts}),
                "message":"No new invitations needed. Use Resend for an unclaimed email."}
    # Send in the existing size-bounded batches, with the usual school-admin
    # gate and contact list validation on every batch.
    deliveries=[]
    for start in range(0,len(prepared),_MAX_SEND):
        outcome=send_invitation_batch(school_id,InvitationBatch(
            guardians=prepared[start:start+_MAX_SEND],
            principal_verified_contact_addresses=True,
            send_individual_invites=True,
        ),authorization)
        deliveries.extend(outcome["invitations"])
    return {"ok":True,"sent":sum(1 for d in deliveries if d["email_accepted"]),
            "failed":sum(1 for d in deliveries if not d["email_accepted"]),
            "existing_invitations":skipped,
            "missing_contact_records":len(refs)-len({r["local_ref"] for r in contacts}),
            "invitations":deliveries,
            "guardian_access_granted":False}


@router.post("/schools/{school_id}/family-invitations/{invite_id}/resend")
def resend_school_family_invitation(school_id: str,invite_id: str,payload: ResendInvitation,
                                    authorization: Optional[str]=Header(default=None)):
    """Rotate an unused invitation code and send another private email.

    Never resend already claimed/approved/revoked invitations. Old codes are
    invalidated immediately; new code is never returned to the browser.
    """
    _enabled(school_id)
    actor=_actor(authorization)
    _schema()
    code=secrets.token_urlsafe(32)
    with get_db() as conn:
        cur=conn.cursor()
        _school_admin(cur,school_id,str(actor["id"]))
        cur.execute("SELECT pg_advisory_xact_lock(hashtext(%s))",("family_invites:"+school_id,))
        cur.execute("""SELECT i.recipient_email,i.approval_status,i.claimed_at,
                   i.revoked_at,i.send_attempts,i.last_sent_at,s.school_name
            FROM verifyd_safety_family_invitations i
            JOIN verifyd_safety_training_schools s ON s.school_id=i.school_id
            WHERE i.school_id=%s AND i.invite_id=%s FOR UPDATE OF i""",
            (school_id,invite_id))
        row=cur.fetchone()
        if not row:
            raise HTTPException(404,detail={"error":"invitation_not_found"})
        if row["approval_status"]!="awaiting_claim" or row["claimed_at"] or row["revoked_at"]:
            raise HTTPException(409,detail={"error":"only_unclaimed_invitations_can_be_resent"})
        if int(row["send_attempts"])>=5:
            raise HTTPException(429,detail={"error":"invitation_resend_limit_contact_school"})
        cur.execute("""UPDATE verifyd_safety_family_invitations
            SET token_hash=%s,expires_at=NOW()+INTERVAL '7 days',
                delivery_status='queued',
                otp_requests=0,verification_attempts=0,
                last_sent_at=NOW(),send_attempts=send_attempts+1
            WHERE school_id=%s AND invite_id=%s
              AND (last_sent_at IS NULL OR last_sent_at<NOW()-INTERVAL '2 minutes')
            RETURNING recipient_email""",
            (_hash(code),school_id,invite_id))
        changed=cur.fetchone()
        if not changed:
            raise HTTPException(429,detail={"error":"wait_two_minutes_before_resending_invitation"})
        school_name=row["school_name"]
        email=row["recipient_email"]
        _audit(cur,school_id,str(actor["id"]),"school_family_invitation_resent",invite_id)
    delivered=_send_invitation(email,school_name,code)
    with get_db() as conn:
        conn.cursor().execute("""UPDATE verifyd_safety_family_invitations
            SET delivery_status=%s WHERE school_id=%s AND invite_id=%s""",
            ("sent" if delivered else "failed",school_id,invite_id))
    return {"ok":True,"email_accepted":delivered,"recipient":_mask(email),
            "prior_invitation_link_revoked":True,"new_code_emailed_not_displayed":True,
            "guardian_access_granted":False}
