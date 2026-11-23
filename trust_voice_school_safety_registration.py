"""Owner-approved school applications and school-scoped roster administration.

This module is TRAINING/SETUP ONLY, never emergency dispatch. No request fields
can assign operator, principal, or backup roles without verified authorization.
It does not create, delete, or edit underlying VeriFYD identity accounts.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import uuid
from urllib import request as urlrequest
from typing import Literal, Optional

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field

from database import get_db, is_valid_email
from trust_voice_school_safety_training import (
    _actor, _audit, _id_for_handle, _membership, _operator, _operators, _schema,
)
from trust_voice_school_safety_consents import _consent_schema, _school_accepted

router = APIRouter(prefix="/trust-voice/school-safety", tags=["School Safety Owner Registration and Roster"])
_notification_log = logging.getLogger("verifyd.safety.registration_notification")
_ready = False
_lock = threading.Lock()


def _application_schema():
    global _ready
    _consent_schema()
    if _ready:
        return
    with _lock:
        if _ready:
            return
        with get_db() as conn:
            cur = conn.cursor()
            cur.execute("""CREATE TABLE IF NOT EXISTS verifyd_safety_school_applications (
                application_id TEXT PRIMARY KEY,
                applicant_id TEXT NOT NULL REFERENCES verifyd_identities(id),
                school_name TEXT NOT NULL,
                district_name TEXT NOT NULL,
                school_address TEXT NOT NULL,
                school_website TEXT NOT NULL DEFAULT '',
                official_email TEXT NOT NULL,
                principal_name TEXT NOT NULL,
                requested_backup_handle TEXT NOT NULL DEFAULT '',
                application_reason TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending' CHECK(status IN ('pending','approved','rejected')),
                reviewed_by TEXT REFERENCES verifyd_identities(id),
                review_note TEXT,
                school_id TEXT REFERENCES verifyd_safety_training_schools(school_id),
                submitted_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                reviewed_at TIMESTAMPTZ
            )""")
            cur.execute("""CREATE INDEX IF NOT EXISTS safety_application_queue
                ON verifyd_safety_school_applications(status,submitted_at DESC)""")
            cur.execute("""CREATE INDEX IF NOT EXISTS safety_application_applicant
                ON verifyd_safety_school_applications(applicant_id,submitted_at DESC)""")
            cur.execute("""CREATE TABLE IF NOT EXISTS verifyd_safety_school_admin_delegations (
                school_id TEXT NOT NULL REFERENCES verifyd_safety_training_schools(school_id) ON DELETE CASCADE,
                identity_id TEXT NOT NULL REFERENCES verifyd_identities(id),
                approved_by TEXT NOT NULL REFERENCES verifyd_identities(id),
                approved_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                accepted_at TIMESTAMPTZ,
                revoked_at TIMESTAMPTZ,
                PRIMARY KEY(school_id,identity_id)
            )""")
            cur.execute("""CREATE UNIQUE INDEX IF NOT EXISTS safety_one_active_backup_per_school
                ON verifyd_safety_school_admin_delegations(school_id)
                WHERE revoked_at IS NULL""")
            cur.execute("""CREATE TABLE IF NOT EXISTS verifyd_safety_school_application_notifications (
                application_id TEXT PRIMARY KEY REFERENCES verifyd_safety_school_applications(application_id),
                status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending','sending','sent','failed')),
                attempts INTEGER NOT NULL DEFAULT 0,
                last_attempt_at TIMESTAMPTZ,
                sent_at TIMESTAMPTZ,
                provider_message_id TEXT
            )""")
        _ready = True


def _owner(identity):
    # ID allowlist is a VeriFYD owner-controlled Render environment variable,
    # not a self-attested role or name/email supplied by the client.
    _operator(identity)


def _school_admin(cur, school_id: str, identity_id: str):
    role = _membership(cur, school_id, identity_id)
    if role == "principal":
        if not _school_accepted(cur, school_id):
            raise HTTPException(403, detail={"error": "school_principal_acknowledgment_required"})
        return "principal"
    if role != "responder":
        raise HTTPException(403, detail={"error": "school_roster_admin_required"})
    cur.execute("""SELECT 1 FROM verifyd_safety_school_admin_delegations
        WHERE school_id=%s AND identity_id=%s AND revoked_at IS NULL AND accepted_at IS NOT NULL""",
        (school_id, identity_id))
    if not cur.fetchone():
        raise HTTPException(403, detail={"error": "approved_backup_administrator_required"})
    if not _school_accepted(cur, school_id):
        raise HTTPException(403, detail={"error": "school_principal_acknowledgment_required"})
    return "backup"


class SchoolApplication(BaseModel):
    school_name: str = Field(min_length=3, max_length=120)
    district_name: str = Field(min_length=3, max_length=150)
    school_address: str = Field(min_length=10, max_length=240)
    official_email: str = Field(min_length=6, max_length=254)
    principal_name: str = Field(min_length=3, max_length=120)
    school_website: str = Field(default="", max_length=200)
    backup_handle: str = Field(default="", max_length=40)
    reason: str = Field(min_length=20, max_length=800)


class ReviewDecision(BaseModel):
    decision: Literal["approve", "reject"]
    verified_official_authority: bool = False
    internal_test_school: bool = False
    review_note: str = Field(min_length=12, max_length=1000)


def _approval_verified(payload: ReviewDecision, school_name: str) -> bool:
    """Explicit independent authority OR owner-controlled fictional test school."""
    return bool(
        payload.verified_official_authority is True
        or (payload.internal_test_school is True and
            str(school_name).strip().upper().endswith("TEST ONLY"))
    )


class BackupApproval(BaseModel):
    backup_handle: str = Field(min_length=3, max_length=40)
    verified_official_authority: Literal[True]


class RosterMember(BaseModel):
    handle: str = Field(min_length=3, max_length=40)
    role: Literal["student", "guardian", "responder"]


class RosterRoleEdit(BaseModel):
    role: Literal["student", "guardian", "responder"]



# A notification is informational only. A message or link never grants school access.
_OWNER_NOTIFICATION_ADDRESS = "support@vfvid.com"
_OWNER_NOTIFICATION_FROM = "VeriFYD Safety <support@vfvid.com>"
_OWNER_DASHBOARD_URL = "https://vfvid.com/Safety"


def _send_owner_application_notification(application_id: str) -> str:
    """Claim one durable notification; send with a stable provider idempotency key.

    Sending must not run inside the application-creation transaction. A failed
    provider request is recorded without losing a successfully saved application.
    """
    _application_schema()
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("""UPDATE verifyd_safety_school_application_notifications
            SET status='sending', attempts=attempts+1, last_attempt_at=NOW()
            WHERE application_id=%s
              AND (status IN ('pending','failed')
                  OR (status='sending'
                      AND last_attempt_at < NOW()-INTERVAL '10 minutes'))
            RETURNING application_id""", (application_id,))
        if not cur.fetchone():
            return "already_sent_or_in_progress"
        cur.execute("""SELECT a.application_id,a.school_name,a.district_name,
                     a.principal_name,a.submitted_at,i.handle AS applicant_handle
            FROM verifyd_safety_school_applications a
            JOIN verifyd_identities i ON i.id=a.applicant_id
            WHERE a.application_id=%s""", (application_id,))
        app = dict(cur.fetchone())

    # Intentional minimal information in email; detailed application is available
    # only after authenticated owner review.
    body = (
        "A VeriFYD Safety Principal application is pending your review.\\n\\n"
        "School: " + app["school_name"] + "\\n"
        "District: " + app["district_name"] + "\\n"
        "Requested Principal: " + app["principal_name"] + "\\n"
        "Applicant handle: @" + app["applicant_handle"] + "\\n"
        "Application reference: " + app["application_id"] + "\\n"
        "Submitted: " + str(app["submitted_at"]) + "\\n\\n"
        "Sign in to review: " + _OWNER_DASHBOARD_URL + "\\n\\n"
        "No school access has been granted. Email links cannot approve applications."
    )
    provider_id = None
    try:
        key = os.getenv("RESEND_API_KEY", "").strip()
        if not key:
            raise RuntimeError("resend_not_configured")
        payload = json.dumps({
            "from": _OWNER_NOTIFICATION_FROM,
            "to": [_OWNER_NOTIFICATION_ADDRESS],
            "subject": "VeriFYD Safety — New Principal Application Pending Approval",
            "text": body,
        }).encode("utf-8")
        req = urlrequest.Request(
            "https://api.resend.com/emails",
            data=payload, method="POST",
            headers={
                "Authorization": "Bearer " + key,
                "Content-Type": "application/json",
                "Idempotency-Key": "verifyd-safety-application-" + application_id,
            },
        )
        with urlrequest.urlopen(req, timeout=8) as resp:
            if not (200 <= resp.status < 300):
                raise RuntimeError("provider_rejected")
            provider_id = json.loads(resp.read(4096).decode("utf-8")).get("id")
        if not provider_id:
            raise RuntimeError("provider_response_missing_id")
    except Exception:
        # Do not log application details, credentials or provider error bodies.
        _notification_log.warning("school_application_owner_notification_failed")
        with get_db() as conn:
            cur = conn.cursor()
            cur.execute("""UPDATE verifyd_safety_school_application_notifications
                SET status='failed' WHERE application_id=%s AND status='sending'""",
                (application_id,))
        return "failed"

    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("""UPDATE verifyd_safety_school_application_notifications
            SET status='sent',sent_at=NOW(),provider_message_id=%s
            WHERE application_id=%s AND status='sending'""",
            (provider_id,application_id))
    return "accepted_by_provider"


@router.post("/owner/notifications/retry")
def retry_owner_notifications(authorization: Optional[str] = Header(default=None)):
    owner = _actor(authorization)
    _owner(owner)
    _application_schema()
    with get_db() as conn:
        cur = conn.cursor()
        # Includes pending applications from before the email feature was added.
        cur.execute("""INSERT INTO verifyd_safety_school_application_notifications (application_id)
            SELECT a.application_id FROM verifyd_safety_school_applications a
            WHERE a.status='pending'
            ON CONFLICT(application_id) DO NOTHING""")
        cur.execute("""SELECT n.application_id
            FROM verifyd_safety_school_application_notifications n
            JOIN verifyd_safety_school_applications a USING(application_id)
            WHERE a.status='pending' AND n.status IN ('pending','failed')
            ORDER BY a.submitted_at LIMIT 20""")
        ids = [row["application_id"] for row in cur.fetchall()]
    counts = {"accepted_by_provider": 0, "failed": 0, "already_sent_or_in_progress": 0}
    for application_id in ids:
        result = _send_owner_application_notification(application_id)
        counts[result] += 1
    return {"ok": True, "attempted": len(ids), "results": counts,
            "school_access_granted": False}


@router.post("/applications")
def submit_school_application(payload: SchoolApplication, authorization: Optional[str] = Header(default=None)):
    applicant = _actor(authorization)
    _application_schema()
    applicant_id = str(applicant["id"])
    school_name = payload.school_name.strip()
    if payload.school_website and not payload.school_website.strip().startswith("https://"):
        raise HTTPException(422, detail={"error": "https_school_website_required"})
    if not is_valid_email(payload.official_email.strip()):
        raise HTTPException(422, detail={"error": "valid_official_email_required"})
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("""SELECT COUNT(*) AS pending FROM verifyd_safety_school_applications
            WHERE applicant_id=%s AND status='pending'""", (applicant_id,))
        if int(cur.fetchone()["pending"]) >= 2:
            raise HTTPException(409, detail={"error": "existing_school_applications_pending"})
        cur.execute("""SELECT COUNT(*) AS recent FROM verifyd_safety_school_applications
           WHERE applicant_id=%s AND submitted_at>NOW()-interval '24 hours'""", (applicant_id,))
        if int(cur.fetchone()["recent"]) >= 3:
            raise HTTPException(429, detail={"error": "school_application_rate_limit"})
        application_id = "ssapp_" + uuid.uuid4().hex
        cur.execute("""INSERT INTO verifyd_safety_school_applications
           (application_id,applicant_id,school_name,district_name,school_address,
            school_website,official_email,principal_name,requested_backup_handle,application_reason)
           VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
           (application_id,applicant_id,school_name,payload.district_name.strip(),
            payload.school_address.strip(),payload.school_website.strip(),
            payload.official_email.strip().lower(),payload.principal_name.strip(),
            payload.backup_handle.strip().lstrip("@"),payload.reason.strip()))
        _audit(cur, "pending_application", applicant_id, "school_application_submitted", application_id)
        cur.execute("""INSERT INTO verifyd_safety_school_application_notifications (application_id)
            VALUES(%s) ON CONFLICT(application_id) DO NOTHING""", (application_id,))
    # Notification failures never grant access or roll back the saved application.
    try:
        _send_owner_application_notification(application_id)
    except Exception:
        _notification_log.warning("school_application_owner_notification_unavailable")
    return {"ok": True, "application_id": application_id, "status": "pending_owner_review",
            "school_access_granted": False, "emergency_alerts_active": False}


@router.get("/applications/mine")
def my_school_applications(authorization: Optional[str] = Header(default=None)):
    actor = _actor(authorization)
    _application_schema()
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("""SELECT application_id,school_name,district_name,status,submitted_at,reviewed_at,school_id
            FROM verifyd_safety_school_applications WHERE applicant_id=%s
            ORDER BY submitted_at DESC LIMIT 20""", (str(actor["id"]),))
        apps = [dict(r) for r in cur.fetchall()]
    return {"ok": True, "applications": apps, "emergency_alerts_active": False}


@router.get("/owner/applications")
def owner_school_applications(authorization: Optional[str] = Header(default=None)):
    owner = _actor(authorization)
    _owner(owner)
    _application_schema()
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("""SELECT a.application_id,a.applicant_id,i.handle AS applicant_handle,
            a.school_name,a.district_name,a.school_address,a.school_website,
            a.official_email,a.principal_name,a.requested_backup_handle,
            a.application_reason,a.status,a.submitted_at,a.reviewed_at,a.review_note,a.school_id
            FROM verifyd_safety_school_applications a JOIN verifyd_identities i ON i.id=a.applicant_id
            ORDER BY CASE a.status WHEN 'pending' THEN 0 ELSE 1 END,a.submitted_at DESC LIMIT 150""")
        apps = [dict(r) for r in cur.fetchall()]
    return {"ok": True, "applications": apps, "owner_approval_required": True}


@router.post("/owner/applications/{application_id}/review")
def review_school_application(application_id: str, payload: ReviewDecision,
                              authorization: Optional[str] = Header(default=None)):
    owner = _actor(authorization)
    _owner(owner)
    _application_schema()
    owner_id = str(owner["id"])
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("""SELECT * FROM verifyd_safety_school_applications
            WHERE application_id=%s FOR UPDATE""", (application_id,))
        app = cur.fetchone()
        if not app:
            raise HTTPException(404, detail={"error": "application_not_found"})
        if app["status"] != "pending":
            raise HTTPException(409, detail={"error": "application_already_reviewed"})
        if payload.decision == "approve":
            # For the explicitly test-only service, the operator may approve
            # a fictional school they control without falsely attesting to
            # independently verified real-world district authority.
            if not _approval_verified(payload, app["school_name"]):
                raise HTTPException(422, detail={"error": "independent_school_authority_verification_or_internal_test_required"})
        school_id = None
        if payload.decision == "approve":
            if str(app["applicant_id"]) == owner_id:
                # Explicit operator-created test schools use /schools; avoid
                # an approval record that pretends to be independent verification.
                raise HTTPException(409, detail={"error": "owner_cannot_self_certify_school_application"})
            school_id = "sschool_" + uuid.uuid4().hex
            cur.execute("""INSERT INTO verifyd_safety_training_schools
               (school_id,school_name,created_by) VALUES(%s,%s,%s)""",
               (school_id,app["school_name"],owner_id))
            cur.execute("""INSERT INTO verifyd_safety_training_members
               (school_id,identity_id,role,verified_by) VALUES(%s,%s,'principal',%s)""",
               (school_id,str(app["applicant_id"]),owner_id))
            _audit(cur,school_id,owner_id,"principal_application_approved",str(app["applicant_id"]))
        cur.execute("""UPDATE verifyd_safety_school_applications SET status=%s,
          reviewed_by=%s,review_note=%s,school_id=%s,reviewed_at=NOW() WHERE application_id=%s""",
          ("approved" if payload.decision=="approve" else "rejected",
           owner_id,payload.review_note.strip(),school_id,application_id))
        _audit(cur,school_id or "application_review",owner_id,
               "school_application_"+payload.decision,application_id)
    return {"ok": True, "status": payload.decision, "school_id": school_id,
            "emergency_alerts_active": False, "backup_admin_approved": False}


@router.post("/owner/schools/{school_id}/backup")
def owner_approve_backup(school_id: str, payload: BackupApproval,
                         authorization: Optional[str] = Header(default=None)):
    owner = _actor(authorization)
    _owner(owner)
    _application_schema()
    owner_id = str(owner["id"])
    candidate_id = _id_for_handle(payload.backup_handle)
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("""SELECT identity_id FROM verifyd_safety_training_members
          WHERE school_id=%s AND role='principal'""", (school_id,))
        principals = [str(row["identity_id"]) for row in cur.fetchall()]
        if not principals:
            raise HTTPException(404, detail={"error": "school_principal_not_found"})
        if candidate_id in principals or candidate_id == owner_id:
            raise HTTPException(422, detail={"error": "backup_must_be_distinct_authorized_person"})
        cur.execute("""SELECT role FROM verifyd_safety_training_members WHERE school_id=%s AND identity_id=%s""",
                    (school_id,candidate_id))
        existing = cur.fetchone()
        if existing and existing["role"] not in ("responder",):
            raise HTTPException(409, detail={"error": "backup_requires_separate_responder_account"})
        # Activate only after owner approval and the designated backup's own
        # acceptance in their authenticated session; previous backup revoked.
        cur.execute("""UPDATE verifyd_safety_school_admin_delegations
          SET revoked_at=NOW(), accepted_at=NULL
          WHERE school_id=%s AND revoked_at IS NULL""", (school_id,))
        if not existing:
            cur.execute("""INSERT INTO verifyd_safety_training_members
              (school_id,identity_id,role,verified_by)
              VALUES(%s,%s,'responder',%s)""", (school_id,candidate_id,owner_id))
        cur.execute("""INSERT INTO verifyd_safety_school_admin_delegations
          (school_id,identity_id,approved_by,approved_at,accepted_at,revoked_at)
          VALUES(%s,%s,%s,NOW(),NULL,NULL)
          ON CONFLICT(school_id,identity_id)
          DO UPDATE SET approved_by=EXCLUDED.approved_by,approved_at=NOW(),
            accepted_at=NULL,revoked_at=NULL""", (school_id,candidate_id,owner_id))
        _audit(cur,school_id,owner_id,"backup_admin_owner_approved",candidate_id)
    return {"ok": True, "status": "awaiting_backup_acceptance",
            "backup_id": candidate_id, "emergency_alerts_active": False}


@router.get("/schools/{school_id}/my-admin-rights")
def my_school_admin_rights(school_id: str, authorization: Optional[str] = Header(default=None)):
    actor = _actor(authorization)
    _application_schema()
    actor_id = str(actor["id"])
    with get_db() as conn:
        cur = conn.cursor()
        role = _membership(cur,school_id,actor_id)
        cur.execute("""SELECT approved_at,accepted_at,revoked_at
          FROM verifyd_safety_school_admin_delegations WHERE school_id=%s AND identity_id=%s""",
          (school_id,actor_id))
        delegation = cur.fetchone()
        # Reuse principal signed training policy gate on roster administration.
        try:
            rights = _school_admin(cur,school_id,actor_id)
        except HTTPException:
            rights = None
    return {"ok": True, "role":role, "roster_management_enabled": rights is not None,
            "admin_kind": rights, "backup_pending_acceptance": bool(
                delegation and delegation["revoked_at"] is None and delegation["accepted_at"] is None)}


@router.post("/schools/{school_id}/backup-accept")
def accept_backup_delegation(school_id: str, authorization: Optional[str] = Header(default=None)):
    actor = _actor(authorization)
    _application_schema()
    actor_id = str(actor["id"])
    with get_db() as conn:
        cur = conn.cursor()
        _membership(cur,school_id,actor_id,{"responder"})
        cur.execute("""UPDATE verifyd_safety_school_admin_delegations
           SET accepted_at=NOW() WHERE school_id=%s AND identity_id=%s
             AND revoked_at IS NULL AND accepted_at IS NULL
           RETURNING identity_id""", (school_id,actor_id))
        if not cur.fetchone():
            raise HTTPException(409,detail={"error":"no_pending_backup_approval"})
        _audit(cur,school_id,actor_id,"backup_admin_accepted",actor_id)
    return {"ok":True,"status":"accepted", "emergency_alerts_active":False}


@router.get("/schools/{school_id}/roster")
def school_roster(school_id: str, authorization: Optional[str] = Header(default=None)):
    actor = _actor(authorization)
    _application_schema()
    with get_db() as conn:
        cur = conn.cursor()
        admin_role = _school_admin(cur,school_id,str(actor["id"]))
        cur.execute("""SELECT m.identity_id,i.handle,i.display_name,m.role,m.verified_at,
            EXISTS(SELECT 1 FROM verifyd_safety_school_admin_delegations x
              WHERE x.school_id=m.school_id AND x.identity_id=m.identity_id
                AND x.revoked_at IS NULL) AS backup_delegated
          FROM verifyd_safety_training_members m
          JOIN verifyd_identities i ON i.id=m.identity_id
          WHERE m.school_id=%s ORDER BY m.role,i.handle""", (school_id,))
        rows=[dict(r) for r in cur.fetchall()]
        _audit(cur,school_id,str(actor["id"]),"school_roster_viewed")
    return {"ok":True,"members":rows,"admin_role":admin_role,
            "account_deletion_allowed":False,"emergency_alerts_active":False}


def _reject_protected_member(cur, school_id: str, target_id: str):
    cur.execute("""SELECT role FROM verifyd_safety_training_members
        WHERE school_id=%s AND identity_id=%s FOR UPDATE""", (school_id,target_id))
    target=cur.fetchone()
    if not target:
        raise HTTPException(404,detail={"error":"school_member_not_found"})
    if target["role"]=="principal":
        raise HTTPException(403,detail={"error":"owner_only_principal_change"})
    cur.execute("""SELECT 1 FROM verifyd_safety_school_admin_delegations
        WHERE school_id=%s AND identity_id=%s AND revoked_at IS NULL""",
        (school_id,target_id))
    if cur.fetchone():
        raise HTTPException(403,detail={"error":"owner_only_backup_change"})
    return target


@router.post("/schools/{school_id}/roster")
def add_school_roster_member(school_id: str,payload: RosterMember,
                             authorization: Optional[str]=Header(default=None)):
    actor=_actor(authorization)
    _application_schema()
    member_id=_id_for_handle(payload.handle)
    if member_id==str(actor["id"]):
        raise HTTPException(403,detail={"error":"cannot_change_own_school_role"})
    with get_db() as conn:
        cur=conn.cursor()
        _school_admin(cur,school_id,str(actor["id"]))
        cur.execute("""SELECT role FROM verifyd_safety_training_members
          WHERE school_id=%s AND identity_id=%s""",(school_id,member_id))
        if cur.fetchone():
            raise HTTPException(409,detail={"error":"member_already_enrolled_use_modify"})
        cur.execute("""INSERT INTO verifyd_safety_training_members
          (school_id,identity_id,role,verified_by) VALUES(%s,%s,%s,%s)""",
          (school_id,member_id,payload.role,str(actor["id"])))
        _audit(cur,school_id,str(actor["id"]),"school_roster_member_added",member_id)
    return {"ok":True,"role":payload.role,"school_only":True}


@router.patch("/schools/{school_id}/roster/{member_id}")
def change_school_roster_member(school_id:str,member_id:str,payload:RosterRoleEdit,
                                authorization:Optional[str]=Header(default=None)):
    actor=_actor(authorization)
    _application_schema()
    if member_id==str(actor["id"]):
        raise HTTPException(403,detail={"error":"cannot_modify_own_school_role"})
    with get_db() as conn:
        cur=conn.cursor()
        _school_admin(cur,school_id,str(actor["id"]))
        _reject_protected_member(cur,school_id,member_id)
        cur.execute("""UPDATE verifyd_safety_training_members SET role=%s,verified_by=%s,verified_at=NOW()
           WHERE school_id=%s AND identity_id=%s""",
           (payload.role,str(actor["id"]),school_id,member_id))
        # Old role-bound school/guardian acknowledgments and links no longer
        # automatically authorize a changed identity.
        cur.execute("""DELETE FROM verifyd_safety_training_guardians
           WHERE school_id=%s AND (child_id=%s OR guardian_id=%s)""",(school_id,member_id,member_id))
        cur.execute("""DELETE FROM verifyd_safety_training_contacts
           WHERE school_id=%s AND (student_id=%s OR primary_id=%s OR backup_id=%s)""",
           (school_id,member_id,member_id,member_id))
        cur.execute("""UPDATE verifyd_safety_training_signoffs SET revoked_at=NOW()
           WHERE school_id=%s AND actor_id=%s AND revoked_at IS NULL""",(school_id,member_id))
        _audit(cur,school_id,str(actor["id"]),"school_roster_member_role_changed",member_id)
    return {"ok":True,"role":payload.role,"school_only":True}


@router.delete("/schools/{school_id}/roster/{member_id}")
def deactivate_school_roster_member(school_id:str,member_id:str,
                                    authorization:Optional[str]=Header(default=None)):
    actor=_actor(authorization)
    _application_schema()
    if member_id==str(actor["id"]):
        raise HTTPException(403,detail={"error":"cannot_remove_own_school_role"})
    with get_db() as conn:
        cur=conn.cursor()
        _school_admin(cur,school_id,str(actor["id"]))
        _reject_protected_member(cur,school_id,member_id)
        cur.execute("""DELETE FROM verifyd_safety_training_guardians
            WHERE school_id=%s AND (child_id=%s OR guardian_id=%s)""",(school_id,member_id,member_id))
        cur.execute("""DELETE FROM verifyd_safety_training_contacts
           WHERE school_id=%s AND (student_id=%s OR primary_id=%s OR backup_id=%s)""",
           (school_id,member_id,member_id,member_id))
        cur.execute("""UPDATE verifyd_safety_training_signoffs SET revoked_at=NOW()
           WHERE school_id=%s AND (actor_id=%s OR subject_id=%s)
             AND revoked_at IS NULL""",
           (school_id,member_id,member_id))
        cur.execute("""DELETE FROM verifyd_safety_training_members
           WHERE school_id=%s AND identity_id=%s""",(school_id,member_id))
        _audit(cur,school_id,str(actor["id"]),"school_membership_deactivated",member_id)
    return {"ok":True,"school_membership_removed":True,
            "verifyd_user_account_deleted":False,"historical_training_audit_retained":True}
