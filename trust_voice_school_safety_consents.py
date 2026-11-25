"""School/guardian/student acknowledgments for VeriFYD Safety TRAINING ONLY.

A click is not proof of legal status or binding district approval. Only provisioned,
operator-verified school members can acknowledge; never enable emergency dispatch.
"""
from __future__ import annotations

import os
import threading
import uuid
from typing import Literal, Optional

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field

from database import get_db
from trust_voice_school_safety_training import (
    _actor, _audit, _membership, _schema, _enabled,
)

router = APIRouter(prefix="/trust-voice/school-safety", tags=["VeriFYD Safety Training Acknowledgments"])
POLICY_VERSION = "2026-11-22.training.v1"
_ready = False
_lock = threading.Lock()

# Scope disclosures are immutable for POLICY_VERSION. On substantive change,
# publish a NEW policy version and require renewed acknowledgment.
TRAINING_POLICIES = {
    "school": {
        "title": "School / principal training authorization",
        "statements": [
            "I am an operator-verified principal acting in the assigned TRAINING school account.",
            "I understand this is a TRAINING RECORD system only; it does not alert staff, families, security, or emergency services.",
            "No actual student, guardian, or staff roster may be provisioned without district review, appropriate written agreement, and privacy/security approvals.",
            "I will use test identities and fictitious incidents. I will not rely on this application for an emergency response.",
            "I will ensure a designated response plan, network/accessibility drills and an alternative emergency reporting method before any later live pilot.",
        ],
    },
    "guardian": {
        "title": "Verified guardian acknowledgment for school training",
        "statements": [
            "My access is linked by an operator to this student's training account; this form does not itself verify legal guardianship.",
            "I understand no real emergency notifications, child-location tracking, or direct messages will be sent through this training system.",
            "I understand test reports and timestamps may be visible to authorized training school staff, with school-specific access rules.",
            "I may revoke this training acknowledgment. Revocation prevents the student's NEW training reports unless another verified guardian has an active acknowledgment.",
            "For a real school launch, the school must publish approved privacy notices, data practices, retention/deletion rules, and any required consent process.",
        ],
    },
    "student": {
        "title": "Student training acknowledgment",
        "statements": [
            "This is a TRAINING activity and not a way to call for help in a real emergency.",
            "Holding the training button for five seconds saves a test record only. Nobody is dispatched or notified by it.",
            "If I need help, I must follow my school's real safety instructions and contact a responsible adult or emergency services.",
            "My training updates and test messages can be viewed by authorized school training staff and linked guardians.",
            "I should not enter real threats, personal location, medical details, or other private information during training.",
        ],
    },
}


def _consent_schema():
    global _ready
    _schema()
    if _ready:
        return
    with _lock:
        if _ready:
            return
        with get_db() as conn:
            cur = conn.cursor()
            cur.execute("""CREATE TABLE IF NOT EXISTS verifyd_safety_training_signoffs (
                signoff_id TEXT PRIMARY KEY,
                school_id TEXT NOT NULL REFERENCES verifyd_safety_training_schools(school_id) ON DELETE CASCADE,
                actor_id TEXT NOT NULL REFERENCES verifyd_identities(id) ON DELETE CASCADE,
                kind TEXT NOT NULL CHECK (kind IN ('school','guardian','student')),
                subject_id TEXT NOT NULL DEFAULT '',
                policy_version TEXT NOT NULL,
                accepted_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                revoked_at TIMESTAMPTZ,
                UNIQUE(school_id, actor_id, kind, subject_id)
            )""")
            cur.execute("""CREATE INDEX IF NOT EXISTS safety_signoff_active
                 ON verifyd_safety_training_signoffs(school_id,kind,subject_id,revoked_at)""")
        _ready = True


def _school_accepted(cur, school_id):
    cur.execute("""SELECT EXISTS(
        SELECT 1 FROM verifyd_safety_training_signoffs x
        JOIN verifyd_safety_training_members m
          ON m.school_id=x.school_id AND m.identity_id=x.actor_id AND m.role='principal'
        WHERE x.school_id=%s AND x.kind='school' AND x.subject_id=''
          AND x.policy_version=%s AND x.revoked_at IS NULL
    ) AS approved""", (school_id, POLICY_VERSION))
    return bool(cur.fetchone()["approved"])


def _student_accepted(cur, school_id, student_id):
    cur.execute("""SELECT EXISTS(
        SELECT 1 FROM verifyd_safety_training_signoffs x
        JOIN verifyd_safety_training_members m
          ON m.school_id=x.school_id AND m.identity_id=x.actor_id AND m.role='student'
        WHERE x.school_id=%s AND x.kind='student' AND x.actor_id=%s
          AND x.subject_id=%s AND x.policy_version=%s AND x.revoked_at IS NULL
    ) AS approved""", (school_id, student_id, student_id, POLICY_VERSION))
    return bool(cur.fetchone()["approved"])


def _guardian_count(cur, school_id, student_id):
    cur.execute("""SELECT COUNT(*) AS count
       FROM verifyd_safety_training_signoffs x
       JOIN verifyd_safety_training_guardians g
         ON g.school_id=x.school_id AND g.guardian_id=x.actor_id AND g.child_id=x.subject_id
       JOIN verifyd_safety_training_members m
         ON m.school_id=x.school_id AND m.identity_id=x.actor_id AND m.role='guardian'
       WHERE x.school_id=%s AND x.kind='guardian' AND x.subject_id=%s
         AND x.policy_version=%s AND x.revoked_at IS NULL""",
       (school_id, student_id, POLICY_VERSION))
    return int(cur.fetchone()["count"])


def require_training_signoffs(cur, school_id: str, student_id: str):
    """Use inside the same transaction as report creation, after schema initialization."""
    if not _school_accepted(cur, school_id):
        raise HTTPException(409, detail={"error": "school_training_signoff_required"})
    if not _student_accepted(cur, school_id, student_id):
        raise HTTPException(409, detail={"error": "student_training_acknowledgment_required"})
    from trust_voice_school_student_conduct import _require_accepted
    _require_accepted(cur,school_id,student_id)
    if not _guardian_count(cur, school_id, student_id):
        raise HTTPException(409, detail={"error": "verified_guardian_training_acknowledgment_required"})


class SignoffInput(BaseModel):
    kind: Literal["school", "guardian", "student"]
    child_id: Optional[str] = Field(default=None, max_length=100)
    policy_version: Literal["2026-11-22.training.v1"]
    accepted: Literal[True]
    conduct_policy_version: Optional[str] = None
    understands_acceptable_use: bool = False
    understands_school_review_consequences: bool = False


class RevokeInput(BaseModel):
    kind: Literal["school", "guardian", "student"]
    child_id: Optional[str] = Field(default=None, max_length=100)


@router.get("/policy")
def training_signoff_policy():
    from trust_voice_school_student_conduct import CONDUCT_POLICY_VERSION, CONDUCT_RULES
    return {
        "policy_version": POLICY_VERSION,
        "student_conduct": {
            "version": CONDUCT_POLICY_VERSION,
            "title": "Student Safety acceptable-use agreement",
            "statements": CONDUCT_RULES,
            "student_signs_separately": True,
            "school_review_required_for_restriction": True,
        },
        "mode": "TRAINING ONLY",
        "live_emergency_delivery": False,
        "location_tracking": False,
        "privacy_review_required": True,
        "policies": TRAINING_POLICIES,
        "note": "These records document in-app training acknowledgments only; they are not district contracts, parental identity verification, legal consent determinations, or authorization to operate a real emergency system.",
    }


def _scope(cur, school_id, actor_id, kind, child_id):
    role = _membership(cur, school_id, actor_id)
    if kind == "school":
        if role != "principal":
            raise HTTPException(403, detail={"error": "verified_principal_required"})
        return ""
    if kind == "student":
        if role != "student" or (child_id and child_id != actor_id):
            raise HTTPException(403, detail={"error": "student_self_acknowledgment_only"})
        return actor_id
    if kind != "guardian" or role != "guardian" or not child_id:
        raise HTTPException(403, detail={"error": "verified_guardian_link_required"})
    cur.execute("""SELECT 1 FROM verifyd_safety_training_guardians
        WHERE school_id=%s AND child_id=%s AND guardian_id=%s""",
        (school_id, child_id, actor_id))
    if not cur.fetchone():
        raise HTTPException(403, detail={"error": "verified_guardian_link_required"})
    _membership(cur, school_id, child_id, {"student"})
    return child_id


@router.get("/schools/{school_id}/signoffs")
def signoffs_for_my_role(school_id: str, authorization: Optional[str] = Header(default=None)):
    actor = _actor(authorization)
    actor_id = str(actor["id"])
    _consent_schema()
    with get_db() as conn:
        cur = conn.cursor()
        role = _membership(cur, school_id, actor_id)
        school_approved = _school_accepted(cur, school_id)
        result = {
            "ok": True, "mode": "TRAINING ONLY", "policy_version": POLICY_VERSION,
            "school_approved": school_approved, "role": role,
            "emergency_delivery_enabled": False,
        }
        if role == "student":
            student_approved = _student_accepted(cur, school_id, actor_id)
            guardian_count = _guardian_count(cur, school_id, actor_id)
            from trust_voice_school_student_conduct import _status as conduct_status
            conduct=conduct_status(cur,school_id,actor_id)
            result.update(student_conduct=conduct,student_acknowledged=student_approved,
                          guardian_approved=guardian_count > 0,
                          can_submit_training_report=bool(school_approved and student_approved
                              and guardian_count and conduct["acknowledged"] and not conduct["restricted"]))
        elif role == "guardian":
            cur.execute("""SELECT g.child_id,i.handle,i.display_name
              FROM verifyd_safety_training_guardians g
              JOIN verifyd_identities i ON i.id=g.child_id
              WHERE g.school_id=%s AND g.guardian_id=%s""", (school_id, actor_id))
            children = []
            for c in cur.fetchall():
                cur.execute("""SELECT EXISTS(SELECT 1 FROM verifyd_safety_training_signoffs
                WHERE school_id=%s AND actor_id=%s AND kind='guardian' AND subject_id=%s
                  AND policy_version=%s AND revoked_at IS NULL) AS yes""",
                  (school_id, actor_id, c["child_id"], POLICY_VERSION))
                children.append({"child_id": c["child_id"], "handle": c["handle"],
                                 "display_name": c["display_name"],
                                 "my_acknowledgment": bool(cur.fetchone()["yes"])})
            result["children"] = children
        elif role in {"principal", "responder"}:
            cur.execute("""SELECT EXISTS(SELECT 1 FROM verifyd_safety_training_signoffs
                WHERE school_id=%s AND actor_id=%s AND kind='school' AND subject_id=''
                  AND policy_version=%s AND revoked_at IS NULL) AS yes""",
                (school_id, actor_id, POLICY_VERSION))
            result["my_school_acknowledgment"] = bool(cur.fetchone()["yes"])
    return result


@router.post("/schools/{school_id}/signoffs")
def accept_signoff(school_id: str, payload: SignoffInput, authorization: Optional[str] = Header(default=None)):
    actor = _actor(authorization)
    actor_id = str(actor["id"])
    _consent_schema()
    with get_db() as conn:
        cur = conn.cursor()
        subject = _scope(cur, school_id, actor_id, payload.kind, payload.child_id)
        if payload.kind == "student":
            from trust_voice_school_student_conduct import _enabled as conduct_enabled, _record as record_conduct, CONDUCT_POLICY_VERSION
            if conduct_enabled(school_id):
                if (payload.conduct_policy_version != CONDUCT_POLICY_VERSION
                    or not payload.understands_acceptable_use
                    or not payload.understands_school_review_consequences):
                    raise HTTPException(422,detail={"error":"student_conduct_acknowledgment_required"})
                record_conduct(cur,school_id,actor_id,payload.conduct_policy_version)
        cur.execute("""INSERT INTO verifyd_safety_training_signoffs
            (signoff_id,school_id,actor_id,kind,subject_id,policy_version)
            VALUES (%s,%s,%s,%s,%s,%s)
            ON CONFLICT(school_id,actor_id,kind,subject_id)
            DO UPDATE SET policy_version=EXCLUDED.policy_version,
                accepted_at=NOW(),revoked_at=NULL""",
            ("sssig_"+uuid.uuid4().hex,school_id,actor_id,payload.kind,subject,POLICY_VERSION))
        _audit(cur, school_id, actor_id, "training_acknowledged_"+payload.kind, subject or school_id)
    return {"ok": True, "mode": "TRAINING ONLY", "policy_version": POLICY_VERSION,
            "recorded": True, "live_emergency_delivery": False}


@router.post("/schools/{school_id}/signoffs/revoke")
def revoke_signoff(school_id: str, payload: RevokeInput, authorization: Optional[str] = Header(default=None)):
    actor = _actor(authorization)
    actor_id = str(actor["id"])
    _consent_schema()
    with get_db() as conn:
        cur = conn.cursor()
        subject = _scope(cur, school_id, actor_id, payload.kind, payload.child_id)
        cur.execute("""UPDATE verifyd_safety_training_signoffs
            SET revoked_at=NOW() WHERE school_id=%s AND actor_id=%s
              AND kind=%s AND subject_id=%s AND revoked_at IS NULL""",
            (school_id, actor_id, payload.kind, subject))
        _audit(cur, school_id, actor_id, "training_revoked_"+payload.kind, subject or school_id)
    return {"ok": True, "mode": "TRAINING ONLY", "revoked": True,
            "live_emergency_delivery": False}
