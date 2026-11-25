"""VeriFYD Safety: ISOLATED school training records, NOT emergency dispatch.

No live delivery, E911, Web Push, parent location, or modification of /voice.
A verified operator must explicitly provision schools and members. Default OFF.
"""
from __future__ import annotations

import hashlib
import os
import threading
import uuid
from datetime import datetime, timedelta, timezone
from typing import Literal, Optional

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field

from database import get_db
from trust_voice_messages import _identity_from_bearer
from trust_voice_signaling import _identity_by_handle, _normalize_handle
from trust_voice_permissions import trust_circle_contains
from trust_voice_blocking import identities_blocked

router = APIRouter(prefix="/trust-voice/school-safety", tags=["School Safety TRAINING ONLY"])
_ready = False
_lock = threading.Lock()
WINDOW_MINUTES = 15
CATEGORIES = ("active_threat", "fire", "medical", "violence", "hazard", "other")


def _enabled():
    return os.getenv("VERIFYD_SAFETY_TRAINING_ENABLED", "0").lower().strip() in {"1", "true", "on", "yes"}


def _require():
    if not _enabled():
        raise HTTPException(503, detail={"error": "school_safety_training_not_enabled", "message": "Training is not available. No report was sent."})


def _actor(authorization: Optional[str]):
    _require()
    identity = _identity_from_bearer(authorization)
    if str(identity.get("status") or "active") != "active":
        raise HTTPException(401, detail={"error": "session_required"})
    return identity


def _operators():
    return {v.strip() for v in os.getenv("VERIFYD_SAFETY_TRAINING_OPERATOR_IDS", "").split(",") if v.strip()}


def _operator(identity):
    if str(identity["id"]) not in _operators():
        raise HTTPException(403, detail={"error": "training_operator_required"})


def _schema():
    """Provision only isolated training tables; never reuse personal distress storage."""
    global _ready
    if _ready:
        return
    with _lock:
        if _ready:
            return
        with get_db() as conn:
            cur = conn.cursor()
            cur.execute("""CREATE TABLE IF NOT EXISTS verifyd_safety_training_schools (
                school_id TEXT PRIMARY KEY, school_name TEXT NOT NULL,
                created_by TEXT NOT NULL REFERENCES verifyd_identities(id),
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )""")
            cur.execute("""CREATE TABLE IF NOT EXISTS verifyd_safety_training_members (
                school_id TEXT NOT NULL REFERENCES verifyd_safety_training_schools(school_id) ON DELETE CASCADE,
                identity_id TEXT NOT NULL REFERENCES verifyd_identities(id) ON DELETE CASCADE,
                role TEXT NOT NULL CHECK(role IN ('student','guardian','principal','responder')),
                verified_by TEXT NOT NULL REFERENCES verifyd_identities(id),
                verified_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                PRIMARY KEY (school_id,identity_id)
            )""")
            cur.execute("CREATE INDEX IF NOT EXISTS safety_training_member_identity ON verifyd_safety_training_members(identity_id,school_id)")
            cur.execute("""CREATE TABLE IF NOT EXISTS verifyd_safety_training_guardians (
                school_id TEXT NOT NULL REFERENCES verifyd_safety_training_schools(school_id) ON DELETE CASCADE,
                child_id TEXT NOT NULL REFERENCES verifyd_identities(id) ON DELETE CASCADE,
                guardian_id TEXT NOT NULL REFERENCES verifyd_identities(id) ON DELETE CASCADE,
                verified_by TEXT NOT NULL REFERENCES verifyd_identities(id),
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                PRIMARY KEY(school_id,child_id,guardian_id)
            )""")
            cur.execute("""CREATE TABLE IF NOT EXISTS verifyd_safety_training_contacts (
                school_id TEXT NOT NULL REFERENCES verifyd_safety_training_schools(school_id) ON DELETE CASCADE,
                student_id TEXT NOT NULL REFERENCES verifyd_identities(id) ON DELETE CASCADE,
                primary_id TEXT NOT NULL REFERENCES verifyd_identities(id),
                backup_id TEXT REFERENCES verifyd_identities(id),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                PRIMARY KEY(school_id,student_id),
                CHECK(primary_id<>backup_id), CHECK(student_id<>primary_id), CHECK(student_id<>backup_id)
            )""")
            cur.execute("""ALTER TABLE verifyd_safety_training_contacts
                ADD COLUMN IF NOT EXISTS route_source TEXT NOT NULL DEFAULT 'manual'""")
            # Old TEST routes sometimes stored the Principal as a family Backup.
            # A school response recipient is not a guardian. Preserve Primary,
            # move staff to its separate recipient role, and keep report history.
            cur.execute("ALTER TABLE verifyd_safety_training_contacts ALTER COLUMN backup_id DROP NOT NULL")
            cur.execute("""CREATE TABLE IF NOT EXISTS verifyd_safety_school_response_team (
                school_id TEXT NOT NULL REFERENCES verifyd_safety_training_schools(school_id) ON DELETE CASCADE,
                responder_id TEXT NOT NULL REFERENCES verifyd_identities(id),
                assigned_by TEXT NOT NULL REFERENCES verifyd_identities(id),
                assigned_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                PRIMARY KEY(school_id,responder_id)
            )""")
            cur.execute("""CREATE TABLE IF NOT EXISTS verifyd_safety_training_incidents (
                incident_id TEXT PRIMARY KEY,
                school_id TEXT NOT NULL REFERENCES verifyd_safety_training_schools(school_id),
                category TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'active'
                  CHECK(status IN ('active','resolved')),
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                resolved_at TIMESTAMPTZ, resolved_by TEXT, resolution_reason TEXT
            )""")
            cur.execute("CREATE INDEX IF NOT EXISTS safety_training_incident_lookup ON verifyd_safety_training_incidents(school_id,category,status,created_at DESC)")
            cur.execute("""CREATE TABLE IF NOT EXISTS verifyd_safety_training_reports (
                report_id TEXT PRIMARY KEY,
                incident_id TEXT NOT NULL REFERENCES verifyd_safety_training_incidents(incident_id),
                school_id TEXT NOT NULL REFERENCES verifyd_safety_training_schools(school_id),
                student_id TEXT NOT NULL REFERENCES verifyd_identities(id),
                client_key TEXT NOT NULL,
                category TEXT NOT NULL,
                description TEXT NOT NULL DEFAULT '',
                student_status TEXT NOT NULL DEFAULT 'need_help',
                closed_at TIMESTAMPTZ,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                UNIQUE(student_id,client_key)
            )""")
            cur.execute("CREATE INDEX IF NOT EXISTS safety_training_report_lookup ON verifyd_safety_training_reports(incident_id,created_at)")
            cur.execute("""CREATE TABLE IF NOT EXISTS verifyd_safety_training_updates (
                update_id TEXT PRIMARY KEY, report_id TEXT NOT NULL REFERENCES verifyd_safety_training_reports(report_id),
                actor_id TEXT NOT NULL REFERENCES verifyd_identities(id),
                status TEXT NOT NULL CHECK(status IN ('need_help','safe','improved')),
                message TEXT NOT NULL DEFAULT '',
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )""")
            cur.execute("""CREATE TABLE IF NOT EXISTS verifyd_safety_training_recipients (
                report_id TEXT NOT NULL REFERENCES verifyd_safety_training_reports(report_id),
                recipient_id TEXT NOT NULL REFERENCES verifyd_identities(id),
                reason TEXT NOT NULL,
                PRIMARY KEY(report_id,recipient_id)
            )""")
            cur.execute("""CREATE TABLE IF NOT EXISTS verifyd_safety_training_audit (
                audit_id TEXT PRIMARY KEY,
                school_id TEXT NOT NULL,
                actor_id TEXT NOT NULL,
                action TEXT NOT NULL,
                object_id TEXT,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )""")
            # One-time conversion of legacy "Principal as backup guardian"
            # assignments. Principal stays in school response, not family.
            cur.execute("""UPDATE verifyd_safety_training_contacts c
                 SET backup_id=NULL,updated_at=NOW()
                 WHERE c.backup_id IS NOT NULL AND EXISTS (
                   SELECT 1 FROM verifyd_safety_training_members m
                   WHERE m.school_id=c.school_id AND m.identity_id=c.backup_id
                     AND m.role IN ('principal','responder'))
                 RETURNING school_id,student_id""")
            for moved in cur.fetchall():
                cur.execute("""INSERT INTO verifyd_safety_training_audit
                  (audit_id,school_id,actor_id,action,object_id)
                  VALUES(%s,%s,%s,%s,%s)""",
                  ("ssa_" + uuid.uuid4().hex, moved["school_id"], "system-migration",
                   "school_staff_removed_from_guardian_backup", moved["student_id"]))
        _ready = True


def _membership(cur, school_id, actor_id, roles=None):
    cur.execute("""SELECT role FROM verifyd_safety_training_members
        WHERE school_id=%s AND identity_id=%s""", (school_id, actor_id))
    row = cur.fetchone()
    if not row or (roles is not None and row["role"] not in roles):
        raise HTTPException(403, detail={"error": "school_membership_required"})
    return row["role"]


def _audit(cur, school_id, actor_id, action, object_id=None):
    cur.execute("""INSERT INTO verifyd_safety_training_audit
      (audit_id,school_id,actor_id,action,object_id) VALUES (%s,%s,%s,%s,%s)""",
      ("ssa_" + uuid.uuid4().hex, school_id, actor_id, action, object_id))


class SchoolInput(BaseModel):
    name: str = Field(min_length=3, max_length=120)


class MemberInput(BaseModel):
    handle: str = Field(min_length=3, max_length=40)
    role: Literal["student", "guardian", "principal", "responder"]


class GuardianLink(BaseModel):
    child_handle: str = Field(min_length=3, max_length=40)
    guardian_handle: str = Field(min_length=3, max_length=40)


class ContactInput(BaseModel):
    primary_id: str = Field(min_length=3, max_length=100)
    backup_id: str = Field(min_length=3, max_length=100)


class ReportInput(BaseModel):
    school_id: str = Field(min_length=3, max_length=80)
    idempotency_key: str = Field(min_length=16, max_length=100)
    category: Literal["active_threat", "fire", "medical", "violence", "hazard", "other"]
    choice_id: str = Field(default="", max_length=40, pattern=r"^(?:[a-z][a-z0-9_]{2,39})?$")
    description: str = Field(default="", max_length=700)


class UpdateInput(BaseModel):
    status: Literal["need_help", "safe", "improved"]
    message: str = Field(default="", max_length=700)


class EndInput(BaseModel):
    confirmation: Literal[True]


class ResolveInput(BaseModel):
    reason: str = Field(min_length=8, max_length=700)


def _id_for_handle(handle):
    person = _identity_by_handle(_normalize_handle(handle))
    if not person or str(person.get("status")) != "active":
        raise HTTPException(404, detail={"error": "identity_not_found"})
    return str(person["id"])


@router.get("/health")
def health():
    # Public, sanitized, no student data, no transport credentials.
    return {"feature": "verifyd_safety_training_v1", "training_enabled": _enabled(),
            "live_emergency_delivery": False, "location_sharing": False, "emergency_services_connected": False}


@router.get("/enrollment")
def enrollment(authorization: Optional[str] = Header(default=None)):
    identity = _actor(authorization)
    _schema()
    from trust_voice_school_safety_consents import _consent_schema
    _consent_schema()
    identity_id = str(identity["id"])
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("""SELECT s.school_id,s.school_name,m.role FROM verifyd_safety_training_members m
            JOIN verifyd_safety_training_schools s ON s.school_id=m.school_id
            WHERE m.identity_id=%s ORDER BY s.school_name""", (identity_id,))
        schools = [dict(x) for x in cur.fetchall()]
        eligible = []
        for school in schools:
            if school["role"] != "student":
                continue
            cur.execute("""SELECT m.identity_id,i.handle,i.display_name,m.role FROM verifyd_safety_training_members m
              JOIN verifyd_identities i ON i.id=m.identity_id
              WHERE m.school_id=%s AND m.role='guardian' ORDER BY i.handle""",
              (school["school_id"],))
            school["eligible_contacts"] = []
            for row in cur.fetchall():
                adult_id = str(row["identity_id"])
                try:
                    _verified_training_contact(cur,school["school_id"],identity_id,adult_id)
                except HTTPException:
                    continue
                school["eligible_contacts"].append({"identity_id": adult_id, "handle": row["handle"],
                    "display_name": row["display_name"], "role": row["role"]})
            cur.execute("""SELECT primary_id,backup_id FROM verifyd_safety_training_contacts
                WHERE school_id=%s AND student_id=%s""", (school["school_id"], identity_id))
            school["contacts"] = dict(cur.fetchone() or {})
    return {"ok": True, "mode": "TRAINING ONLY", "schools": schools,
            "is_operator": identity_id in _operators(), "live_emergency_delivery": False}


@router.post("/schools")
def create_training_school(payload: SchoolInput, authorization: Optional[str] = Header(default=None)):
    actor = _actor(authorization)
    _operator(actor)
    _schema()
    school_id = "sschool_" + uuid.uuid4().hex
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("""INSERT INTO verifyd_safety_training_schools(school_id,school_name,created_by)
            VALUES (%s,%s,%s)""", (school_id, payload.name.strip(), str(actor["id"])))
        _audit(cur, school_id, str(actor["id"]), "training_school_created", school_id)
    return {"ok": True, "school_id": school_id, "mode": "TRAINING ONLY"}


@router.post("/schools/{school_id}/members")
def assign_training_member(school_id: str, payload: MemberInput, authorization: Optional[str] = Header(default=None)):
    actor = _actor(authorization)
    _operator(actor)  # no self-enrollment or student-created responders
    _schema()
    member_id = _id_for_handle(payload.handle)
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("SELECT 1 FROM verifyd_safety_training_schools WHERE school_id=%s", (school_id,))
        if not cur.fetchone():
            raise HTTPException(404, detail={"error": "school_not_found"})
        cur.execute("""INSERT INTO verifyd_safety_training_members(school_id,identity_id,role,verified_by)
            VALUES(%s,%s,%s,%s)
            ON CONFLICT(school_id,identity_id) DO UPDATE SET
            role=EXCLUDED.role, verified_by=EXCLUDED.verified_by, verified_at=NOW()""",
            (school_id, member_id, payload.role, str(actor["id"])))
        _audit(cur, school_id, str(actor["id"]), "training_member_assigned", member_id)
    return {"ok": True, "role": payload.role, "mode": "TRAINING ONLY"}


@router.post("/schools/{school_id}/guardian-links")
def add_training_guardian(school_id: str, payload: GuardianLink, authorization: Optional[str] = Header(default=None)):
    actor = _actor(authorization)
    _operator(actor)
    _schema()
    child_id = _id_for_handle(payload.child_handle)
    guardian_id = _id_for_handle(payload.guardian_handle)
    if child_id == guardian_id:
        raise HTTPException(422, detail={"error": "distinct_people_required"})
    with get_db() as conn:
        cur = conn.cursor()
        _membership(cur, school_id, child_id, {"student"})
        _membership(cur, school_id, guardian_id, {"guardian"})
        cur.execute("""INSERT INTO verifyd_safety_training_guardians
          (school_id,child_id,guardian_id,verified_by) VALUES (%s,%s,%s,%s)
          ON CONFLICT DO NOTHING""", (school_id, child_id, guardian_id, str(actor["id"])))
        _audit(cur, school_id, str(actor["id"]), "training_guardian_linked", child_id)
    return {"ok": True, "mode": "TRAINING ONLY"}


@router.put("/schools/{school_id}/contacts")
def set_training_contacts(school_id: str, payload: ContactInput, authorization: Optional[str] = Header(default=None)):
    # Old student-side route cannot grant school alert recipients.
    _actor(authorization)
    raise HTTPException(403, detail={"error": "school_contact_routing_managed_by_principal"})


def _verified_training_contact(cur, school_id, student_id, adult_id):
    """Only an independently linked, acknowledged guardian; NEVER school staff."""
    if not adult_id or student_id == adult_id or identities_blocked(student_id, adult_id):
        raise HTTPException(403, detail={"error": "guardian_not_authorized_for_student"})
    _membership(cur, school_id, adult_id, {"guardian"})
    from trust_voice_school_safety_consents import POLICY_VERSION
    cur.execute("""SELECT 1 FROM verifyd_safety_training_guardians g
      JOIN verifyd_safety_training_signoffs x
        ON x.school_id=g.school_id AND x.actor_id=g.guardian_id
        AND x.subject_id=g.child_id AND x.kind='guardian'
        AND x.policy_version=%s AND x.revoked_at IS NULL
      JOIN verifyd_identities i ON i.id=g.guardian_id AND i.status='active'
      WHERE g.school_id=%s AND g.child_id=%s AND g.guardian_id=%s""",
      (POLICY_VERSION,school_id,student_id,adult_id))
    if not cur.fetchone():
        raise HTTPException(403, detail={"error": "verified_guardian_training_acknowledgment_required"})
    return "guardian"


def _school_response_authorized(cur, school_id, staff_id):
    """Access to INITIAL TEST incident only, not school roster administration."""
    from trust_voice_school_safety_registration import _school_admin
    role = _membership(cur, school_id, staff_id, {"principal", "responder"})
    cur.execute("SELECT status FROM verifyd_identities WHERE id=%s", (staff_id,))
    person = cur.fetchone()
    if not person or person["status"] != "active":
        raise HTTPException(403, detail={"error": "school_response_staff_inactive"})
    if role == "principal":
        _school_admin(cur, school_id, staff_id)
        return role
    cur.execute("""SELECT 1 FROM verifyd_safety_school_response_team
      WHERE school_id=%s AND responder_id=%s""", (school_id, staff_id))
    if not cur.fetchone():
        raise HTTPException(403, detail={"error": "school_response_staff_not_designated"})
    return role


def _school_response_recipients(cur, school_id):
    cur.execute("""SELECT identity_id FROM verifyd_safety_training_members
        WHERE school_id=%s AND role='principal'
        UNION SELECT responder_id FROM verifyd_safety_school_response_team
        WHERE school_id=%s""", (school_id, school_id))
    candidates = [str(row["identity_id"]) for row in cur.fetchall()]
    approved = []
    for adult_id in candidates:
        try: _school_response_authorized(cur, school_id, adult_id)
        except HTTPException: continue
        approved.append(adult_id)
    if not approved:
        raise HTTPException(409, detail={"error": "verified_school_principal_required"})
    return approved


class SchoolVerifiedContactRoute(BaseModel):
    primary_id: str = Field(min_length=3, max_length=100)
    backup_id: Optional[str] = Field(default=None, min_length=3, max_length=100)
    independently_verified: Literal[True]


class SchoolResponseTeamSelection(BaseModel):
    responder_ids: list[str] = Field(default_factory=list, max_length=20)
    independently_verified: Literal[True]


@router.get("/schools/{school_id}/response-team")
def get_verified_school_response_team(school_id: str, authorization: Optional[str] = Header(default=None)):
    actor = _actor(authorization)
    _schema()
    from trust_voice_school_safety_registration import _school_admin
    from trust_voice_school_safety_consents import _consent_schema
    _consent_schema()
    with get_db() as conn:
        cur = conn.cursor()
        _school_admin(cur, school_id, str(actor["id"]))
        cur.execute("""SELECT m.identity_id,i.handle,i.display_name,m.role
            FROM verifyd_safety_training_members m JOIN verifyd_identities i ON i.id=m.identity_id
            WHERE m.school_id=%s AND m.role IN ('principal','responder') AND i.status='active'
            ORDER BY CASE WHEN m.role='principal' THEN 0 ELSE 1 END,i.handle""", (school_id,))
        members = [dict(x) for x in cur.fetchall()]
        cur.execute("""SELECT responder_id FROM verifyd_safety_school_response_team WHERE school_id=%s""", (school_id,))
        designated = {str(r["responder_id"]) for r in cur.fetchall()}
    for item in members:
        item["identity_id"] = str(item["identity_id"])
        item["selected"] = item["role"] == "principal" or item["identity_id"] in designated
        item["fixed_principal"] = item["role"] == "principal"
    return {"ok": True, "mode": "TRAINING ONLY", "school_id": school_id, "members": members,
            "staff_receive_initial_incident_only": True, "private_family_texts_visible": False}


@router.put("/schools/{school_id}/response-team")
def update_verified_school_response_team(school_id: str, payload: SchoolResponseTeamSelection,
                                        authorization: Optional[str] = Header(default=None)):
    actor = _actor(authorization)
    _schema()
    from trust_voice_school_safety_registration import _school_admin
    from trust_voice_school_safety_consents import _consent_schema
    _consent_schema()
    selected = [str(v) for v in payload.responder_ids]
    if len(set(selected)) != len(selected):
        raise HTTPException(422, detail={"error": "duplicate_school_response_staff"})
    with get_db() as conn:
        cur = conn.cursor()
        _school_admin(cur, school_id, str(actor["id"]))
        if selected:
            cur.execute("""SELECT m.identity_id FROM verifyd_safety_training_members m
                 JOIN verifyd_identities i ON i.id=m.identity_id AND i.status='active'
                 WHERE m.school_id=%s AND m.role='responder' AND m.identity_id=ANY(%s)""", (school_id, selected))
            eligible = {str(r["identity_id"]) for r in cur.fetchall()}
            if eligible != set(selected):
                raise HTTPException(403, detail={"error": "school_response_staff_must_be_verified_responder"})
        cur.execute("DELETE FROM verifyd_safety_school_response_team WHERE school_id=%s", (school_id,))
        for adult_id in selected:
            cur.execute("""INSERT INTO verifyd_safety_school_response_team
                 (school_id,responder_id,assigned_by) VALUES(%s,%s,%s)""",
                 (school_id, adult_id, str(actor["id"])))
        _audit(cur, school_id, str(actor["id"]), "school_response_team_designated",
               "responders:" + str(len(selected)))
    return {"ok": True, "mode": "TRAINING ONLY", "designated_responders": len(selected),
            "principal_always_included": True, "new_notifications_sent": False,
            "private_family_texts_visible": False}


@router.get("/schools/{school_id}/contact-routing")
def get_verified_school_contact_routing(school_id: str, authorization: Optional[str] = Header(default=None)):
    actor = _actor(authorization)
    _schema()
    from trust_voice_school_safety_consents import _consent_schema
    from trust_voice_school_safety_registration import _school_admin
    _consent_schema()
    with get_db() as conn:
        cur = conn.cursor()
        _school_admin(cur, school_id, str(actor["id"]))
        from trust_voice_school_auto_routing import _enabled as _auto_enabled, sync_approved_guardian_route
        if _auto_enabled(school_id):
            cur.execute("""SELECT student_identity_id FROM verifyd_safety_managed_students
                WHERE school_id=%s AND enrollment_status='school_approved'
                  AND student_identity_id IS NOT NULL ORDER BY local_ref LIMIT 2000""",(school_id,))
            student_ids=[str(row["student_identity_id"]) for row in cur.fetchall()]
            for sid in student_ids:
                sync_approved_guardian_route(cur,school_id,sid,str(actor["id"]))
        cur.execute("""SELECT m.identity_id AS student_id,i.handle,i.display_name
          FROM verifyd_safety_training_members m JOIN verifyd_identities i ON i.id=m.identity_id
          WHERE m.school_id=%s AND m.role='student' ORDER BY i.handle""", (school_id,))
        students = [dict(x) for x in cur.fetchall()]
        result = []
        for student in students:
            sid = str(student["student_id"])
            cur.execute("""SELECT m.identity_id,i.handle,i.display_name,m.role
              FROM verifyd_safety_training_guardians g
              JOIN verifyd_safety_training_members m
                ON m.school_id=g.school_id AND m.identity_id=g.guardian_id AND m.role='guardian'
              JOIN verifyd_identities i ON i.id=m.identity_id
              WHERE g.school_id=%s AND g.child_id=%s ORDER BY i.handle""", (school_id,sid))
            primary = []
            for row in cur.fetchall():
                person = dict(row)
                person["identity_id"] = str(person["identity_id"])
                try: _verified_training_contact(cur,school_id,sid,person["identity_id"])
                except HTTPException: continue
                primary.append(person)
            # Family recipients can only be verified guardians, never school staff.
            backup = list(primary)
            cur.execute("""SELECT primary_id,backup_id,route_source FROM verifyd_safety_training_contacts
              WHERE school_id=%s AND student_id=%s""", (school_id,sid))
            current = dict(cur.fetchone() or {})
            result.append({"student_id":sid,"handle":student["handle"],
              "display_name":student["display_name"],"primary_guardians":primary,
              "backup_guardians":backup,"backup_contacts":backup,
              "current_contacts":current})
    return {"ok":True,"mode":"TRAINING ONLY","students":result}


@router.put("/schools/{school_id}/contact-routing/{student_id}")
def save_verified_school_contact_routing(
    school_id: str, student_id: str, payload: SchoolVerifiedContactRoute,
    authorization: Optional[str] = Header(default=None),
):
    actor = _actor(authorization)
    if student_id == payload.primary_id or (payload.backup_id and
        (student_id == payload.backup_id or payload.primary_id == payload.backup_id)):
        raise HTTPException(422,detail={"error":"guardians_must_be_distinct_from_student_and_each_other"})
    _schema()
    from trust_voice_school_safety_consents import _consent_schema
    from trust_voice_school_safety_registration import _school_admin
    _consent_schema()
    with get_db() as conn:
        cur = conn.cursor()
        _school_admin(cur,school_id,str(actor["id"]))
        _membership(cur,school_id,student_id,{"student"})
        _verified_training_contact(cur,school_id,student_id,payload.primary_id)
        if payload.backup_id:
            _verified_training_contact(cur,school_id,student_id,payload.backup_id)
        cur.execute("""INSERT INTO verifyd_safety_training_contacts
          (school_id,student_id,primary_id,backup_id,route_source) VALUES (%s,%s,%s,%s,'manual')
          ON CONFLICT(school_id,student_id) DO UPDATE SET
          primary_id=EXCLUDED.primary_id,backup_id=EXCLUDED.backup_id,
          route_source='manual',updated_at=NOW()""",
          (school_id,student_id,payload.primary_id,payload.backup_id))
        _audit(cur,school_id,str(actor["id"]),"verified_training_contact_route_saved",student_id)
    return {"ok":True,"mode":"TRAINING ONLY","saved":True,
            "notifications_sent":False,"emergency_services_notified":False}



@router.delete("/schools/{school_id}/contact-routing/{student_id}")
def disable_verified_school_contact_routing(
    school_id: str, student_id: str,
    authorization: Optional[str] = Header(default=None),
):
    """Clear the family guardian route for future TEST incidents only.

    Primary guardian is required; Backup is optional but MUST also be a
    verified guardian. Removing the route blocks NEW training reports.
    It does not change the separate Principal/designated staff response team,
    historical incident records, or school membership.
    """
    actor = _actor(authorization)
    _schema()
    from trust_voice_school_safety_registration import _school_admin
    with get_db() as conn:
        cur = conn.cursor()
        _school_admin(cur, school_id, str(actor["id"]))
        _membership(cur, school_id, student_id, {"student"})
        cur.execute("""DELETE FROM verifyd_safety_training_contacts
            WHERE school_id=%s AND student_id=%s
            RETURNING primary_id, backup_id""", (school_id, student_id))
        removed = cur.fetchone() is not None
        if removed:
            _audit(cur, school_id, str(actor["id"]),
                   "verified_training_contact_route_disabled", student_id)
    return {
        "ok": True, "mode": "TRAINING ONLY", "removed": removed,
        "test_reporting_disabled_until_new_route": True,
        "accounts_deleted": False, "prior_reports_deleted": False,
        "notifications_sent": False, "emergency_services_notified": False,
    }

def _report_receipt(row, count):
    return {"ok": True, "mode": "TRAINING ONLY", "report_id": row["report_id"],
        "incident_id": row["incident_id"], "incident_reports": count,
        "stored_on_server": True, "staff_notified": False, "parents_notified": False,
        "emergency_services_notified": False, "delivery_status": "TRAINING_RECORD_ONLY"}


@router.post("/reports")
def create_training_report(payload: ReportInput, authorization: Optional[str] = Header(default=None)):
    actor = _actor(authorization)
    return _create_training_report_for_student(payload, str(actor["id"]))


def _create_training_report_for_student(payload: ReportInput, student_id: str):
    """Shared TEST record path. Caller MUST authenticate the current student.

    The standard endpoint uses /voice bearer auth; the supervised device route
    uses only its own narrow, expiring school-device token. Neither route skips
    role, consent, guardian, recipient or school membership checks below.
    """
    _schema()
    from trust_voice_school_incident_chat import _schema as _incident_schema,queue_report,dispatch_report
    from trust_voice_school_choice_catalog import _catalog_schema, resolve_report_choice
    _incident_schema()
    _catalog_schema()
    from trust_voice_school_safety_consents import _consent_schema, require_training_signoffs
    _consent_schema()  # DDL before report transaction, never while holding row locks
    with get_db() as conn:
        cur = conn.cursor()
        _membership(cur, payload.school_id, student_id, {"student"})
        require_training_signoffs(cur, payload.school_id, student_id)
        initial_description = payload.description.strip()
        if payload.choice_id:
            # A customized choice is validated using the server copy, not the client label.
            selected_choice = resolve_report_choice(cur, payload.school_id, payload.choice_id)
            if selected_choice["category"] != payload.category:
                raise HTTPException(409, detail={"error": "school_safety_choice_changed"})
            if selected_choice["id"] == "other" and not initial_description:
                raise HTTPException(422, detail={"error": "school_safety_details_required"})
            initial_description = ("TEST choice: " + selected_choice["label"] + "\n" + initial_description)[:700]
        cur.execute("""SELECT primary_id, backup_id FROM verifyd_safety_training_contacts
            WHERE school_id=%s AND student_id=%s""", (payload.school_id, student_id))
        contacts = cur.fetchone()
        if not contacts:
            raise HTTPException(409, detail={"error": "configure_primary_and_backup_first"})
        # Recipient domains are disjoint: guardians vs Principal/designated staff.
        # This also handles old staff-as-Backup rows safely during migration.
        try:
            _verified_training_contact(cur,payload.school_id,student_id,str(contacts["primary_id"]))
            if contacts["backup_id"]:
                _verified_training_contact(cur,payload.school_id,student_id,str(contacts["backup_id"]))
        except HTTPException:
            raise HTTPException(409, detail={"error": "selected_guardian_no_longer_eligible"})
        school_team = _school_response_recipients(cur, payload.school_id)
        # Serialize attempts with the same per-student idempotency key, even if
        # clients retry with different categories at the same time.
        replay_lock_hash = hashlib.sha256(
            (student_id + ":" + payload.idempotency_key).encode()
        ).digest()
        replay_lock = int.from_bytes(replay_lock_hash[:8], "big") % (2**63 - 1)
        cur.execute("SELECT pg_advisory_xact_lock(%s)", (replay_lock,))
        cur.execute("""SELECT report_id,incident_id FROM verifyd_safety_training_reports
             WHERE student_id=%s AND client_key=%s""", (student_id, payload.idempotency_key))
        previous = cur.fetchone()
        if previous:
            cur.execute("""SELECT COUNT(*) AS count FROM verifyd_safety_training_reports
                WHERE incident_id=%s""", (previous["incident_id"],))
            return _report_receipt(previous, cur.fetchone()["count"])
        # Per-school/per-category transactional lock guarantees one group under concurrent reports.
        raw_lock = hashlib.sha256((payload.school_id + ":" + payload.category).encode()).digest()
        lock = int.from_bytes(raw_lock[:8], "big") % (2**63 - 1)
        cur.execute("SELECT pg_advisory_xact_lock(%s)", (lock,))
        cur.execute("""SELECT incident_id FROM verifyd_safety_training_incidents
            WHERE school_id=%s AND category=%s AND status='active'
              AND created_at>NOW()-(%s * interval '1 minute')
            ORDER BY created_at DESC LIMIT 1""", (payload.school_id, payload.category, WINDOW_MINUTES))
        group = cur.fetchone()
        incident_id = group["incident_id"] if group else "ssinc_" + uuid.uuid4().hex
        if not group:
            cur.execute("""INSERT INTO verifyd_safety_training_incidents
                (incident_id,school_id,category) VALUES(%s,%s,%s)""",
                (incident_id,payload.school_id,payload.category))
        report_id = "ssrep_" + uuid.uuid4().hex
        cur.execute("""INSERT INTO verifyd_safety_training_reports
           (report_id,incident_id,school_id,student_id,client_key,category,description)
           VALUES (%s,%s,%s,%s,%s,%s,%s)""", (report_id,incident_id,payload.school_id,
           student_id,payload.idempotency_key,payload.category,initial_description))
        recipients = set(school_team) | {str(contacts["primary_id"])}
        if contacts["backup_id"]:
            recipients.add(str(contacts["backup_id"]))
        for recipient in recipients:
            cur.execute("""INSERT INTO verifyd_safety_training_recipients
                (report_id,recipient_id,reason) VALUES(%s,%s,%s) ON CONFLICT DO NOTHING""",
                (report_id,recipient,"TRAINING_ONLY_NEVER_SENT"))
        queue_report(cur, report_id, payload.school_id, student_id, recipients)
        _audit(cur, payload.school_id, student_id, "training_report_created", report_id)
        cur.execute("SELECT COUNT(*) AS count FROM verifyd_safety_training_reports WHERE incident_id=%s", (incident_id,))
        count = cur.fetchone()["count"]
    # A database receipt is NOT a device receipt. The test push outbox is
    # durable even if this best-effort dispatch fails.
    push={"attempted_recipients":0,"push_provider_accepted":0}
    try:
        push=dispatch_report(report_id)
    except Exception:
        pass
    return {**_report_receipt({"report_id":report_id,"incident_id":incident_id}, count),
            "test_notifications":push}


def _report_access(cur, report_id, actor_id):
    cur.execute("""SELECT r.report_id,r.incident_id,r.school_id,r.student_id,r.student_status,
        r.closed_at,i.status AS incident_status FROM verifyd_safety_training_reports r
        JOIN verifyd_safety_training_incidents i ON i.incident_id=r.incident_id
        WHERE r.report_id=%s""", (report_id,))
    row = cur.fetchone()
    if not row:
        raise HTTPException(404, detail={"error": "report_not_found"})
    return row


@router.get("/my-reports")
def my_training_reports(authorization: Optional[str] = Header(default=None)):
    actor = _actor(authorization)
    _schema()
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("""SELECT r.report_id,r.school_id,r.incident_id,r.category,r.description,
            r.student_status,r.closed_at,r.created_at,i.status AS incident_status
            FROM verifyd_safety_training_reports r JOIN verifyd_safety_training_incidents i
            ON r.incident_id=i.incident_id WHERE r.student_id=%s
            ORDER BY r.created_at DESC LIMIT 20""", (str(actor["id"]),))
        result = [dict(r) for r in cur.fetchall()]
    return {"ok": True, "mode": "TRAINING ONLY", "reports": result}


@router.post("/reports/{report_id}/updates")
def update_training_report(report_id: str, payload: UpdateInput, authorization: Optional[str] = Header(default=None)):
    actor = _actor(authorization)
    _schema()
    with get_db() as conn:
        cur = conn.cursor()
        row = _report_access(cur, report_id, str(actor["id"]))
        if row["student_id"] != str(actor["id"]):
            raise HTTPException(403, detail={"error": "student_report_owner_only"})
        if row["closed_at"] or row["incident_status"] == "resolved":
            raise HTTPException(409, detail={"error": "report_closed"})
        cur.execute("""INSERT INTO verifyd_safety_training_updates
          (update_id,report_id,actor_id,status,message) VALUES(%s,%s,%s,%s,%s)""",
          ("ssupd_"+uuid.uuid4().hex, report_id, str(actor["id"]), payload.status, payload.message.strip()))
        cur.execute("""UPDATE verifyd_safety_training_reports SET student_status=%s
            WHERE report_id=%s""", (payload.status, report_id))
        _audit(cur, row["school_id"], str(actor["id"]), "training_student_update", report_id)
    return {"ok": True, "mode": "TRAINING ONLY", "staff_notified": False}


@router.post("/reports/{report_id}/end")
def end_training_report(report_id: str, payload: EndInput, authorization: Optional[str] = Header(default=None)):
    actor = _actor(authorization)
    _schema()
    with get_db() as conn:
        cur = conn.cursor()
        row = _report_access(cur, report_id, str(actor["id"]))
        if row["student_id"] != str(actor["id"]):
            raise HTTPException(403, detail={"error": "student_report_owner_only"})
        cur.execute("UPDATE verifyd_safety_training_reports SET closed_at=NOW() WHERE report_id=%s AND closed_at IS NULL", (report_id,))
        _audit(cur, row["school_id"], str(actor["id"]), "training_student_report_ended", report_id)
    return {"ok": True, "mode": "TRAINING ONLY", "school_incident_resolved": False}


@router.get("/schools/{school_id}/incidents")
def training_school_incidents(school_id: str, authorization: Optional[str] = Header(default=None)):
    actor = _actor(authorization)
    _schema()
    with get_db() as conn:
        cur = conn.cursor()
        _school_response_authorized(cur, school_id, str(actor["id"]))
        # Staff see incident grouping and initial reports, never student follow-up status.
        cur.execute("""SELECT i.incident_id,i.category,i.status,i.created_at,i.resolved_at,
            COUNT(r.report_id) AS report_count
            FROM verifyd_safety_training_incidents i LEFT JOIN verifyd_safety_training_reports r
            ON r.incident_id=i.incident_id WHERE i.school_id=%s
            GROUP BY i.incident_id ORDER BY i.created_at DESC LIMIT 60""", (school_id,))
        incidents = [dict(v) for v in cur.fetchall()]
        _audit(cur, school_id, str(actor["id"]), "training_incidents_viewed")
    return {"ok": True, "mode": "TRAINING ONLY", "incidents": incidents, "real_notifications_sent": 0}


@router.get("/incidents/{incident_id}/reports")
def incident_training_reports(incident_id: str, authorization: Optional[str] = Header(default=None)):
    actor = _actor(authorization)
    _schema()
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("SELECT school_id FROM verifyd_safety_training_incidents WHERE incident_id=%s", (incident_id,))
        incident = cur.fetchone()
        if not incident:
            raise HTTPException(404, detail={"error": "incident_not_found"})
        _school_response_authorized(cur, incident["school_id"], str(actor["id"]))
        # Initial submission metadata only: no ongoing student updates or closure state.
        cur.execute("""SELECT r.report_id,r.student_id,r.category,r.description,
            r.created_at FROM verifyd_safety_training_reports r
            WHERE r.incident_id=%s ORDER BY r.created_at DESC LIMIT 1000""", (incident_id,))
        reports = [dict(v) for v in cur.fetchall()]
        _audit(cur, incident["school_id"], str(actor["id"]), "training_reports_viewed", incident_id)
    return {"ok": True, "mode": "TRAINING ONLY", "reports": reports}


@router.post("/incidents/{incident_id}/resolve")
def resolve_training_incident(incident_id: str, payload: ResolveInput, authorization: Optional[str] = Header(default=None)):
    actor = _actor(authorization)
    _schema()
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("SELECT school_id,status FROM verifyd_safety_training_incidents WHERE incident_id=%s FOR UPDATE", (incident_id,))
        record = cur.fetchone()
        if not record:
            raise HTTPException(404, detail={"error": "incident_not_found"})
        _school_response_authorized(cur, record["school_id"], str(actor["id"]))
        if record["status"] != "active":
            raise HTTPException(409, detail={"error": "already_resolved"})
        cur.execute("""UPDATE verifyd_safety_training_incidents SET status='resolved',
            resolved_at=NOW(),resolved_by=%s,resolution_reason=%s WHERE incident_id=%s""",
            (str(actor["id"]),payload.reason.strip(),incident_id))
        _audit(cur, record["school_id"], str(actor["id"]), "training_official_resolution", incident_id)
    return {"ok": True, "mode": "TRAINING ONLY", "real_notifications_sent": 0}


@router.get("/family/children")
def guardian_training_children(authorization: Optional[str] = Header(default=None)):
    actor = _actor(authorization)
    _schema()
    guardian_id = str(actor["id"])
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("""SELECT g.school_id,g.child_id,i.display_name,i.handle
          FROM verifyd_safety_training_guardians g
          JOIN verifyd_identities i ON i.id=g.child_id
          JOIN verifyd_safety_training_members m
          ON m.school_id=g.school_id AND m.identity_id=g.guardian_id AND m.role='guardian'
          WHERE g.guardian_id=%s ORDER BY i.handle""", (guardian_id,))
        children = [dict(row) for row in cur.fetchall()]
    return {"ok": True, "mode": "TRAINING ONLY", "children": children}


@router.get("/family/my-child/{student_id}")
def guardian_training_view(student_id: str, authorization: Optional[str] = Header(default=None)):
    actor = _actor(authorization)
    _schema()
    guardian_id = str(actor["id"])
    from trust_voice_school_safety_consents import _consent_schema
    _consent_schema()
    with get_db() as conn:
        cur = conn.cursor()
        # A historical guardian link is insufficient once a person loses the
        # verified school guardian role or withdraws their own acknowledgment.
        # This prevents access via a bookmarked direct URL after revocation.
        from trust_voice_school_safety_consents import POLICY_VERSION
        cur.execute("""SELECT g.school_id
           FROM verifyd_safety_training_guardians g
           JOIN verifyd_safety_training_members m
             ON m.school_id=g.school_id AND m.identity_id=g.guardian_id AND m.role='guardian'
           JOIN verifyd_safety_training_signoffs x
             ON x.school_id=g.school_id AND x.actor_id=g.guardian_id
                AND x.subject_id=g.child_id AND x.kind='guardian'
                AND x.policy_version=%s AND x.revoked_at IS NULL
           WHERE g.child_id=%s AND g.guardian_id=%s""",
           (POLICY_VERSION,student_id,guardian_id))
        school_ids = [row["school_id"] for row in cur.fetchall()]
        if not school_ids:
            raise HTTPException(404, detail={"error": "no_verified_guardian_link"})
        cur.execute("""SELECT r.report_id,r.school_id,r.student_status,r.created_at,i.status AS incident_status
           FROM verifyd_safety_training_reports r JOIN verifyd_safety_training_incidents i
           ON r.incident_id=i.incident_id WHERE r.student_id=%s AND r.school_id=ANY(%s)
           ORDER BY r.created_at DESC LIMIT 20""", (student_id,school_ids))
        reports = [dict(r) for r in cur.fetchall()]
        for school_id in school_ids:
            _audit(cur, school_id,guardian_id,"training_guardian_view",student_id)
    return {"ok": True, "mode": "TRAINING ONLY", "student_id": student_id,
        "reports": reports, "location_available": False}
