"""School-scoped, training-only guardian applications and non-emergency updates.

Access is never granted from an email/claim alone. School staff must confirm
guardian authority out of band and record that verification in an audit trail.
No push, SMS, live alerts, student location, or emergency dispatch is provided.
"""
from __future__ import annotations
import threading
import uuid
from typing import Literal, Optional
from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field
from database import get_db
from trust_voice_school_safety_training import _actor, _audit, _id_for_handle, _membership
from trust_voice_school_safety_registration import _application_schema, _school_admin
from trust_voice_school_safety_consents import _consent_schema, POLICY_VERSION

router = APIRouter(prefix="/trust-voice/school-safety", tags=["School Safety Family Pilot"])
_ready = False
_lock = threading.Lock()


def _family_schema():
    global _ready
    _application_schema()
    if _ready:
        return
    with _lock:
        if _ready:
            return
        with get_db() as conn:
            cur = conn.cursor()
            cur.execute("""CREATE TABLE IF NOT EXISTS verifyd_safety_family_requests (
                request_id TEXT PRIMARY KEY,
                school_id TEXT NOT NULL REFERENCES verifyd_safety_training_schools(school_id),
                student_id TEXT NOT NULL REFERENCES verifyd_identities(id),
                guardian_id TEXT NOT NULL REFERENCES verifyd_identities(id),
                relationship TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending'
                    CHECK(status IN ('pending','approved','rejected','cancelled')),
                submitted_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                reviewed_at TIMESTAMPTZ,
                reviewed_by TEXT REFERENCES verifyd_identities(id),
                review_note TEXT NOT NULL DEFAULT '',
                CHECK(student_id <> guardian_id)
            )""")
            cur.execute("""CREATE UNIQUE INDEX IF NOT EXISTS safety_one_pending_family_request
                ON verifyd_safety_family_requests(school_id,student_id,guardian_id)
                WHERE status='pending'""")
            cur.execute("""CREATE INDEX IF NOT EXISTS safety_family_requests_school_status
                ON verifyd_safety_family_requests(school_id,status,submitted_at DESC)""")
            cur.execute("""CREATE TABLE IF NOT EXISTS verifyd_safety_school_updates (
                update_id TEXT PRIMARY KEY,
                school_id TEXT NOT NULL REFERENCES verifyd_safety_training_schools(school_id),
                author_id TEXT NOT NULL REFERENCES verifyd_identities(id),
                title TEXT NOT NULL,
                body TEXT NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                archived_at TIMESTAMPTZ,
                CHECK(length(title) BETWEEN 3 AND 160),
                CHECK(length(body) BETWEEN 5 AND 3000)
            )""")
            cur.execute("""CREATE INDEX IF NOT EXISTS safety_school_updates_by_school
                ON verifyd_safety_school_updates(school_id,created_at DESC)""")
        _ready = True


class FamilyRequest(BaseModel):
    school_id: str = Field(min_length=8, max_length=100)
    student_handle: str = Field(min_length=3, max_length=40)
    relationship: Literal["parent","legal_guardian","authorized_caregiver"]


class FamilyReview(BaseModel):
    decision: Literal["approve","reject"]
    verified_relationship: bool = False
    note: str = Field(min_length=12, max_length=700)


class SchoolUpdate(BaseModel):
    title: str = Field(min_length=3, max_length=160)
    body: str = Field(min_length=5, max_length=3000)


@router.get("/family/schools")
def available_family_schools(authorization: Optional[str] = Header(default=None)):
    _actor(authorization)
    _family_schema()
    with get_db() as conn:
        cur = conn.cursor()
        # Only schools registered through owner-reviewed Principal applications.
        cur.execute("""SELECT DISTINCT s.school_id,s.school_name
            FROM verifyd_safety_training_schools s
            JOIN verifyd_safety_school_applications a
              ON a.school_id=s.school_id AND a.status='approved'
            ORDER BY s.school_name LIMIT 100""")
        rows = [dict(r) for r in cur.fetchall()]
    return {"ok": True, "schools": rows, "training_only": True}


@router.post("/family/requests")
def request_family_link(payload: FamilyRequest, authorization: Optional[str] = Header(default=None)):
    guardian = _actor(authorization)
    _family_schema()
    guardian_id = str(guardian["id"])
    student_id = _id_for_handle(payload.student_handle)
    if guardian_id == student_id:
        raise HTTPException(422, detail={"error":"distinct_people_required"})
    with get_db() as conn:
        cur = conn.cursor()
        # Never expose whether a child exists except to an approved student roster.
        _membership(cur, payload.school_id, student_id, {"student"})
        cur.execute("""SELECT role FROM verifyd_safety_training_members
            WHERE school_id=%s AND identity_id=%s""",(payload.school_id, guardian_id))
        row = cur.fetchone()
        if row and row["role"] != "guardian":
            raise HTTPException(403, detail={"error":"existing_non_guardian_school_role"})
        cur.execute("""SELECT 1 FROM verifyd_safety_training_guardians
            WHERE school_id=%s AND guardian_id=%s AND child_id=%s""",
            (payload.school_id, guardian_id, student_id))
        if cur.fetchone():
            raise HTTPException(409, detail={"error":"family_link_already_approved"})
        cur.execute("""SELECT count(*) AS count FROM verifyd_safety_family_requests
            WHERE guardian_id=%s AND submitted_at>NOW()-INTERVAL '24 hours'""",(guardian_id,))
        if cur.fetchone()["count"] >= 5:
            raise HTTPException(429, detail={"error":"family_request_rate_limit"})
        request_id = "ssfam_" + uuid.uuid4().hex
        try:
            cur.execute("""INSERT INTO verifyd_safety_family_requests
                (request_id,school_id,student_id,guardian_id,relationship)
                VALUES(%s,%s,%s,%s,%s)""",
                (request_id,payload.school_id,student_id,guardian_id,payload.relationship))
        except Exception as exc:
            # A duplicate request must not create new approval privileges.
            if getattr(exc,"pgcode",None)=="23505":
                raise HTTPException(409,detail={"error":"family_request_already_pending"})
            raise
        _audit(cur,payload.school_id,guardian_id,"family_link_requested",request_id)
    return {"ok":True,"request_id":request_id,"status":"pending","emergency_alerts_active":False}


@router.get("/family/requests/mine")
def my_family_requests(authorization: Optional[str] = Header(default=None)):
    guardian = _actor(authorization)
    _family_schema()
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("""SELECT r.request_id,r.school_id,s.school_name,
            r.relationship,r.status,r.submitted_at,r.reviewed_at
            FROM verifyd_safety_family_requests r
            JOIN verifyd_safety_training_schools s USING(school_id)
            WHERE r.guardian_id=%s ORDER BY r.submitted_at DESC LIMIT 30""",(str(guardian["id"]),))
        rows=[dict(r) for r in cur.fetchall()]
    return {"ok":True,"requests":rows,"training_only":True}


@router.get("/schools/{school_id}/family-requests")
def staff_family_requests(school_id: str, authorization: Optional[str] = Header(default=None)):
    actor = _actor(authorization)
    _family_schema()
    with get_db() as conn:
        cur = conn.cursor()
        _school_admin(cur,school_id,str(actor["id"]))
        cur.execute("""SELECT r.request_id,r.relationship,r.status,r.submitted_at,
            st.handle AS student_handle,ga.handle AS guardian_handle
            FROM verifyd_safety_family_requests r
            JOIN verifyd_identities st ON st.id=r.student_id
            JOIN verifyd_identities ga ON ga.id=r.guardian_id
            WHERE r.school_id=%s ORDER BY r.submitted_at DESC LIMIT 100""",(school_id,))
        rows=[dict(r) for r in cur.fetchall()]
    return {"ok":True,"requests":rows,"training_only":True}


@router.post("/schools/{school_id}/family-requests/{request_id}/review")
def review_family_request(school_id: str,request_id: str,payload: FamilyReview,
                          authorization: Optional[str] = Header(default=None)):
    actor = _actor(authorization)
    _family_schema()
    actor_id = str(actor["id"])
    with get_db() as conn:
        cur = conn.cursor()
        _school_admin(cur,school_id,actor_id)
        cur.execute("""SELECT * FROM verifyd_safety_family_requests
            WHERE request_id=%s AND school_id=%s FOR UPDATE""",(request_id,school_id))
        r = cur.fetchone()
        if not r:
            raise HTTPException(404,detail={"error":"family_request_not_found"})
        if r["status"]!="pending":
            raise HTTPException(409,detail={"error":"family_request_already_reviewed"})
        if r["guardian_id"]==actor_id or r["student_id"]==actor_id:
            raise HTTPException(403,detail={"error":"cannot_approve_own_family_link"})
        if payload.decision=="approve":
            if not payload.verified_relationship:
                raise HTTPException(422,detail={"error":"independent_guardianship_verification_required"})
            _membership(cur,school_id,str(r["student_id"]),{"student"})
            cur.execute("""SELECT role FROM verifyd_safety_training_members
                WHERE school_id=%s AND identity_id=%s FOR UPDATE""",
                (school_id,str(r["guardian_id"])))
            existing=cur.fetchone()
            if existing and existing["role"]!="guardian":
                raise HTTPException(403,detail={"error":"cannot_replace_staff_or_student_role"})
            if not existing:
                cur.execute("""INSERT INTO verifyd_safety_training_members
                    (school_id,identity_id,role,verified_by) VALUES(%s,%s,'guardian',%s)""",
                    (school_id,str(r["guardian_id"]),actor_id))
            cur.execute("""INSERT INTO verifyd_safety_training_guardians
                (school_id,child_id,guardian_id,verified_by) VALUES(%s,%s,%s,%s)
                ON CONFLICT DO NOTHING""",
                (school_id,str(r["student_id"]),str(r["guardian_id"]),actor_id))
            _audit(cur,school_id,actor_id,"family_link_approved",request_id)
        else:
            _audit(cur,school_id,actor_id,"family_link_rejected",request_id)
        cur.execute("""UPDATE verifyd_safety_family_requests
            SET status=%s,reviewed_at=NOW(),reviewed_by=%s,review_note=%s
            WHERE request_id=%s""",
            ("approved" if payload.decision=="approve" else "rejected",
             actor_id,payload.note.strip(),request_id))
    return {"ok":True,"status":payload.decision,"school_only":True,
            "emergency_alerts_active":False}


@router.get("/family/portal")
def family_portal(authorization: Optional[str] = Header(default=None)):
    actor = _actor(authorization)
    _family_schema()
    guardian_id = str(actor["id"])
    with get_db() as conn:
        cur = conn.cursor()
        # Both a current guardian membership and current linked-child consent
        # are needed. Never reveal an unlinked student's details.
        cur.execute("""SELECT g.school_id,s.school_name,g.child_id,
            child.handle AS child_handle,child.display_name AS child_name
            FROM verifyd_safety_training_guardians g
            JOIN verifyd_safety_training_schools s ON s.school_id=g.school_id
            JOIN verifyd_safety_training_members m ON m.school_id=g.school_id
              AND m.identity_id=g.guardian_id AND m.role='guardian'
            JOIN verifyd_identities child ON child.id=g.child_id
            JOIN verifyd_safety_training_signoffs x ON x.school_id=g.school_id
              AND x.actor_id=g.guardian_id AND x.subject_id=g.child_id
              AND x.kind='guardian' AND x.policy_version=%s AND x.revoked_at IS NULL
            WHERE g.guardian_id=%s ORDER BY s.school_name,child.handle""",
            (POLICY_VERSION,guardian_id))
        children=[dict(r) for r in cur.fetchall()]
        school_ids=sorted({c["school_id"] for c in children})
        updates=[]
        if school_ids:
            cur.execute("""SELECT u.update_id,u.school_id,u.title,u.body,u.created_at,
                s.school_name
                FROM verifyd_safety_school_updates u
                JOIN verifyd_safety_training_schools s USING(school_id)
                WHERE u.school_id=ANY(%s) AND u.archived_at IS NULL
                ORDER BY u.created_at DESC LIMIT 40""",(school_ids,))
            updates=[dict(r) for r in cur.fetchall()]
    return {"ok":True,"children":children,"updates":updates,
            "training_only":True,"live_emergency_delivery":False}


@router.get("/schools/{school_id}/updates")
def staff_school_updates(school_id: str, authorization: Optional[str] = Header(default=None)):
    actor=_actor(authorization)
    _family_schema()
    with get_db() as conn:
        cur=conn.cursor()
        _school_admin(cur,school_id,str(actor["id"]))
        cur.execute("""SELECT update_id,title,body,created_at FROM verifyd_safety_school_updates
            WHERE school_id=%s AND archived_at IS NULL
            ORDER BY created_at DESC LIMIT 50""",(school_id,))
        rows=[dict(r) for r in cur.fetchall()]
    return {"ok":True,"updates":rows,"training_only":True}


@router.post("/schools/{school_id}/updates")
def publish_school_update(school_id: str,payload: SchoolUpdate,
                          authorization: Optional[str] = Header(default=None)):
    actor=_actor(authorization)
    _family_schema()
    if any(term in (payload.title+" "+payload.body).lower() for term in
           ("911","emergency alert","active shooter","evacuate now","lockdown now")):
        raise HTTPException(422,detail={"error":"use_official_emergency_channels"})
    with get_db() as conn:
        cur=conn.cursor()
        _school_admin(cur,school_id,str(actor["id"]))
        update_id="ssupd_"+uuid.uuid4().hex
        cur.execute("""INSERT INTO verifyd_safety_school_updates
            (update_id,school_id,author_id,title,body)
            VALUES(%s,%s,%s,%s,%s)""",
            (update_id,school_id,str(actor["id"]),payload.title.strip(),payload.body.strip()))
        _audit(cur,school_id,str(actor["id"]),"school_general_update_published",update_id)
    return {"ok":True,"update_id":update_id,"in_app_only":True,
            "emergency_alerts_active":False}
