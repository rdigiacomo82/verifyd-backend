"""VeriFYD Circle Classroom — isolated opt-in assignment pilot.

Never imports or mutates existing Homework, messages, calls, or voicemail.
Teachers may opt into adult-only, self-reported educator registration when
explicitly enabled. Enrollment still requires classroom teacher approval. Raw file uploads are stored in a separate private R2 quarantine bucket
and are NEVER returned, downloaded, rendered, analyzed or considered safe here.
Feature flag: VERIFYD_CIRCLE_CLASSROOM_ENABLED=1 (default OFF).
"""
from __future__ import annotations

import hashlib
import os
import re
import secrets
import threading
import uuid
from datetime import datetime, timedelta, timezone
from typing import Literal, Optional

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field

from database import get_db
from trust_voice_messages import _identity_from_bearer

router = APIRouter(prefix="/trust-voice/classroom", tags=["Circle Classroom Pilot"])
_READY = False
_LOCK = threading.Lock()
_BANDS = frozenset(("K-2", "3-5", "6-8", "9-12", "Adult", "Custom"))


def _enabled():
    return os.getenv("VERIFYD_CIRCLE_CLASSROOM_ENABLED", "0").lower().strip() in ("1", "true", "yes", "on")


def _require():
    if not _enabled():
        raise HTTPException(404, detail={"error": "classroom_unavailable"})


def _identity(authorization):
    _require()
    person = _identity_from_bearer(authorization)
    if person.get("status") != "active":
        raise HTTPException(401, detail={"error": "session_required"})
    return str(person["id"])


def _operator_teacher(identity_id):
    return identity_id in {value.strip() for value in os.getenv(
        "VERIFYD_CIRCLE_CLASSROOM_TEACHER_IDS", "").split(",") if value.strip()}


def _teacher_allowed(identity_id, cur=None):
    if _operator_teacher(identity_id):
        return True
    _schema()
    if cur is not None:
        cur.execute("SELECT status FROM verifyd_circle_teacher_profiles WHERE identity_id=%s", (identity_id,))
        row = cur.fetchone()
    else:
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT status FROM verifyd_circle_teacher_profiles WHERE identity_id=%s", (identity_id,))
            row = cursor.fetchone()
    return bool(row and row["status"] == "active")


def _approved_teacher(identity_id, cur=None):
    if not _teacher_allowed(identity_id, cur):
        raise HTTPException(403, detail={"error": "teacher_registration_required"})


def _schema():
    global _READY
    if _READY:
        return
    with _LOCK:
        if _READY:
            return
        with get_db() as conn:
            cur = conn.cursor()
            cur.execute("""CREATE TABLE IF NOT EXISTS verifyd_circle_classes (
                class_id TEXT PRIMARY KEY,
                teacher_id TEXT NOT NULL REFERENCES verifyd_identities(id) ON DELETE CASCADE,
                title TEXT NOT NULL, subject TEXT NOT NULL, grade_band TEXT NOT NULL,
                join_hash TEXT NOT NULL UNIQUE,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )""")
            cur.execute("""CREATE INDEX IF NOT EXISTS idx_circle_class_teacher
                ON verifyd_circle_classes(teacher_id)""")
            cur.execute("""CREATE TABLE IF NOT EXISTS verifyd_circle_class_members (
                class_id TEXT NOT NULL REFERENCES verifyd_circle_classes(class_id) ON DELETE CASCADE,
                student_id TEXT NOT NULL REFERENCES verifyd_identities(id) ON DELETE CASCADE,
                status TEXT NOT NULL DEFAULT 'pending' CHECK(status IN ('pending','approved','removed')),
                requested_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                approved_at TIMESTAMPTZ,
                PRIMARY KEY(class_id,student_id)
            )""")
            cur.execute("""CREATE INDEX IF NOT EXISTS idx_circle_class_student
                ON verifyd_circle_class_members(student_id,status)""")
            cur.execute("""CREATE TABLE IF NOT EXISTS verifyd_circle_class_assignments (
                assignment_id TEXT PRIMARY KEY,
                class_id TEXT NOT NULL REFERENCES verifyd_circle_classes(class_id) ON DELETE CASCADE,
                author_id TEXT NOT NULL REFERENCES verifyd_identities(id) ON DELETE CASCADE,
                title TEXT NOT NULL, instructions TEXT NOT NULL,
                due_at TIMESTAMPTZ,
                publish_at TIMESTAMPTZ,
                points_possible INTEGER NOT NULL DEFAULT 100 CHECK(points_possible BETWEEN 1 AND 1000),
                status TEXT NOT NULL DEFAULT 'draft' CHECK(status IN ('draft','published','archived')),
                rank_points INTEGER NOT NULL DEFAULT 0 CHECK(rank_points BETWEEN 0 AND 5),
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )""")
            cur.execute("ALTER TABLE verifyd_circle_class_assignments ADD COLUMN IF NOT EXISTS publish_at TIMESTAMPTZ")
            cur.execute("ALTER TABLE verifyd_circle_class_assignments ADD COLUMN IF NOT EXISTS rank_points INTEGER NOT NULL DEFAULT 0 CHECK(rank_points BETWEEN 0 AND 5)")
            cur.execute("""CREATE INDEX IF NOT EXISTS idx_circle_class_assignments_class
                ON verifyd_circle_class_assignments(class_id,status)""")
            cur.execute("""CREATE TABLE IF NOT EXISTS verifyd_circle_class_submissions (
                assignment_id TEXT NOT NULL REFERENCES verifyd_circle_class_assignments(assignment_id) ON DELETE CASCADE,
                student_id TEXT NOT NULL REFERENCES verifyd_identities(id) ON DELETE CASCADE,
                response TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL DEFAULT 'draft' CHECK(status IN ('draft','submitted','returned')),
                points_awarded INTEGER,
                feedback TEXT NOT NULL DEFAULT '',
                submitted_at TIMESTAMPTZ, returned_at TIMESTAMPTZ,
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                PRIMARY KEY(assignment_id,student_id)
            )""")
            cur.execute("""CREATE TABLE IF NOT EXISTS verifyd_circle_teacher_profiles (
                identity_id TEXT PRIMARY KEY REFERENCES verifyd_identities(id) ON DELETE CASCADE,
                display_name TEXT NOT NULL,
                organization TEXT NOT NULL DEFAULT '',
                teaching_role TEXT NOT NULL,
                subjects TEXT NOT NULL DEFAULT '',
                credentials TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active','suspended')),
                adult_attested_at TIMESTAMPTZ NOT NULL,
                terms_accepted_at TIMESTAMPTZ NOT NULL,
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )""")
        _READY = True


def _teacher(cur, class_id, identity_id):
    cur.execute("SELECT teacher_id FROM verifyd_circle_classes WHERE class_id=%s", (class_id,))
    record = cur.fetchone()
    if not record or record["teacher_id"] != identity_id:
        raise HTTPException(404, detail={"error": "class_not_found"})
    _approved_teacher(identity_id, cur)
    return record


def _member(cur, class_id, identity_id):
    cur.execute("""SELECT c.teacher_id, m.status FROM verifyd_circle_classes c
        LEFT JOIN verifyd_circle_class_members m ON m.class_id=c.class_id AND m.student_id=%s
        WHERE c.class_id=%s""", (identity_id,class_id))
    record = cur.fetchone()
    if not record or (record["teacher_id"] != identity_id and record["status"] != "approved"):
        raise HTTPException(404, detail={"error": "class_not_found"})
    if record["teacher_id"] == identity_id:
        _approved_teacher(identity_id, cur)  # Revoked educators cannot read student content.
    return record


def _assignment(cur, class_id, assignment_id):
    cur.execute("""SELECT assignment_id,class_id,title,instructions,due_at,publish_at,points_possible,rank_points,status
        FROM verifyd_circle_class_assignments WHERE class_id=%s AND assignment_id=%s""",
        (class_id,assignment_id))
    record = cur.fetchone()
    if not record:
        raise HTTPException(404, detail={"error": "assignment_not_found"})
    return record


def _display(row):
    result = dict(row)
    for field in ("created_at", "requested_at", "approved_at", "due_at", "publish_at", "submitted_at", "returned_at", "updated_at"):
        value = result.get(field)
        if isinstance(value, datetime):
            result[field] = value.isoformat()
    return result


class TeacherRegistration(BaseModel):
    display_name: str = Field(min_length=2, max_length=100)
    organization: str = Field(default="", max_length=120)
    teaching_role: Literal["Teacher", "Tutor", "Homeschool Educator", "Therapist", "Instructor", "Other"]
    subjects: str = Field(default="", max_length=160)
    credentials: str = Field(default="", max_length=500)
    adult_attestation: bool = False
    accept_terms: bool = False


@router.get("/teachers/me")
def my_teacher_profile(authorization: Optional[str] = Header(default=None)):
    identity_id = _identity(authorization)
    _schema()
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("""SELECT display_name,organization,teaching_role,subjects,credentials,status
            FROM verifyd_circle_teacher_profiles WHERE identity_id=%s""", (identity_id,))
        profile = cur.fetchone()
    allowed = _operator_teacher(identity_id) or bool(profile and profile["status"] == "active")
    return {"can_create_classes": allowed,
            "status": "operator_approved" if _operator_teacher(identity_id) else (
                "self_registered" if allowed else "suspended" if profile else "not_registered"),
            "independently_verified": False,
            "registration_available": os.getenv("VERIFYD_CIRCLE_CLASSROOM_SELF_REGISTER_ENABLED", "0").strip().lower() in ("1", "true", "yes", "on"),
            "profile": dict(profile) if profile else None}


@router.post("/teachers/register")
def register_teacher(payload: TeacherRegistration, authorization: Optional[str] = Header(default=None)):
    identity_id = _identity(authorization)
    if os.getenv("VERIFYD_CIRCLE_CLASSROOM_SELF_REGISTER_ENABLED", "0").strip().lower() not in ("1", "true", "yes", "on"):
        raise HTTPException(403, detail={"error": "teacher_self_registration_disabled"})
    if not payload.adult_attestation or not payload.accept_terms:
        raise HTTPException(422, detail={"error": "adult_attestation_and_terms_required"})
    _schema()
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("SELECT status FROM verifyd_circle_teacher_profiles WHERE identity_id=%s FOR UPDATE", (identity_id,))
        row = cur.fetchone()
        if row and row["status"] == "suspended":
            raise HTTPException(403, detail={"error": "teacher_account_suspended"})
        cur.execute("""INSERT INTO verifyd_circle_teacher_profiles
            (identity_id,display_name,organization,teaching_role,subjects,credentials,adult_attested_at,terms_accepted_at)
            VALUES (%s,%s,%s,%s,%s,%s,NOW(),NOW())
            ON CONFLICT (identity_id) DO UPDATE SET
                display_name=EXCLUDED.display_name,organization=EXCLUDED.organization,
                teaching_role=EXCLUDED.teaching_role,subjects=EXCLUDED.subjects,
                credentials=EXCLUDED.credentials,updated_at=NOW()
            WHERE verifyd_circle_teacher_profiles.status='active'""",
            (identity_id,payload.display_name.strip(),payload.organization.strip(),
             payload.teaching_role,payload.subjects.strip(),payload.credentials.strip()))
    return {"status": "self_registered", "can_create_classes": True,
            "independently_verified": False,
            "note": "Credentials are self-reported. Restricted adult-only text classroom pilot."}


class ClassCreate(BaseModel):
    title: str = Field(min_length=3, max_length=120)
    subject: str = Field(min_length=2, max_length=80)
    grade_band: Literal["K-2","3-5","6-8","9-12","Adult","Custom"]


class JoinRequest(BaseModel):
    join_code: str = Field(min_length=16, max_length=48)
    adult_attestation: bool = False


class AssignmentCreate(BaseModel):
    title: str = Field(min_length=3, max_length=160)
    instructions: str = Field(min_length=1, max_length=16000)
    due_at: Optional[datetime] = None
    points_possible: int = Field(default=100, ge=1, le=1000)
    rank_points: int = Field(default=0, ge=0, le=5)  # 0 for legacy clients; 1–5 when explicitly selected


class AssignmentEdit(BaseModel):
    """Only unpublished assignments may be edited; grades and point rewards are locked after publish."""
    title: str = Field(min_length=3, max_length=160)
    instructions: str = Field(min_length=1, max_length=16000)
    due_at: Optional[datetime] = None
    rank_points: Optional[int] = Field(default=None, ge=0, le=5)


class ScheduleAssignment(BaseModel):
    publish_at: datetime


class Work(BaseModel):
    response: str = Field(default="", max_length=30000)
    action: Literal["save","turn_in"] = "save"


class Review(BaseModel):
    points_awarded: int = Field(ge=0, le=1000)
    feedback: str = Field(default="", max_length=12000)


def _prepare_classroom_audience():
    # Set up optional assignment restrictions before opening a student's transaction.
    from trust_voice_classroom_invitations import prepare_audience
    prepare_audience()


def _classroom_assignment_visible(cur, assignment_id, student_id):
    from trust_voice_classroom_invitations import audience_allows
    return audience_allows(cur, assignment_id, student_id)


@router.get("/health")
def health():
    _require()
    return {"service":"circle_classroom_pilot","enabled":True,
            "teacher_access":"operator-or-self-registered-adult","files":"quarantine_only",
            "scan_service":"not_connected","unsafe_files_downloadable":False}


@router.post("/classes")
def create_class(payload: ClassCreate, authorization: Optional[str] = Header(default=None)):
    user = _identity(authorization)
    _approved_teacher(user)
    if payload.grade_band != "Adult" and not _operator_teacher(user):
        raise HTTPException(403, detail={"error": "self_registered_adult_classes_only"})
    _schema()
    class_id = str(uuid.uuid4())
    join_code = secrets.token_urlsafe(24)
    join_hash = hashlib.sha256(join_code.encode()).hexdigest()
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("SELECT count(*) AS count FROM verifyd_circle_classes WHERE teacher_id=%s", (user,))
        if cur.fetchone()["count"] >= 50:
            raise HTTPException(429, detail={"error":"teacher_class_limit"})
        cur.execute("""INSERT INTO verifyd_circle_classes
            (class_id,teacher_id,title,subject,grade_band,join_hash)
            VALUES (%s,%s,%s,%s,%s,%s)""",
            (class_id,user,payload.title,payload.subject,payload.grade_band,join_hash))
    return {"class_id":class_id,"title":payload.title,"subject":payload.subject,
            "grade_band":payload.grade_band,"join_code":join_code,
            "join_instruction":"Keep the code private. Teacher approval is required after joining."}


@router.post("/classes/join")
def request_to_join(payload: JoinRequest, authorization: Optional[str] = Header(default=None)):
    user = _identity(authorization)
    _schema()
    digest = hashlib.sha256(payload.join_code.encode()).hexdigest()
    with get_db() as conn:
        cur=conn.cursor()
        cur.execute("SELECT class_id,teacher_id,grade_band FROM verifyd_circle_classes WHERE join_hash=%s", (digest,))
        record=cur.fetchone()
        if not record:
            raise HTTPException(404, detail={"error":"invalid_class_code"})
        if record["grade_band"] == "Adult" and not payload.adult_attestation:
            raise HTTPException(422, detail={"error": "adult_join_attestation_required"})
        if record["teacher_id"] == user:
            raise HTTPException(409, detail={"error":"already_teacher"})
        cur.execute("""SELECT count(*) AS n FROM verifyd_circle_class_members WHERE student_id=%s AND status='pending'""", (user,))
        if cur.fetchone()["n"] >= 20:
            raise HTTPException(429, detail={"error":"too_many_pending_requests"})
        cur.execute("""INSERT INTO verifyd_circle_class_members(class_id,student_id)
            VALUES (%s,%s) ON CONFLICT(class_id,student_id) DO UPDATE
            SET status=CASE WHEN verifyd_circle_class_members.status='removed'
            THEN 'pending' ELSE verifyd_circle_class_members.status END
            RETURNING status""", (record["class_id"],user))
        status=cur.fetchone()["status"]
    return {"status":status,"teacher_approval_required":status=="pending"}


@router.get("/classes")
def my_classes(authorization: Optional[str] = Header(default=None)):
    user=_identity(authorization)
    _schema()
    teacher_allowed = _teacher_allowed(user)
    with get_db() as conn:
        cur=conn.cursor()
        cur.execute("""SELECT c.class_id,c.title,c.subject,c.grade_band,c.created_at,
            CASE WHEN c.teacher_id=%s THEN 'teacher' ELSE 'student' END AS my_role
            FROM verifyd_circle_classes c LEFT JOIN verifyd_circle_class_members m
            ON m.class_id=c.class_id AND m.student_id=%s
            WHERE (c.teacher_id=%s AND %s) OR m.status='approved'
            ORDER BY c.created_at DESC LIMIT 100""",(user,user,user,teacher_allowed))
        rows=[_display(x) for x in cur.fetchall()]
    return {"classes":rows}


@router.delete("/classes/{class_id}/members/me")
def leave_class_and_erase_my_work(class_id: str, authorization: Optional[str] = Header(default=None)):
    """Member-directed deletion of their own class enrollment and submissions."""
    user = _identity(authorization)
    _schema()
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("""SELECT 1 FROM verifyd_circle_class_members
                       WHERE class_id=%s AND student_id=%s""", (class_id, user))
        if not cur.fetchone():
            raise HTTPException(404, detail={"error": "membership_not_found"})
        cur.execute("""DELETE FROM verifyd_circle_class_submissions
                       WHERE student_id=%s AND assignment_id IN
                       (SELECT assignment_id FROM verifyd_circle_class_assignments WHERE class_id=%s)""",
                    (user, class_id))
        cur.execute("""DELETE FROM verifyd_circle_class_members
                       WHERE student_id=%s AND class_id=%s""", (user, class_id))
    return {"left": True, "my_classwork_deleted": True}


@router.delete("/classes/{class_id}")
def delete_teacher_class(class_id: str, authorization: Optional[str] = Header(default=None)):
    """Erase an entire teacher-owned classroom and related text-based records."""
    user = _identity(authorization)
    _approved_teacher(user)
    _schema()
    with get_db() as conn:
        cur = conn.cursor()
        _teacher(cur, class_id, user)
        cur.execute("DELETE FROM verifyd_circle_classes WHERE class_id=%s AND teacher_id=%s RETURNING class_id",
                    (class_id, user))
        if not cur.fetchone():
            raise HTTPException(404, detail={"error": "class_not_found"})
    return {"deleted": True, "class_id": class_id}


@router.get("/classes/{class_id}/members")
def members(class_id: str, authorization: Optional[str] = Header(default=None)):
    user=_identity(authorization)
    # A revoked teacher must not retain read access to enrolled student IDs.
    _approved_teacher(user)
    _schema()
    with get_db() as conn:
        cur=conn.cursor()
        _teacher(cur,class_id,user)
        cur.execute("""SELECT m.student_id,m.status,m.requested_at,m.approved_at,i.handle AS student_handle
            FROM verifyd_circle_class_members m
            JOIN verifyd_identities i ON i.id=m.student_id AND i.status='active'
            WHERE m.class_id=%s ORDER BY m.requested_at LIMIT 300""",(class_id,))
        rows=[_display(x) for x in cur.fetchall()]
    return {"members":rows}


@router.post("/classes/{class_id}/members/{student_id}/approve")
def approve_member(class_id: str,student_id: str, authorization: Optional[str] = Header(default=None)):
    user=_identity(authorization)
    _approved_teacher(user)
    _schema()
    with get_db() as conn:
        cur=conn.cursor()
        _teacher(cur,class_id,user)
        cur.execute("""UPDATE verifyd_circle_class_members SET status='approved',approved_at=NOW()
            WHERE class_id=%s AND student_id=%s AND status='pending' RETURNING student_id""",
            (class_id,student_id))
        if not cur.fetchone():
            raise HTTPException(404,detail={"error":"pending_request_not_found"})
    return {"approved":True}


@router.post("/classes/{class_id}/assignments")
def create_assignment(class_id: str,payload: AssignmentCreate,
                      authorization: Optional[str] = Header(default=None)):
    user=_identity(authorization)
    _approved_teacher(user)
    _schema()
    assignment_id=str(uuid.uuid4())
    with get_db() as conn:
        cur=conn.cursor()
        _teacher(cur,class_id,user)
        cur.execute("""INSERT INTO verifyd_circle_class_assignments
            (assignment_id,class_id,author_id,title,instructions,due_at,points_possible,rank_points)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s)""",(assignment_id,class_id,user,payload.title,
            payload.instructions,payload.due_at,payload.points_possible,payload.rank_points))
    return {"assignment_id":assignment_id,"status":"draft"}


@router.put("/classes/{class_id}/assignments/{assignment_id}")
def edit_assignment(class_id: str, assignment_id: str, payload: AssignmentEdit,
                    authorization: Optional[str] = Header(default=None)):
    """Edit the teacher's draft; published student-visible content is immutable."""
    user = _identity(authorization)
    _approved_teacher(user)
    _schema()
    with get_db() as conn:
        cur = conn.cursor()
        _teacher(cur, class_id, user)
        cur.execute("""UPDATE verifyd_circle_class_assignments
            SET title=%s, instructions=%s, due_at=%s, rank_points=COALESCE(%s,rank_points)
            WHERE assignment_id=%s AND class_id=%s AND status='draft'
            RETURNING assignment_id,class_id,title,instructions,due_at,publish_at,points_possible,rank_points,status""",
            (payload.title.strip(), payload.instructions.strip(), payload.due_at, payload.rank_points, assignment_id, class_id))
        row = cur.fetchone()
        if not row:
            raise HTTPException(409, detail={"error": "assignment_not_draft"})
    return {"assignment": _display(row)}


@router.post("/classes/{class_id}/assignments/{assignment_id}/publish")
def publish_assignment(class_id: str,assignment_id: str,
                       authorization: Optional[str] = Header(default=None)):
    user=_identity(authorization)
    _approved_teacher(user)
    _schema()
    with get_db() as conn:
        cur=conn.cursor()
        _teacher(cur,class_id,user)
        cur.execute("""UPDATE verifyd_circle_class_assignments SET status='published',publish_at=NULL
            WHERE class_id=%s AND assignment_id=%s AND status='draft' RETURNING assignment_id""",
            (class_id,assignment_id))
        if not cur.fetchone():
            raise HTTPException(409,detail={"error":"assignment_not_draft"})
    return {"status":"published"}


@router.post("/classes/{class_id}/assignments/{assignment_id}/schedule")
def schedule_assignment(class_id: str, assignment_id: str, payload: ScheduleAssignment,
                        authorization: Optional[str] = Header(default=None)):
    user = _identity(authorization)
    _approved_teacher(user)
    _schema()
    when = payload.publish_at
    if when.tzinfo is None or when.utcoffset() is None:
        raise HTTPException(400, detail={"error": "schedule_requires_timezone"})
    now = datetime.now(timezone.utc)
    when = when.astimezone(timezone.utc)
    if not now + timedelta(minutes=1) <= when <= now + timedelta(days=365):
        raise HTTPException(400, detail={"error": "invalid_schedule_time"})
    with get_db() as conn:
        cur = conn.cursor()
        _teacher(cur, class_id, user)
        rec = _assignment(cur, class_id, assignment_id)
        if rec["status"] != "draft":
            raise HTTPException(409, detail={"error": "assignment_not_draft"})
        if rec["due_at"] and rec["due_at"] < when:
            raise HTTPException(400, detail={"error": "due_date_before_publish"})
        cur.execute("""UPDATE verifyd_circle_class_assignments
            SET status='published',publish_at=%s
            WHERE class_id=%s AND assignment_id=%s AND status='draft'
            RETURNING assignment_id""", (when, class_id, assignment_id))
        if not cur.fetchone():
            raise HTTPException(409, detail={"error": "assignment_not_draft"})
    return {"status": "scheduled", "publish_at": when.isoformat()}


@router.get("/classes/{class_id}/assignments")
def list_assignments(class_id: str, authorization: Optional[str] = Header(default=None)):
    user=_identity(authorization)
    _prepare_classroom_audience()
    _schema()
    with get_db() as conn:
        cur=conn.cursor()
        rec=_member(cur,class_id,user)
        show_drafts=rec["teacher_id"]==user
        cur.execute("""SELECT assignment_id,title,instructions,due_at,publish_at,points_possible,rank_points,status,created_at
            FROM verifyd_circle_class_assignments WHERE class_id=%s
            AND (%s OR (status='published' AND (publish_at IS NULL OR publish_at<=NOW())))
            ORDER BY created_at DESC LIMIT 150""",
            (class_id,show_drafts))
        rows=[_display(x) for x in cur.fetchall()]
        if not show_drafts:
            rows=[item for item in rows if _classroom_assignment_visible(cur,item["assignment_id"],user)]
    return {"assignments":rows}


@router.put("/classes/{class_id}/assignments/{assignment_id}/my-work")
def save_or_turn_in(class_id: str,assignment_id: str,payload: Work,
                    authorization: Optional[str] = Header(default=None)):
    user=_identity(authorization)
    _prepare_classroom_audience()
    _schema()
    with get_db() as conn:
        cur=conn.cursor()
        rec=_member(cur,class_id,user)
        if rec["teacher_id"]==user:
            raise HTTPException(403,detail={"error":"students_only"})
        assignment=_assignment(cur,class_id,assignment_id)
        if not _classroom_assignment_visible(cur,assignment_id,user):
            raise HTTPException(404,detail={"error":"assignment_not_found"})
        if (assignment["status"] != "published" or
                (assignment["publish_at"] is not None and assignment["publish_at"] > datetime.now(timezone.utc))):
            raise HTTPException(409,detail={"error":"assignment_unavailable"})
        cur.execute("""INSERT INTO verifyd_circle_class_submissions(assignment_id,student_id)
            VALUES (%s,%s) ON CONFLICT DO NOTHING""",(assignment_id,user))
        cur.execute("""SELECT status FROM verifyd_circle_class_submissions
            WHERE assignment_id=%s AND student_id=%s FOR UPDATE""",(assignment_id,user))
        previous=cur.fetchone()
        if previous["status"]=="submitted":
            raise HTTPException(409,detail={"error":"already_turned_in"})
        if not payload.response.strip() and payload.action=="turn_in":
            raise HTTPException(400,detail={"error":"submission_text_required"})
        status="submitted" if payload.action=="turn_in" else "draft"
        cur.execute("""UPDATE verifyd_circle_class_submissions SET
            response=%s,status=%s,points_awarded=NULL,feedback='',
            submitted_at=CASE WHEN %s='submitted' THEN NOW() ELSE NULL END,
            returned_at=NULL,updated_at=NOW()
            WHERE assignment_id=%s AND student_id=%s
            RETURNING assignment_id,student_id,status,points_awarded,feedback,
                submitted_at,returned_at,updated_at""",
            (payload.response,status,status,assignment_id,user))
        result=_display(cur.fetchone())
    return {"work":result}


@router.get("/classes/{class_id}/assignments/{assignment_id}/my-work")
def my_work(class_id: str,assignment_id: str,authorization: Optional[str] = Header(default=None)):
    user=_identity(authorization)
    _prepare_classroom_audience()
    _schema()
    with get_db() as conn:
        cur=conn.cursor()
        membership=_member(cur,class_id,user)
        _assignment(cur,class_id,assignment_id)
        if membership["teacher_id"] != user and not _classroom_assignment_visible(cur,assignment_id,user):
            raise HTTPException(404,detail={"error":"assignment_not_found"})
        cur.execute("""SELECT assignment_id,status,response,points_awarded,feedback,
            submitted_at,returned_at,updated_at FROM verifyd_circle_class_submissions
            WHERE assignment_id=%s AND student_id=%s""",(assignment_id,user))
        row=cur.fetchone()
    return {"work":_display(row) if row else None}


@router.get("/classes/{class_id}/assignments/{assignment_id}/submissions")
def teacher_submissions(class_id: str,assignment_id: str,
                        authorization: Optional[str] = Header(default=None)):
    user=_identity(authorization)
    _approved_teacher(user)
    _schema()
    with get_db() as conn:
        cur=conn.cursor()
        _teacher(cur,class_id,user)
        _assignment(cur,class_id,assignment_id)
        cur.execute("""SELECT s.student_id, i.handle AS student_handle,
            s.status,s.response,s.points_awarded,s.feedback,
            s.submitted_at,s.returned_at,s.updated_at
            FROM verifyd_circle_class_submissions s
            JOIN verifyd_identities i ON i.id=s.student_id
            WHERE s.assignment_id=%s AND s.status IN ('submitted','returned')
            ORDER BY s.submitted_at DESC LIMIT 300""",(assignment_id,))
        rows=[_display(x) for x in cur.fetchall()]
    return {"submissions":rows}


@router.post("/classes/{class_id}/assignments/{assignment_id}/submissions/{student_id}/review")
def review_submission(class_id: str,assignment_id: str,student_id: str,payload: Review,
                      authorization: Optional[str] = Header(default=None)):
    """Teacher review completes a submitted assignment and awards configured Rank credit once.

    Rank credit is separate from academic grade points. It is never computed from
    client-supplied point amounts, and is atomically committed with the review.
    """
    user=_identity(authorization)
    _approved_teacher(user)
    _schema()
    # Ensure rank tables before starting the review transaction; never open a
    # nested connection while a submission row is locked.
    from trust_voice_ranks import _schema as ensure_rank_schema, enabled as ranking_enabled
    rank_available = ranking_enabled()
    if rank_available:
        ensure_rank_schema()
    points_added = 0
    with get_db() as conn:
        cur=conn.cursor()
        _teacher(cur,class_id,user)
        assignment=_assignment(cur,class_id,assignment_id)
        if payload.points_awarded > assignment["points_possible"]:
            raise HTTPException(400,detail={"error":"grade_exceeds_max"})
        rank_points=int(assignment.get("rank_points") or 0)
        if rank_points and not rank_available:
            raise HTTPException(503,detail={"error":"rank_service_unavailable_for_completion"})
        # Only an APPROVED member may earn Classroom Rank points.
        cur.execute("""SELECT 1 FROM verifyd_circle_class_members
            WHERE class_id=%s AND student_id=%s AND status='approved'""", (class_id,student_id))
        if not cur.fetchone():
            raise HTTPException(403,detail={"error":"student_not_enrolled"})
        # A completed review transitions submitted -> returned exactly once.
        cur.execute("""UPDATE verifyd_circle_class_submissions SET
            status='returned', points_awarded=%s,feedback=%s,returned_at=NOW(),updated_at=NOW()
            WHERE assignment_id=%s AND student_id=%s AND status='submitted'
            RETURNING status,points_awarded,feedback,returned_at""",
            (payload.points_awarded,payload.feedback,assignment_id,student_id))
        rec=cur.fetchone()
        if not rec:
            raise HTTPException(409,detail={"error":"submission_not_ready"})
        if rank_points:
            cur.execute("""INSERT INTO verifyd_circle_rank_awards
                (award_id, identity_id, challenge_id, kind, points, evidence_ref)
                VALUES (%s,%s,%s,'challenge',%s,%s)
                ON CONFLICT DO NOTHING RETURNING award_id""",
                (str(uuid.uuid4()),student_id,f"classroom:{assignment_id}",rank_points,
                 f"classroom-review:{assignment_id}:{student_id}"))
            points_added=rank_points if cur.fetchone() else 0
    return {"review":_display(rec),"trust_rank_points_added":points_added,
            "rank_credit": "awarded_once" if points_added else "none",
            "note":"Trust Rank points reflect a completed teacher review, not proof of independent verification or academic achievement."}
