"""VeriFYD Circle Classroom — isolated opt-in assignment pilot.

Never imports or mutates existing Homework, messages, calls, or voicemail.
Teachers must be explicitly preapproved by the operator. Joining needs teacher
approval. Raw file uploads are stored in a separate private R2 quarantine bucket
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
from datetime import datetime
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


def _approved_teacher(identity_id):
    approved = {value.strip() for value in os.getenv("VERIFYD_CIRCLE_CLASSROOM_TEACHER_IDS", "").split(",") if value.strip()}
    if identity_id not in approved:
        raise HTTPException(403, detail={"error": "teacher_approval_required"})


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
                points_possible INTEGER NOT NULL DEFAULT 100 CHECK(points_possible BETWEEN 1 AND 1000),
                status TEXT NOT NULL DEFAULT 'draft' CHECK(status IN ('draft','published','archived')),
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )""")
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
        _READY = True


def _teacher(cur, class_id, identity_id):
    cur.execute("SELECT teacher_id FROM verifyd_circle_classes WHERE class_id=%s", (class_id,))
    record = cur.fetchone()
    if not record or record["teacher_id"] != identity_id:
        raise HTTPException(404, detail={"error": "class_not_found"})
    return record


def _member(cur, class_id, identity_id):
    cur.execute("""SELECT c.teacher_id, m.status FROM verifyd_circle_classes c
        LEFT JOIN verifyd_circle_class_members m ON m.class_id=c.class_id AND m.student_id=%s
        WHERE c.class_id=%s""", (identity_id,class_id))
    record = cur.fetchone()
    if not record or (record["teacher_id"] != identity_id and record["status"] != "approved"):
        raise HTTPException(404, detail={"error": "class_not_found"})
    return record


def _assignment(cur, class_id, assignment_id):
    cur.execute("""SELECT assignment_id,class_id,title,instructions,due_at,points_possible,status
        FROM verifyd_circle_class_assignments WHERE class_id=%s AND assignment_id=%s""",
        (class_id,assignment_id))
    record = cur.fetchone()
    if not record:
        raise HTTPException(404, detail={"error": "assignment_not_found"})
    return record


def _display(row):
    result = dict(row)
    for field in ("created_at", "requested_at", "approved_at", "due_at", "submitted_at", "returned_at", "updated_at"):
        value = result.get(field)
        if isinstance(value, datetime):
            result[field] = value.isoformat()
    return result


class ClassCreate(BaseModel):
    title: str = Field(min_length=3, max_length=120)
    subject: str = Field(min_length=2, max_length=80)
    grade_band: Literal["K-2","3-5","6-8","9-12","Adult","Custom"]


class JoinRequest(BaseModel):
    join_code: str = Field(min_length=16, max_length=48)


class AssignmentCreate(BaseModel):
    title: str = Field(min_length=3, max_length=160)
    instructions: str = Field(min_length=1, max_length=16000)
    due_at: Optional[datetime] = None
    points_possible: int = Field(default=100, ge=1, le=1000)


class Work(BaseModel):
    response: str = Field(default="", max_length=30000)
    action: Literal["save","turn_in"] = "save"


class Review(BaseModel):
    points_awarded: int = Field(ge=0, le=1000)
    feedback: str = Field(default="", max_length=12000)


@router.get("/health")
def health():
    _require()
    return {"service":"circle_classroom_pilot","enabled":True,
            "teacher_access":"operator-approved","files":"quarantine_only",
            "scan_service":"not_connected","unsafe_files_downloadable":False}


@router.post("/classes")
def create_class(payload: ClassCreate, authorization: Optional[str] = Header(default=None)):
    user = _identity(authorization)
    _approved_teacher(user)
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
        cur.execute("SELECT class_id,teacher_id FROM verifyd_circle_classes WHERE join_hash=%s", (digest,))
        record=cur.fetchone()
        if not record:
            raise HTTPException(404, detail={"error":"invalid_class_code"})
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
    with get_db() as conn:
        cur=conn.cursor()
        cur.execute("""SELECT c.class_id,c.title,c.subject,c.grade_band,c.created_at,
            CASE WHEN c.teacher_id=%s THEN 'teacher' ELSE 'student' END AS my_role
            FROM verifyd_circle_classes c LEFT JOIN verifyd_circle_class_members m
            ON m.class_id=c.class_id AND m.student_id=%s
            WHERE c.teacher_id=%s OR m.status='approved'
            ORDER BY c.created_at DESC LIMIT 100""",(user,user,user))
        rows=[_display(x) for x in cur.fetchall()]
    return {"classes":rows}


@router.get("/classes/{class_id}/members")
def members(class_id: str, authorization: Optional[str] = Header(default=None)):
    user=_identity(authorization)
    _schema()
    with get_db() as conn:
        cur=conn.cursor()
        _teacher(cur,class_id,user)
        cur.execute("""SELECT student_id,status,requested_at,approved_at
            FROM verifyd_circle_class_members WHERE class_id=%s ORDER BY requested_at LIMIT 300""",(class_id,))
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
            (assignment_id,class_id,author_id,title,instructions,due_at,points_possible)
            VALUES (%s,%s,%s,%s,%s,%s,%s)""",(assignment_id,class_id,user,payload.title,
            payload.instructions,payload.due_at,payload.points_possible))
    return {"assignment_id":assignment_id,"status":"draft"}


@router.post("/classes/{class_id}/assignments/{assignment_id}/publish")
def publish_assignment(class_id: str,assignment_id: str,
                       authorization: Optional[str] = Header(default=None)):
    user=_identity(authorization)
    _approved_teacher(user)
    _schema()
    with get_db() as conn:
        cur=conn.cursor()
        _teacher(cur,class_id,user)
        cur.execute("""UPDATE verifyd_circle_class_assignments SET status='published'
            WHERE class_id=%s AND assignment_id=%s AND status='draft' RETURNING assignment_id""",
            (class_id,assignment_id))
        if not cur.fetchone():
            raise HTTPException(409,detail={"error":"assignment_not_draft"})
    return {"status":"published"}


@router.get("/classes/{class_id}/assignments")
def list_assignments(class_id: str, authorization: Optional[str] = Header(default=None)):
    user=_identity(authorization)
    _schema()
    with get_db() as conn:
        cur=conn.cursor()
        rec=_member(cur,class_id,user)
        show_drafts=rec["teacher_id"]==user
        cur.execute("""SELECT assignment_id,title,instructions,due_at,points_possible,status,created_at
            FROM verifyd_circle_class_assignments WHERE class_id=%s
            AND (%s OR status='published') ORDER BY created_at DESC LIMIT 150""",
            (class_id,show_drafts))
        rows=[_display(x) for x in cur.fetchall()]
    return {"assignments":rows}


@router.put("/classes/{class_id}/assignments/{assignment_id}/my-work")
def save_or_turn_in(class_id: str,assignment_id: str,payload: Work,
                    authorization: Optional[str] = Header(default=None)):
    user=_identity(authorization)
    _schema()
    with get_db() as conn:
        cur=conn.cursor()
        rec=_member(cur,class_id,user)
        if rec["teacher_id"]==user:
            raise HTTPException(403,detail={"error":"students_only"})
        assignment=_assignment(cur,class_id,assignment_id)
        if assignment["status"]!="published":
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
    _schema()
    with get_db() as conn:
        cur=conn.cursor()
        _member(cur,class_id,user)
        _assignment(cur,class_id,assignment_id)
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
        cur.execute("""SELECT student_id,status,response,points_awarded,feedback,
            submitted_at,returned_at,updated_at FROM verifyd_circle_class_submissions
            WHERE assignment_id=%s AND status IN ('submitted','returned')
            ORDER BY submitted_at DESC LIMIT 300""",(assignment_id,))
        rows=[_display(x) for x in cur.fetchall()]
    return {"submissions":rows}


@router.post("/classes/{class_id}/assignments/{assignment_id}/submissions/{student_id}/review")
def review_submission(class_id: str,assignment_id: str,student_id: str,payload: Review,
                      authorization: Optional[str] = Header(default=None)):
    user=_identity(authorization)
    _approved_teacher(user)
    _schema()
    with get_db() as conn:
        cur=conn.cursor()
        _teacher(cur,class_id,user)
        assignment=_assignment(cur,class_id,assignment_id)
        if payload.points_awarded > assignment["points_possible"]:
            raise HTTPException(400,detail={"error":"grade_exceeds_max"})
        cur.execute("""UPDATE verifyd_circle_class_submissions SET
            status='returned', points_awarded=%s,feedback=%s,returned_at=NOW(),updated_at=NOW()
            WHERE assignment_id=%s AND student_id=%s AND status='submitted'
            RETURNING status,points_awarded,feedback,returned_at""",
            (payload.points_awarded,payload.feedback,assignment_id,student_id))
        rec=cur.fetchone()
        if not rec:
            raise HTTPException(409,detail={"error":"submission_not_ready"})
    return {"review":_display(rec),"trust_rank_points_added":0,
            "note":"Teacher grades do not automatically change Trust Rank."}
