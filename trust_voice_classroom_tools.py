"""Additional Classroom tools; same owner-based access, separate pilot flag."""
from __future__ import annotations

import hashlib
import secrets
from datetime import datetime
import threading
from typing import Optional

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field

from database import get_db
from trust_voice_classrooms import (
    _identity, _schema as class_schema, _teacher, _member, _assignment, _approved_teacher, _display,
    _prepare_classroom_audience, _classroom_assignment_visible
)

router = APIRouter(prefix="/trust-voice/classroom", tags=["Circle Classroom Tools"])
_READY = False
_LOCK = threading.Lock()


def schema():
    global _READY
    if _READY:
        return
    with _LOCK:
        if _READY:
            return
        class_schema()
        with get_db() as conn:
            cur = conn.cursor()
            cur.execute("""CREATE TABLE IF NOT EXISTS verifyd_circle_class_announcements(
                announcement_id TEXT PRIMARY KEY,
                class_id TEXT NOT NULL REFERENCES verifyd_circle_classes(class_id) ON DELETE CASCADE,
                author_id TEXT NOT NULL REFERENCES verifyd_identities(id) ON DELETE CASCADE,
                body TEXT NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )""")
            cur.execute("""CREATE INDEX IF NOT EXISTS idx_class_announcement_created
                ON verifyd_circle_class_announcements(class_id,created_at DESC)""")
            cur.execute("""CREATE TABLE IF NOT EXISTS verifyd_circle_assignment_rubrics(
                assignment_id TEXT PRIMARY KEY REFERENCES verifyd_circle_class_assignments(assignment_id) ON DELETE CASCADE,
                criteria JSONB NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )""")
        _READY = True

class Announcement(BaseModel):
    body: str = Field(min_length=1, max_length=4000)


class RubricCriterion(BaseModel):
    label: str = Field(min_length=2, max_length=120)
    explanation: str = Field(default="", max_length=700)
    max_points: int = Field(ge=1, le=1000)


class Rubric(BaseModel):
    criteria: list[RubricCriterion] = Field(min_length=1, max_length=12)


@router.post("/classes/{class_id}/announcements")
def post_announcement(class_id: str, payload: Announcement,
                      authorization: Optional[str] = Header(default=None)):
    user = _identity(authorization)
    _approved_teacher(user)
    schema()
    key = secrets.token_hex(16)
    with get_db() as conn:
        cur=conn.cursor()
        _teacher(cur,class_id,user)
        cur.execute("""INSERT INTO verifyd_circle_class_announcements
            (announcement_id,class_id,author_id,body) VALUES (%s,%s,%s,%s)""",
            (key,class_id,user,payload.body))
    return {"announcement_id":key,"posted":True}


@router.get("/classes/{class_id}/announcements")
def list_announcements(class_id: str,
                       authorization: Optional[str] = Header(default=None)):
    user=_identity(authorization)
    schema()
    with get_db() as conn:
        cur=conn.cursor()
        _member(cur,class_id,user)
        cur.execute("""SELECT announcement_id,body,created_at
            FROM verifyd_circle_class_announcements WHERE class_id=%s
            ORDER BY created_at DESC LIMIT 100""",(class_id,))
        rows=[_display(x) for x in cur.fetchall()]
    return {"announcements":rows}


@router.post("/classes/{class_id}/rotate-code")
def rotate_join_code(class_id: str, authorization: Optional[str] = Header(default=None)):
    user=_identity(authorization)
    _approved_teacher(user)
    class_schema()
    new_code=secrets.token_urlsafe(24)
    digest=hashlib.sha256(new_code.encode()).hexdigest()
    with get_db() as conn:
        cur=conn.cursor()
        _teacher(cur,class_id,user)
        cur.execute("UPDATE verifyd_circle_classes SET join_hash=%s WHERE class_id=%s",
            (digest,class_id))
    return {"join_code":new_code,"previous_code_invalidated":True}


@router.post("/classes/{class_id}/members/{student_id}/remove")
def remove_member(class_id: str, student_id: str,
                  authorization: Optional[str] = Header(default=None)):
    user=_identity(authorization)
    _approved_teacher(user)
    class_schema()
    with get_db() as conn:
        cur=conn.cursor()
        _teacher(cur,class_id,user)
        cur.execute("""UPDATE verifyd_circle_class_members SET status='removed'
            WHERE class_id=%s AND student_id=%s AND status IN ('pending','approved')
            RETURNING student_id""",(class_id,student_id))
        if not cur.fetchone():
            raise HTTPException(404,detail={"error":"member_not_found"})
    return {"removed":True,
            "note":"Access revoked. Existing submissions remain stored until an explicit deletion and retention workflow is implemented."}


@router.get("/classes/{class_id}/my-todo")
def student_todo(class_id: str, authorization: Optional[str] = Header(default=None)):
    user=_identity(authorization)
    _prepare_classroom_audience()
    class_schema()
    with get_db() as conn:
        cur=conn.cursor()
        rec=_member(cur,class_id,user)
        if rec["teacher_id"]==user:
            raise HTTPException(403,detail={"error":"students_only"})
        cur.execute("""SELECT a.assignment_id,a.title,a.due_at,a.points_possible,
            COALESCE(s.status,'not_started') AS my_status
            FROM verifyd_circle_class_assignments a
            LEFT JOIN verifyd_circle_class_submissions s
            ON s.assignment_id=a.assignment_id AND s.student_id=%s
            WHERE a.class_id=%s AND a.status='published'
            AND (a.publish_at IS NULL OR a.publish_at<=NOW())
            ORDER BY a.due_at NULLS LAST,a.created_at DESC LIMIT 200""",(user,class_id))
        items=[]
        now=datetime.now().astimezone()
        for rec in cur.fetchall():
            if not _classroom_assignment_visible(cur,rec["assignment_id"],user):
                continue
            item=_display(rec)
            item["overdue"]=bool(rec["due_at"] and rec["due_at"]<now
                                and rec["my_status"] in ("not_started","draft"))
            items.append(item)
    return {"assignments":items}


@router.get("/classes/{class_id}/gradebook")
def teacher_gradebook(class_id: str, authorization: Optional[str] = Header(default=None)):
    user=_identity(authorization)
    _approved_teacher(user)
    _prepare_classroom_audience()
    class_schema()
    with get_db() as conn:
        cur=conn.cursor()
        _teacher(cur,class_id,user)
        cur.execute("""SELECT a.assignment_id,a.title,a.points_possible,
            m.student_id,i.handle AS student_handle,a.rank_points,
            COALESCE(s.status,'not_started') AS status,
            s.points_awarded,s.submitted_at,s.returned_at
            FROM verifyd_circle_class_assignments a
            JOIN verifyd_circle_class_members m
                ON m.class_id=a.class_id AND m.status='approved'
            JOIN verifyd_identities i ON i.id=m.student_id
            LEFT JOIN verifyd_circle_class_submissions s
                ON s.assignment_id=a.assignment_id AND s.student_id=m.student_id
            WHERE a.class_id=%s AND a.status='published'
            ORDER BY a.created_at DESC,m.student_id LIMIT 5000""",(class_id,))
        rows=[_display(x) for x in cur.fetchall()]
        rows=[item for item in rows if _classroom_assignment_visible(cur,item["assignment_id"],item["student_id"])]
    return {"gradebook":rows,"teacher_only":True}


@router.post("/classes/{class_id}/assignments/{assignment_id}/rubric")
def create_rubric(class_id: str,assignment_id: str,payload: Rubric,
                  authorization: Optional[str] = Header(default=None)):
    user=_identity(authorization)
    _approved_teacher(user)
    schema()
    with get_db() as conn:
        cur=conn.cursor()
        _teacher(cur,class_id,user)
        assignment=_assignment(cur,class_id,assignment_id)
        if assignment["status"]!="draft":
            raise HTTPException(409,detail={"error":"rubric_locked_after_publish"})
        if sum(x.max_points for x in payload.criteria)!=assignment["points_possible"]:
            raise HTTPException(400,detail={"error":"rubric_points_must_equal_assignment_total"})
        import json
        criteria=[c.model_dump() for c in payload.criteria]
        cur.execute("""INSERT INTO verifyd_circle_assignment_rubrics(assignment_id,criteria)
            VALUES (%s,%s::jsonb) ON CONFLICT(assignment_id)
            DO UPDATE SET criteria=EXCLUDED.criteria""",
            (assignment_id,json.dumps(criteria)))
    return {"criteria_count":len(criteria),"total_points":assignment["points_possible"]}


@router.get("/classes/{class_id}/assignments/{assignment_id}/rubric")
def get_rubric(class_id: str,assignment_id: str,
               authorization: Optional[str] = Header(default=None)):
    user=_identity(authorization)
    _prepare_classroom_audience()
    schema()
    with get_db() as conn:
        cur=conn.cursor()
        rec=_member(cur,class_id,user)
        assignment=_assignment(cur,class_id,assignment_id)
        if rec["teacher_id"] != user and not _classroom_assignment_visible(cur,assignment_id,user):
            raise HTTPException(404,detail={"error":"rubric_not_found"})
        if rec["teacher_id"] != user and (assignment["status"] != "published" or (assignment["publish_at"] is not None and assignment["publish_at"] > datetime.now().astimezone())):
            raise HTTPException(404,detail={"error":"rubric_not_found"})
        cur.execute("SELECT criteria FROM verifyd_circle_assignment_rubrics WHERE assignment_id=%s",
                    (assignment_id,))
        row=cur.fetchone()
    return {"rubric":row["criteria"] if row else None,
            "note":"Criteria are visible before submission. Teacher-awarded grades remain human-reviewed."}
