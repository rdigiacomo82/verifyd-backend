"""Private-by-default, moderated lesson sharing for VeriFYD Circle educators.

Shares only an educator-authored lesson draft after explicit opt-in and operator
approval. No Classroom submissions, grades, personal coursework or uploaded
documents are read. All author credentials remain self-reported.
"""
from __future__ import annotations

import os
import re
import uuid
from typing import Literal, Optional

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field

from database import get_db
from trust_voice_classrooms import _identity, _approved_teacher, _schema as classroom_schema

router = APIRouter(prefix="/trust-voice/classroom/library", tags=["Circle Educator Library"])

_CATEGORIES = (
    "Mathematics", "Literacy & Reading", "Writing & Language", "Science",
    "Social Studies", "Digital Literacy", "Communication & AAC",
    "Executive Function", "Social Skills", "Independent Living",
    "Creative Arts", "Health & Wellbeing", "Other",
)
_SKILL_LIMIT = 8
_PRIVATE_PATTERN = re.compile(
    r"(?i)(?:[\w.+-]+@[\w.-]+\.[a-z]{2,}|(?:\+?1[-. ]?)?\(?\d{3}\)?[-. ]?\d{3}[-. ]?\d{4}\b|"
    r"\b(?:ssn|student id|patient id|date of birth|home address)\s*[:#])"
)


def _enabled():
    return os.getenv("VERIFYD_CIRCLE_CLASSROOM_LIBRARY_ENABLED", "0").strip().lower() in ("1", "true", "yes", "on")


def _access(auth):
    person = _identity(auth)
    if not _enabled():
        raise HTTPException(404, detail={"error": "lesson_library_unavailable"})
    return person


def _moderator(user):
    ids = {x.strip() for x in os.getenv(
        "VERIFYD_CIRCLE_CLASSROOM_LIBRARY_MODERATOR_IDS", "").split(",") if x.strip()}
    if user not in ids:
        raise HTTPException(403, detail={"error": "library_moderator_required"})


def _schema():
    classroom_schema()
    with get_db() as conn:
        conn.cursor().execute("""CREATE TABLE IF NOT EXISTS verifyd_circle_shared_lessons (
            lesson_id TEXT PRIMARY KEY,
            teacher_id TEXT NOT NULL REFERENCES verifyd_identities(id) ON DELETE CASCADE,
            author_label TEXT NOT NULL,
            title TEXT NOT NULL,
            instructions TEXT NOT NULL,
            grade_band TEXT NOT NULL,
            subject TEXT NOT NULL,
            category TEXT NOT NULL,
            skills TEXT NOT NULL DEFAULT '',
            status TEXT NOT NULL DEFAULT 'pending'
                CHECK (status IN ('pending','approved','rejected','withdrawn')),
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            reviewed_at TIMESTAMPTZ,
            reviewed_by TEXT REFERENCES verifyd_identities(id)
        )""")
        conn.cursor().execute("""CREATE INDEX IF NOT EXISTS idx_circle_shared_lessons_public
            ON verifyd_circle_shared_lessons(status, created_at DESC)""")


class ShareLesson(BaseModel):
    title: str = Field(min_length=3, max_length=160)
    instructions: str = Field(min_length=20, max_length=16000)
    grade_band: Literal["K-2", "3-5", "6-8", "9-12", "Adult", "Custom"]
    subject: str = Field(min_length=2, max_length=80)
    category: Literal[
        "Mathematics", "Literacy & Reading", "Writing & Language", "Science",
        "Social Studies", "Digital Literacy", "Communication & AAC",
        "Executive Function", "Social Skills", "Independent Living",
        "Creative Arts", "Health & Wellbeing", "Other"
    ]
    skills: list[str] = Field(default_factory=list, max_length=_SKILL_LIMIT)
    share_with_community: bool = False
    confirm_rights_and_privacy: bool = False


class ReviewDecision(BaseModel):
    decision: Literal["approved", "rejected"]


def _public(row):
    row = dict(row)
    row.pop("teacher_id", None)
    row.pop("reviewed_by", None)
    if row.get("created_at") is not None:
        row["created_at"] = row["created_at"].isoformat()
    if row.get("reviewed_at") is not None:
        row["reviewed_at"] = row["reviewed_at"].isoformat()
    row["skills"] = [x for x in (row.get("skills") or "").split("|") if x]
    row["author_verification"] = "Self-reported professional; credentials not independently verified"
    return row


@router.get("/catalog")
def catalog(subject: str = "", grade_band: str = "", category: str = "",
            authorization: Optional[str] = Header(default=None)):
    _access(authorization)
    _schema()
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("""SELECT lesson_id,author_label,title,instructions,grade_band,
            subject,category,skills,status,created_at,reviewed_at
            FROM verifyd_circle_shared_lessons
            WHERE status='approved'
              AND (%s='' OR LOWER(subject)=LOWER(%s))
              AND (%s='' OR grade_band=%s)
              AND (%s='' OR category=%s)
            ORDER BY reviewed_at DESC, created_at DESC LIMIT 100""",
            (subject, subject, grade_band, grade_band, category, category))
        rows = [_public(row) for row in cur.fetchall()]
    return {"lessons": rows, "total_shown": len(rows),
            "note": "Approved educator-contributed lessons only. Attribution and qualifications are self-reported; VeriFYD does not verify authorship or instructional quality."}


@router.get("/suggestions")
def suggestions(subject: str = "", grade_band: str = "",
                authorization: Optional[str] = Header(default=None)):
    _access(authorization)
    _schema()
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("""SELECT category,COUNT(*) AS uses
            FROM verifyd_circle_shared_lessons
            WHERE status='approved'
              AND (%s='' OR LOWER(subject)=LOWER(%s))
              AND (%s='' OR grade_band=%s)
            GROUP BY category ORDER BY uses DESC, category ASC LIMIT 8""",
            (subject, subject, grade_band, grade_band))
        categories = [{"name": row["category"], "uses": int(row["uses"])} for row in cur.fetchall()]
    return {"suggestions": categories, "categories": list(_CATEGORIES),
            "method": "Counts of educator-selected categories on moderator-approved shared lessons; suggestions require teacher confirmation. No private lesson plans used."}


@router.get("/mine")
def mine(authorization: Optional[str] = Header(default=None)):
    user = _access(authorization)
    _schema()
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("""SELECT lesson_id,author_label,title,instructions,grade_band,
            subject,category,skills,status,created_at,reviewed_at
            FROM verifyd_circle_shared_lessons WHERE teacher_id=%s
            ORDER BY created_at DESC LIMIT 100""",(user,))
        rows = [_public(row) for row in cur.fetchall()]
    return {"submissions": rows}


@router.post("/submissions")
def submit(payload: ShareLesson, authorization: Optional[str] = Header(default=None)):
    user = _access(authorization)
    _approved_teacher(user)
    if not payload.share_with_community or not payload.confirm_rights_and_privacy:
        raise HTTPException(422, detail={"error": "explicit_sharing_consent_required"})
    text = payload.title.strip() + "\n" + payload.instructions.strip()
    if _PRIVATE_PATTERN.search(text):
        raise HTTPException(422, detail={"error": "possible_personal_information_in_lesson"})
    cleaned_skills = []
    for value in payload.skills:
        value = re.sub(r"[\x00-\x1f|]", " ", value.strip())[:60].strip()
        if value and value.lower() not in [x.lower() for x in cleaned_skills]:
            cleaned_skills.append(value)
    if not cleaned_skills:
        raise HTTPException(422, detail={"error": "skill_tag_required"})
    _schema()
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("""SELECT COUNT(*) AS count FROM verifyd_circle_shared_lessons
            WHERE teacher_id=%s AND created_at > NOW()-INTERVAL '24 hours'""",(user,))
        if int(cur.fetchone()["count"]) >= 10:
            raise HTTPException(429, detail={"error": "library_daily_submission_limit"})
        cur.execute("""SELECT COUNT(*) AS count FROM verifyd_circle_shared_lessons
            WHERE teacher_id=%s AND status IN ('pending','approved')""",(user,))
        if int(cur.fetchone()["count"]) >= 100:
            raise HTTPException(429, detail={"error": "library_submission_limit"})
        cur.execute("""SELECT display_name FROM verifyd_circle_teacher_profiles
            WHERE identity_id=%s""",(user,))
        profile = cur.fetchone()
        if profile and profile["display_name"]:
            author = profile["display_name"][:100]
        else:
            cur.execute("SELECT display_name FROM verifyd_identities WHERE id=%s",(user,))
            record = cur.fetchone()
            author = ((record or {}).get("display_name") or "Circle educator")[:100]
        lesson_id = str(uuid.uuid4())
        cur.execute("""INSERT INTO verifyd_circle_shared_lessons
            (lesson_id,teacher_id,author_label,title,instructions,grade_band,subject,category,skills)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
            (lesson_id,user,author,payload.title.strip(),payload.instructions.strip(),
             payload.grade_band,payload.subject.strip(),payload.category,
             "|".join(cleaned_skills)))
    return {"lesson_id": lesson_id, "status": "pending", "public": False,
            "note": "Lesson submitted for moderation. It is private until independently approved; classroom/student data was not accessed."}


@router.delete("/mine/{lesson_id}")
def withdraw(lesson_id: str, authorization: Optional[str] = Header(default=None)):
    user = _access(authorization)
    _schema()
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("""UPDATE verifyd_circle_shared_lessons SET status='withdrawn'
            WHERE lesson_id=%s AND teacher_id=%s AND status IN ('pending','approved')
            RETURNING lesson_id""",(lesson_id,user))
        if not cur.fetchone():
            raise HTTPException(404,detail={"error": "shared_lesson_not_found"})
    return {"withdrawn": True, "lesson_id": lesson_id}


@router.get("/moderation")
def moderation(authorization: Optional[str] = Header(default=None)):
    user = _access(authorization)
    _moderator(user)
    _schema()
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("""SELECT lesson_id,author_label,title,instructions,grade_band,
            subject,category,skills,status,created_at,reviewed_at
            FROM verifyd_circle_shared_lessons WHERE status='pending'
            ORDER BY created_at ASC LIMIT 100""")
        rows = [_public(row) for row in cur.fetchall()]
    return {"pending": rows, "moderator": True}


@router.post("/moderation/{lesson_id}/decision")
def decide(lesson_id: str, payload: ReviewDecision,
           authorization: Optional[str] = Header(default=None)):
    user = _access(authorization)
    _moderator(user)
    _schema()
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("""UPDATE verifyd_circle_shared_lessons
            SET status=%s,reviewed_at=NOW(),reviewed_by=%s
            WHERE lesson_id=%s AND status='pending'
            RETURNING lesson_id""",(payload.decision,user,lesson_id))
        if not cur.fetchone():
            raise HTTPException(409,detail={"error": "lesson_not_pending"})
    return {"lesson_id": lesson_id, "status": payload.decision}
