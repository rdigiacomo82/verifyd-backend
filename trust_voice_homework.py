"""VeriFYD Circle Homework v0.1 — standards-informed practice assignments.

Optional feature, disabled by default. Drafts and submissions belong to the
authenticated identity. Only the first server-checked passing completion can
receive a points award. Homework is not a clinical assessment or school grade.
There is no public API to set scores or grant ranks.
"""
from __future__ import annotations

import json
import os
import threading
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field

from database import get_db
from trust_voice_messages import _identity_from_bearer

router = APIRouter(prefix="/trust-voice/homework", tags=["Circle Homework"])
_catalog = json.loads(Path(__file__).with_name("trust_voice_homework_catalog.json").read_text(encoding="utf-8"))
ASSIGNMENTS = {item["id"]: item for item in _catalog["assignments"]}
_schema_lock = threading.Lock()
_schema_ready = False


def enabled():
    return os.getenv("VERIFYD_CIRCLE_HOMEWORK_ENABLED", "0").strip().lower() in {"1", "true", "yes", "on"}


def _require():
    if not enabled():
        raise HTTPException(status_code=404, detail={"error": "circle_homework_unavailable"})


def _owner(auth: Optional[str]):
    _require()
    identity = _identity_from_bearer(auth)
    if identity.get("status") != "active":
        raise HTTPException(401, detail={"error": "session_required"})
    return identity["id"]


def _schema():
    global _schema_ready
    if _schema_ready:
        return
    with _schema_lock:
        if _schema_ready:
            return
        with get_db() as conn:
            cur = conn.cursor()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS verifyd_circle_homework_work (
                    identity_id TEXT NOT NULL REFERENCES verifyd_identities(id) ON DELETE CASCADE,
                    assignment_id TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'in_progress'
                        CHECK(status IN ('in_progress','completed')),
                    answers JSONB NOT NULL DEFAULT '{}'::jsonb,
                    reflection TEXT NOT NULL DEFAULT '',
                    attempt_count INTEGER NOT NULL DEFAULT 0,
                    last_score INTEGER,
                    completed_at TIMESTAMPTZ,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    PRIMARY KEY(identity_id, assignment_id)
                )
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_circle_homework_work_identity
                ON verifyd_circle_homework_work(identity_id, status)
            """)
        _schema_ready = True


def _public_assignment(item):
    """Do not ship the answer keys to a browser."""
    return {key: value for key, value in item.items() if key != "questions"} | {
        "questions": [
            {key: value for key, value in question.items() if key != "answer"}
            for question in item["questions"]
        ]
    }


def _find(assignment_id):
    item = ASSIGNMENTS.get(assignment_id)
    if item is None:
        raise HTTPException(404, detail={"error": "assignment_not_found"})
    return item


def grade_answers(item, answers):
    questions = item["questions"]
    if not isinstance(answers, dict) or set(answers) != {q["id"] for q in questions}:
        raise HTTPException(400, detail={"error": "answer_every_question"})
    for q in questions:
        if answers[q["id"]] not in {c["id"] for c in q["choices"]}:
            raise HTTPException(400, detail={"error": "invalid_answer_choice"})
    correct = sum(answers[q["id"]] == q["answer"] for q in questions)
    return {
        "correct": correct, "total": len(questions),
        "score_percent": round(correct * 100 / len(questions)),
        "passed": correct * 3 >= 2 * len(questions),
    }


class HomeworkDraft(BaseModel):
    answers: dict[str, str] = Field(default_factory=dict)
    reflection: str = Field(default="", max_length=2000)


class HomeworkSubmit(BaseModel):
    answers: dict[str, str]
    reflection: str = Field(default="", max_length=2000)


def _validate_partial(item, answers):
    allowed = {q["id"]: {c["id"] for c in q["choices"]} for q in item["questions"]}
    if any(q not in allowed or choice not in allowed[q] for q, choice in answers.items()):
        raise HTTPException(400, detail={"error": "invalid_answer_choice"})


def _work_item(row):
    if row is None:
        return None
    record = dict(row)
    for field in ("completed_at", "updated_at"):
        if record.get(field) is not None:
            record[field] = record[field].isoformat()
    return record


@router.get("/catalog")
def catalog():
    _require()
    return {
        "version": _catalog["version"],
        "grade_bands": _catalog["grade_bands"],
        "subjects": _catalog["subjects"],
        "assignments": [_public_assignment(x) for x in ASSIGNMENTS.values()],
        "note": "Grades reflect learning level, not the user's age. Homework is private by default.",
    }


@router.get("/me")
def my_work(authorization: Optional[str] = Header(default=None)):
    identity_id = _owner(authorization)
    _schema()
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT assignment_id, status, answers, reflection, attempt_count,
                   last_score, completed_at, updated_at
            FROM verifyd_circle_homework_work WHERE identity_id=%s
            ORDER BY updated_at DESC
        """, (identity_id,))
        work = [_work_item(row) for row in cur.fetchall() if row["assignment_id"] in ASSIGNMENTS]
    completed = sum(x["status"] == "completed" for x in work)
    started = sum(x["status"] == "in_progress" for x in work)
    return {
        "work": work,
        "summary": {
            "completed": completed, "in_progress": started,
            "not_started": len(ASSIGNMENTS) - completed - started,
            "total": len(ASSIGNMENTS),
        },
    }


@router.put("/me/{assignment_id}/draft")
def save_draft(assignment_id: str, payload: HomeworkDraft,
               authorization: Optional[str] = Header(default=None)):
    identity_id = _owner(authorization)
    item = _find(assignment_id)
    _validate_partial(item, payload.answers)
    _schema()
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO verifyd_circle_homework_work(identity_id, assignment_id, answers, reflection)
            VALUES (%s,%s,%s::jsonb,%s)
            ON CONFLICT(identity_id, assignment_id) DO UPDATE SET
                answers=EXCLUDED.answers,
                reflection=EXCLUDED.reflection,
                updated_at=NOW()
            WHERE verifyd_circle_homework_work.status='in_progress'
            RETURNING assignment_id, status, answers, reflection, attempt_count,
                      last_score, completed_at, updated_at
        """, (identity_id, assignment_id, json.dumps(payload.answers), payload.reflection))
        row = cur.fetchone()
    if row is None:
        raise HTTPException(409, detail={"error": "already_completed"})
    return {"work": _work_item(row)}


@router.post("/me/{assignment_id}/submit")
def submit(assignment_id: str, payload: HomeworkSubmit,
           authorization: Optional[str] = Header(default=None)):
    identity_id = _owner(authorization)
    item = _find(assignment_id)
    result = grade_answers(item, payload.answers)
    _schema()
    # Initialize rank schema BEFORE locking the user row: no nested pool request
    # while the account's homework transaction has uncommitted changes.
    from trust_voice_ranks import _schema as ensure_rank_schema
    ensure_rank_schema()

    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO verifyd_circle_homework_work(identity_id, assignment_id)
            VALUES (%s,%s) ON CONFLICT DO NOTHING
        """, (identity_id, assignment_id))
        cur.execute("""
            SELECT status, attempt_count FROM verifyd_circle_homework_work
            WHERE identity_id=%s AND assignment_id=%s FOR UPDATE
        """, (identity_id, assignment_id))
        previous = cur.fetchone()
        if previous["status"] == "completed":
            raise HTTPException(409, detail={"error": "already_completed"})
        status = "completed" if result["passed"] else "in_progress"
        cur.execute("""
            UPDATE verifyd_circle_homework_work SET
                status=%s, answers=%s::jsonb, reflection=%s,
                attempt_count=attempt_count+1,
                last_score=%s,
                completed_at=CASE WHEN %s THEN NOW() ELSE NULL END,
                updated_at=NOW()
            WHERE identity_id=%s AND assignment_id=%s
            RETURNING assignment_id, status, answers, reflection, attempt_count,
                      last_score, completed_at, updated_at
        """, (status, json.dumps(payload.answers), payload.reflection,
              result["score_percent"], result["passed"], identity_id, assignment_id))
        work = _work_item(cur.fetchone())
        points_added = 0
        if result["passed"]:
            cur.execute("""
                INSERT INTO verifyd_circle_rank_awards
                    (award_id, identity_id, challenge_id, kind, points, evidence_ref)
                VALUES (%s,%s,%s,'challenge',%s,%s)
                ON CONFLICT DO NOTHING RETURNING award_id
            """, (str(uuid.uuid4()), identity_id, f"homework:{assignment_id}", item["points"],
                  f"verified-homework:{identity_id}:{assignment_id}"))
            if cur.fetchone():
                points_added = item["points"]
    return {
        "work": work, "result": result, "points_added": points_added,
        "rank_credit": "awarded_once" if points_added else "none",
    }
