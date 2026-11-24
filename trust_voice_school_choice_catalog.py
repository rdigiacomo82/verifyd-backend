"""School-scoped TEST-only Safety Button choices.

Stored per approved school; client labels never grant permissions or dispatch.
The backend retains its six canonical incident categories for grouping.
"""
from __future__ import annotations

import json
import re
import threading
from typing import Literal, Optional

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field, model_validator

from database import get_db
from trust_voice_school_safety_training import _actor, _audit, _membership, _schema
from trust_voice_school_safety_registration import _school_admin

router = APIRouter(prefix="/trust-voice/school-safety", tags=["School Safety TEST Choice Catalog"])
_lock = threading.Lock()
_ready = False

DEFAULT_CHOICES = (
    ("weapon", "Weapon or immediate threat", "active_threat", "I observed a possible weapon or immediate threat at school."),
    ("unsafe_person", "Unknown or threatening person", "active_threat", "There is an unfamiliar person or someone making threats near me."),
    ("fight", "Fight or physical violence", "violence", "I witnessed a fight or someone being physically hurt."),
    ("bullying", "Bullying or harassment", "violence", "A student is being bullied, threatened, or harassed."),
    ("medical", "Medical issue or injury", "medical", "A student needs assistance for an injury or a medical problem."),
    ("fire", "Fire, smoke, or evacuation concern", "fire", "I see smoke, fire, or an unsafe evacuation situation."),
    ("hazard", "Unsafe school area or hazard", "hazard", "I noticed a dangerous condition at school that needs attention."),
    ("transportation", "Transportation or dismissal concern", "other", "There is a problem with dismissal, my bus, or my approved pickup."),
    ("pickup", "Please pick me up", "other", "Please pick me up through the approved school dismissal process."),
    ("safe", "I am safe", "other", "I am safe. Please share this update with my verified guardian."),
    ("school", "Please contact the school", "other", "Please contact the school about my current situation."),
    ("other", "Other — type your concern", "other", ""),
)

def default_choices():
    return [{"id": key, "label": label, "category": category, "hint": hint}
            for key, label, category, hint in DEFAULT_CHOICES]


class SafetyChoice(BaseModel):
    id: str = Field(pattern=r"^[a-z][a-z0-9_]{2,39}$")
    label: str = Field(min_length=3, max_length=80)
    category: Literal["active_threat", "fire", "medical", "violence", "hazard", "other"]
    hint: str = Field(default="", max_length=180)

    @model_validator(mode="after")
    def valid_text(self):
        if self.label != self.label.strip() or not self.label.strip():
            raise ValueError("A choice must have a trimmed label.")
        if self.id == "other" and (self.category != "other" or self.hint.strip()):
            raise ValueError("The always-available Other choice must use 'other' and require typed details.")
        return self


class SafetyChoiceUpdate(BaseModel):
    expected_revision: int = Field(ge=0)
    choices: list[SafetyChoice] = Field(min_length=1, max_length=20)

    @model_validator(mode="after")
    def valid_set(self):
        ids = [item.id for item in self.choices]
        if len(ids) != len(set(ids)):
            raise ValueError("Choice IDs must be unique.")
        if "other" not in ids:
            raise ValueError("The 'Other — type your concern' choice cannot be removed.")
        return self


def _catalog_schema():
    global _ready
    _schema()
    if _ready:
        return
    with _lock:
        if _ready:
            return
        with get_db() as conn:
            conn.cursor().execute("""CREATE TABLE IF NOT EXISTS verifyd_safety_choice_catalog(
                school_id TEXT PRIMARY KEY REFERENCES verifyd_safety_training_schools(school_id) ON DELETE CASCADE,
                revision INTEGER NOT NULL DEFAULT 1,
                choices JSONB NOT NULL,
                updated_by TEXT NOT NULL REFERENCES verifyd_identities(id),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )""")
        _ready = True


def _read_catalog(cur, school_id):
    cur.execute("SELECT revision, choices FROM verifyd_safety_choice_catalog WHERE school_id=%s",
                (school_id,))
    row = cur.fetchone()
    return {"revision": row["revision"], "choices": row["choices"]} if row else {
        "revision": 0, "choices": default_choices()
    }


def resolve_report_choice(cur, school_id, choice_id):
    """Resolve server-authoritative label/category; fail closed on stale IDs."""
    result = _read_catalog(cur, school_id)
    for choice in result["choices"]:
        if choice["id"] == choice_id:
            return choice
    raise HTTPException(409, detail={"error": "school_safety_choice_changed"})


@router.get("/choices/me")
def my_school_safety_choices(authorization: Optional[str] = Header(default=None)):
    """The hold launcher may load choices only for the user's verified school(s)."""
    actor = _actor(authorization)
    _catalog_schema()
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("""SELECT m.school_id, s.school_name, m.role
            FROM verifyd_safety_training_members m
            JOIN verifyd_safety_training_schools s ON s.school_id=m.school_id
            WHERE m.identity_id=%s ORDER BY s.school_name""", (str(actor["id"]),))
        schools = [dict(row) for row in cur.fetchall()]
        for school in schools:
            school.update(_read_catalog(cur, school["school_id"]))
    return {"ok": True, "test_only": True, "schools": schools}


@router.get("/schools/{school_id}/choices")
def school_safety_choices(school_id: str, authorization: Optional[str] = Header(default=None)):
    actor = _actor(authorization)
    _catalog_schema()
    with get_db() as conn:
        cur = conn.cursor()
        _membership(cur, school_id, str(actor["id"]))
        data = _read_catalog(cur, school_id)
    return {"ok": True, "test_only": True, "school_id": school_id, **data}


@router.put("/schools/{school_id}/choices")
def update_school_safety_choices(
    school_id: str, payload: SafetyChoiceUpdate,
    authorization: Optional[str] = Header(default=None)
):
    actor = _actor(authorization)
    _catalog_schema()
    with get_db() as conn:
        cur = conn.cursor()
        # A principal's verified acceptance, or an owner-approved AND accepted
        # backup administration delegation, is mandatory; roster role alone is insufficient.
        _school_admin(cur, school_id, str(actor["id"]))
        cur.execute("SELECT revision FROM verifyd_safety_choice_catalog WHERE school_id=%s FOR UPDATE",
                    (school_id,))
        row = cur.fetchone()
        current = int(row["revision"]) if row else 0
        if current != payload.expected_revision:
            raise HTTPException(409, detail={"error": "school_safety_choices_updated_elsewhere"})
        items = [choice.model_dump() for choice in payload.choices]
        cur.execute("""INSERT INTO verifyd_safety_choice_catalog
            (school_id,revision,choices,updated_by) VALUES(%s,%s,%s::jsonb,%s)
            ON CONFLICT(school_id) DO UPDATE SET
              revision=EXCLUDED.revision, choices=EXCLUDED.choices,
              updated_by=EXCLUDED.updated_by, updated_at=NOW()""",
            (school_id, current + 1, json.dumps(items), str(actor["id"])))
        _audit(cur, school_id, str(actor["id"]), "school_safety_choices_updated")
    return {"ok": True, "test_only": True, "school_id": school_id,
            "revision": current + 1, "choices": items,
            "notifications_sent": False, "emergency_dispatch": False}
