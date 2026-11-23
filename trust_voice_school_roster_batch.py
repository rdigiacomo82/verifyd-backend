"""Training-only school roster batch intake and school-scoped reference labels.

The @handle identifies an existing VeriFYD account. The short reference is a
random school-local display label, not an authentication credential or a claim
that student information is outside FERPA/NYS ED Law 2-d.
"""
from __future__ import annotations
import re
import secrets
import threading
from typing import Literal, Optional
from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field
from database import get_db
from trust_voice_school_safety_training import _actor, _audit, _id_for_handle
from trust_voice_school_safety_registration import _application_schema, _school_admin

router = APIRouter(prefix="/trust-voice/school-safety", tags=["School Safety Roster Batch"])
_lock=threading.Lock()
_ready=False
_HANDLE=re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.-]{2,39}$")


def _batch_schema():
    global _ready
    _application_schema()
    if _ready:
        return
    with _lock:
        if _ready:
            return
        with get_db() as conn:
            conn.cursor().execute("""CREATE TABLE IF NOT EXISTS verifyd_safety_roster_references (
                school_id TEXT NOT NULL REFERENCES verifyd_safety_training_schools(school_id),
                identity_id TEXT NOT NULL REFERENCES verifyd_identities(id),
                reference TEXT NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                PRIMARY KEY(school_id,identity_id),
                UNIQUE(school_id,reference)
            )""")
        _ready=True


class BatchInput(BaseModel):
    handles: list[str] = Field(min_length=1, max_length=50)
    role: Literal["student","guardian"] = "student"


def _normalized(items):
    normalized=[]
    seen=set()
    for item in items:
        h=str(item).strip().lstrip("@")
        if not _HANDLE.fullmatch(h):
            raise HTTPException(422,detail={"error":"invalid_handle_in_batch"})
        if h.lower() in seen:
            raise HTTPException(422,detail={"error":"duplicate_handle_in_batch"})
        seen.add(h.lower())
        normalized.append(h)
    return normalized


def _resolve(cur,school_id,handles):
    rows=[]
    for handle in handles:
        uid=_id_for_handle(handle)
        cur.execute("""SELECT i.handle,i.display_name,m.role,r.reference
            FROM verifyd_identities i
            LEFT JOIN verifyd_safety_training_members m
              ON m.identity_id=i.id AND m.school_id=%s
            LEFT JOIN verifyd_safety_roster_references r
              ON r.identity_id=i.id AND r.school_id=%s
            WHERE i.id=%s""",(school_id,school_id,uid))
        row=cur.fetchone()
        if not row:
            raise HTTPException(404,detail={"error":"identity_not_found"})
        rows.append({"identity_id":uid,"handle":row["handle"],
                     "display_name":row["display_name"] or "",
                     "current_role":row["role"],"reference":row["reference"]})
    return rows


@router.post("/schools/{school_id}/roster/batch/preview")
def preview_batch(school_id: str,payload: BatchInput,authorization: Optional[str]=Header(default=None)):
    actor=_actor(authorization)
    _batch_schema()
    handles=_normalized(payload.handles)
    with get_db() as conn:
        cur=conn.cursor()
        _school_admin(cur,school_id,str(actor["id"]))
        rows=_resolve(cur,school_id,handles)
    # Never return emails, birthdays, DOE student numbers, addresses or tokens.
    for row in rows:
        row["action"]="add" if row["current_role"] is None else "already_enrolled"
        row.pop("identity_id")
    return {"ok":True,"rows":rows,"requested_role":payload.role,
            "training_only":True,"emergency_alerts_active":False}


@router.post("/schools/{school_id}/roster/batch/commit")
def commit_batch(school_id: str,payload: BatchInput,authorization: Optional[str]=Header(default=None)):
    actor=_actor(authorization)
    _batch_schema()
    actor_id=str(actor["id"])
    handles=_normalized(payload.handles)
    with get_db() as conn:
        cur=conn.cursor()
        _school_admin(cur,school_id,actor_id)
        rows=_resolve(cur,school_id,handles)
        if any(row["identity_id"]==actor_id for row in rows):
            raise HTTPException(403,detail={"error":"cannot_change_own_school_role"})
        if any(row["current_role"] is not None and row["current_role"]!=payload.role for row in rows):
            raise HTTPException(409,detail={"error":"conflicting_existing_role_review_each_account"})
        added=0
        skipped=0
        for row in rows:
            if row["current_role"]:
                skipped+=1
                continue
            cur.execute("""INSERT INTO verifyd_safety_training_members
              (school_id,identity_id,role,verified_by)
              VALUES(%s,%s,%s,%s) ON CONFLICT DO NOTHING""",
              (school_id,row["identity_id"],payload.role,actor_id))
            if cur.rowcount:
                added+=1
                _audit(cur,school_id,actor_id,"school_batch_member_added",row["identity_id"])
            else:
                skipped+=1
        # Non-guessable school-specific display references. Uniqueness enforced by DB.
        for row in rows:
            if row["current_role"] is None or row["current_role"]==payload.role:
                for attempt in range(4):
                    ref="VS-"+secrets.token_hex(4).upper()
                    cur.execute("""INSERT INTO verifyd_safety_roster_references
                        (school_id,identity_id,reference) VALUES(%s,%s,%s)
                        ON CONFLICT(school_id,identity_id) DO NOTHING""",
                        (school_id,row["identity_id"],ref))
                    if cur.rowcount or row["reference"]:
                        break
        _audit(cur,school_id,actor_id,"school_batch_intake_completed",str(added))
    return {"ok":True,"added":added,"skipped":skipped,"training_only":True,
            "emergency_alerts_active":False}


@router.get("/schools/{school_id}/roster/references")
def list_roster_references(school_id: str,authorization: Optional[str]=Header(default=None)):
    actor=_actor(authorization)
    _batch_schema()
    with get_db() as conn:
        cur=conn.cursor()
        _school_admin(cur,school_id,str(actor["id"]))
        cur.execute("""SELECT m.identity_id,i.handle,i.display_name,m.role,r.reference
            FROM verifyd_safety_training_members m
            JOIN verifyd_identities i ON i.id=m.identity_id
            LEFT JOIN verifyd_safety_roster_references r
              ON r.school_id=m.school_id AND r.identity_id=m.identity_id
            WHERE m.school_id=%s ORDER BY m.role,i.handle""",(school_id,))
        rows=[dict(x) for x in cur.fetchall()]
    return {"ok":True,"members":rows,"training_only":True}
