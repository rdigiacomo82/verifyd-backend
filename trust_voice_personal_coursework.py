"""User-private coursework journal.

Available to any active VeriFYD Circle account when classroom pilot is enabled.
This is intentionally separate from shared classes and existing Homework.
Private notes are never exposed to teachers, classmates, or public profiles.
"""
import uuid
import threading
from datetime import datetime
from typing import Literal, Optional

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field

from database import get_db
from trust_voice_classrooms import _identity, _schema as class_schema

router = APIRouter(prefix="/trust-voice/classroom", tags=["Circle Personal Learning"])
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
            cur=conn.cursor()
            cur.execute("""CREATE TABLE IF NOT EXISTS verifyd_circle_personal_coursework(
                coursework_id TEXT PRIMARY KEY,
                identity_id TEXT NOT NULL REFERENCES verifyd_identities(id) ON DELETE CASCADE,
                title TEXT NOT NULL,
                assignment TEXT NOT NULL DEFAULT '',
                response TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL DEFAULT 'in_progress'
                    CHECK (status IN ('in_progress','completed')),
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )""")
            cur.execute("""CREATE INDEX IF NOT EXISTS idx_circle_personal_coursework_owner
                ON verifyd_circle_personal_coursework(identity_id,updated_at DESC)""")
        _READY = True

class PersonalCreate(BaseModel):
    title: str = Field(min_length=3,max_length=160)
    assignment: str = Field(default="",max_length=16000)


class PersonalUpdate(BaseModel):
    response: str = Field(default="",max_length=30000)
    action: Literal["save","complete"] = "save"


def public(row):
    result = dict(row)
    for key in ("created_at","updated_at"):
        if isinstance(result.get(key), datetime):
            result[key] = result[key].isoformat()
    return result


@router.get("/personal")
def list_private(authorization: Optional[str] = Header(default=None)):
    user=_identity(authorization)
    schema()
    with get_db() as conn:
        cur=conn.cursor()
        cur.execute("""SELECT coursework_id,title,assignment,response,status,created_at,updated_at
            FROM verifyd_circle_personal_coursework
            WHERE identity_id=%s ORDER BY updated_at DESC LIMIT 200""",(user,))
        rows=[public(row) for row in cur.fetchall()]
    return {"private_work":rows, "teacher_access":False,"auto_rank_points":False}


@router.post("/personal")
def create_private(payload: PersonalCreate,authorization: Optional[str] = Header(default=None)):
    user=_identity(authorization)
    schema()
    key=str(uuid.uuid4())
    with get_db() as conn:
        cur=conn.cursor()
        cur.execute("SELECT COUNT(*) AS n FROM verifyd_circle_personal_coursework WHERE identity_id=%s",(user,))
        if cur.fetchone()["n"]>=250:
            raise HTTPException(429,detail={"error":"personal_coursework_limit"})
        cur.execute("""INSERT INTO verifyd_circle_personal_coursework
            (coursework_id,identity_id,title,assignment)
            VALUES (%s,%s,%s,%s)""",(key,user,payload.title,payload.assignment))
    return {"coursework_id":key,"status":"in_progress"}


@router.put("/personal/{coursework_id}")
def update_private(coursework_id: str,payload: PersonalUpdate,
                   authorization: Optional[str] = Header(default=None)):
    user=_identity(authorization)
    schema()
    new_status="completed" if payload.action=="complete" else "in_progress"
    with get_db() as conn:
        cur=conn.cursor()
        cur.execute("""UPDATE verifyd_circle_personal_coursework
            SET response=%s,status=%s,updated_at=NOW()
            WHERE coursework_id=%s AND identity_id=%s
            RETURNING coursework_id,title,assignment,response,status,created_at,updated_at""",
            (payload.response,new_status,coursework_id,user))
        item=cur.fetchone()
        if not item:
            raise HTTPException(404,detail={"error":"private_work_not_found"})
    return {"work":public(item),"auto_rank_points":0}


@router.delete("/personal/{coursework_id}")
def delete_private(coursework_id: str,authorization: Optional[str] = Header(default=None)):
    user=_identity(authorization)
    schema()
    with get_db() as conn:
        cur=conn.cursor()
        cur.execute("""DELETE FROM verifyd_circle_personal_coursework
            WHERE coursework_id=%s AND identity_id=%s RETURNING coursework_id""",(coursework_id,user))
        if not cur.fetchone():
            raise HTTPException(404,detail={"error":"private_work_not_found"})
    return {"deleted":True}
