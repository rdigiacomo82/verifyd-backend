"""Separate student acceptable-use acknowledgment for controlled TEST-school access.

Deliberately NOT parental consent and not a general 911 reporting restriction.
A school may review and restrict optional TEST privileges after due process,
while retaining alternative school safety reporting methods.
"""
from __future__ import annotations
import os
import threading
import uuid
from typing import Literal, Optional
from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field
from database import get_db
from trust_voice_school_safety_training import _actor, _schema as _training_schema, _membership, _audit
from trust_voice_school_safety_registration import _school_admin

router=APIRouter(prefix="/trust-voice/school-safety",tags=["TEST Student Acceptable Use"])
CONDUCT_POLICY_VERSION="2026-11-25.student-conduct-v1"
CONDUCT_RULES=[
    "I will use VeriFYD Safety only to report a real concern or an approved school practice activity. I will not send fake, prank, threatening, or intentionally misleading reports.",
    "I will not bully, harass, impersonate another person, share another student's information, or share my school-device code or session.",
    "I will not use the app to play games with emergency procedures, interrupt classes, or contact people without a legitimate school safety purpose.",
    "I understand a good-faith request for help will not be treated as misconduct simply because the concern turns out to be mistaken.",
    "If I need immediate help, I will use my school's real emergency procedures or contact a trusted adult; this TEST service does not call 911 or guarantee delivery.",
    "My school may review suspected misuse fairly, contact my parent or guardian as appropriate, and temporarily restrict or remove TEST app privileges under its written policy. I can ask the school to review its decision.",
    "If my school restricts app access, it must provide another appropriate way for me to ask for help. Signing this does not replace parent permission or school approval.",
]
_lock=threading.Lock()
_ready=False

def _enabled(school_id:str)->bool:
    on=os.getenv("VERIFYD_SAFETY_STUDENT_CONDUCT_ENABLED","0").lower().strip() in ("1","true","on","yes")
    allowed={s.strip() for s in os.getenv("VERIFYD_SAFETY_STUDENT_CONDUCT_SCHOOL_IDS","").split(",") if s.strip()}
    return bool(on and school_id in allowed)

def _schema():
    global _ready
    _training_schema()
    if _ready:return
    with _lock:
        if _ready:return
        with get_db() as conn:
            conn.cursor().execute("""CREATE TABLE IF NOT EXISTS verifyd_safety_student_conduct_signoffs(
                school_id TEXT NOT NULL REFERENCES verifyd_safety_training_schools(school_id),
                student_id TEXT NOT NULL REFERENCES verifyd_identities(id),
                policy_version TEXT NOT NULL,
                acknowledged_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                restricted_at TIMESTAMPTZ,
                restricted_by TEXT REFERENCES verifyd_identities(id),
                review_reason TEXT NOT NULL DEFAULT '',
                reinstated_at TIMESTAMPTZ,
                reinstated_by TEXT REFERENCES verifyd_identities(id),
                PRIMARY KEY(school_id,student_id)
            )""")
        _ready=True

def _record(cur,school_id:str,student_id:str,version:str):
    if not _enabled(school_id):
        return False
    if version != CONDUCT_POLICY_VERSION:
        raise HTTPException(409,detail={"error":"student_conduct_policy_outdated"})
    _membership(cur,school_id,student_id,{"student"})
    # A suspended student cannot self-reinstate by signing again.
    cur.execute("""SELECT restricted_at FROM verifyd_safety_student_conduct_signoffs
        WHERE school_id=%s AND student_id=%s FOR UPDATE""",(school_id,student_id))
    previous=cur.fetchone()
    if previous and previous["restricted_at"] is not None:
        raise HTTPException(403,detail={"error":"school_student_practice_access_restricted"})
    cur.execute("""INSERT INTO verifyd_safety_student_conduct_signoffs
        (school_id,student_id,policy_version)
        VALUES(%s,%s,%s) ON CONFLICT(school_id,student_id)
        DO UPDATE SET policy_version=EXCLUDED.policy_version,acknowledged_at=NOW()""",
        (school_id,student_id,version))
    _audit(cur,school_id,student_id,"student_conduct_acknowledged",student_id)
    return True

def _status(cur,school_id:str,student_id:str):
    if not _enabled(school_id):
        return {"required":False,"acknowledged":True,"restricted":False}
    _schema()
    cur.execute("""SELECT policy_version,restricted_at,acknowledged_at FROM
        verifyd_safety_student_conduct_signoffs
        WHERE school_id=%s AND student_id=%s""",(school_id,student_id))
    row=cur.fetchone()
    return {"required":True,
            "acknowledged":bool(row and row["policy_version"]==CONDUCT_POLICY_VERSION),
            "restricted":bool(row and row["restricted_at"] is not None),
            "policy_version":CONDUCT_POLICY_VERSION}

def _require_accepted(cur,school_id:str,student_id:str):
    if not _enabled(school_id):
        return
    state=_status(cur,school_id,student_id)
    if state["restricted"]:
        raise HTTPException(403,detail={"error":"school_student_practice_access_restricted"})
    if not state["acknowledged"]:
        raise HTTPException(409,detail={"error":"student_conduct_acknowledgment_required"})

class StaffConductDecision(BaseModel):
    decision:Literal["restrict_test_privileges","reinstate_test_privileges"]
    reason:str=Field(min_length=20,max_length=400)
    school_confirms_alternative_safety_reporting:Literal[True]
    school_confirms_fair_review:Literal[True]

@router.get("/schools/{school_id}/student-conduct/status")
def admin_student_conduct_status(school_id:str,authorization:Optional[str]=Header(default=None)):
    if not _enabled(school_id):
        raise HTTPException(404,detail={"error":"student_conduct_pilot_not_enabled"})
    actor=_actor(authorization);_schema()
    with get_db() as conn:
        cur=conn.cursor();_school_admin(cur,school_id,str(actor["id"]))
        cur.execute("""SELECT m.identity_id,i.display_name,i.handle
            FROM verifyd_safety_training_members m
            JOIN verifyd_identities i ON i.id=m.identity_id
            WHERE m.school_id=%s AND m.role='student' ORDER BY i.handle LIMIT 1000""",
            (school_id,))
        students=[]
        for x in cur.fetchall():
            students.append({**dict(x),**_status(cur,school_id,str(x["identity_id"]))})
    return {"ok":True,"students":students,"policy_version":CONDUCT_POLICY_VERSION,
            "test_only":True,"emergency_access_not_guaranteed":True}

@router.post("/schools/{school_id}/student-conduct/{student_id}/decision")
def admin_student_conduct_decision(school_id:str,student_id:str,payload:StaffConductDecision,
    authorization:Optional[str]=Header(default=None)):
    if not _enabled(school_id):
        raise HTTPException(404,detail={"error":"student_conduct_pilot_not_enabled"})
    actor=_actor(authorization);_schema()
    with get_db() as conn:
        cur=conn.cursor()
        _school_admin(cur,school_id,str(actor["id"]))
        _membership(cur,school_id,student_id,{"student"})
        cur.execute("""SELECT restricted_at FROM verifyd_safety_student_conduct_signoffs
            WHERE school_id=%s AND student_id=%s FOR UPDATE""",(school_id,student_id))
        existing=cur.fetchone()
        if not existing:
            raise HTTPException(409,detail={"error":"student_conduct_first_acknowledgment_required"})
        if payload.decision=="restrict_test_privileges":
            if existing["restricted_at"] is not None:
                raise HTTPException(409,detail={"error":"student_practice_already_restricted"})
            cur.execute("""UPDATE verifyd_safety_student_conduct_signoffs
                SET restricted_at=NOW(),restricted_by=%s,review_reason=%s,
                    reinstated_at=NULL,reinstated_by=NULL
                WHERE school_id=%s AND student_id=%s""",
                (str(actor["id"]),payload.reason,school_id,student_id))
        else:
            if existing["restricted_at"] is None:
                raise HTTPException(409,detail={"error":"student_practice_not_restricted"})
            cur.execute("""UPDATE verifyd_safety_student_conduct_signoffs
                SET restricted_at=NULL,reinstated_at=NOW(),reinstated_by=%s,
                    review_reason=%s
                WHERE school_id=%s AND student_id=%s""",
                (str(actor["id"]),payload.reason,school_id,student_id))
        _audit(cur,school_id,str(actor["id"]),
               "student_conduct_"+payload.decision,student_id)
    return {"ok":True,"test_only":True,"decision":payload.decision,
            "school_alternative_reporting_required":True,
            "real_emergency_services_not_affected":True}
