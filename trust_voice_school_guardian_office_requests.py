"""Guardian-to-school OFFICE CHECK-IN requests, isolated TRAINING pilot.

This is not a direct distress signal to a minor. A verified guardian may request
staff to contact their student. Only a verified school responder can approve a
neutral student-device instruction. Staff must follow school procedures; no
911, outbound SMS, guaranteed push or automatic emergency dispatch.
"""
from __future__ import annotations
import os
import threading
import uuid
from datetime import datetime, timezone
from typing import Literal, Optional
from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field
from database import get_db
from trust_voice_school_safety_training import (
    _actor, _schema as _training_schema, _audit,
    _membership, _verified_training_contact, _school_response_authorized,
)
from trust_voice_school_safety_consents import _consent_schema

router = APIRouter(prefix="/trust-voice/school-safety", tags=["TEST Guardian Office Requests"])
_lock=threading.Lock()
_ready=False

def _gate(school_id:str):
    on=os.getenv("VERIFYD_SAFETY_GUARDIAN_OFFICE_REQUEST_ENABLED","0").lower() in ("1","true","on","yes")
    schools={x.strip() for x in os.getenv("VERIFYD_SAFETY_GUARDIAN_OFFICE_REQUEST_SCHOOL_IDS","").split(",") if x.strip()}
    if not on or school_id not in schools:
        raise HTTPException(404,detail={"error":"guardian_office_requests_not_enabled_for_school"})

def _schema():
    global _ready
    _training_schema()
    _consent_schema()
    if _ready:return
    with _lock:
        if _ready:return
        with get_db() as conn:
            conn.cursor().execute("""CREATE TABLE IF NOT EXISTS verifyd_safety_guardian_office_requests(
                request_id TEXT PRIMARY KEY,
                school_id TEXT NOT NULL REFERENCES verifyd_safety_training_schools(school_id),
                student_id TEXT NOT NULL REFERENCES verifyd_identities(id),
                guardian_id TEXT NOT NULL REFERENCES verifyd_identities(id),
                purpose TEXT NOT NULL CHECK(purpose IN ('urgent_family_contact','school_office_check_in','pickup_coordination')),
                status TEXT NOT NULL DEFAULT 'pending_school'
                  CHECK(status IN ('pending_school','approved_to_notify','declined','completed')),
                abuse_policy_acknowledged_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                requested_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                reviewed_at TIMESTAMPTZ,
                reviewed_by TEXT REFERENCES verifyd_identities(id),
                school_note TEXT NOT NULL DEFAULT '',
                student_viewed_at TIMESTAMPTZ,
                completed_at TIMESTAMPTZ
            )""")
            conn.cursor().execute("""CREATE INDEX IF NOT EXISTS safety_office_guardian_rate
                ON verifyd_safety_guardian_office_requests
                  (school_id,guardian_id,requested_at DESC)""")
            conn.cursor().execute("""CREATE INDEX IF NOT EXISTS safety_office_staff_queue
                ON verifyd_safety_guardian_office_requests
                  (school_id,status,requested_at DESC)""")
        _ready=True

class GuardianOfficeRequest(BaseModel):
    school_id:str=Field(min_length=8,max_length=100)
    student_id:str=Field(min_length=8,max_length=120)
    purpose:Literal["urgent_family_contact","school_office_check_in","pickup_coordination"]
    understands_school_mediated_not_emergency:Literal[True]
    agrees_not_to_abuse_school_requests:Literal[True]

class StaffDecision(BaseModel):
    decision:Literal["approved_to_notify","declined","completed"]
    staff_confirms_school_follow_up:Literal[True]
    note:str=Field(default="",max_length=250)

def _guardian_children(cur,guardian_id):
    cur.execute("""SELECT g.school_id,g.child_id,s.school_name,
         i.display_name AS child_name,i.handle AS child_handle
       FROM verifyd_safety_training_guardians g
       JOIN verifyd_safety_training_schools s ON s.school_id=g.school_id
       JOIN verifyd_identities i ON i.id=g.child_id AND i.status='active'
       WHERE g.guardian_id=%s ORDER BY s.school_name,i.display_name""",(guardian_id,))
    eligible=[]
    for row in cur.fetchall():
        child=dict(row)
        try:
            _gate(str(child["school_id"]))
            _verified_training_contact(cur,str(child["school_id"]),
                                       str(child["child_id"]),guardian_id)
        except HTTPException:
            continue
        eligible.append(child)
    return eligible

@router.get("/family/office-requests")
def my_office_requests(authorization:Optional[str]=Header(default=None)):
    actor=_actor(authorization); actor_id=str(actor["id"])
    _schema()
    with get_db() as conn:
        cur=conn.cursor()
        children=_guardian_children(cur,actor_id)
        cur.execute("""SELECT request_id,school_id,student_id,purpose,status,
               requested_at,reviewed_at,completed_at,student_viewed_at
            FROM verifyd_safety_guardian_office_requests
            WHERE guardian_id=%s ORDER BY requested_at DESC LIMIT 40""",(actor_id,))
        requests=[dict(row) for row in cur.fetchall()]
    return {"ok":True,"eligible_children":children,"requests":requests,
            "school_mediated":True,"direct_student_distress":False,
            "live_emergency_dispatch":False,"device_delivery_guaranteed":False}

@router.post("/family/office-requests")
def create_office_request(payload:GuardianOfficeRequest,authorization:Optional[str]=Header(default=None)):
    _gate(payload.school_id)
    actor=_actor(authorization);guardian_id=str(actor["id"])
    _schema()
    with get_db() as conn:
        cur=conn.cursor()
        _verified_training_contact(cur,payload.school_id,payload.student_id,guardian_id)
        cur.execute("SELECT pg_advisory_xact_lock(hashtext(%s))",
                    ("guardian_office:"+payload.school_id+":"+guardian_id,))
        cur.execute("""SELECT COUNT(*) AS n FROM verifyd_safety_guardian_office_requests
            WHERE school_id=%s AND guardian_id=%s AND requested_at>NOW()-INTERVAL '24 hours'""",
            (payload.school_id,guardian_id))
        if int(cur.fetchone()["n"])>=2:
            raise HTTPException(429,detail={"error":"guardian_office_daily_limit_use_school_phone"})
        cur.execute("""SELECT COUNT(*) AS n FROM verifyd_safety_guardian_office_requests
            WHERE school_id=%s AND guardian_id=%s AND student_id=%s
             AND status IN ('pending_school','approved_to_notify')""",
            (payload.school_id,guardian_id,payload.student_id))
        if int(cur.fetchone()["n"]):
            raise HTTPException(409,detail={"error":"guardian_office_request_already_open"})
        # A verified school responder MUST exist. Never promise a delivery
        # when no school is staffed for this TEST child.
        cur.execute("""SELECT identity_id FROM verifyd_safety_training_members
            WHERE school_id=%s AND role='principal'
            UNION SELECT responder_id AS identity_id
              FROM verifyd_safety_school_response_team WHERE school_id=%s""",
            (payload.school_id,payload.school_id))
        candidates=[str(r["identity_id"]) for r in cur.fetchall()]
        responders=0
        for candidate in candidates:
            try:
                _school_response_authorized(cur,payload.school_id,candidate)
                responders+=1
            except HTTPException:
                continue
        if not responders:
            raise HTTPException(409,detail={"error":"no_verified_school_response_recipients"})
        request_id="sfoffice_"+uuid.uuid4().hex
        cur.execute("""INSERT INTO verifyd_safety_guardian_office_requests
            (request_id,school_id,student_id,guardian_id,purpose)
            VALUES(%s,%s,%s,%s,%s)""",
            (request_id,payload.school_id,payload.student_id,guardian_id,payload.purpose))
        _audit(cur,payload.school_id,guardian_id,"guardian_office_request_created",request_id)
    return {"ok":True,"request_id":request_id,"status":"pending_school",
            "school_review_required":True,"student_not_notified":True,
            "emergency_services_not_contacted":True}

@router.get("/schools/{school_id}/guardian-office-requests")
def staff_office_queue(school_id:str,authorization:Optional[str]=Header(default=None)):
    _gate(school_id)
    actor=_actor(authorization);actor_id=str(actor["id"])
    _schema()
    with get_db() as conn:
        cur=conn.cursor()
        _school_response_authorized(cur,school_id,actor_id)
        cur.execute("""SELECT q.request_id,q.student_id,q.purpose,q.status,
                q.requested_at,q.reviewed_at,q.student_viewed_at,q.completed_at,
                child.display_name AS child_name,child.handle AS child_handle
            FROM verifyd_safety_guardian_office_requests q
            JOIN verifyd_identities child ON child.id=q.student_id
            WHERE q.school_id=%s ORDER BY q.requested_at DESC LIMIT 100""",
            (school_id,))
        items=[dict(x) for x in cur.fetchall()]
    return {"ok":True,"requests":items,"school_mediated":True,
            "private_family_conversations_exposed":False}

@router.post("/schools/{school_id}/guardian-office-requests/{request_id}/review")
def staff_review_office_request(school_id:str,request_id:str,payload:StaffDecision,
    authorization:Optional[str]=Header(default=None)):
    _gate(school_id)
    actor=_actor(authorization);actor_id=str(actor["id"])
    _schema()
    with get_db() as conn:
        cur=conn.cursor()
        _school_response_authorized(cur,school_id,actor_id)
        cur.execute("""SELECT * FROM verifyd_safety_guardian_office_requests
            WHERE school_id=%s AND request_id=%s FOR UPDATE""",(school_id,request_id))
        request=cur.fetchone()
        if not request:
            raise HTTPException(404,detail={"error":"office_request_not_found"})
        status=str(request["status"])
        if payload.decision in ("approved_to_notify","declined"):
            if status!="pending_school":
                raise HTTPException(409,detail={"error":"office_request_already_reviewed"})
        elif payload.decision=="completed" and status!="approved_to_notify":
            raise HTTPException(409,detail={"error":"office_request_not_yet_approved"})
        cur.execute("""UPDATE verifyd_safety_guardian_office_requests SET
            status=%s,reviewed_by=%s,reviewed_at=NOW(),school_note=%s,
            completed_at=CASE WHEN %s='completed' THEN NOW() ELSE completed_at END
            WHERE request_id=%s""",
            (payload.decision,actor_id,payload.note.strip(),payload.decision,request_id))
        _audit(cur,school_id,actor_id,"guardian_office_request_"+payload.decision,request_id)
    return {"ok":True,"status":payload.decision,"school_staff_action_required":True,
            "device_delivery_guaranteed":False,"emergency_dispatch":False}

@router.get("/student/office-notices")
def student_office_notices(authorization:Optional[str]=Header(default=None)):
    actor=_actor(authorization);student_id=str(actor["id"])
    _schema()
    with get_db() as conn:
        cur=conn.cursor()
        cur.execute("""SELECT q.request_id,q.school_id,s.school_name,q.requested_at,
               q.reviewed_at,q.student_viewed_at
            FROM verifyd_safety_guardian_office_requests q
            JOIN verifyd_safety_training_members m
              ON m.school_id=q.school_id AND m.identity_id=q.student_id AND m.role='student'
            JOIN verifyd_safety_training_schools s ON s.school_id=q.school_id
            WHERE q.student_id=%s AND q.status='approved_to_notify'
              AND q.reviewed_at>NOW()-INTERVAL '1 day'
            ORDER BY q.reviewed_at DESC LIMIT 10""",(student_id,))
        notices=[dict(x) for x in cur.fetchall()]
    return {"ok":True,"notices":notices,"staff_approved_only":True,
            "student_device_delivery_guaranteed":False}

@router.post("/student/office-notices/{request_id}/seen")
def student_seen_office_notice(request_id:str,authorization:Optional[str]=Header(default=None)):
    actor=_actor(authorization);student_id=str(actor["id"])
    _schema()
    with get_db() as conn:
        cur=conn.cursor()
        cur.execute("""UPDATE verifyd_safety_guardian_office_requests q
            SET student_viewed_at=COALESCE(student_viewed_at,NOW())
            WHERE q.request_id=%s AND q.student_id=%s AND q.status='approved_to_notify'
              AND EXISTS(SELECT 1 FROM verifyd_safety_training_members m
                  WHERE m.school_id=q.school_id AND m.identity_id=%s AND m.role='student')
            RETURNING q.school_id""",(request_id,student_id,student_id))
        found=cur.fetchone()
        if not found:
            raise HTTPException(403,detail={"error":"student_office_notice_not_authorized"})
        _audit(cur,str(found["school_id"]),student_id,"student_office_notice_seen",request_id)
    return {"ok":True,"student_viewed":True,"staff_confirmation_still_required":True}
