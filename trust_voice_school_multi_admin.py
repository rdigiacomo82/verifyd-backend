"""Multi-administrator nominations, owner approvals and school-scoped access.

Principals nominate EXISTING school response accounts, VeriFYD operator
approves, and each nominee accepts. Nomination never grants access.
No real emergency alert transmission is enabled by these endpoints.
"""
from __future__ import annotations
import threading
import uuid
from typing import Optional, Literal
from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field
from database import get_db
from trust_voice_school_safety_training import _actor,_audit,_membership,_id_for_handle
from trust_voice_school_safety_registration import _application_schema,_owner,_school_admin

router=APIRouter(prefix="/trust-voice/school-safety",tags=["School Multi-Administrator Authorization"])
_lock=threading.Lock()
_ready=False

def _schema():
    global _ready
    _application_schema()
    if _ready:return
    with _lock:
        if _ready:return
        with get_db() as conn:
            conn.cursor().execute("""CREATE TABLE IF NOT EXISTS verifyd_safety_school_admin_nominations(
               nomination_id TEXT PRIMARY KEY,
               school_id TEXT NOT NULL REFERENCES verifyd_safety_training_schools(school_id),
               candidate_id TEXT NOT NULL REFERENCES verifyd_identities(id),
               nominated_by TEXT NOT NULL REFERENCES verifyd_identities(id),
               status TEXT NOT NULL CHECK(status IN ('pending','approved','rejected','cancelled')),
               reason TEXT NOT NULL,
               created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
               reviewed_by TEXT REFERENCES verifyd_identities(id),
               reviewed_at TIMESTAMPTZ
            )""")
            conn.cursor().execute("""CREATE UNIQUE INDEX IF NOT EXISTS safety_one_pending_nomination_per_admin
                ON verifyd_safety_school_admin_nominations(school_id,candidate_id)
                WHERE status='pending'""")
        _ready=True

class Nominate(BaseModel):
    candidate_handle: str=Field(min_length=3,max_length=40)
    reason: str=Field(min_length=12,max_length=500)
class Decide(BaseModel):
    decision: Literal["approve","reject"]
    verified_staff_authority: bool=False
class Revoke(BaseModel):
    reason: str=Field(min_length=12,max_length=500)

@router.get("/schools/{school_id}/administrators")
def school_administrators(school_id:str,authorization:Optional[str]=Header(default=None)):
    actor=_actor(authorization);_schema()
    with get_db() as conn:
        cur=conn.cursor()
        # Principal sees nominations, accepted delegates see current admin list.
        role=_school_admin(cur,school_id,str(actor["id"]))
        cur.execute("""SELECT d.identity_id,i.handle,i.display_name,
          d.accepted_at,d.approved_at
          FROM verifyd_safety_school_admin_delegations d
          JOIN verifyd_identities i ON i.id=d.identity_id
          WHERE d.school_id=%s AND d.revoked_at IS NULL
          ORDER BY d.approved_at DESC""",(school_id,))
        admins=[dict(x) for x in cur.fetchall()]
        nominations=[]
        candidates=[]
        if role=="principal":
            cur.execute("""SELECT n.nomination_id,n.candidate_id,i.handle,i.display_name,
              n.status,n.reason,n.created_at
              FROM verifyd_safety_school_admin_nominations n
              JOIN verifyd_identities i ON i.id=n.candidate_id
              WHERE n.school_id=%s ORDER BY n.created_at DESC LIMIT 100""",(school_id,))
            nominations=[dict(x) for x in cur.fetchall()]
            cur.execute("""SELECT m.identity_id,i.handle,i.display_name
              FROM verifyd_safety_training_members m
              JOIN verifyd_identities i ON i.id=m.identity_id
              WHERE m.school_id=%s AND m.role='responder' AND i.status='active'
                AND NOT EXISTS (SELECT 1 FROM verifyd_safety_school_admin_delegations d
                   WHERE d.school_id=m.school_id AND d.identity_id=m.identity_id
                     AND d.revoked_at IS NULL)
                AND NOT EXISTS (SELECT 1 FROM verifyd_safety_school_admin_nominations n
                   WHERE n.school_id=m.school_id AND n.candidate_id=m.identity_id
                     AND n.status='pending')
              ORDER BY i.handle LIMIT 100""",(school_id,))
            candidates=[dict(x) for x in cur.fetchall()]
        _audit(cur,school_id,str(actor["id"]),"school_administrator_directory_viewed",school_id)
    return {"ok":True,"administrators":admins,"nominations":nominations,"candidates":candidates,
            "max_additional_administrators":10,
            "emergency_delivery_active":False}

@router.post("/schools/{school_id}/administrators/nominate")
def nominate_administrator(school_id:str,payload:Nominate,authorization:Optional[str]=Header(default=None)):
    actor=_actor(authorization);_schema()
    candidate_id=_id_for_handle(payload.candidate_handle)
    aid=str(actor["id"])
    if candidate_id==aid:raise HTTPException(403,detail={"error":"cannot_nominate_self"})
    with get_db() as conn:
        cur=conn.cursor()
        if _school_admin(cur,school_id,aid)!="principal":
            raise HTTPException(403,detail={"error":"principal_nomination_required"})
        _membership(cur,school_id,candidate_id,{"responder"})
        cur.execute("SELECT 1 FROM verifyd_safety_school_admin_delegations WHERE school_id=%s AND identity_id=%s AND revoked_at IS NULL",(school_id,candidate_id))
        if cur.fetchone():raise HTTPException(409,detail={"error":"candidate_already_approved"})
        cur.execute("""SELECT 1 FROM verifyd_safety_school_admin_nominations WHERE school_id=%s
            AND candidate_id=%s AND status='pending'""",(school_id,candidate_id))
        if cur.fetchone():raise HTTPException(409,detail={"error":"nomination_already_pending"})
        nom_id="ssadm_"+uuid.uuid4().hex
        cur.execute("""INSERT INTO verifyd_safety_school_admin_nominations(
           nomination_id,school_id,candidate_id,nominated_by,status,reason)
           VALUES(%s,%s,%s,%s,'pending',%s)""",(nom_id,school_id,candidate_id,aid,payload.reason.strip()))
        _audit(cur,school_id,aid,"additional_school_admin_nominated",nom_id)
    return {"ok":True,"nomination_id":nom_id,"status":"pending_owner_review",
            "administrator_access_granted":False,"emergency_delivery_active":False}

@router.get("/owner/administrator-nominations")
def operator_admin_nominations(authorization:Optional[str]=Header(default=None)):
    owner=_actor(authorization);_owner(owner);_schema()
    with get_db() as conn:
        cur=conn.cursor()
        cur.execute("""SELECT n.nomination_id,n.school_id,s.school_name,n.status,n.reason,
             n.created_at,nom.handle AS nominated_by_handle,
             i.handle AS candidate_handle,i.display_name AS candidate_name
             FROM verifyd_safety_school_admin_nominations n
             JOIN verifyd_safety_training_schools s ON s.school_id=n.school_id
             JOIN verifyd_identities i ON i.id=n.candidate_id
             JOIN verifyd_identities nom ON nom.id=n.nominated_by
             WHERE n.status='pending' ORDER BY n.created_at ASC LIMIT 100""")
        rows=[dict(x) for x in cur.fetchall()]
    return {"ok":True,"nominations":rows}

@router.post("/owner/administrator-nominations/{nomination_id}/decide")
def operator_decide(nomination_id:str,payload:Decide,authorization:Optional[str]=Header(default=None)):
    owner=_actor(authorization);_owner(owner);_schema()
    if payload.decision=="approve" and not payload.verified_staff_authority:
        raise HTTPException(422,detail={"error":"independent_staff_verification_required"})
    owner_id=str(owner["id"])
    with get_db() as conn:
        cur=conn.cursor()
        cur.execute("""SELECT * FROM verifyd_safety_school_admin_nominations
              WHERE nomination_id=%s FOR UPDATE""",(nomination_id,))
        n=cur.fetchone()
        if not n:raise HTTPException(404,detail={"error":"nomination_not_found"})
        if n["status"]!="pending":raise HTTPException(409,detail={"error":"nomination_already_decided"})
        cur.execute("SELECT school_id FROM verifyd_safety_training_schools WHERE school_id=%s FOR UPDATE",(n["school_id"],))
        _membership(cur,n["school_id"],n["candidate_id"],{"responder"})
        if payload.decision=="approve":
            if str(n["candidate_id"])==owner_id:
                raise HTTPException(403,detail={"error":"owner_cannot_delegate_self"})
            cur.execute("""SELECT count(*) AS n FROM verifyd_safety_school_admin_delegations
                WHERE school_id=%s AND revoked_at IS NULL""",(n["school_id"],))
            count=int(cur.fetchone()["n"])
            cur.execute("""SELECT 1 FROM verifyd_safety_school_admin_delegations
                WHERE school_id=%s AND identity_id=%s AND revoked_at IS NULL""",
                (n["school_id"],n["candidate_id"]))
            active=bool(cur.fetchone())
            if count>=10 and not active:
                raise HTTPException(409,detail={"error":"school_admin_limit_reached"})
            if active:raise HTTPException(409,detail={"error":"candidate_already_approved"})
            cur.execute("""INSERT INTO verifyd_safety_school_admin_delegations
               (school_id,identity_id,approved_by,approved_at,accepted_at,revoked_at)
               VALUES(%s,%s,%s,NOW(),NULL,NULL)
               ON CONFLICT(school_id,identity_id) DO UPDATE SET
                  approved_by=EXCLUDED.approved_by,approved_at=NOW(),
                  accepted_at=NULL,revoked_at=NULL""",
                (n["school_id"],n["candidate_id"],owner_id))
        cur.execute("""UPDATE verifyd_safety_school_admin_nominations
            SET status=%s,reviewed_by=%s,reviewed_at=NOW() WHERE nomination_id=%s""",
            ("approved" if payload.decision=="approve" else "rejected",owner_id,nomination_id))
        _audit(cur,n["school_id"],owner_id,"additional_school_admin_"+payload.decision,nomination_id)
    return {"ok":True,"status":"awaiting_candidate_acceptance" if payload.decision=="approve" else "rejected",
            "emergency_delivery_active":False}

@router.post("/owner/schools/{school_id}/administrators/{identity_id}/revoke")
def owner_revoke_administrator(school_id:str,identity_id:str,payload:Revoke,
    authorization:Optional[str]=Header(default=None)):
    owner=_actor(authorization);_owner(owner);_schema()
    with get_db() as conn:
        cur=conn.cursor()
        cur.execute("""UPDATE verifyd_safety_school_admin_delegations
            SET revoked_at=NOW(),accepted_at=NULL
            WHERE school_id=%s AND identity_id=%s AND revoked_at IS NULL RETURNING identity_id""",
            (school_id,identity_id))
        if not cur.fetchone():raise HTTPException(404,detail={"error":"delegation_not_found"})
        _audit(cur,school_id,str(owner["id"]),"additional_school_admin_revoked",identity_id)
    return {"ok":True,"revoked":True,"emergency_delivery_active":False}
