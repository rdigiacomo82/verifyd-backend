"""Guardian-requested school contact priority changes.

A verified linked guardian may request contact-order changes for up to five
children in one school. The other proposed guardian must confirm, and an
authorized school administrator must independently verify and approve.
Current contacts never change before both checks. No emergency dispatch.
"""
from __future__ import annotations

import threading
import uuid
from typing import Optional, Literal
from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field
from database import get_db
from emailer import _send, FROM_NAME, FROM_ADDRESS
from trust_voice_school_safety_training import _actor, _audit, _schema as _school_schema
from trust_voice_school_safety_registration import _school_admin
from trust_voice_school_safety_consents import _consent_schema, POLICY_VERSION

router = APIRouter(prefix="/trust-voice/school-safety", tags=["Verified Family Contact Changes"])
_ready = False
_lock = threading.Lock()

def _schema():
    global _ready
    _consent_schema()
    if _ready: return
    with _lock:
        if _ready: return
        with get_db() as conn:
            cur=conn.cursor()
            cur.execute("""CREATE TABLE IF NOT EXISTS verifyd_safety_guardian_contact_changes (
                change_id TEXT PRIMARY KEY,
                school_id TEXT NOT NULL REFERENCES verifyd_safety_training_schools(school_id),
                student_id TEXT NOT NULL REFERENCES verifyd_identities(id),
                requested_by TEXT NOT NULL REFERENCES verifyd_identities(id),
                proposed_primary_id TEXT NOT NULL REFERENCES verifyd_identities(id),
                proposed_backup_id TEXT NOT NULL REFERENCES verifyd_identities(id),
                previous_primary_id TEXT REFERENCES verifyd_identities(id),
                previous_backup_id TEXT REFERENCES verifyd_identities(id),
                reason TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending'
                    CHECK(status IN ('pending','approved','rejected','cancelled','superseded')),
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                reviewed_by TEXT REFERENCES verifyd_identities(id),
                reviewed_at TIMESTAMPTZ,
                review_note TEXT,
                CHECK(proposed_primary_id<>proposed_backup_id)
            )""")
            cur.execute("""CREATE UNIQUE INDEX IF NOT EXISTS safety_one_pending_guardian_change
                ON verifyd_safety_guardian_contact_changes(school_id,student_id)
                WHERE status='pending'""")
            cur.execute("""CREATE TABLE IF NOT EXISTS verifyd_safety_guardian_contact_confirmations (
                change_id TEXT NOT NULL REFERENCES verifyd_safety_guardian_contact_changes(change_id) ON DELETE CASCADE,
                guardian_id TEXT NOT NULL REFERENCES verifyd_identities(id),
                decision TEXT NOT NULL CHECK(decision IN ('accepted','declined')),
                recorded_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                PRIMARY KEY(change_id,guardian_id)
            )""")
        _ready=True

def _verified_link(cur, school_id, child_id, guardian_id):
    cur.execute("""SELECT 1 FROM verifyd_safety_training_guardians g
        JOIN verifyd_safety_training_members gm
           ON gm.school_id=g.school_id AND gm.identity_id=g.guardian_id AND gm.role='guardian'
        JOIN verifyd_safety_training_members sm
           ON sm.school_id=g.school_id AND sm.identity_id=g.child_id AND sm.role='student'
        JOIN verifyd_safety_training_signoffs x
           ON x.school_id=g.school_id AND x.actor_id=g.guardian_id
           AND x.subject_id=g.child_id AND x.kind='guardian'
           AND x.policy_version=%s AND x.revoked_at IS NULL
        JOIN verifyd_identities gi ON gi.id=g.guardian_id AND gi.status='active'
        WHERE g.school_id=%s AND g.child_id=%s AND g.guardian_id=%s""",
        (POLICY_VERSION,school_id,child_id,guardian_id))
    return bool(cur.fetchone())

class FamilyContactRequest(BaseModel):
    school_id: str=Field(min_length=8,max_length=100)
    student_ids: list[str]=Field(min_length=1,max_length=5)
    primary_guardian_id: str=Field(min_length=5,max_length=100)
    backup_guardian_id: str=Field(min_length=5,max_length=100)
    reason: str=Field(min_length=12,max_length=500)
    guardian_authority_acknowledged: bool=False

class Confirmation(BaseModel):
    decision: Literal["accept","decline"]

class AdminReview(BaseModel):
    decision: Literal["approve","reject"]
    independently_verified: bool=False
    note: str=Field(min_length=12,max_length=700)

@router.get("/family/guardian-contacts")
def my_guardian_contacts(authorization:Optional[str]=Header(default=None)):
    actor=_actor(authorization);_schema()
    gid=str(actor["id"])
    with get_db() as conn:
        cur=conn.cursor()
        cur.execute("""SELECT g.school_id,s.school_name,g.child_id,i.handle AS child_handle,
            i.display_name AS child_name
            FROM verifyd_safety_training_guardians g
            JOIN verifyd_safety_training_schools s ON s.school_id=g.school_id
            JOIN verifyd_identities i ON i.id=g.child_id
            WHERE g.guardian_id=%s ORDER BY s.school_name,i.handle""",(gid,))
        children=[]
        for row in cur.fetchall():
            r=dict(row)
            if not _verified_link(cur,r["school_id"],r["child_id"],gid):
                continue
            cur.execute("""SELECT g.guardian_id,i.handle,i.display_name
                FROM verifyd_safety_training_guardians g
                JOIN verifyd_identities i ON i.id=g.guardian_id
                WHERE g.school_id=%s AND g.child_id=%s ORDER BY i.handle""",
                (r["school_id"],r["child_id"]))
            r["eligible_guardians"]=[dict(x) for x in cur.fetchall()
                if _verified_link(cur,r["school_id"],r["child_id"],str(x["guardian_id"]))]
            cur.execute("""SELECT primary_id,backup_id,updated_at FROM verifyd_safety_training_contacts
                WHERE school_id=%s AND student_id=%s""",(r["school_id"],r["child_id"]))
            v=cur.fetchone()
            r["current_contacts"]=dict(v) if v else None
            children.append(r)
        cur.execute("""SELECT c.change_id,c.school_id,c.student_id,c.proposed_primary_id,
          c.proposed_backup_id,c.reason,c.status,c.created_at,c.review_note,c.requested_by,
          x.decision AS my_confirmation
          FROM verifyd_safety_guardian_contact_changes c
          LEFT JOIN verifyd_safety_guardian_contact_confirmations x
             ON x.change_id=c.change_id AND x.guardian_id=%s
          WHERE EXISTS (SELECT 1 FROM verifyd_safety_training_guardians g
             WHERE g.school_id=c.school_id AND g.child_id=c.student_id AND g.guardian_id=%s)
            AND (c.requested_by=%s OR c.proposed_primary_id=%s OR c.proposed_backup_id=%s)
          ORDER BY c.created_at DESC LIMIT 80""",(gid,gid,gid,gid,gid))
        changes=[dict(x) for x in cur.fetchall() if
          _verified_link(cur,x["school_id"],x["student_id"],gid)]
        for item in changes:
            item["my_action_required"]=bool(item["status"]=="pending" and
                item["my_confirmation"] is None and gid in
                (item["proposed_primary_id"],item["proposed_backup_id"]))
    return {"ok":True,"children":children,"changes":changes,
            "live_emergency_delivery":False}

@router.post("/family/guardian-contacts/requests")
def request_guardian_contact_change(payload:FamilyContactRequest,
            authorization:Optional[str]=Header(default=None)):
    actor=_actor(authorization);_schema()
    gid=str(actor["id"])
    if not payload.guardian_authority_acknowledged:
        raise HTTPException(422,detail={"error":"guardian_authority_acknowledgment_required"})
    if payload.primary_guardian_id==payload.backup_guardian_id:
        raise HTTPException(422,detail={"error":"two_distinct_guardians_required"})
    if len(set(payload.student_ids))!=len(payload.student_ids):
        raise HTTPException(422,detail={"error":"duplicate_student_ids"})
    ids=[]
    with get_db() as conn:
        cur=conn.cursor()
        for student_id in sorted(payload.student_ids):
            if not _verified_link(cur,payload.school_id,student_id,gid):
                raise HTTPException(403,detail={"error":"verified_current_guardian_required"})
            for candidate in (payload.primary_guardian_id,payload.backup_guardian_id):
                if not _verified_link(cur,payload.school_id,student_id,candidate):
                    raise HTTPException(409,detail={"error":"verify_and_acknowledge_both_guardians_first"})
            cur.execute("SELECT student_id FROM verifyd_safety_training_contacts WHERE school_id=%s AND student_id=%s FOR UPDATE",
                        (payload.school_id,student_id))
            # Serialize even if no contact row is configured yet.
            cur.execute("SELECT pg_advisory_xact_lock(hashtext(%s))",
                        ("guardian_contacts:"+payload.school_id+":"+student_id,))
            cur.execute("""SELECT 1 FROM verifyd_safety_guardian_contact_changes
                WHERE school_id=%s AND student_id=%s AND status='pending'""",
                (payload.school_id,student_id))
            if cur.fetchone():
                raise HTTPException(409,detail={"error":"contact_change_already_pending","student_id":student_id})
            cur.execute("""SELECT primary_id,backup_id FROM verifyd_safety_training_contacts
                WHERE school_id=%s AND student_id=%s""",(payload.school_id,student_id))
            existing=cur.fetchone()
            if existing and existing["primary_id"]==payload.primary_guardian_id and existing["backup_id"]==payload.backup_guardian_id:
                raise HTTPException(409,detail={"error":"contacts_already_match_request","student_id":student_id})
            cid="ssgcontact_"+uuid.uuid4().hex
            cur.execute("""INSERT INTO verifyd_safety_guardian_contact_changes(
                change_id,school_id,student_id,requested_by,
                proposed_primary_id,proposed_backup_id,previous_primary_id,previous_backup_id,reason)
                VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                (cid,payload.school_id,student_id,gid,payload.primary_guardian_id,
                 payload.backup_guardian_id,existing["primary_id"] if existing else None,
                 existing["backup_id"] if existing else None,payload.reason.strip()))
            # Requester's explicit signed-in acknowledgment also constitutes acceptance
            # when that guardian is among the two proposed recipients.
            if gid in (payload.primary_guardian_id,payload.backup_guardian_id):
                cur.execute("""INSERT INTO verifyd_safety_guardian_contact_confirmations
                    (change_id,guardian_id,decision) VALUES(%s,%s,'accepted')""",(cid,gid))
            _audit(cur,payload.school_id,gid,"guardian_contact_change_requested",cid)
            ids.append(cid)
    return {"ok":True,"request_ids":ids,"status":"pending_guardian_and_school_review",
            "contacts_changed":False,"live_emergency_delivery":False}

@router.post("/family/guardian-contacts/requests/{change_id}/confirm")
def confirm_guardian_contact_change(change_id:str,payload:Confirmation,
            authorization:Optional[str]=Header(default=None)):
    actor=_actor(authorization);_schema()
    gid=str(actor["id"])
    with get_db() as conn:
        cur=conn.cursor()
        cur.execute("""SELECT * FROM verifyd_safety_guardian_contact_changes
            WHERE change_id=%s FOR UPDATE""",(change_id,))
        row=cur.fetchone()
        if not row:raise HTTPException(404,detail={"error":"contact_request_not_found"})
        if row["status"]!="pending":
            raise HTTPException(409,detail={"error":"contact_request_not_pending"})
        if gid not in (row["proposed_primary_id"],row["proposed_backup_id"]):
            raise HTTPException(403,detail={"error":"not_a_proposed_guardian"})
        if not _verified_link(cur,row["school_id"],row["student_id"],gid):
            raise HTTPException(403,detail={"error":"guardian_authorization_expired"})
        cur.execute("""INSERT INTO verifyd_safety_guardian_contact_confirmations
          (change_id,guardian_id,decision) VALUES(%s,%s,%s)
          ON CONFLICT(change_id,guardian_id) DO UPDATE SET
            decision=EXCLUDED.decision,recorded_at=NOW()""",
            (change_id,gid,"accepted" if payload.decision=="accept" else "declined"))
        if payload.decision=="decline":
            cur.execute("UPDATE verifyd_safety_guardian_contact_changes SET status='cancelled' WHERE change_id=%s",(change_id,))
        _audit(cur,row["school_id"],gid,"guardian_contact_"+payload.decision,change_id)
    return {"ok":True,"status":"cancelled" if payload.decision=="decline" else "guardian_confirmation_recorded",
            "contacts_changed":False}

@router.get("/schools/{school_id}/guardian-contact-requests")
def staff_contact_change_requests(school_id:str,authorization:Optional[str]=Header(default=None)):
    actor=_actor(authorization);_schema()
    with get_db() as conn:
        cur=conn.cursor()
        _school_admin(cur,school_id,str(actor["id"]))
        cur.execute("""SELECT c.change_id,c.student_id,st.handle AS child_handle,
          proposer.handle AS requested_by_handle,
          p.handle AS proposed_primary_handle,b.handle AS proposed_backup_handle,
          c.previous_primary_id,c.previous_backup_id,c.reason,c.status,c.created_at,
          c.reviewed_at,c.review_note,
          (SELECT count(*) FROM verifyd_safety_guardian_contact_confirmations x
             WHERE x.change_id=c.change_id AND x.decision='accepted'
               AND x.guardian_id IN (c.proposed_primary_id,c.proposed_backup_id)) AS approvals_received
          FROM verifyd_safety_guardian_contact_changes c
          JOIN verifyd_identities st ON st.id=c.student_id
          JOIN verifyd_identities proposer ON proposer.id=c.requested_by
          JOIN verifyd_identities p ON p.id=c.proposed_primary_id
          JOIN verifyd_identities b ON b.id=c.proposed_backup_id
          WHERE c.school_id=%s ORDER BY c.created_at DESC LIMIT 100""",(school_id,))
        requests=[dict(x) for x in cur.fetchall()]
    return {"ok":True,"requests":requests,"live_emergency_delivery":False}

@router.post("/schools/{school_id}/guardian-contact-requests/{change_id}/review")
def staff_review_contact_change(school_id:str,change_id:str,payload:AdminReview,
        authorization:Optional[str]=Header(default=None)):
    actor=_actor(authorization);_schema()
    aid=str(actor["id"])
    if payload.decision=="approve" and not payload.independently_verified:
        raise HTTPException(422,detail={"error":"independent_school_verification_required"})
    with get_db() as conn:
        cur=conn.cursor()
        _school_admin(cur,school_id,aid)
        cur.execute("""SELECT * FROM verifyd_safety_guardian_contact_changes
          WHERE school_id=%s AND change_id=%s FOR UPDATE""",(school_id,change_id))
        row=cur.fetchone()
        if not row:raise HTTPException(404,detail={"error":"contact_request_not_found"})
        if row["status"]!="pending":
            raise HTTPException(409,detail={"error":"contact_request_not_pending"})
        if aid in (row["requested_by"],row["proposed_primary_id"],row["proposed_backup_id"],row["student_id"]):
            raise HTTPException(403,detail={"error":"self_approval_not_allowed"})
        outcome="rejected"
        if payload.decision=="approve":
            for gid in (row["proposed_primary_id"],row["proposed_backup_id"]):
                if not _verified_link(cur,school_id,row["student_id"],gid):
                    raise HTTPException(409,detail={"error":"guardian_link_or_acknowledgment_invalid"})
                cur.execute("""SELECT decision FROM verifyd_safety_guardian_contact_confirmations
                    WHERE change_id=%s AND guardian_id=%s""",(change_id,gid))
                ok=cur.fetchone()
                if not ok or ok["decision"]!="accepted":
                    raise HTTPException(409,detail={"error":"both_guardians_must_confirm"})
            cur.execute("SELECT pg_advisory_xact_lock(hashtext(%s))",
                        ("guardian_contacts:"+school_id+":"+row["student_id"],))
            cur.execute("""SELECT primary_id,backup_id FROM verifyd_safety_training_contacts
                WHERE school_id=%s AND student_id=%s FOR UPDATE""",(school_id,row["student_id"]))
            current=cur.fetchone()
            previous=(row["previous_primary_id"],row["previous_backup_id"])
            actual=(current["primary_id"],current["backup_id"]) if current else (None,None)
            if actual!=previous:
                raise HTTPException(409,detail={"error":"contact_priorities_changed_since_request"})
            cur.execute("""INSERT INTO verifyd_safety_training_contacts
                (school_id,student_id,primary_id,backup_id,updated_at)
                VALUES(%s,%s,%s,%s,NOW())
                ON CONFLICT(school_id,student_id) DO UPDATE SET
                  primary_id=EXCLUDED.primary_id,backup_id=EXCLUDED.backup_id,
                  updated_at=NOW()""",(school_id,row["student_id"],
                       row["proposed_primary_id"],row["proposed_backup_id"]))
            outcome="approved"
        cur.execute("""UPDATE verifyd_safety_guardian_contact_changes
            SET status=%s,reviewed_by=%s,reviewed_at=NOW(),review_note=%s
            WHERE change_id=%s""",(outcome,aid,payload.note.strip(),change_id))
        _audit(cur,school_id,aid,"guardian_contact_change_"+outcome,change_id)
    # Notify verified, affected adults after the transaction commits.
    # In-app status remains authoritative if a message fails.
    emails_sent=0
    emails_failed=0
    if outcome=="approved":
        targets={str(x) for x in (row["requested_by"],row["proposed_primary_id"],row["proposed_backup_id"],row["previous_primary_id"],row["previous_backup_id"]) if x}
        with get_db() as conn:
            cur=conn.cursor()
            cur.execute("""SELECT email_lower FROM verifyd_identities
                WHERE id=ANY(%s) AND status='active' AND email_verified=TRUE""",(list(targets),))
            recipients=sorted({r["email_lower"] for r in cur.fetchall() if r["email_lower"]})
        for recipient in recipients:
            delivered=_send({
                "from":f"{FROM_NAME} <{FROM_ADDRESS}>",
                "to":[recipient],
                "subject":"VeriFYD Safety — school guardian contacts updated",
                "html":"<p>A school administrator approved a change to the primary or backup guardian contact designation.</p>"
                       "<p>Sign in to VeriFYD Safety and open Manage Guardian Contacts to review the updated details.</p>"
                       "<p>Your verified family relationship is not changed by this notification.</p>"})
            emails_sent+=1 if delivered else 0
            emails_failed+=0 if delivered else 1
    return {"ok":True,"status":outcome,"contacts_changed":outcome=="approved",
            "in_app_confirmation_available":True,
            "confirmation_emails_sent":emails_sent,"confirmation_emails_failed":emails_failed,
            "push_sent":False,"live_emergency_delivery":False}
