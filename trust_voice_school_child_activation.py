"""Guardian-confirmed, school-verified supervised child identity activation.

TRAINING ONLY. This creates a globally unique but non-login, non-discoverable
supervised identity. Ordinary /voice authentication only accepts status=active,
so the child cannot call, message, or independently sign in.
"""
from __future__ import annotations
import html
import os
import threading
import uuid
import secrets
import hashlib
import hmac
from datetime import datetime, timedelta, timezone
from typing import Optional
from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field
from database import get_db
from emailer import _send, FROM_NAME, FROM_ADDRESS
from trust_voice_school_family_onboarding import _schema as _intake_schema
from trust_voice_school_safety_training import _actor, _audit
from trust_voice_school_safety_registration import _school_admin
from trust_voice_school_safety_consents import _school_accepted

router=APIRouter(prefix="/trust-voice/school-safety",tags=["Supervised Child Identity"])
_ready=False
_lock=threading.Lock()

def _require_pilot():
    if os.getenv("VERIFYD_SAFETY_CHILD_ACTIVATION_ENABLED","0").strip().lower() not in ("1","true","yes","on"):
        raise HTTPException(503,detail={"error":"supervised_child_activation_not_enabled"})

def _allowed_school(school_id: str):
    allowed={x.strip() for x in os.getenv("VERIFYD_SAFETY_CHILD_ACTIVATION_SCHOOL_IDS","").split(",") if x.strip()}
    if not allowed or school_id not in allowed:
        raise HTTPException(403,detail={"error":"child_activation_school_not_authorized"})

def _schema():
    global _ready
    _intake_schema()
    if _ready:return
    with _lock:
        if _ready:return
        with get_db() as conn:
            conn.cursor().execute("""CREATE TABLE IF NOT EXISTS verifyd_safety_child_activation (
                request_id TEXT PRIMARY KEY REFERENCES verifyd_safety_family_intake(request_id),
                school_id TEXT NOT NULL REFERENCES verifyd_safety_training_schools(school_id),
                guardian_id TEXT NOT NULL REFERENCES verifyd_identities(id),
                verified_by TEXT NOT NULL REFERENCES verifyd_identities(id),
                verification_note TEXT NOT NULL,
                verified_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                child_identity_id TEXT UNIQUE REFERENCES verifyd_identities(id),
                activated_at TIMESTAMPTZ
            )""")
            conn.cursor().execute("""CREATE TABLE IF NOT EXISTS verifyd_safety_child_activation_codes (
                guardian_id TEXT PRIMARY KEY REFERENCES verifyd_identities(id),
                salt TEXT NOT NULL, code_hash TEXT NOT NULL,
                sent_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                expires_at TIMESTAMPTZ NOT NULL,
                attempts INTEGER NOT NULL DEFAULT 0
            )""")
            conn.cursor().execute("""ALTER TABLE verifyd_safety_child_activation ADD COLUMN IF NOT EXISTS approval_email_status TEXT NOT NULL DEFAULT 'pending'""")
            conn.cursor().execute("""ALTER TABLE verifyd_safety_child_activation ADD COLUMN IF NOT EXISTS approval_email_sent_at TIMESTAMPTZ""")
            conn.cursor().execute("""ALTER TABLE verifyd_safety_child_activation ADD COLUMN IF NOT EXISTS approval_email_attempts INTEGER NOT NULL DEFAULT 0""")
        _ready=True

def _digest(salt: str, code: str) -> str:
    return hashlib.sha256((salt+":"+code).encode("utf-8")).hexdigest()


def _notify_parent_after_approval(school_id:str, request_ids:list[str]):
    """Send one informational guardian email after a committed school approval."""
    if not request_ids:return {"status":"none","sent":False}
    with get_db() as conn:
        cur=conn.cursor()
        cur.execute("""SELECT r.request_id,r.guardian_id,r.suggested_handle,
            s.school_name,g.email_lower,g.email_verified,a.approval_email_status
            FROM verifyd_safety_child_activation a
            JOIN verifyd_safety_family_intake r ON r.request_id=a.request_id
            JOIN verifyd_safety_training_schools s ON s.school_id=r.school_id
            JOIN verifyd_identities g ON g.id=r.guardian_id
            WHERE a.school_id=%s AND r.request_id=ANY(%s)
              AND a.activated_at IS NULL AND r.status='reviewed'""",
            (school_id,request_ids))
        records=[dict(x) for x in cur.fetchall()]
    if len(records)!=len(request_ids) or len({str(x["guardian_id"]) for x in records})!=1:
        return {"status":"unavailable","sent":False}
    pending=[x for x in records if x["approval_email_status"]!="sent"]
    if not pending:return {"status":"already_sent","sent":True}
    email=pending[0]["email_lower"]
    if not email or not pending[0]["email_verified"]:
        return {"status":"guardian_email_not_verified","sent":False}
    handles=", ".join("@"+html.escape(x["suggested_handle"]) for x in pending)
    school=html.escape(pending[0]["school_name"])
    delivered=_send({
        "from":f"{FROM_NAME} <{FROM_ADDRESS}>",
        "to":[email],
        "subject":"VeriFYD Safety — school verification completed; parent action needed",
        "html":("<p>Your school, <strong>"+school+"</strong>, has verified the "
          "guardian relationship for these requested child handles: "+handles+".</p>"
          "<p><strong>Your next step:</strong> sign in to "
          '<a href="https://vfvid.com/voice">VeriFYD Circle</a>, open VeriFYD Safety, '
          "then select the approved children and request a fresh parent activation code.</p>"
          "<p>Approval does not activate a child account. The parent must complete "
          "the separate confirmation step. Emergency delivery is not enabled.</p>"
          "<p>If you did not submit these requests, contact the school directly.</p>")
    })
    with get_db() as conn:
        cur=conn.cursor()
        cur.execute("""UPDATE verifyd_safety_child_activation
            SET approval_email_status=%s,approval_email_attempts=approval_email_attempts+1,
                approval_email_sent_at=CASE WHEN %s THEN NOW() ELSE approval_email_sent_at END
            WHERE request_id=ANY(%s) AND school_id=%s
                AND approval_email_status<>'sent'""",
            ("sent" if delivered else "failed",delivered,[x["request_id"] for x in pending],school_id))
    return {"status":"sent" if delivered else "failed","sent":bool(delivered),
            "requested_children":len(pending)}

class SchoolVerify(BaseModel):
    independently_verified: bool=False
    verification_note: str=Field(default='',max_length=500)

class ActivateChildren(BaseModel):
    request_ids: list[str]=Field(min_length=1,max_length=5)
    email_code: str=Field(min_length=6,max_length=12)
    supervised_account_acknowledged: bool=False

@router.post("/schools/{school_id}/family/onboarding/{request_id}/verify")
def verify_child(school_id:str,request_id:str,payload:SchoolVerify,
                 authorization:Optional[str]=Header(default=None)):
    actor=_actor(authorization);_schema();_require_pilot();_allowed_school(school_id)
    if not payload.independently_verified:
        raise HTTPException(422,detail={"error":"independent_guardian_verification_required"})
    aid=str(actor["id"])
    with get_db() as conn:
        cur=conn.cursor();_school_admin(cur,school_id,aid)
        cur.execute("""SELECT guardian_id,status FROM verifyd_safety_family_intake
            WHERE request_id=%s AND school_id=%s FOR UPDATE""",(request_id,school_id))
        req=cur.fetchone()
        if not req: raise HTTPException(404,detail={"error":"request_not_found"})
        if req["guardian_id"]==aid: raise HTTPException(403,detail={"error":"self_verification_denied"})
        if req["status"]=="cancelled": raise HTTPException(409,detail={"error":"request_cancelled"})
        cur.execute("SELECT 1 FROM verifyd_safety_child_activation WHERE request_id=%s",(request_id,))
        if cur.fetchone(): raise HTTPException(409,detail={"error":"request_already_verified"})
        cur.execute("""INSERT INTO verifyd_safety_child_activation
            (request_id,school_id,guardian_id,verified_by,verification_note)
            VALUES(%s,%s,%s,%s,%s)""",
            (request_id,school_id,req["guardian_id"],aid,payload.verification_note.strip()))
        cur.execute("""UPDATE verifyd_safety_family_intake
            SET status='reviewed',reviewed_by=%s,reviewed_at=NOW()
            WHERE request_id=%s""",(aid,request_id))
        _audit(cur,school_id,aid,"child_identity_independently_verified",request_id)
    delivery=_notify_parent_after_approval(school_id,[request_id])
    return {"ok":True,"verified":True,"activated":False,"guardian_confirmation_required":True,
            "parent_approval_email":delivery}


class FamilyBatchItem(BaseModel):
    request_id: str=Field(min_length=8,max_length=100)
    independently_verified: bool=False
    verification_note: str=Field(default='',max_length=500)

class FamilyBatchVerify(BaseModel):
    # Each child has its own independent verification note and checkbox.
    requests: list[FamilyBatchItem]=Field(min_length=1,max_length=20)

@router.post("/schools/{school_id}/family/child-activation/verify-batch")
def verify_family_batch(school_id:str,payload:FamilyBatchVerify,
    authorization:Optional[str]=Header(default=None)):
    actor=_actor(authorization);_schema();_require_pilot();_allowed_school(school_id)
    aid=str(actor["id"])
    request_ids=[item.request_id for item in payload.requests]
    if len(set(request_ids))!=len(request_ids):
        raise HTTPException(422,detail={"error":"duplicate_request_ids"})
    if any(not item.independently_verified for item in payload.requests):
        raise HTTPException(422,detail={"error":"each_child_must_be_independently_verified"})
    with get_db() as conn:
        cur=conn.cursor()
        _school_admin(cur,school_id,aid)
        cur.execute("""SELECT request_id,guardian_id,status FROM verifyd_safety_family_intake
            WHERE school_id=%s AND request_id=ANY(%s) ORDER BY request_id FOR UPDATE""",
            (school_id,sorted(request_ids)))
        records={r["request_id"]:r for r in cur.fetchall()}
        if len(records)!=len(request_ids):
            raise HTTPException(404,detail={"error":"one_or_more_children_not_found"})
        guardian_ids={str(r["guardian_id"]) for r in records.values()}
        if len(guardian_ids)!=1:
            raise HTTPException(422,detail={"error":"batch_requires_same_parent"})
        if aid in guardian_ids:
            raise HTTPException(403,detail={"error":"self_verification_denied"})
        for item in payload.requests:
            record=records[item.request_id]
            if record["status"]=="cancelled":
                raise HTTPException(409,detail={"error":"request_cancelled"})
            cur.execute("""SELECT 1 FROM verifyd_safety_child_activation
                WHERE request_id=%s""",(item.request_id,))
            if cur.fetchone():
                raise HTTPException(409,detail={"error":"request_already_verified",
                    "request_id":item.request_id})
        for item in payload.requests:
            record=records[item.request_id]
            cur.execute("""INSERT INTO verifyd_safety_child_activation
              (request_id,school_id,guardian_id,verified_by,verification_note)
              VALUES(%s,%s,%s,%s,%s)""",
              (item.request_id,school_id,record["guardian_id"],aid,item.verification_note.strip()))
            cur.execute("""UPDATE verifyd_safety_family_intake
                SET status='reviewed',reviewed_by=%s,reviewed_at=NOW()
                WHERE request_id=%s""",(aid,item.request_id))
            _audit(cur,school_id,aid,"child_identity_independently_verified",item.request_id)
    delivery=_notify_parent_after_approval(school_id,request_ids)
    return {"ok":True,"verified_request_ids":request_ids,
            "verified_count":len(request_ids),"activated_count":0,
            "guardian_email_confirmation_required":True,"parent_approval_email":delivery,
            "emergency_delivery_active":False}

class ApprovalEmailRetry(BaseModel):
    request_ids: list[str]=Field(min_length=1,max_length=20)

@router.post("/schools/{school_id}/family/child-activation/approval-email/retry")
def retry_parent_approval_email(school_id:str,payload:ApprovalEmailRetry,
    authorization:Optional[str]=Header(default=None)):
    actor=_actor(authorization);_schema();_require_pilot();_allowed_school(school_id)
    with get_db() as conn:
        cur=conn.cursor()
        _school_admin(cur,school_id,str(actor["id"]))
    if len(set(payload.request_ids))!=len(payload.request_ids):
        raise HTTPException(422,detail={"error":"duplicate_request_ids"})
    return {"ok":True,"parent_approval_email":_notify_parent_after_approval(school_id,payload.request_ids)}

@router.get("/family/child-activation/status")
def my_activation_status(authorization:Optional[str]=Header(default=None)):
    guardian=_actor(authorization);_schema()
    with get_db() as conn:
        cur=conn.cursor()
        cur.execute("""SELECT r.request_id,r.school_id,s.school_name,r.suggested_handle,r.child_label,
            r.status,a.verified_at,a.activated_at,a.child_identity_id
            FROM verifyd_safety_family_intake r
            JOIN verifyd_safety_training_schools s ON s.school_id=r.school_id
            LEFT JOIN verifyd_safety_child_activation a ON a.request_id=r.request_id
            WHERE r.guardian_id=%s ORDER BY r.created_at DESC LIMIT 50""",(str(guardian["id"]),))
        rows=[dict(x) for x in cur.fetchall()]
    for row in rows:
        row["activation_ready"]=bool(row["verified_at"] and not row["activated_at"] and row["status"]=="reviewed")
        row.pop("child_identity_id",None)
    return {"ok":True,"children":rows,"supervised_voice_login_available":False}

@router.post("/family/child-activation/request-code")
def request_activation_code(authorization:Optional[str]=Header(default=None)):
    guardian=_actor(authorization);_schema();_require_pilot()
    gid=str(guardian["id"])
    with get_db() as conn:
        cur=conn.cursor()
        cur.execute("SELECT email_lower FROM verifyd_identities WHERE id=%s AND status='active'",(gid,))
        account=cur.fetchone()
        if not account or not account["email_lower"]:
            raise HTTPException(403,detail={"error":"active_verified_guardian_required"})
        cur.execute("""SELECT 1 FROM verifyd_safety_child_activation a
            JOIN verifyd_safety_family_intake r ON r.request_id=a.request_id
            WHERE r.guardian_id=%s AND r.status='reviewed' AND a.activated_at IS NULL AND r.school_id=ANY(%s) LIMIT 1""",(gid, list({x.strip() for x in os.getenv("VERIFYD_SAFETY_CHILD_ACTIVATION_SCHOOL_IDS","").split(",") if x.strip()})))
        if not cur.fetchone():
            raise HTTPException(409,detail={"error":"no_verified_child_ready"})
        cur.execute("SELECT sent_at FROM verifyd_safety_child_activation_codes WHERE guardian_id=%s FOR UPDATE",(gid,))
        previous=cur.fetchone()
        if previous and previous["sent_at"]>datetime.now(timezone.utc)-timedelta(seconds=60):
            raise HTTPException(429,detail={"error":"activation_code_recently_sent"})
        code=str(secrets.randbelow(900000)+100000)
        salt=secrets.token_hex(16)
        cur.execute("""INSERT INTO verifyd_safety_child_activation_codes
           (guardian_id,salt,code_hash,sent_at,expires_at,attempts)
           VALUES(%s,%s,%s,NOW(),NOW()+INTERVAL '10 minutes',0)
           ON CONFLICT(guardian_id) DO UPDATE SET
            salt=EXCLUDED.salt, code_hash=EXCLUDED.code_hash,
            sent_at=NOW(), expires_at=NOW()+INTERVAL '10 minutes', attempts=0""",
            (gid,salt,_digest(salt,code)))
        email=account["email_lower"]
    delivered=_send({"from":f"{FROM_NAME} <{FROM_ADDRESS}>","to":[email],
                     "subject":"VeriFYD Safety supervised child activation code",
                     "html":"<p>Your one-time school-supervised child activation code is <strong>"+
                         html.escape(code)+"</strong>.</p><p>It expires in 10 minutes. "+
                         "Do not share it. This code cannot sign you into VeriFYD Circle.</p>"})
    if not delivered:
        with get_db() as conn:
            conn.cursor().execute("DELETE FROM verifyd_safety_child_activation_codes WHERE guardian_id=%s AND code_hash=%s",
                                  (gid,_digest(salt,code)))
        raise HTTPException(503,detail={"error":"activation_email_delivery_failed"})
    return {"ok":True,"email_sent":True,"expires_minutes":10}

@router.post("/family/child-activation/activate")
def activate_children(payload:ActivateChildren,authorization:Optional[str]=Header(default=None)):
    guardian=_actor(authorization);_schema();_require_pilot()
    gid=str(guardian["id"])
    if not payload.supervised_account_acknowledged:
        raise HTTPException(422,detail={"error":"supervised_account_consent_required"})
    if len(set(payload.request_ids))!=len(payload.request_ids):
        raise HTTPException(422,detail={"error":"duplicate_request_ids"})
    with get_db() as conn:
        cur=conn.cursor()
        cur.execute("SELECT email_lower FROM verifyd_identities WHERE id=%s AND status='active'",(gid,))
        a=cur.fetchone()
        if not a or not a["email_lower"]:
            raise HTTPException(403,detail={"error":"guardian_account_unavailable"})
        email=a["email_lower"]
        cur.execute("""SELECT r.request_id,r.school_id,r.guardian_id,r.suggested_handle,r.handle_lower,
            r.child_label,r.status,a.verified_by,a.activated_at
            FROM verifyd_safety_family_intake r
            JOIN verifyd_safety_child_activation a ON a.request_id=r.request_id
            WHERE r.request_id=ANY(%s) AND r.guardian_id=%s
            ORDER BY r.request_id FOR UPDATE OF r,a""",(sorted(payload.request_ids),gid))
        requests=[dict(x) for x in cur.fetchall()]
        if len(requests)!=len(payload.request_ids):
            raise HTTPException(403,detail={"error":"all_children_must_be_verified_for_guardian"})
        for req in requests:
            _allowed_school(req["school_id"])
            if req["status"]!="reviewed" or req["activated_at"]:
                raise HTTPException(409,detail={"error":"child_request_not_activation_ready"})
            if not _school_accepted(cur,req["school_id"]):
                raise HTTPException(403,detail={"error":"school_authorization_required"})
            cur.execute("""SELECT role FROM verifyd_safety_training_members
                WHERE school_id=%s AND identity_id=%s""",(req["school_id"],gid))
            member=cur.fetchone()
            if member and member["role"]!="guardian":
                raise HTTPException(403,detail={"error":"guardian_has_another_school_role"})
            req["guardian_already_member"]=bool(member)
            cur.execute("SELECT 1 FROM verifyd_identities WHERE handle_lower=%s",(req["handle_lower"],))
            if cur.fetchone():raise HTTPException(409,detail={"error":"proposed_handle_already_claimed",
                                                               "handle":"@"+req["suggested_handle"]})
        # Purpose-scoped, attempt-limited challenge; normal /voice login codes
        # cannot be used to activate supervised child identities.
        cur.execute("""SELECT salt,code_hash,expires_at,attempts
            FROM verifyd_safety_child_activation_codes
            WHERE guardian_id=%s FOR UPDATE""",(gid,))
        challenge=cur.fetchone()
        if not challenge:
            raise HTTPException(403,detail={"error":"activation_code_required"})
        if challenge["expires_at"]<datetime.now(timezone.utc) or challenge["attempts"]>=5:
            raise HTTPException(403,detail={"error":"activation_code_expired_or_locked"})
        if not hmac.compare_digest(challenge["code_hash"],_digest(challenge["salt"],payload.email_code)):
            cur.execute("""UPDATE verifyd_safety_child_activation_codes SET attempts=attempts+1
                WHERE guardian_id=%s""",(gid,))
            # Commit the failed-attempt counter before raising (FastAPI errors
            # otherwise roll back the surrounding database transaction).
            conn.commit()
            raise HTTPException(403,detail={"error":"invalid_activation_code"})
        cur.execute("DELETE FROM verifyd_safety_child_activation_codes WHERE guardian_id=%s",(gid,))
        created=[]
        for req in requests:
            child_id="vfyd_usr_"+uuid.uuid4().hex
            hidden_email="supervised-"+uuid.uuid4().hex+"@child.verifyd.invalid"
            cur.execute("""INSERT INTO verifyd_identities (
                id,email_lower,handle,handle_lower,display_name,display_emoji,
                email_verified,identity_verified,organization_verified,
                verification_level,can_receive_calls,call_privacy,status,
                created_at,updated_at)
                VALUES(%s,%s,%s,%s,%s,'',FALSE,FALSE,FALSE,
                  'guardian_supervised',FALSE,'nobody','supervised_child',
                  NOW()::text,NOW()::text)""",
                (child_id,hidden_email,req["suggested_handle"],req["handle_lower"],req["child_label"]))
            cur.execute("""INSERT INTO verifyd_safety_training_members
              (school_id,identity_id,role,verified_by) VALUES(%s,%s,'student',%s)
              ON CONFLICT(school_id,identity_id) DO NOTHING""",
              (req["school_id"],child_id,req["verified_by"]))
            if not req["guardian_already_member"]:
                cur.execute("""INSERT INTO verifyd_safety_training_members
                  (school_id,identity_id,role,verified_by) VALUES(%s,%s,'guardian',%s)
                  ON CONFLICT(school_id,identity_id) DO NOTHING""",
                  (req["school_id"],gid,req["verified_by"]))
            cur.execute("""INSERT INTO verifyd_safety_training_guardians
               (school_id,child_id,guardian_id,verified_by) VALUES(%s,%s,%s,%s)""",
               (req["school_id"],child_id,gid,req["verified_by"]))
            cur.execute("""UPDATE verifyd_safety_child_activation
               SET child_identity_id=%s,activated_at=NOW()
               WHERE request_id=%s""",(child_id,req["request_id"]))
            _audit(cur,req["school_id"],gid,"supervised_child_identity_created",req["request_id"])
            created.append({"request_id":req["request_id"],"handle":"@"+req["suggested_handle"],
                            "identity_status":"supervised_child"})
    # Send informational receipts after commit, never child credentials.
    delivery=[]
    for item in created:
        sent=_send({"from":f"{FROM_NAME} <{FROM_ADDRESS}>","to":[email],
                    "subject":"VeriFYD Safety — supervised child account created",
                    "html":"<p>A supervised child account for "+html.escape(item["handle"])+
                           " was created following school verification and your confirmation.</p>"+
                           "<p>The child cannot independently sign in, receive calls, send messages or appear in public lookup. "+
                           "School access and policy acknowledgments remain separately controlled.</p>"})
        delivery.append({"handle":item["handle"],"confirmation_email_sent":sent})
    return {"ok":True,"children":created,"confirmations":delivery,
            "global_handles_reserved":True,
            "school_memberships_created":True,
            "supervised_voice_login_available":False,
            "calls_enabled":False,"messaging_enabled":False,
            "emergency_alerts_active":False}

@router.get("/schools/{school_id}/family/child-activation")
def school_activation_status(school_id:str,authorization:Optional[str]=Header(default=None)):
    actor=_actor(authorization);_schema()
    with get_db() as conn:
        cur=conn.cursor()
        _school_admin(cur,school_id,str(actor["id"]))
        cur.execute("""SELECT r.request_id,r.suggested_handle,r.child_label,
            guardian.handle AS guardian_handle,a.verified_at,a.activated_at,
            a.approval_email_status,a.approval_email_sent_at,a.approval_email_attempts,
            child.handle AS global_child_handle,child.status AS identity_status
            FROM verifyd_safety_family_intake r
            JOIN verifyd_identities guardian ON guardian.id=r.guardian_id
            LEFT JOIN verifyd_safety_child_activation a ON a.request_id=r.request_id
            LEFT JOIN verifyd_identities child ON child.id=a.child_identity_id
            WHERE r.school_id=%s ORDER BY r.created_at DESC LIMIT 150""",(school_id,))
        rows=[dict(x) for x in cur.fetchall()]
    return {"ok":True,"children":rows,"training_only":True}

@router.get("/family/child-activation/readiness/{school_id}")
def activation_readiness(school_id:str,authorization:Optional[str]=Header(default=None)):
    """Authenticated, sanitized end-to-end readiness indicators (no secrets)."""
    guardian=_actor(authorization);_schema()
    gid=str(guardian["id"])
    with get_db() as conn:
        cur=conn.cursor()
        cur.execute("""SELECT 1 FROM verifyd_safety_school_applications
            WHERE school_id=%s AND status='approved' LIMIT 1""",(school_id,))
        approved=bool(cur.fetchone())
        cur.execute("SELECT email_verified FROM verifyd_identities WHERE id=%s AND status='active'",(gid,))
        identity=cur.fetchone()
        cur.execute("""SELECT count(*) AS n FROM verifyd_safety_family_intake
            WHERE school_id=%s AND guardian_id=%s AND status IN ('pending','reviewed')""",(school_id,gid))
        requested=int(cur.fetchone()["n"])
        cur.execute("""SELECT count(*) AS n FROM verifyd_safety_child_activation a
          JOIN verifyd_safety_family_intake r ON r.request_id=a.request_id
          WHERE a.school_id=%s AND a.guardian_id=%s
            AND r.status='reviewed' AND a.activated_at IS NULL""",(school_id,gid))
        verified=int(cur.fetchone()["n"])
        cur.execute("""SELECT count(*) AS n FROM verifyd_safety_child_activation a
            WHERE a.school_id=%s AND a.guardian_id=%s AND a.activated_at IS NOT NULL""",(school_id,gid))
        activated=int(cur.fetchone()["n"])
        school_ack=_school_accepted(cur,school_id) if approved else False
    enabled=os.getenv("VERIFYD_SAFETY_CHILD_ACTIVATION_ENABLED","0").lower().strip() in ("1","true","yes","on")
    allowed={x.strip() for x in os.getenv("VERIFYD_SAFETY_CHILD_ACTIVATION_SCHOOL_IDS","").split(",") if x.strip()}
    return {"ok":True,"pilot_enabled_for_school":bool(enabled and school_id in allowed),
            "approved_school":approved,"school_acknowledgment_recorded":school_ack,
            "guardian_email_verified":bool(identity and identity["email_verified"]),
            "pending_or_verified_requests":requested,
            "ready_for_parent_confirmation":verified,"activated_children":activated,
            "child_login_enabled":False,"emergency_alerts_active":False}
