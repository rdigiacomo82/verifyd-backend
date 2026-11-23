"""Unified guardian and multi-child school enrollment application.

One school form collects parent identity and proposed child handles. Parent
email must be OTP-verified. Only the parent identity is activated in /voice;
all child handles remain school-reviewed PROPOSALS. No emergency privileges.
"""
from __future__ import annotations
import html
import uuid
from typing import Literal
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from database import get_db, verify_otp, is_valid_email
from emailer import _send, FROM_NAME, FROM_ADDRESS
from trust_voice_identity import (_require_enabled as _voice_enabled,
    _validate_handle,_validate_display_name,_ensure_identity_schema,_now_iso)
from trust_voice_school_family_onboarding import (_schema as _family_schema,
    _availability)
from trust_voice_school_safety_training import _enabled as _school_enabled, _audit

router=APIRouter(prefix="/trust-voice/school-safety",tags=["Unified Family Registration"])

class ChildChoice(BaseModel):
    handle: str=Field(min_length=3,max_length=25)
    display_label: str=Field(min_length=1,max_length=60)

class UnifiedFamilyApplication(BaseModel):
    school_id: str=Field(min_length=8,max_length=100)
    email: str=Field(min_length=3,max_length=254)
    email_code: str=Field(min_length=6,max_length=12)
    parent_handle: str=Field(min_length=3,max_length=25)
    parent_display_name: str=Field(min_length=1,max_length=80)
    relationship: Literal["parent","legal_guardian","authorized_caregiver"]="parent"
    children: list[ChildChoice]=Field(min_length=1,max_length=5)
    parental_authority_acknowledged: bool=False
    supervised_creation_consent: bool=False

@router.get("/family/onboarding/open-schools")
def open_schools():
    if not _school_enabled():
        raise HTTPException(503,detail={"error":"school_registration_not_enabled"})
    _family_schema()
    with get_db() as conn:
        cur=conn.cursor()
        cur.execute("""SELECT DISTINCT s.school_id,s.school_name
            FROM verifyd_safety_training_schools s
            JOIN verifyd_safety_school_applications a
              ON a.school_id=s.school_id AND a.status='approved'
            ORDER BY s.school_name LIMIT 100""")
        rows=[dict(x) for x in cur.fetchall()]
    return {"ok":True,"schools":rows,"student_accounts_not_activated":True}

@router.post("/family/onboarding/register-family")
def register_family(payload:UnifiedFamilyApplication):
    # The school and parent identity features must both be explicitly enabled.
    _voice_enabled()
    if not _school_enabled():
        raise HTTPException(503,detail={"error":"school_registration_not_enabled"})
    if not payload.parental_authority_acknowledged:
        raise HTTPException(422,detail={"error":"parental_authority_confirmation_required"})
    email=payload.email.strip().lower()
    if not is_valid_email(email):
        raise HTTPException(400,detail={"error":"invalid_email"})
    parent_handle,parent_lower=_validate_handle(payload.parent_handle)
    parent_name=_validate_display_name(payload.parent_display_name)
    children=[]
    for c in payload.children:
        label=" ".join(c.display_label.strip().split())
        if not label: raise HTTPException(422,detail={"error":"child_label_required"})
        handle,lower=_validate_handle(c.handle)
        children.append((handle,lower,label))
    lowers=[parent_lower]+[c[1] for c in children]
    if len(lowers)!=len(set(lowers)):
        raise HTTPException(422,detail={"error":"duplicate_family_handles"})
    # OTP verifies parent ownership before account creation.
    valid,reason=verify_otp(email,payload.email_code)
    if not valid:
        raise HTTPException(403,detail={"error":"invalid_parent_email_code"})
    _ensure_identity_schema()
    _family_schema()
    identity_id="vfyd_usr_"+uuid.uuid4().hex
    ids=[]
    with get_db() as conn:
        cur=conn.cursor()
        cur.execute("""SELECT 1 FROM verifyd_safety_school_applications
            WHERE school_id=%s AND status='approved' LIMIT 1""",(payload.school_id,))
        if not cur.fetchone():
            raise HTTPException(404,detail={"error":"school_not_accepting_applications"})
        # Serialize family handles against other unified school applications.
        # Database unique constraints also protect all global /voice identities.
        for lower in sorted(lowers):
            cur.execute("SELECT pg_advisory_xact_lock(hashtext(%s))",("safety_handle_"+lower,))
        cur.execute("SELECT 1 FROM verifyd_identities WHERE email_lower=%s",(email,))
        if cur.fetchone():
            raise HTTPException(409,detail={"error":"parent_email_already_registered_sign_in"})
        for lower in lowers:
            if not _availability(cur,lower):
                raise HTTPException(409,detail={"error":"handle_unavailable","handle":"@"+lower})
        now=_now_iso()
        cur.execute("""INSERT INTO verifyd_identities(
           id,email_lower,handle,handle_lower,display_name,display_emoji,
           email_verified,identity_verified,organization_verified,
           verification_level,can_receive_calls,call_privacy,status,
           created_at,updated_at)
           VALUES(%s,%s,%s,%s,%s,'',TRUE,FALSE,FALSE,
              'email_verified',FALSE,'verified_users','active',%s,%s)""",
           (identity_id,email,parent_handle,parent_lower,parent_name,now,now))
        for handle,lower,label in children:
            rid="ssint_"+uuid.uuid4().hex
            cur.execute("""INSERT INTO verifyd_safety_family_intake(
              request_id,school_id,guardian_id,suggested_handle,handle_lower,
              child_label,relationship)
              VALUES(%s,%s,%s,%s,%s,%s,%s)""",
              (rid,payload.school_id,identity_id,handle,lower,label,payload.relationship,bool(payload.supervised_creation_consent)))
            _audit(cur,payload.school_id,identity_id,"family_handle_intake_submitted",rid)
            ids.append((rid,handle))
    receipts=[]
    for rid,handle in ids:
        sent=_send({
            "from":f"{FROM_NAME} <{FROM_ADDRESS}>","to":[email],
            "subject":"VeriFYD Safety — child registration request received",
            "html":"<p>Your school enrollment request for proposed handle "+
               html.escape("@"+handle)+" has been received.</p>"+
               "<p>A school administrator must independently verify family authority. "+
               "This is not an activated student account or emergency notification service.</p>"
        })
        with get_db() as conn:
            conn.cursor().execute("""UPDATE verifyd_safety_family_intake
                SET email_status=%s WHERE request_id=%s""",
                ("sent" if sent else "failed",rid))
        receipts.append({"request_id":rid,"requested_handle":"@"+handle,
                         "confirmation_email_sent":sent,"status":"pending"})
    return {"ok":True,"parent_account_created":True,
            "parent_handle":"@"+parent_handle,
            "children":receipts,"children_activated":False,
            "next_step":"sign_in_to_safety_with_verified_email",
            "emergency_delivery_active":False}
