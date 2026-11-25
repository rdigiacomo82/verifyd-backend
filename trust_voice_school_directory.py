"""Authenticated TEST school contact directory.

The school application's Principal email / official_email is intentionally never
queried or returned. Only approved family members and verified school staff can
read this directory. School admins may add nonpersonal public office phones.
"""
from __future__ import annotations
import re
import threading
from typing import Optional
from fastapi import APIRouter,Header,HTTPException
from pydantic import BaseModel,Field,field_validator
from database import get_db
from trust_voice_school_safety_training import _actor,_audit,_membership
from trust_voice_school_safety_consents import POLICY_VERSION

router=APIRouter(tags=["TEST Verified School Contact Directory"])
_ready=False
_lock=threading.Lock()
_PHONE=re.compile(r"^[+0-9() .-]{10,32}$")
_EMAIL=re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")


def _schema():
    global _ready
    if _ready:return
    from trust_voice_school_safety_registration import _application_schema
    _application_schema()
    with _lock:
        if _ready:return
        with get_db() as conn:
            conn.cursor().execute("""CREATE TABLE IF NOT EXISTS verifyd_safety_school_directory (
                school_id TEXT PRIMARY KEY REFERENCES verifyd_safety_training_schools(school_id),
                main_phone TEXT NOT NULL DEFAULT '',
                office_phone TEXT NOT NULL DEFAULT '',
                public_instructions TEXT NOT NULL DEFAULT '',
                updated_by TEXT REFERENCES verifyd_identities(id),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )""")
        _ready=True


class DirectoryUpdate(BaseModel):
    main_phone:str=Field(default="",max_length=32)
    office_phone:str=Field(default="",max_length=32)
    public_instructions:str=Field(default="",max_length=350)

    @field_validator("main_phone","office_phone")
    @classmethod
    def check_phone(cls,v):
        x=" ".join(v.strip().split())
        if x and (not _PHONE.fullmatch(x) or not 10<=len(re.sub(r"\D","",x))<=15):
            raise ValueError("valid_school_contact_phone_required")
        return x

    @field_validator("public_instructions")
    @classmethod
    def no_private_email(cls,v):
        x=" ".join(v.split())
        if _EMAIL.search(x) or "<" in x or ">" in x:
            raise ValueError("public_school_instructions_cannot_contain_email_or_html")
        return x


def _details(cur,school_id:str)->dict:
    cur.execute("""SELECT school_name FROM verifyd_safety_training_schools
        WHERE school_id=%s""",(school_id,))
    school=cur.fetchone()
    if not school:
        raise HTTPException(404,detail={"error":"school_not_found"})
    cur.execute("""SELECT district_name,school_address,school_website,principal_name
        FROM verifyd_safety_school_applications
        WHERE school_id=%s AND status='approved'
        ORDER BY reviewed_at DESC LIMIT 1""",(school_id,))
    app=cur.fetchone()
    cur.execute("""SELECT main_phone,office_phone,public_instructions
        FROM verifyd_safety_school_directory WHERE school_id=%s""",(school_id,))
    contact=cur.fetchone()
    cur.execute("""SELECT i.handle,i.display_name
        FROM verifyd_safety_training_members m
        JOIN verifyd_identities i ON i.id=m.identity_id AND i.status='active'
        WHERE m.school_id=%s AND m.role='principal'
        ORDER BY i.handle LIMIT 1""",(school_id,))
    principal=cur.fetchone()
    cur.execute("""SELECT i.handle,i.display_name
        FROM verifyd_safety_school_admin_delegations d
        JOIN verifyd_safety_training_members m
          ON m.school_id=d.school_id AND m.identity_id=d.identity_id
         AND m.role='responder'
        JOIN verifyd_identities i ON i.id=d.identity_id AND i.status='active'
        WHERE d.school_id=%s AND d.revoked_at IS NULL AND d.accepted_at IS NOT NULL
        ORDER BY i.handle LIMIT 10""",(school_id,))
    admins=[{"handle":str(x["handle"]),"name":str(x["display_name"] or "")}
            for x in cur.fetchall()]
    return {
        "school_id":school_id,
        "school_name":str(school["school_name"]),
        "district_name":str(app["district_name"] or "") if app else "",
        "school_address":str(app["school_address"] or "") if app else "",
        "school_website":str(app["school_website"] or "") if app else "",
        "principal_name":str(app["principal_name"] or "") if app else "",
        "principal_handle":str(principal["handle"]) if principal else "",
        "school_main_phone":str(contact["main_phone"] or "") if contact else "",
        "school_office_phone":str(contact["office_phone"] or "") if contact else "",
        "school_contact_instructions":str(contact["public_instructions"] or "") if contact else "",
        "approved_backup_administrators":admins,
        "contact_phone_missing":not bool(contact and (contact["main_phone"] or contact["office_phone"])),
        "training_only":True,
        "principal_email_disclosed":False,
    }


def _require_directory_read(cur,school_id:str,actor_id:str)->str:
    role=_membership(cur,school_id,actor_id)
    if role in ("principal","responder"):
        from trust_voice_school_safety_registration import _school_admin
        _school_admin(cur,school_id,actor_id)
        return role
    if role=="student":
        return role
    if role!="guardian":
        raise HTTPException(403,detail={"error":"verified_school_family_member_required"})
    # A membership alone is insufficient: current child link AND signoff.
    cur.execute("""SELECT 1 FROM verifyd_safety_training_guardians g
        JOIN verifyd_safety_training_signoffs a
          ON a.school_id=g.school_id AND a.actor_id=g.guardian_id
         AND a.subject_id=g.child_id AND a.kind='guardian'
         AND a.policy_version=%s AND a.revoked_at IS NULL
        WHERE g.school_id=%s AND g.guardian_id=%s LIMIT 1""",
        (POLICY_VERSION,school_id,actor_id))
    if not cur.fetchone():
        raise HTTPException(403,detail={"error":"approved_guardian_family_link_required"})
    return role


@router.get("/schools/{school_id}/directory")
def get_verified_school_directory(school_id:str,authorization:Optional[str]=Header(default=None)):
    actor=_actor(authorization);_schema()
    with get_db() as conn:
        cur=conn.cursor()
        _require_directory_read(cur,school_id,str(actor["id"]))
        data=_details(cur,school_id)
    return {"ok":True,"directory":data}


@router.put("/schools/{school_id}/directory")
def update_school_public_directory(school_id:str,payload:DirectoryUpdate,
                                    authorization:Optional[str]=Header(default=None)):
    actor=_actor(authorization);_schema()
    from trust_voice_school_safety_registration import _school_admin
    with get_db() as conn:
        cur=conn.cursor()
        _school_admin(cur,school_id,str(actor["id"]))
        cur.execute("""INSERT INTO verifyd_safety_school_directory
            (school_id,main_phone,office_phone,public_instructions,updated_by)
            VALUES(%s,%s,%s,%s,%s)
            ON CONFLICT(school_id) DO UPDATE
             SET main_phone=EXCLUDED.main_phone,
                 office_phone=EXCLUDED.office_phone,
                 public_instructions=EXCLUDED.public_instructions,
                 updated_by=EXCLUDED.updated_by,updated_at=NOW()""",
            (school_id,payload.main_phone,payload.office_phone,
             payload.public_instructions,str(actor["id"])))
        _audit(cur,school_id,str(actor["id"]),"school_public_directory_updated",school_id)
        data=_details(cur,school_id)
    return {"ok":True,"directory":data,"notifications_sent":False,"test_only":True}
