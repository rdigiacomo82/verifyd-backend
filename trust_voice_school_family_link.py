"""Shareable school entry link; individually verified family invitations only.

A schoolwide link is an ENTRY POINT, not a student credential or household
authorization. A registered guardian must be preassigned by school staff to
an approved child; only their specified email receives a private invitation,
and guardian OTP + school final verification are still mandatory.
"""
from __future__ import annotations
import secrets
import threading
import uuid
from typing import Optional
from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field
from database import get_db
from trust_voice_school_safety_training import _actor, _audit
from trust_voice_school_safety_registration import _school_admin
from trust_voice_school_family_invitations import (
    _schema as _invites_schema, _enabled as _invites_enabled,
    _hash, _send_invitation, is_valid_email,
)

router=APIRouter(prefix="/trust-voice/school-safety",
                 tags=["Verified School Family Registration Entry"])
_ready=False
_lock=threading.Lock()

def _schema():
    global _ready
    _invites_schema()
    if _ready:return
    with _lock:
        if _ready:return
        with get_db() as conn:
            cur=conn.cursor()
            cur.execute("""CREATE TABLE IF NOT EXISTS verifyd_safety_school_family_links(
              school_id TEXT PRIMARY KEY REFERENCES verifyd_safety_training_schools(school_id),
              public_code TEXT NOT NULL UNIQUE,
              created_by TEXT NOT NULL REFERENCES verifyd_identities(id),
              created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
              rotated_at TIMESTAMPTZ
            )""")
            cur.execute("""CREATE TABLE IF NOT EXISTS verifyd_safety_school_family_entry_requests(
              id TEXT PRIMARY KEY,
              school_id TEXT NOT NULL REFERENCES verifyd_safety_training_schools(school_id),
              recipient_hash TEXT NOT NULL,
              requested_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )""")
            cur.execute("""CREATE INDEX IF NOT EXISTS safety_family_entry_request_rate
              ON verifyd_safety_school_family_entry_requests
              (school_id,recipient_hash,requested_at)""")
        _ready=True


def _new_school_code():
    return "SC_" + secrets.token_urlsafe(18)


@router.get("/schools/{school_id}/family-invitations/school-link")
def get_or_create_school_registration_link(
    school_id:str,authorization:Optional[str]=Header(default=None)):
    """Only authenticated school administrators see the public share URL."""
    _invites_enabled(school_id)
    actor=_actor(authorization);_schema()
    with get_db() as conn:
        cur=conn.cursor()
        _school_admin(cur,school_id,str(actor["id"]))
        cur.execute("""INSERT INTO verifyd_safety_school_family_links
            (school_id,public_code,created_by)
            VALUES(%s,%s,%s) ON CONFLICT(school_id) DO NOTHING""",
            (school_id,_new_school_code(),str(actor["id"])))
        cur.execute("""SELECT public_code FROM verifyd_safety_school_family_links
            WHERE school_id=%s""",(school_id,))
        code=cur.fetchone()["public_code"]
    return {"ok":True,"url":"https://vfvid.com/Safety#school-family="+code,
            "public_code":code,"entry_only":True,
            "family_verification_required":True,
            "individual_guardian_email_invitation_required":True}


class SchoolLinkRotation(BaseModel):
    principal_confirms_new_link: bool

@router.post("/schools/{school_id}/family-invitations/school-link/rotate")
def rotate_school_registration_link(
    school_id:str,payload:SchoolLinkRotation,
    authorization:Optional[str]=Header(default=None)):
    """Optional Principal-only invalidation of an old schoolwide entry link."""
    if payload.principal_confirms_new_link is not True:
        raise HTTPException(422,detail={"error":"principal_confirmation_required"})
    _invites_enabled(school_id)
    actor=_actor(authorization);_schema()
    with get_db() as conn:
        cur=conn.cursor()
        if _school_admin(cur,school_id,str(actor["id"]))!="principal":
            raise HTTPException(403,detail={"error":"school_principal_required"})
        new_code=_new_school_code()
        cur.execute("""INSERT INTO verifyd_safety_school_family_links
            (school_id,public_code,created_by,rotated_at)
            VALUES(%s,%s,%s,NOW())
            ON CONFLICT(school_id) DO UPDATE
            SET public_code=EXCLUDED.public_code,rotated_at=NOW()""",
            (school_id,new_code,str(actor["id"])))
        _audit(cur,school_id,str(actor["id"]),"school_family_public_link_rotated",school_id)
    return {"ok":True,"url":"https://vfvid.com/Safety#school-family="+new_code,
            "prior_public_link_invalidated":True,"individual_invitations_unchanged":True}


class FamilyEmailLookup(BaseModel):
    public_code:str=Field(min_length=16,max_length=96)
    guardian_email:str=Field(min_length=5,max_length=254)


@router.post("/family-invitations/school-link/request")
def request_personal_invitation_via_school_link(payload:FamilyEmailLookup):
    """Request individual email invites from verified preassigned recipient list.

    Response is ALWAYS generic to prevent discovering a student's school
    membership or whether an email has a guardian assignment.
    """
    generic={"ok":True,"message":"If your email is on the school's verified, approved guardian list, private invitations will be emailed. Check spam or contact the school if none arrive.",
             "student_information_disclosed":False,"family_access_granted":False}
    email=payload.guardian_email.strip().lower()
    if not is_valid_email(email):return generic
    _schema()
    sends=[]
    with get_db() as conn:
        cur=conn.cursor()
        cur.execute("""SELECT school_id FROM verifyd_safety_school_family_links
            WHERE public_code=%s""",(payload.public_code.strip(),))
        row=cur.fetchone()
        if not row:return generic
        school_id=str(row["school_id"])
        # Restrict to schools enabled by VeriFYD operator for family onboarding.
        try:_invites_enabled(school_id)
        except HTTPException:return generic
        cur.execute("SELECT pg_advisory_xact_lock(hashtext(%s))",
                    ("school_family_public_entry:"+school_id,))
        lookup_hash=_hash(school_id+":"+email)
        cur.execute("""SELECT count(*) AS n FROM verifyd_safety_school_family_entry_requests
            WHERE school_id=%s AND recipient_hash=%s
              AND requested_at>NOW()-INTERVAL '24 hours'""",
            (school_id,lookup_hash))
        if int(cur.fetchone()["n"])>=2:return generic
        cur.execute("""SELECT count(*) AS n FROM verifyd_safety_school_family_entry_requests
            WHERE school_id=%s AND requested_at>NOW()-INTERVAL '24 hours'""",
            (school_id,))
        if int(cur.fetchone()["n"])>=300:return generic
        cur.execute("""INSERT INTO verifyd_safety_school_family_entry_requests
            (id,school_id,recipient_hash) VALUES(%s,%s,%s)""",
            ("ssfentry_"+uuid.uuid4().hex,school_id,lookup_hash))
        cur.execute("""SELECT c.local_ref,c.slot,c.guardian_name,c.relationship,
              r.student_identity_id,s.school_name
            FROM verifyd_safety_family_contact_drafts c
            JOIN verifyd_safety_managed_students r
              ON r.school_id=c.school_id AND r.local_ref=c.local_ref
            JOIN verifyd_safety_training_schools s ON s.school_id=c.school_id
            WHERE c.school_id=%s AND c.guardian_email=%s
              AND r.enrollment_status='school_approved'
              AND r.student_identity_id IS NOT NULL
            ORDER BY c.local_ref,c.slot LIMIT 10""",(school_id,email))
        assigned=[dict(r) for r in cur.fetchall()]
        for contact in assigned:
            cur.execute("""SELECT 1 FROM verifyd_safety_family_invitations
                WHERE school_id=%s AND local_ref=%s AND slot=%s AND revoked_at IS NULL
                  AND (claimed_at IS NOT NULL OR expires_at>NOW())
                LIMIT 1""",(school_id,contact["local_ref"],contact["slot"]))
            if cur.fetchone():continue
            code=secrets.token_urlsafe(32)
            invitation_id="ssinv_"+uuid.uuid4().hex
            cur.execute("""INSERT INTO verifyd_safety_family_invitations
                (invite_id,school_id,local_ref,student_id,slot,recipient_email,
                guardian_name,relationship,token_hash,created_by)
                VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                (invitation_id,school_id,contact["local_ref"],
                 contact["student_identity_id"],contact["slot"],email,
                 contact["guardian_name"],contact["relationship"],_hash(code),
                 # record the staff member who verified recipient identity,
                 # not an anonymous user, as invitation's responsible sponsor:
                 _verified_contact_creator(cur,school_id,contact["local_ref"],contact["slot"])))
            sends.append((invitation_id,email,contact["school_name"],code))
    for invitation_id,recipient,school_name,code in sends:
        ok=False
        try:ok=bool(_send_invitation(recipient,school_name,code))
        except Exception:ok=False
        with get_db() as conn:
            conn.cursor().execute("""UPDATE verifyd_safety_family_invitations
                SET delivery_status=%s,send_attempts=send_attempts+1,
                    last_sent_at=NOW() WHERE invite_id=%s""",
                ("sent" if ok else "failed",invitation_id))
    return generic


def _verified_contact_creator(cur,school_id,local_ref,slot):
    cur.execute("""SELECT verified_by FROM verifyd_safety_family_contact_drafts
        WHERE school_id=%s AND local_ref=%s AND slot=%s""",(school_id,local_ref,slot))
    return str(cur.fetchone()["verified_by"])
