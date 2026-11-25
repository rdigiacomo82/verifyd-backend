"""Privacy-preserving family intake for households not pre-registered by email.

The school's public shareable link permits adults to request school contact
using a phone OR email and describe multiple children. Name claims are not
matched or disclosed to the visitor. Only authenticated school admins can see
the requests; staff must independently verify relationship and student roster
before any private guardian invitation, linkage or approval.
"""
from __future__ import annotations
import hashlib
import re
import threading
import uuid
from typing import Literal,Optional
from fastapi import APIRouter,Header,HTTPException
from pydantic import BaseModel,Field,field_validator
from database import get_db,is_valid_email
from trust_voice_school_safety_training import _actor,_audit
from trust_voice_school_safety_registration import _school_admin
from trust_voice_school_family_link import _schema as _link_schema,_invites_enabled
from trust_voice_school_data_notice import DATA_NOTICE_VERSION

router=APIRouter(prefix="/trust-voice/school-safety",tags=["School Family Intake"])
_ready=False
_lock=threading.Lock()
_NAME=re.compile(r"^[^\x00-\x1F\x7F<>]{2,100}$")
_PHONE=re.compile(r"^\+?[0-9 ()\-]{10,24}$")


def _schema():
    global _ready
    _link_schema()
    if _ready:return
    with _lock:
        if _ready:return
        with get_db() as conn:
            conn.cursor().execute("""CREATE TABLE IF NOT EXISTS verifyd_safety_school_family_interests(
                request_id TEXT PRIMARY KEY,
                school_id TEXT NOT NULL REFERENCES verifyd_safety_training_schools(school_id),
                guardian_name TEXT NOT NULL,
                relationship TEXT NOT NULL,
                child_names TEXT[] NOT NULL,
                contact_method TEXT NOT NULL CHECK(contact_method IN ('email','phone')),
                contact_value TEXT NOT NULL,
                contact_fingerprint TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'awaiting_school_verification'
                    CHECK(status IN ('awaiting_school_verification','reviewed','closed')),
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                reviewed_at TIMESTAMPTZ,
                reviewed_by TEXT REFERENCES verifyd_identities(id),
                review_note TEXT NOT NULL DEFAULT ''
            )""")
            conn.cursor().execute("""ALTER TABLE verifyd_safety_school_family_interests
                ADD COLUMN IF NOT EXISTS data_notice_version TEXT NOT NULL DEFAULT ''""")
            conn.cursor().execute("""ALTER TABLE verifyd_safety_school_family_interests
                ADD COLUMN IF NOT EXISTS data_notice_acknowledged_at TIMESTAMPTZ""")
            conn.cursor().execute("""ALTER TABLE verifyd_safety_school_family_interests
                ADD COLUMN IF NOT EXISTS app_use_notice_acknowledged_at TIMESTAMPTZ""")
            conn.cursor().execute("""CREATE INDEX IF NOT EXISTS school_family_interest_recent
                ON verifyd_safety_school_family_interests
                (school_id,contact_fingerprint,created_at)""")
        _ready=True


class FamilyInterest(BaseModel):
    public_code:str=Field(min_length=16,max_length=96)
    guardian_name:str=Field(min_length=2,max_length=100)
    children:list[str]=Field(min_length=1,max_length=6)
    relationship:Literal["parent","legal_guardian","authorized_caregiver"]="parent"
    contact_method:Literal["email","phone"]
    contact_value:str=Field(min_length=5,max_length=254)
    permission_for_school_follow_up:Literal[True]
    data_notice_version:Optional[str]=Field(default=None,max_length=100)
    acknowledge_registration_data_use:bool=False
    acknowledge_app_use_instructions:bool=False

    @field_validator("guardian_name")
    @classmethod
    def check_guardian_name(cls,value):
        value=" ".join(value.split())
        if not _NAME.fullmatch(value):raise ValueError("invalid_contact_name")
        return value

    @field_validator("children")
    @classmethod
    def check_child_names(cls,values):
        result=[]
        for name in values:
            clean=" ".join(str(name).strip().split())
            if not _NAME.fullmatch(clean):raise ValueError("invalid_child_name")
            result.append(clean)
        if len({x.lower() for x in result})!=len(result):
            raise ValueError("duplicate_child_name_in_request")
        return result


class FamilyInterestDecision(BaseModel):
    status:Literal["reviewed","closed"]
    review_note:str=Field(default="",max_length=500)


class FamilyInterestBulkDecision(BaseModel):
    request_ids:list[str]=Field(min_length=1,max_length=50)
    status:Literal["reviewed","closed"]="reviewed"
    review_note:str=Field(default="",max_length=500)
    school_staff_confirms_independent_review:Literal[True]
    require_electronic_acknowledgment:bool=True


def _matched_electronic_ack(cur,school_id:str,row:dict,cache:dict|None=None):
    """Verified email ownership plus a separate school-held child assignment.

    Self-reported names never verify guardianship. Only admins see match counts.
    """
    children={str(name).strip().casefold() for name in row.get("child_names",[]) if str(name).strip()}
    empty={"electronic_acknowledgment_count":0,
           "electronic_acknowledgment_complete":False,
           "electronic_acknowledged_at":None}
    if row["contact_method"]!="email" or not children:
        return empty
    email=str(row["contact_value"]).strip().lower()
    if cache is not None and email in cache:
        matches=cache[email]
    else:
        cur.execute("""SELECT lower(s.first_name || ' ' || s.last_name) AS child_name,
                 MAX(a.acknowledged_at) AS acknowledged_at
              FROM verifyd_safety_electronic_guardian_ack a
              JOIN verifyd_safety_managed_students s
                ON s.school_id=a.school_id AND s.local_ref=a.local_ref
              WHERE a.school_id=%s AND lower(a.guardian_email)=%s
                AND a.verified_at IS NOT NULL
                AND a.status IN ('pending_school_review','approved')
              GROUP BY lower(s.first_name || ' ' || s.last_name)""",
              (school_id,email))
        matches={str(x["child_name"]).casefold():x["acknowledged_at"]
                 for x in cur.fetchall()}
        if cache is not None:cache[email]=matches
    acknowledged=[matches[name] for name in children if name in matches]
    return {"electronic_acknowledgment_count":len(acknowledged),
            "electronic_acknowledgment_complete":len(acknowledged)==len(children),
            "electronic_acknowledged_at":max(acknowledged) if acknowledged else None}



def _contact(method,value):
    value=value.strip()
    if method=="email":
        if not is_valid_email(value.lower()):
            raise HTTPException(422,detail={"error":"invalid_email_for_school_callback"})
        return value.lower()
    if not _PHONE.fullmatch(value):
        raise HTTPException(422,detail={"error":"enter_valid_phone_for_school_callback"})
    digits=re.sub(r"\D","",value)
    if len(digits) not in (10,11,12,13,14,15):
        raise HTTPException(422,detail={"error":"phone_should_include_area_code"})
    return "+"+digits


@router.post("/family-invitations/school-link/interest")
def submit_family_contact_request(payload:FamilyInterest):
    """Untrusted self-asserted interest, never parent verification."""
    generic={"ok":True,
             "message":"Your request was received. The school must independently verify guardianship before any child details or family access can be released.",
             "verified_guardian":False,"student_information_disclosed":False,
             "student_account_created":False,"family_access_granted":False}
    if (bool(payload.data_notice_version) != bool(payload.acknowledge_registration_data_use)
        or (payload.data_notice_version and payload.data_notice_version != DATA_NOTICE_VERSION)
        or (payload.data_notice_version and not payload.acknowledge_app_use_instructions)
        or (payload.acknowledge_app_use_instructions and not payload.data_notice_version)):
        raise HTTPException(409,detail={"error":"school_data_notice_version_changed_reload_registration"})
    _schema()
    contact=_contact(payload.contact_method,payload.contact_value)
    with get_db() as conn:
        cur=conn.cursor()
        cur.execute("""SELECT school_id FROM verifyd_safety_school_family_links
            WHERE public_code=%s""",(payload.public_code.strip(),))
        row=cur.fetchone()
        if not row:return generic
        school_id=str(row["school_id"])
        try:_invites_enabled(school_id)
        except HTTPException:return generic
        fingerprint=hashlib.sha256((school_id+":"+payload.contact_method+":"+contact).encode()).hexdigest()
        # psycopg2 requires a one-item tuple. The prior bare string raised
        # TypeError and surfaced as the browser's misleading "Failed to fetch".
        cur.execute("SELECT pg_advisory_xact_lock(hashtext(%s))",
                    ("school_family_interest:"+school_id,))
        # Aggregate anti-abuse controls. No claim-matching or student details
        # are ever returned, even if submitted names match a real roster.
        cur.execute("""SELECT count(*) AS n FROM verifyd_safety_school_family_interests
            WHERE school_id=%s AND created_at>NOW()-INTERVAL '24 hours'""",
            (school_id,))
        if int(cur.fetchone()["n"])>=250:return generic
        cur.execute("""SELECT count(*) AS n FROM verifyd_safety_school_family_interests
            WHERE school_id=%s AND contact_fingerprint=%s
              AND created_at>NOW()-INTERVAL '24 hours'""",
            (school_id,fingerprint))
        if int(cur.fetchone()["n"])>=2:return generic
        cur.execute("""DELETE FROM verifyd_safety_school_family_interests
            WHERE created_at<NOW()-INTERVAL '30 days'
              AND status!='awaiting_school_verification'""")
        cur.execute("""INSERT INTO verifyd_safety_school_family_interests
          (request_id,school_id,guardian_name,relationship,child_names,
           contact_method,contact_value,contact_fingerprint,
           data_notice_version,data_notice_acknowledged_at,
           app_use_notice_acknowledged_at)
          VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,
                 CASE WHEN %s THEN NOW() ELSE NULL END,
                 CASE WHEN %s THEN NOW() ELSE NULL END)""",
          ("ssfinterest_"+uuid.uuid4().hex,school_id,payload.guardian_name,
           payload.relationship,payload.children,payload.contact_method,
           contact,fingerprint,payload.data_notice_version or "",
           bool(payload.acknowledge_registration_data_use),
           bool(payload.acknowledge_app_use_instructions)))
    return generic


@router.get("/schools/{school_id}/family-invitations/family-interest")
def list_school_family_interests(school_id:str,
                                  view:Literal["pending","history","all"]="pending",
                                  authorization:Optional[str]=Header(default=None)):
    actor=_actor(authorization);_schema()
    with get_db() as conn:
        cur=conn.cursor()
        _school_admin(cur,school_id,str(actor["id"]))
        where=("" if view=="all" else
               " AND status='awaiting_school_verification'" if view=="pending"
               else " AND status IN ('reviewed','closed')")
        cur.execute("""SELECT request_id,guardian_name,relationship,child_names,
            contact_method,contact_value,status,created_at,reviewed_at,review_note,
            data_notice_version,data_notice_acknowledged_at,
            app_use_notice_acknowledged_at
            FROM verifyd_safety_school_family_interests WHERE school_id=%s"""
            +where+" ORDER BY created_at DESC LIMIT 150",(school_id,))
        rows=[dict(r) for r in cur.fetchall()]
        cache={}
        for row in rows:
            row.update(_matched_electronic_ack(cur,school_id,row,cache))
    return {"ok":True,"requests":rows,"view":view,
      "children_verified":False,
      "requires_separate_school_verification":True,
      "ordinary_mobile_sms_available":False}


@router.post("/schools/{school_id}/family-invitations/family-interest/{request_id}/review")
def review_family_interest(school_id:str,request_id:str,
                           payload:FamilyInterestDecision,
                           authorization:Optional[str]=Header(default=None)):
    actor=_actor(authorization);_schema()
    with get_db() as conn:
        cur=conn.cursor()
        _school_admin(cur,school_id,str(actor["id"]))
        cur.execute("""UPDATE verifyd_safety_school_family_interests
          SET status=%s,review_note=%s,reviewed_at=NOW(),reviewed_by=%s
          WHERE school_id=%s AND request_id=%s
            AND status='awaiting_school_verification'
          RETURNING request_id""",
          (payload.status,payload.review_note.strip(),str(actor["id"]),
           school_id,request_id))
        if not cur.fetchone():
            raise HTTPException(409,detail={"error":"family_contact_request_already_reviewed_or_missing"})
        _audit(cur,school_id,str(actor["id"]),"school_family_contact_reviewed",request_id)
    return {"ok":True,"status":payload.status,"guardian_verified":False,
            "family_access_granted":False,"student_record_modified":False}


@router.post("/schools/{school_id}/family-invitations/family-interest/bulk-review")
def bulk_review_family_interests(school_id:str,payload:FamilyInterestBulkDecision,
    authorization:Optional[str]=Header(default=None)):
    """Review at most 50 school callback requests; never activate/approve a child.

    Staff must already have independently verified the actual family records.
    For 'reviewed' bulk actions, only consider requests whose school-confirmed
    email matches an OTP acknowledgment for EVERY named child; phone-only and
    partial family matches must be individually processed by staff.
    """
    actor=_actor(authorization);_schema()
    ids=[str(x).strip() for x in payload.request_ids]
    if any(not x.startswith("ssfinterest_") or len(x)>90 for x in ids) or len(set(ids))!=len(ids):
        raise HTTPException(422,detail={"error":"invalid_or_duplicate_family_request_id"})
    with get_db() as conn:
        cur=conn.cursor()
        _school_admin(cur,school_id,str(actor["id"]))
        cur.execute("""SELECT request_id,contact_method,contact_value,child_names,status
            FROM verifyd_safety_school_family_interests
            WHERE school_id=%s AND request_id=ANY(%s)
            FOR UPDATE""",(school_id,ids))
        rows=[dict(x) for x in cur.fetchall()]
        if len(rows)!=len(ids) or any(x["status"]!="awaiting_school_verification" for x in rows):
            raise HTTPException(409,detail={"error":"family_callback_selection_changed_refresh"})
        if payload.status=="reviewed" and payload.require_electronic_acknowledgment:
            cache={}
            for row in rows:
                match=_matched_electronic_ack(cur,school_id,row,cache)
                if not match["electronic_acknowledgment_complete"]:
                    raise HTTPException(409,detail={"error":"family_callback_missing_verified_electronic_acknowledgment"})
        cur.execute("""UPDATE verifyd_safety_school_family_interests
           SET status=%s,review_note=%s,reviewed_at=NOW(),reviewed_by=%s
           WHERE school_id=%s AND request_id=ANY(%s)
              AND status='awaiting_school_verification'""",
           (payload.status,payload.review_note.strip(),str(actor["id"]),school_id,ids))
        for request_id in ids:
            _audit(cur,school_id,str(actor["id"]),"school_family_callback_bulk_"+payload.status,request_id)
    return {"ok":True,"reviewed_count":len(ids),"status":payload.status,
            "recorded_in_history":True,"student_or_guardian_approved":False,
            "family_access_granted":False,"student_record_modified":False}


# School-mediated guardian office check-in routes (TEST-only and gated).
from trust_voice_school_guardian_office_requests import router as _school_office_router
router.include_router(_school_office_router)
