"""Principal-initiated school staff account invitation; no owner second review.

A Principal enters a verified school employee's name, official email and
position. A unique school-prefixed handle is RESERVED in an invitation record.
The actual VeriFYD identity is created (or the existing identity reused) only
after the invited adult personally verifies that email via OTP AND accepts
school-scoped administrator responsibility. Never email login passwords.
"""
from __future__ import annotations
import hashlib
import html
import re
import secrets
import threading
import uuid
from datetime import datetime,timezone
from typing import Literal,Optional
from fastapi import APIRouter,Header,HTTPException
from pydantic import BaseModel,Field
from database import get_db,create_otp,verify_otp,is_valid_email
from emailer import _send,FROM_NAME,FROM_ADDRESS,send_otp_email
from trust_voice_school_safety_registration import _school_admin
from trust_voice_school_multi_admin import _schema as _admin_schema
from trust_voice_school_managed_roster import _schema as _roster_schema
from trust_voice_school_safety_training import _actor,_audit
from trust_voice_identity import _ensure_identity_schema,_recent_otp_request

router=APIRouter(prefix="/trust-voice/school-safety",tags=["School Administrator Invitations"])
_lock=threading.Lock()
_ready=False

def _hash(v):return hashlib.sha256(v.encode("utf-8")).hexdigest()

def _schema():
    global _ready
    _admin_schema();_roster_schema();_ensure_identity_schema()
    if _ready:return
    with _lock:
        if _ready:return
        with get_db() as conn:
            conn.cursor().execute("""CREATE TABLE IF NOT EXISTS verifyd_safety_staff_invites (
                invite_id TEXT PRIMARY KEY,
                school_id TEXT NOT NULL REFERENCES verifyd_safety_training_schools(school_id),
                staff_name TEXT NOT NULL,
                staff_email TEXT NOT NULL,
                position TEXT NOT NULL,
                proposed_handle TEXT NOT NULL,
                token_hash TEXT NOT NULL UNIQUE,
                invited_by TEXT NOT NULL REFERENCES verifyd_identities(id),
                delivery_status TEXT NOT NULL DEFAULT 'queued',
                otp_requests INTEGER NOT NULL DEFAULT 0,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                expires_at TIMESTAMPTZ NOT NULL DEFAULT NOW()+INTERVAL '7 days',
                accepted_at TIMESTAMPTZ,
                accepted_identity_id TEXT REFERENCES verifyd_identities(id),
                revoked_at TIMESTAMPTZ
            )""")
            conn.cursor().execute("""CREATE UNIQUE INDEX IF NOT EXISTS safety_staff_email_pending_unique
                ON verifyd_safety_staff_invites(school_id,lower(staff_email))
                WHERE accepted_at IS NULL AND revoked_at IS NULL""")
            conn.cursor().execute("""CREATE UNIQUE INDEX IF NOT EXISTS safety_staff_handle_pending_unique
                ON verifyd_safety_staff_invites(lower(proposed_handle))
                WHERE accepted_at IS NULL AND revoked_at IS NULL""")
        _ready=True


def _ensure_school_naming(cur, school_id: str, principal_id: str):
    """Initialize a collision-safe prefix for a certified Principal on first staff invite.

    Preserve manually configured naming exactly. No enrollment, identity,
    policy acknowledgment, or emergency notification is created here.
    The global advisory lock is shared with save_managed_roster_settings.
    """
    cur.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", ("safety_school_code_settings",))
    cur.execute("""SELECT school_code FROM verifyd_safety_managed_roster_settings
        WHERE school_id=%s""", (school_id,))
    existing = cur.fetchone()
    if existing:
        return str(existing["school_code"]).upper(), False
    cur.execute("""SELECT school_name FROM verifyd_safety_training_schools
        WHERE school_id=%s""", (school_id,))
    school = cur.fetchone()
    if not school:
        raise HTTPException(404, detail={"error": "school_not_found"})
    name = re.findall(r"[A-Za-z0-9]+", str(school["school_name"]))
    initials = "".join(part[0] for part in name if part and part[0].isalpha()).upper()
    if len(initials) < 2:
        initials = "".join(ch for ch in str(school["school_name"]).upper() if ch.isalpha())[:5]
    base = initials[:8] if len(initials) >= 2 else "VS"
    digest = hashlib.sha256(school_id.encode("utf-8")).hexdigest().upper()
    candidates = [base] + [base[:5] + digest[i:i+4] for i in range(0, 28, 4)]
    for candidate in candidates:
        cur.execute("""SELECT 1 FROM verifyd_safety_managed_roster_settings
            WHERE school_code=%s""", (candidate,))
        if cur.fetchone():
            continue
        cur.execute("""INSERT INTO verifyd_safety_managed_roster_settings
            (school_id,school_code,handle_style,configured_by)
            VALUES(%s,%s,'name',%s)""", (school_id,candidate,principal_id))
        _audit(cur,school_id,principal_id,"school_naming_auto_initialized_for_staff",candidate)
        return candidate, True
    raise HTTPException(409, detail={"error": "school_prefix_unavailable_contact_operator"})


class StaffInvite(BaseModel):
    staff_name:str=Field(min_length=3,max_length=80)
    official_email:str=Field(min_length=5,max_length=254)
    position:str=Field(min_length=3,max_length=100)
    principal_verified_employment_and_authority:Literal[True]


class StaffInviteCode(BaseModel):
    invitation_code:str=Field(min_length=32,max_length=128)


class StaffAccept(StaffInviteCode):
    email_code:str=Field(min_length=6,max_length=12)
    accept_school_administrator_responsibility:Literal[True]


def _active_invitation(cur,code,lock=False):
    cur.execute("""SELECT * FROM verifyd_safety_staff_invites
        WHERE token_hash=%s """+("FOR UPDATE" if lock else ""),
        (_hash(code.strip()),))
    row=cur.fetchone()
    if (not row or row["revoked_at"] or row["accepted_at"]
        or row["expires_at"]<=datetime.now(timezone.utc)):
        raise HTTPException(403,detail={"error":"invalid_or_expired_staff_invitation"})
    return dict(row)


@router.get("/schools/{school_id}/staff-invitations")
def list_staff_invites(school_id:str,authorization:Optional[str]=Header(default=None)):
    actor=_actor(authorization);_schema()
    with get_db() as conn:
        cur=conn.cursor()
        if _school_admin(cur,school_id,str(actor["id"]))!="principal":
            raise HTTPException(403,detail={"error":"principal_required"})
        cur.execute("""SELECT invite_id,staff_name,staff_email,position,proposed_handle,
            delivery_status,created_at,expires_at,accepted_at,revoked_at
            FROM verifyd_safety_staff_invites WHERE school_id=%s
            ORDER BY created_at DESC LIMIT 60""",(school_id,))
        rows=[dict(x) for x in cur.fetchall()]
    return {"ok":True,"invitations":rows,"real_account_created_before_acceptance":False}


@router.post("/schools/{school_id}/staff-invitations")
def create_staff_invite(school_id:str,payload:StaffInvite,
                        authorization:Optional[str]=Header(default=None)):
    actor=_actor(authorization);_schema()
    email=payload.official_email.strip().lower()
    if not is_valid_email(email):
        raise HTTPException(422,detail={"error":"invalid_staff_email"})
    staff_name=" ".join(payload.staff_name.strip().split())
    position=" ".join(payload.position.strip().split())
    code=secrets.token_urlsafe(32)
    invite_id="ssstaff_"+uuid.uuid4().hex
    proposed=None
    school_name=""
    with get_db() as conn:
        cur=conn.cursor()
        if _school_admin(cur,school_id,str(actor["id"]))!="principal":
            raise HTTPException(403,detail={"error":"principal_required"})
        cur.execute("SELECT pg_advisory_xact_lock(hashtext(%s))",("safety_staff_registration:"+school_id,))
        prefix, prefix_created = _ensure_school_naming(cur,school_id,str(actor["id"]))
        # Expired, unused invitations should not permanently block a staff email.
        cur.execute("""UPDATE verifyd_safety_staff_invites
            SET revoked_at=NOW(),delivery_status='expired'
            WHERE school_id=%s AND accepted_at IS NULL AND revoked_at IS NULL
              AND expires_at<=NOW()""",(school_id,))
        cur.execute("""SELECT count(*) AS n FROM verifyd_safety_school_admin_delegations
            WHERE school_id=%s AND revoked_at IS NULL""",(school_id,))
        assigned=int(cur.fetchone()["n"])
        cur.execute("""SELECT count(*) AS n FROM verifyd_safety_staff_invites
            WHERE school_id=%s AND accepted_at IS NULL AND revoked_at IS NULL
              AND expires_at>NOW()""",(school_id,))
        pending=int(cur.fetchone()["n"])
        if assigned+pending>=10:
            raise HTTPException(409,detail={"error":"school_admin_limit_reached"})
        cur.execute("""SELECT 1 FROM verifyd_safety_staff_invites
            WHERE school_id=%s AND lower(staff_email)=%s
              AND accepted_at IS NULL AND revoked_at IS NULL""",(school_id,email))
        if cur.fetchone():
            raise HTTPException(409,detail={"error":"staff_email_already_invited"})
        cur.execute("""SELECT m.role,d.identity_id FROM verifyd_identities i
            LEFT JOIN verifyd_safety_training_members m
             ON m.school_id=%s AND m.identity_id=i.id
            LEFT JOIN verifyd_safety_school_admin_delegations d
             ON d.school_id=%s AND d.identity_id=i.id AND d.revoked_at IS NULL
            WHERE i.email_lower=%s""",(school_id,school_id,email))
        membership=cur.fetchone()
        if membership and (membership["role"] in ("student","guardian") or membership["identity_id"]):
            raise HTTPException(409,detail={"error":"staff_email_is_existing_student_guardian_or_admin"})
        for _ in range(20):
            candidate=prefix+"_ADM"+secrets.token_hex(3).upper()
            if len(candidate)>24:continue
            cur.execute("""SELECT 1 FROM verifyd_identities WHERE handle_lower=%s
                UNION SELECT 1 FROM verifyd_safety_staff_invites
                WHERE lower(proposed_handle)=%s AND revoked_at IS NULL LIMIT 1""",
                (candidate.lower(),candidate.lower()))
            if not cur.fetchone():proposed=candidate;break
        if not proposed:
            raise HTTPException(409,detail={"error":"could_not_allocate_school_staff_handle"})
        cur.execute("""SELECT school_name FROM verifyd_safety_training_schools
            WHERE school_id=%s""",(school_id,))
        school_name=cur.fetchone()["school_name"]
        cur.execute("""INSERT INTO verifyd_safety_staff_invites
          (invite_id,school_id,staff_name,staff_email,position,proposed_handle,token_hash,invited_by)
          VALUES(%s,%s,%s,%s,%s,%s,%s,%s)""",
          (invite_id,school_id,staff_name,email,position,proposed,_hash(code),str(actor["id"])))
        _audit(cur,school_id,str(actor["id"]),"principal_staff_invitation_issued",invite_id)
    url="https://vfvid.com/Safety#school-staff="+code
    delivered=False
    try:
        delivered=bool(_send({
          "from":f"{FROM_NAME} <{FROM_ADDRESS}>",
          "to":[email],
          "subject":"VeriFYD Safety — Your school administrator invitation",
          "html":"<p>You have been invited to accept a school-only VeriFYD Safety role at "
              +html.escape(school_name)+".</p><p><a href=\""+html.escape(url,quote=True)
              +"\">Verify email and accept role</a></p><p>Expires in 7 days. "
              "Do not forward. Verify the invitation with your school if unexpected.</p>"
        }))
    except Exception:delivered=False
    with get_db() as conn:
        conn.cursor().execute("""UPDATE verifyd_safety_staff_invites
            SET delivery_status=%s WHERE invite_id=%s""",
            ("accepted_by_provider" if delivered else "failed",invite_id))
    return {"ok":True,"invite_id":invite_id,"proposed_handle":"@"+proposed,
            "school_prefix":prefix,"school_prefix_auto_initialized":prefix_created,
            "delivery_status":"accepted_by_provider" if delivered else "failed",
            "new_account_active":False,"pending_staff_email_verification":True,
            "administrator_access_granted":False}


@router.post("/staff-invitations/request-code")
def staff_invite_email_code(payload:StaffInviteCode):
    _schema()
    with get_db() as conn:
        cur=conn.cursor()
        invite=_active_invitation(cur,payload.invitation_code,True)
        if invite["otp_requests"]>=6:
            raise HTTPException(429,detail={"error":"staff_invitation_code_limit"})
        cur.execute("""UPDATE verifyd_safety_staff_invites
            SET otp_requests=otp_requests+1 WHERE invite_id=%s""",(invite["invite_id"],))
        email=invite["staff_email"]
    if _recent_otp_request(email):
        raise HTTPException(429,detail={"error":"verification_code_recently_sent"})
    otp=create_otp(email)
    if not send_otp_email(email,otp):
        raise HTTPException(503,detail={"error":"verification_email_not_sent"})
    return {"ok":True,"sent":True,"expires_minutes":10}


@router.post("/staff-invitations/accept")
def staff_accept_invitation(payload:StaffAccept):
    """Staff-owned OTP + personal role acceptance activates the account."""
    _schema()
    with get_db() as conn:
        cur=conn.cursor()
        invite=_active_invitation(cur,payload.invitation_code)
        email=invite["staff_email"]
    passed,_=verify_otp(email,payload.email_code)
    if not passed:
        raise HTTPException(403,detail={"error":"invalid_staff_email_verification_code"})
    with get_db() as conn:
        cur=conn.cursor()
        invite=_active_invitation(cur,payload.invitation_code,True)
        school_id=invite["school_id"]
        cur.execute("SELECT pg_advisory_xact_lock(hashtext(%s))",
                    ("safety_staff_registration:"+school_id,))
        # Refuse if the school Principal is no longer an approved Principal.
        if _school_admin(cur,school_id,str(invite["invited_by"]))!="principal":
            raise HTTPException(403,detail={"error":"inviting_principal_no_longer_authorized"})
        cur.execute("""SELECT id,handle,status,email_verified FROM verifyd_identities
            WHERE email_lower=%s FOR UPDATE""",(email,))
        existing=cur.fetchone()
        if existing and (existing["status"]!="active" or not existing["email_verified"]):
            raise HTTPException(409,detail={"error":"existing_account_requires_manual_verification"})
        identity_id=str(existing["id"]) if existing else "vfyd_usr_"+uuid.uuid4().hex
        handle=str(existing["handle"]) if existing else str(invite["proposed_handle"])
        cur.execute("""SELECT role FROM verifyd_safety_training_members
            WHERE school_id=%s AND identity_id=%s""",(school_id,identity_id))
        member=cur.fetchone()
        if member and member["role"]!="responder":
            raise HTTPException(409,detail={"error":"cannot_promote_guardian_or_student"})
        cur.execute("""SELECT 1 FROM verifyd_safety_school_admin_delegations
            WHERE school_id=%s AND identity_id=%s AND revoked_at IS NULL""",
            (school_id,identity_id))
        if cur.fetchone():
            raise HTTPException(409,detail={"error":"already_school_administrator"})
        cur.execute("""SELECT count(*) AS n FROM verifyd_safety_school_admin_delegations
            WHERE school_id=%s AND revoked_at IS NULL""",(school_id,))
        if int(cur.fetchone()["n"])>=10:
            raise HTTPException(409,detail={"error":"school_admin_limit_reached"})
        if not existing:
            cur.execute("""SELECT 1 FROM verifyd_identities WHERE handle_lower=%s""",
                        (handle.lower(),))
            if cur.fetchone():
                raise HTTPException(409,detail={"error":"reserved_staff_handle_no_longer_available"})
            cur.execute("""INSERT INTO verifyd_identities
                (id,email_lower,handle,handle_lower,display_name,display_emoji,
                 email_verified,identity_verified,organization_verified,
                 verification_level,can_receive_calls,call_privacy,status,created_at,updated_at)
                VALUES(%s,%s,%s,%s,%s,'',TRUE,FALSE,FALSE,'email_verified',
                    FALSE,'nobody','active',NOW()::text,NOW()::text)""",
                (identity_id,email,handle,handle.lower(),invite["staff_name"]))
        if not member:
            cur.execute("""INSERT INTO verifyd_safety_training_members
                (school_id,identity_id,role,verified_by)
                VALUES(%s,%s,'responder',%s)""",
                (school_id,identity_id,invite["invited_by"]))
        cur.execute("""INSERT INTO verifyd_safety_school_admin_delegations
            (school_id,identity_id,approved_by,approved_at,accepted_at,revoked_at)
            VALUES(%s,%s,%s,NOW(),NOW(),NULL)
            ON CONFLICT(school_id,identity_id) DO UPDATE SET
                approved_by=EXCLUDED.approved_by,approved_at=NOW(),
                accepted_at=NOW(),revoked_at=NULL""",
            (school_id,identity_id,invite["invited_by"]))
        cur.execute("""UPDATE verifyd_safety_staff_invites
           SET accepted_at=NOW(),accepted_identity_id=%s
           WHERE invite_id=%s""",(identity_id,invite["invite_id"]))
        _audit(cur,school_id,identity_id,"staff_verified_email_and_accepted_role",
               invite["invite_id"])
    return {"ok":True,"status":"active_school_administrator",
            "handle":"@"+handle,"school_only":True,
            "account_created":existing is None,
            "email_verified":True,"emergency_dispatch_active":False}


@router.post("/schools/{school_id}/staff-invitations/{invite_id}/revoke")
def principal_revoke_pending_staff_invitation(school_id:str,invite_id:str,
    authorization:Optional[str]=Header(default=None)):
    actor=_actor(authorization);_schema()
    with get_db() as conn:
        cur=conn.cursor()
        if _school_admin(cur,school_id,str(actor["id"]))!="principal":
            raise HTTPException(403,detail={"error":"principal_required"})
        cur.execute("""UPDATE verifyd_safety_staff_invites SET revoked_at=NOW()
            WHERE school_id=%s AND invite_id=%s AND accepted_at IS NULL AND revoked_at IS NULL
            RETURNING invite_id""",(school_id,invite_id))
        if not cur.fetchone():
            raise HTTPException(404,detail={"error":"pending_staff_invite_not_found"})
        _audit(cur,school_id,str(actor["id"]),"principal_staff_invitation_revoked",invite_id)
    return {"ok":True,"pending_invitation_revoked":True,"account_deleted":False}
