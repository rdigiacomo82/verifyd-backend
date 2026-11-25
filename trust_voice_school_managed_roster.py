"""School-managed enrollment: handle generation, roster staging and evidence-based
bulk enrollment. This is an enduring school product capability, not a TEST
enrollment path. It never enables emergency dispatch or broad /voice access.

School officials may attest to *individual* paper signatures or documented
school authority and approve many verified rows in one transaction. Upload alone
cannot authorize students or guardians. Adult account claim and family messaging
remain independently verified, and district authorization cannot masquerade as
an individual's guardian signature.
"""
from __future__ import annotations
import json
import os
from datetime import date, datetime
import re
import secrets
import threading
import unicodedata
import uuid
from typing import Literal, Optional
from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field, field_validator
from database import get_db, is_valid_email
from trust_voice_identity import _validate_handle
from trust_voice_school_safety_training import _actor, _audit
from trust_voice_school_safety_registration import _application_schema, _school_admin

router = APIRouter(prefix="/trust-voice/school-safety", tags=["School Managed Enrollment"])
_lock = threading.Lock()
_ready = False
_CODE = re.compile(r"^[A-Z][A-Z0-9]{1,9}$")
_REF = re.compile(r"^[A-Za-z0-9_-]{3,64}$")


def _require_enrollment_enabled(school_id):
    active = os.getenv("VERIFYD_SAFETY_MANAGED_ROSTER_ENABLED", "").lower() in ("1", "true", "yes", "on")
    scopes = {x.strip() for x in os.getenv("VERIFYD_SAFETY_MANAGED_ROSTER_SCHOOL_IDS", "").split(",") if x.strip()}
    if not active or school_id not in scopes:
        raise HTTPException(404, detail={"error": "managed_roster_not_enabled_for_school"})


def _schema():
    global _ready
    _application_schema()
    if _ready:
        return
    with _lock:
        if _ready:
            return
        with get_db() as conn:
            cur = conn.cursor()
            cur.execute("""CREATE TABLE IF NOT EXISTS verifyd_safety_managed_roster_settings (
                school_id TEXT PRIMARY KEY REFERENCES verifyd_safety_training_schools(school_id),
                school_code TEXT NOT NULL UNIQUE,
                handle_style TEXT NOT NULL CHECK(handle_style IN ('name','code')),
                configured_by TEXT NOT NULL REFERENCES verifyd_identities(id),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )""")
            cur.execute("""CREATE TABLE IF NOT EXISTS verifyd_safety_managed_students (
                school_id TEXT NOT NULL REFERENCES verifyd_safety_training_schools(school_id),
                local_ref TEXT NOT NULL,
                first_name TEXT NOT NULL,
                last_name TEXT NOT NULL,
                proposed_handle TEXT NOT NULL,
                handle_lower TEXT NOT NULL UNIQUE,
                imported_by TEXT NOT NULL REFERENCES verifyd_identities(id),
                student_identity_id TEXT UNIQUE REFERENCES verifyd_identities(id),
                enrollment_status TEXT NOT NULL DEFAULT 'staged'
                  CHECK(enrollment_status IN ('staged','school_approved')),
                approved_at TIMESTAMPTZ,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                PRIMARY KEY(school_id,local_ref)
            )""")
            cur.execute("""ALTER TABLE verifyd_safety_managed_students
                ADD COLUMN IF NOT EXISTS tracking_reference TEXT""")
            # A school-local, permanent reference for tracking the enrollment
            # record. It does NOT itself constitute guardian consent or
            # legal authorization evidence.
            cur.execute("""UPDATE verifyd_safety_managed_students
                SET tracking_reference='ENROLL-' || upper(local_ref)
                WHERE tracking_reference IS NULL""")
            cur.execute("""CREATE UNIQUE INDEX IF NOT EXISTS safety_managed_school_tracking_unique
                ON verifyd_safety_managed_students(school_id, lower(tracking_reference))""")
            cur.execute("""CREATE TABLE IF NOT EXISTS verifyd_safety_managed_student_contacts(
                school_id TEXT NOT NULL,
                local_ref TEXT NOT NULL,
                student_email TEXT NOT NULL DEFAULT '',
                student_phone TEXT NOT NULL DEFAULT '',
                primary_guardian_email TEXT NOT NULL DEFAULT '',
                primary_guardian_phone TEXT NOT NULL DEFAULT '',
                backup_guardian_email TEXT NOT NULL DEFAULT '',
                backup_guardian_phone TEXT NOT NULL DEFAULT '',
                last_recorded_by TEXT NOT NULL REFERENCES verifyd_identities(id),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                PRIMARY KEY(school_id,local_ref),
                FOREIGN KEY(school_id,local_ref) REFERENCES
                    verifyd_safety_managed_students(school_id,local_ref)
            )""")
            cur.execute("""CREATE INDEX IF NOT EXISTS verifyd_safety_managed_student_school
                ON verifyd_safety_managed_students(school_id,last_name,first_name)""")
            cur.execute("""CREATE TABLE IF NOT EXISTS verifyd_safety_managed_enrollment_evidence (
                school_id TEXT NOT NULL,
                local_ref TEXT NOT NULL,
                method TEXT NOT NULL CHECK(method IN ('signed_paper','school_authority')),
                evidence_ref TEXT NOT NULL,
                guardian_name TEXT,
                signed_date DATE,
                authority_reference TEXT,
                declaration TEXT NOT NULL,
                verified_by TEXT NOT NULL REFERENCES verifyd_identities(id),
                verified_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                PRIMARY KEY(school_id,local_ref),
                FOREIGN KEY(school_id,local_ref) REFERENCES
                  verifyd_safety_managed_students(school_id,local_ref)
            )""")
            cur.execute("""ALTER TABLE verifyd_safety_managed_enrollment_evidence
                ADD COLUMN IF NOT EXISTS review_date DATE""")
            # Retain paper and documented authority, add audited electronic OTP.
            cur.execute("""ALTER TABLE verifyd_safety_managed_enrollment_evidence
                DROP CONSTRAINT IF EXISTS verifyd_safety_managed_enrollment_evidence_method_check""")
            cur.execute("""ALTER TABLE verifyd_safety_managed_enrollment_evidence
                ADD CONSTRAINT verifyd_safety_managed_enrollment_evidence_method_check
                CHECK(method IN ('signed_paper','school_authority','electronic'))""")
            cur.execute("""CREATE TABLE IF NOT EXISTS verifyd_safety_managed_roster_previews (
                preview_id TEXT PRIMARY KEY,
                school_id TEXT NOT NULL REFERENCES verifyd_safety_training_schools(school_id),
                admin_id TEXT NOT NULL REFERENCES verifyd_identities(id),
                school_code TEXT NOT NULL,
                handle_style TEXT NOT NULL,
                rows_json JSONB NOT NULL,
                expires_at TIMESTAMPTZ NOT NULL DEFAULT (NOW() + INTERVAL '30 minutes'),
                consumed_at TIMESTAMPTZ
            )""")
        _ready = True


class SchoolHandleSettings(BaseModel):
    school_code: str = Field(min_length=2, max_length=10)
    handle_style: Literal["name", "code"]
    confirmed_school_naming_convention: Literal[True]


class SchoolStudentContacts(BaseModel):
    student_email:str = Field(default='',max_length=254)
    student_phone:str = Field(default='',max_length=32)
    primary_guardian_email:str = Field(default='',max_length=254)
    primary_guardian_phone:str = Field(default='',max_length=32)
    backup_guardian_email:str = Field(default='',max_length=254)
    backup_guardian_phone:str = Field(default='',max_length=32)

    @field_validator('student_email','primary_guardian_email','backup_guardian_email')
    @classmethod
    def validate_contact_email(cls,value):
        email=value.strip().lower()
        if email and not is_valid_email(email):
            raise ValueError('invalid_roster_contact_email')
        return email

    @field_validator('student_phone','primary_guardian_phone','backup_guardian_phone')
    @classmethod
    def validate_contact_phone(cls,value):
        phone=value.strip()
        if phone:
            if not re.fullmatch(r'\+?[0-9 ()-]{10,32}',phone):
                raise ValueError('invalid_roster_contact_phone')
            digits=re.sub(r'\D','',phone)
            if len(digits)<10 or len(digits)>15:
                raise ValueError('invalid_roster_contact_phone_length')
        return phone


class ManagedStudentRow(SchoolStudentContacts):
    local_ref: str = Field(min_length=3, max_length=64)
    first_name: str = Field(min_length=1, max_length=60)
    last_name: str = Field(min_length=1, max_length=60)


class ManagedRosterPreview(BaseModel):
    students: list[ManagedStudentRow] = Field(min_length=1, max_length=250)
    reserved_handles: list[str] = Field(default_factory=list, max_length=5000)


class ManagedRosterCommit(BaseModel):
    preview_id: str = Field(min_length=8, max_length=100)
    confirm_pending_only: Literal[True]


class SchoolAuthorizationItem(BaseModel):
    local_ref: str = Field(min_length=3, max_length=64)
    method: Literal["signed_paper", "school_authority", "electronic"]
    evidence_ref: str = Field(min_length=8, max_length=180)
    guardian_name: Optional[str] = Field(default=None, max_length=100)
    signed_date: Optional[date] = None
    authority_reference: Optional[str] = Field(default=None, max_length=180)
    individually_verified: Literal[True]

    @field_validator("signed_date", mode="before")
    @classmethod
    def normalize_spreadsheet_date(cls, value):
        # Older school exports use US dates; normalize to a real date rather
        # than allowing a string to pass the frontend eligibility check and
        # fail later with an opaque FastAPI 422.
        if isinstance(value, str):
            trimmed = value.strip()
            if not trimmed:
                return None
            if "/" in trimmed:
                try:
                    return datetime.strptime(trimmed, "%m/%d/%Y").date()
                except ValueError as exc:
                    raise ValueError("Invalid signature date; use YYYY-MM-DD or MM/DD/YYYY") from exc
        return value


class SchoolAuthorizationBatch(BaseModel):
    approvals: list[SchoolAuthorizationItem] = Field(min_length=1, max_length=250)
    principal_confirms_evidence_on_file: Literal[True]
    school_use_only: Literal[True]
    # One declared administrator review date for the entire approval batch.
    # This is NOT the paper signature date of any guardian.
    review_date: date = Field(default_factory=date.today)


class SchoolTrackingReferenceUpdate(BaseModel):
    tracking_reference: str = Field(min_length=6, max_length=80,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{5,79}$")


def _check_evidence(item):
    if not _REF.fullmatch(item.local_ref):
        raise HTTPException(422, detail={"error": "invalid_local_reference"})
    evidence = item.evidence_ref.strip()
    if len(evidence) < 8:
        raise HTTPException(422, detail={"error": "individual_evidence_reference_required"})
    if item.method == "signed_paper":
        if not item.guardian_name or len(item.guardian_name.strip()) < 3 or not item.signed_date:
            raise HTTPException(422, detail={"error": "paper_requires_guardian_name_and_signed_date"})
        if item.signed_date > date.today():
            raise HTTPException(422, detail={"error": "paper_signature_date_in_future"})
    elif item.method == "electronic":
        if not evidence.startswith("EACK-") or item.guardian_name or item.signed_date or item.authority_reference:
            raise HTTPException(422, detail={"error":"electronic_acknowledgment_must_use_system_evidence"})
    elif not item.authority_reference or len(item.authority_reference.strip()) < 8:
        raise HTTPException(422, detail={"error": "school_authority_reference_required"})



def _clean_name(value):
    cleaned = " ".join(str(value or "").strip().split())
    if not cleaned or len(cleaned) > 60 or any(ord(ch) < 32 for ch in cleaned):
        raise HTTPException(422, detail={"error": "invalid_student_name"})
    return cleaned


def _ascii(value):
    ascii_name = unicodedata.normalize("NFKD", value)
    return "".join(ch for ch in ascii_name if ch.isascii() and ch.isalnum())


def _free_handle(cur, name_lower, used):
    if name_lower in used:
        return False
    cur.execute("SELECT 1 FROM verifyd_identities WHERE handle_lower=%s", (name_lower,))
    if cur.fetchone():
        return False
    cur.execute("SELECT 1 FROM verifyd_safety_managed_students WHERE handle_lower=%s", (name_lower,))
    return not bool(cur.fetchone())


def _new_handle(cur, school_code, style, first_name, last_name, used):
    if style == "name":
        letters = _ascii(first_name) + _ascii(last_name)
        if not letters:
            raise HTTPException(422, detail={"error": "student_name_cannot_form_handle"})
        base = school_code + "_" + letters
        for suffix in [""] + [str(n) for n in range(2, 1002)]:
            candidate = base[:24-len(suffix)] + suffix
            try: _validate_handle(candidate)
            except HTTPException: continue
            if _free_handle(cur, candidate.lower(), used):
                used.add(candidate.lower())
                return candidate
    else:
        for _ in range(200):
            candidate = school_code + "_S" + str(secrets.randbelow(900000) + 100000)
            if len(candidate) <= 24 and _free_handle(cur, candidate.lower(), used):
                used.add(candidate.lower())
                return candidate
    raise HTTPException(409, detail={"error": "handle_namespace_exhausted"})


def _config_row(cur, school_id):
    cur.execute("""SELECT school_code,handle_style FROM verifyd_safety_managed_roster_settings
      WHERE school_id=%s""", (school_id,))
    row = cur.fetchone()
    if not row:
        raise HTTPException(409, detail={"error": "configure_school_handle_style_first"})
    return dict(row)


@router.get("/schools/{school_id}/managed-roster/settings")
def get_managed_roster_settings(school_id: str, authorization: Optional[str] = Header(default=None)):
    _require_enrollment_enabled(school_id)
    actor = _actor(authorization)
    _schema()
    with get_db() as conn:
        cur = conn.cursor()
        _school_admin(cur, school_id, str(actor["id"]))
        cur.execute("""SELECT school_code,handle_style FROM verifyd_safety_managed_roster_settings
            WHERE school_id=%s""", (school_id,))
        config = cur.fetchone()
        cur.execute("SELECT COUNT(*) AS n FROM verifyd_safety_managed_students WHERE school_id=%s", (school_id,))
        count = int(cur.fetchone()["n"])
    return {"ok": True, "school_managed_enrollment": True, "configured": bool(config),
            "config": dict(config) if config else None, "staged_count": count,
            "identities_created": 0, "guardian_access_granted": False}


@router.put("/schools/{school_id}/managed-roster/settings")
def save_managed_roster_settings(school_id: str, payload: SchoolHandleSettings,
                                 authorization: Optional[str] = Header(default=None)):
    _require_enrollment_enabled(school_id)
    actor = _actor(authorization)
    code = payload.school_code.strip().upper()
    if not _CODE.fullmatch(code):
        raise HTTPException(422, detail={"error": "school_code_must_be_2_to_10_uppercase_letters_or_digits"})
    _schema()
    with get_db() as conn:
        cur = conn.cursor()
        _school_admin(cur, school_id, str(actor["id"]))
        cur.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", ("safety_school_code_settings",))
        cur.execute("SELECT COUNT(*) AS n FROM verifyd_safety_managed_students WHERE school_id=%s", (school_id,))
        if int(cur.fetchone()["n"]) > 0:
            existing = _config_row(cur, school_id)
            if existing != {"school_code": code, "handle_style": payload.handle_style}:
                raise HTTPException(409, detail={"error": "handle_style_locked_after_roster_import"})
        cur.execute("SELECT school_id FROM verifyd_safety_managed_roster_settings WHERE school_code=%s", (code,))
        owner = cur.fetchone()
        if owner and str(owner["school_id"]) != school_id:
            raise HTTPException(409, detail={"error": "school_abbreviation_already_reserved"})
        cur.execute("""INSERT INTO verifyd_safety_managed_roster_settings
           (school_id,school_code,handle_style,configured_by)
           VALUES (%s,%s,%s,%s)
           ON CONFLICT(school_id) DO UPDATE SET school_code=EXCLUDED.school_code,
           handle_style=EXCLUDED.handle_style,configured_by=EXCLUDED.configured_by,updated_at=NOW()""",
           (school_id,code,payload.handle_style,str(actor["id"])))
        _audit(cur, school_id, str(actor["id"]), "managed_roster_school_style_configured", code+":"+payload.handle_style)
    return {"ok": True, "school_managed_enrollment": True, "config": {"school_code": code,
            "handle_style": payload.handle_style}, "student_accounts_activated": 0}


@router.post("/schools/{school_id}/managed-roster/preview")
def preview_managed_roster(school_id: str, payload: ManagedRosterPreview,
                           authorization: Optional[str] = Header(default=None)):
    _require_enrollment_enabled(school_id)
    actor = _actor(authorization)
    _schema()
    with get_db() as conn:
        cur = conn.cursor()
        _school_admin(cur, school_id, str(actor["id"]))
        # Discard outdated uncommitted roster previews containing student PII.
        cur.execute("""DELETE FROM verifyd_safety_managed_roster_previews
            WHERE school_id=%s AND expires_at<NOW()-INTERVAL '1 day'""",(school_id,))
        config = _config_row(cur, school_id)
        # When a large CSV is previewed in multiple chunks, exclude handles
        # proposed by preceding chunks even before those previews are committed.
        used = set()
        for candidate in payload.reserved_handles:
            _, lower = _validate_handle(candidate)
            if not lower.startswith(config["school_code"].lower() + "_"):
                raise HTTPException(422, detail={"error":"reserved_handle_outside_school_prefix"})
            used.add(lower)
        rows, refs = [], set()
        # Soft warnings for identical names, never block legitimate twins/
        # namesakes; hard uniqueness remains school reference + global handle.
        cur.execute("""SELECT lower(first_name) AS first_lower,
            lower(last_name) AS last_lower,local_ref
            FROM verifyd_safety_managed_students WHERE school_id=%s""",(school_id,))
        roster_names = {}
        for existing in cur.fetchall():
            key=(existing["first_lower"].strip(),existing["last_lower"].strip())
            roster_names.setdefault(key,set()).add(str(existing["local_ref"]).upper())
        incoming_names = {}
        for student in payload.students:
            ref = student.local_ref.strip().upper()
            if not _REF.fullmatch(ref):
                raise HTTPException(422, detail={"error": "invalid_school_local_reference"})
            if ref.lower() in refs:
                raise HTTPException(422, detail={"error": "duplicate_local_reference_in_upload"})
            refs.add(ref.lower())
            first, last = _clean_name(student.first_name), _clean_name(student.last_name)
            cur.execute("""SELECT first_name,last_name,proposed_handle FROM verifyd_safety_managed_students
                  WHERE school_id=%s AND lower(local_ref)=%s""", (school_id, ref.lower()))
            old = cur.fetchone()
            if old:
                if old["first_name"] != first or old["last_name"] != last:
                    raise HTTPException(409, detail={"error": "local_reference_already_assigned_to_different_name"})
                handle, action = old["proposed_handle"], "already_staged"
            else:
                handle = _new_handle(cur, config["school_code"], config["handle_style"], first, last, used)
                action = "stage_pending"
            duplicate_refs=sorted((roster_names.get((first.lower(),last.lower()),set())
              | incoming_names.get((first.lower(),last.lower()),set()))-{ref})
            warning=("Same student name as school record(s) " + ", ".join(duplicate_refs[:3])
                     + ". Check reference and date of birth in the school system; "
                     + "distinct students may keep distinct assigned handles."
                     if duplicate_refs else "")
            contact_fields=student.model_dump(include=set(SchoolStudentContacts.model_fields))
            rows.append({"local_ref": ref, "first_name": first, "last_name": last,
                         "proposed_handle": handle, "action": action,
                         "possible_same_name_warning":warning,**contact_fields})
            incoming_names.setdefault((first.lower(),last.lower()),set()).add(ref)
        preview_id = "srprev_" + secrets.token_hex(18)
        cur.execute("""INSERT INTO verifyd_safety_managed_roster_previews
           (preview_id,school_id,admin_id,school_code,handle_style,rows_json)
           VALUES (%s,%s,%s,%s,%s,%s::jsonb)""",
           (preview_id,school_id,str(actor["id"]),config["school_code"],
            config["handle_style"],json.dumps(rows)))
    return {"ok": True, "school_managed_enrollment": True, "preview_id": preview_id,
            "expires_in_minutes": 30, "students": rows,
            "accounts_created": 0, "guardian_access_granted": False}


@router.post("/schools/{school_id}/managed-roster/commit")
def commit_managed_roster(school_id: str, payload: ManagedRosterCommit,
                          authorization: Optional[str] = Header(default=None)):
    _require_enrollment_enabled(school_id)
    actor = _actor(authorization)
    _schema()
    with get_db() as conn:
        cur = conn.cursor()
        _school_admin(cur, school_id, str(actor["id"]))
        # Serializes TEST commits and the school style namespace.
        cur.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", ("safety_managed_roster_commit",))
        cur.execute("""SELECT admin_id,school_code,handle_style,rows_json,consumed_at,
              expires_at>NOW() AS valid FROM verifyd_safety_managed_roster_previews
              WHERE preview_id=%s AND school_id=%s FOR UPDATE""",
              (payload.preview_id,school_id))
        preview = cur.fetchone()
        if not preview or str(preview["admin_id"]) != str(actor["id"]):
            raise HTTPException(404, detail={"error": "roster_preview_not_found_for_administrator"})
        if preview["consumed_at"]:
            return {"ok": True, "school_managed_enrollment": True, "already_committed": True,
                    "accounts_created": 0, "guardian_access_granted": False}
        if not preview["valid"]:
            raise HTTPException(409, detail={"error": "roster_preview_expired_start_over"})
        current = _config_row(cur,school_id)
        if current != {"school_code": preview["school_code"], "handle_style": preview["handle_style"]}:
            raise HTTPException(409, detail={"error": "school_handle_settings_changed_repreview"})
        rows = preview["rows_json"]
        if isinstance(rows,str):
            rows = json.loads(rows)
        added, skipped = 0, 0
        for row in rows:
            cur.execute("""SELECT first_name,last_name,proposed_handle FROM verifyd_safety_managed_students
                WHERE school_id=%s AND local_ref=%s""", (school_id,row["local_ref"]))
            old = cur.fetchone()
            if old:
                if old["first_name"] != row["first_name"] or old["last_name"] != row["last_name"] or old["proposed_handle"] != row["proposed_handle"]:
                    raise HTTPException(409, detail={"error": "roster_changed_since_preview_repreview"})
                skipped += 1
                continue
            cur.execute("""SELECT 1 FROM verifyd_identities WHERE handle_lower=%s
                    UNION SELECT 1 FROM verifyd_safety_managed_students
                    WHERE handle_lower=%s LIMIT 1""",
                    (row["proposed_handle"].lower(), row["proposed_handle"].lower()))
            if cur.fetchone():
                raise HTTPException(409, detail={"error": "handle_unavailable_repreview"})
            cur.execute("""INSERT INTO verifyd_safety_managed_students
                (school_id,local_ref,first_name,last_name,proposed_handle,handle_lower,
                 imported_by,tracking_reference)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s)""",
                (school_id,row["local_ref"],row["first_name"],row["last_name"],
                 row["proposed_handle"],row["proposed_handle"].lower(),str(actor["id"]),
                 "ENROLL-"+row["local_ref"].upper()))
            contact_values=[str(row.get(k) or '') for k in SchoolStudentContacts.model_fields]
            if any(contact_values):
                cur.execute("""INSERT INTO verifyd_safety_managed_student_contacts
                    (school_id,local_ref,student_email,student_phone,
                     primary_guardian_email,primary_guardian_phone,
                     backup_guardian_email,backup_guardian_phone,last_recorded_by)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                    (school_id,row["local_ref"],*contact_values,str(actor["id"])))
            added += 1
        cur.execute("UPDATE verifyd_safety_managed_roster_previews SET consumed_at=NOW() WHERE preview_id=%s", (payload.preview_id,))
        _audit(cur,school_id,str(actor["id"]),"managed_roster_pending_handles_staged",str(added))
    return {"ok": True, "school_managed_enrollment": True, "staged": added, "already_staged": skipped,
            "accounts_created": 0, "guardian_access_granted": False, "reports_enabled": False}


@router.get("/schools/{school_id}/managed-roster/students")
def list_managed_roster(school_id: str, authorization: Optional[str] = Header(default=None)):
    _require_enrollment_enabled(school_id)
    actor = _actor(authorization)
    _schema()
    with get_db() as conn:
        cur = conn.cursor()
        _school_admin(cur, school_id, str(actor["id"]))
        cur.execute("""SELECT s.local_ref,s.first_name,s.last_name,s.proposed_handle,
            s.student_identity_id,s.enrollment_status AS status,s.created_at,
            s.tracking_reference,s.approved_at,
            e.method AS acknowledgment_method,e.evidence_ref,e.signed_date,
            e.verified_at,e.review_date,e.authority_reference,
            c.student_email,c.student_phone,c.primary_guardian_email,
            c.primary_guardian_phone,c.backup_guardian_email,c.backup_guardian_phone
            FROM verifyd_safety_managed_students s
            LEFT JOIN verifyd_safety_managed_student_contacts c
                ON c.school_id=s.school_id AND c.local_ref=s.local_ref
            LEFT JOIN verifyd_safety_managed_enrollment_evidence e
                ON e.school_id=s.school_id AND e.local_ref=s.local_ref
            WHERE s.school_id=%s
            ORDER BY lower(s.last_name),lower(s.first_name),s.local_ref LIMIT 3000""",
            (school_id,))
        records = [dict(row) for row in cur.fetchall()]
    return {"ok": True, "school_managed_enrollment": True, "students": records,
            "automatic_parent_approval": False, "emergency_dispatch_enabled": False}


@router.post("/schools/{school_id}/managed-roster/approve-batch")
def approve_managed_school_enrollment(
    school_id: str, payload: SchoolAuthorizationBatch,
    authorization: Optional[str] = Header(default=None),
):
    """Approve a school roster in bulk, preserving independently reviewed evidence.

    Paper forms and legally established school authorization are alternative
    *enrollment* attestations. Neither implies a guardian signed digitally,
    nor do they authorize unrestricted messaging or emergency service dispatch.
    """
    _require_enrollment_enabled(school_id)
    actor = _actor(authorization)
    if payload.review_date > date.today():
        raise HTTPException(422,detail={"error":"approval_review_date_cannot_be_in_future"})
    refs = [item.local_ref.strip().lower() for item in payload.approvals]
    if len(refs) != len(set(refs)):
        raise HTTPException(422, detail={"error": "duplicate_student_in_approval_batch"})
    for item in payload.approvals:
        _check_evidence(item)
    if any(item.method == "electronic" for item in payload.approvals):
        from trust_voice_school_electronic_ack import _require_electronic_enabled
        _require_electronic_enabled(school_id)
    # Each claimed paper consent must have its own auditable reference; a
    # district policy can be shared, but a single paper signature cannot.
    paper_refs = [item.evidence_ref.strip().lower() for item in payload.approvals
                  if item.method == "signed_paper"]
    if len(paper_refs) != len(set(paper_refs)):
        raise HTTPException(422, detail={"error": "paper_evidence_reference_must_identify_each_student"})
    _schema()
    from trust_voice_school_safety_training import _schema as _school_schema
    from trust_voice_school_electronic_ack import _schema as _electronic_schema
    from trust_voice_school_safety_consents import POLICY_VERSION
    _school_schema()
    _electronic_schema()
    with get_db() as conn:
        cur = conn.cursor()
        _school_admin(cur, school_id, str(actor["id"]))
        cur.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", ("safety_managed_roster_commit",))
        cur.execute("""SELECT local_ref,first_name,last_name,proposed_handle,
            student_identity_id,enrollment_status
            FROM verifyd_safety_managed_students
            WHERE school_id=%s AND lower(local_ref)=ANY(%s)
            ORDER BY local_ref FOR UPDATE""", (school_id, refs))
        pending = {row["local_ref"].lower(): dict(row) for row in cur.fetchall()}
        if len(pending) != len(refs):
            raise HTTPException(404, detail={"error": "one_or_more_students_not_staged"})
        electronic_claims = {}
        for item in payload.approvals:
            row = pending[item.local_ref.strip().lower()]
            if row["student_identity_id"] or row["enrollment_status"] != "staged":
                raise HTTPException(409, detail={"error": "student_already_approved"})
            cur.execute("SELECT 1 FROM verifyd_identities WHERE handle_lower=%s",
                        (row["proposed_handle"].lower(),))
            if cur.fetchone():
                raise HTTPException(409, detail={"error": "handle_taken_repreview_and_correct"})
            if item.method == "electronic":
                cur.execute("""SELECT a.*, c.guardian_email AS verified_email,
                    c.guardian_name AS verified_name
                    FROM verifyd_safety_electronic_guardian_ack a
                    JOIN verifyd_safety_family_contact_drafts c
                      ON c.school_id=a.school_id AND c.local_ref=a.local_ref
                     AND c.slot=a.slot
                    WHERE a.school_id=%s AND a.local_ref=%s AND a.slot=1
                      AND a.evidence_ref=%s AND a.status='pending_school_review'
                    FOR UPDATE OF a""",
                    (school_id,row["local_ref"],item.evidence_ref.strip()))
                ack=cur.fetchone()
                if (not ack or ack["policy_version"] != POLICY_VERSION
                    or ack["guardian_email"] != ack["verified_email"]
                    or ack["guardian_name"] != ack["verified_name"]):
                    raise HTTPException(409,detail={"error":"verified_electronic_guardian_acknowledgment_required"})
                cur.execute("""SELECT 1 FROM verifyd_identities
                    WHERE id=%s AND email_lower=%s""",
                    (str(actor["id"]),ack["guardian_email"]))
                if cur.fetchone():
                    raise HTTPException(403,detail={"error":"administrator_cannot_approve_own_guardian_acknowledgment"})
                electronic_claims[row["local_ref"]] = dict(ack)
        created = []
        guardian_accounts_created = 0
        for item in payload.approvals:
            row = pending[item.local_ref.strip().lower()]
            student_id = "vfyd_usr_" + uuid.uuid4().hex
            # Existing supervised-child identity architecture uses a random,
            # non-routable internal email sentinel; it is not a guardian email.
            internal_email = "school-supervised-" + uuid.uuid4().hex + "@child.verifyd.invalid"
            cur.execute("""INSERT INTO verifyd_identities (
                id,email_lower,handle,handle_lower,display_name,display_emoji,
                email_verified,identity_verified,organization_verified,
                verification_level,can_receive_calls,call_privacy,status,created_at,updated_at)
                VALUES (%s,%s,%s,%s,%s,'',FALSE,FALSE,FALSE,
                'school_supervised',FALSE,'nobody','supervised_child',
                NOW()::text,NOW()::text)""",
                (student_id,internal_email,row["proposed_handle"],
                 row["proposed_handle"].lower(),
                 (row["first_name"]+" "+row["last_name"])[:80]))
            cur.execute("""INSERT INTO verifyd_safety_training_members
                 (school_id,identity_id,role,verified_by)
                 VALUES (%s,%s,'student',%s)""",
                 (school_id,student_id,str(actor["id"])))
            statement = ("Individual guardian-signed paper record verified by authorized school staff"
                if item.method == "signed_paper" else
                "Guardian electronically acknowledged after mailbox OTP; school admin independently verified the relationship"
                if item.method == "electronic" else
                "School/district authority documented for school-only education services; not guardian consent")
            cur.execute("""INSERT INTO verifyd_safety_managed_enrollment_evidence
                (school_id,local_ref,method,evidence_ref,guardian_name,signed_date,
                 authority_reference,declaration,verified_by,review_date)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                (school_id,row["local_ref"],item.method,item.evidence_ref.strip(),
                 item.guardian_name.strip() if item.guardian_name else None,
                 item.signed_date,item.authority_reference.strip()
                 if item.authority_reference else None,statement,str(actor["id"]),
                 payload.review_date))
            cur.execute("""UPDATE verifyd_safety_managed_students
                SET student_identity_id=%s,enrollment_status='school_approved',
                    approved_at=NOW() WHERE school_id=%s AND local_ref=%s""",
                (student_id,school_id,row["local_ref"]))
            if item.method == "electronic":
                # Link and sign off ONLY after independent school approval, never
                # when a guardian simply enters an email or presses Acknowledge.
                from trust_voice_school_family_invitations import _new_guardian_identity
                ack=electronic_claims[row["local_ref"]]
                guardian_id,guardian_created=_new_guardian_identity(cur,ack["guardian_email"],ack["guardian_name"],school_id)
                guardian_accounts_created += int(guardian_created)
                cur.execute("""SELECT role FROM verifyd_safety_training_members
                    WHERE school_id=%s AND identity_id=%s""",(school_id,guardian_id))
                existing_member=cur.fetchone()
                if existing_member and existing_member["role"]!="guardian":
                    raise HTTPException(403,detail={"error":"school_staff_cannot_claim_guardian_role"})
                cur.execute("""INSERT INTO verifyd_safety_training_members
                    (school_id,identity_id,role,verified_by)
                    VALUES(%s,%s,'guardian',%s) ON CONFLICT DO NOTHING""",
                    (school_id,guardian_id,str(actor["id"])))
                cur.execute("""INSERT INTO verifyd_safety_training_guardians
                    (school_id,child_id,guardian_id,verified_by)
                    VALUES(%s,%s,%s,%s) ON CONFLICT DO NOTHING""",
                    (school_id,student_id,guardian_id,str(actor["id"])))
                cur.execute("""INSERT INTO verifyd_safety_training_signoffs
                    (signoff_id,school_id,actor_id,kind,subject_id,policy_version,accepted_at)
                    VALUES(%s,%s,%s,'guardian',%s,%s,%s)
                    ON CONFLICT(school_id,actor_id,kind,subject_id)
                    DO UPDATE SET policy_version=EXCLUDED.policy_version,
                        accepted_at=EXCLUDED.accepted_at,revoked_at=NULL""",
                    ("sssig_"+uuid.uuid4().hex,school_id,guardian_id,student_id,
                     POLICY_VERSION,ack["acknowledged_at"]))
                cur.execute("""UPDATE verifyd_safety_electronic_guardian_ack
                    SET status='approved',approved_at=NOW(),approved_by=%s,
                        guardian_identity_id=%s WHERE school_id=%s AND local_ref=%s AND slot=1""",
                    (str(actor["id"]),guardian_id,school_id,row["local_ref"]))
                from trust_voice_school_auto_routing import sync_approved_guardian_route
                sync_approved_guardian_route(cur,school_id,student_id,str(actor["id"]))
                _audit(cur,school_id,str(actor["id"]),"electronic_guardian_school_approved",row["local_ref"])
            _audit(cur,school_id,str(actor["id"]),"managed_student_enrollment_approved",
                   row["local_ref"]+":"+item.method)
            created.append({"local_ref":row["local_ref"],
                            "proposed_handle":row["proposed_handle"],
                            "status":"school_approved"})
        _audit(cur,school_id,str(actor["id"]),
               "managed_school_roster_group_approved",str(len(created)))
    return {"ok":True,"school_managed_enrollment":True,"approved":len(created),
            "students":created,"ordinary_voice_login_enabled":False,
            "guardian_accounts_created":guardian_accounts_created,
            "electronically_verified_guardian_links_approved":len(electronic_claims),
            "guardian_private_message_access_granted":False,
            "student_reporting_unlocked":False,
            "emergency_dispatch_enabled":False}


@router.delete("/schools/{school_id}/managed-roster/students/{local_ref}")
def discard_unapproved_managed_student(school_id: str, local_ref: str,
                                       authorization: Optional[str] = Header(default=None)):
    """Allow correcting a mistaken *staged* upload, never erase approved identity.

    Approved students must be deactivated using ordinary school-roster
    administration, which retains audit/history and prevents silent deletion.
    """
    _require_enrollment_enabled(school_id)
    actor = _actor(authorization)
    ref = local_ref.strip().upper()
    if not _REF.fullmatch(ref):
        raise HTTPException(422, detail={"error": "invalid_school_local_reference"})
    _schema()
    with get_db() as conn:
        cur = conn.cursor()
        _school_admin(cur, school_id, str(actor["id"]))
        cur.execute("""SELECT student_identity_id,enrollment_status
            FROM verifyd_safety_managed_students
            WHERE school_id=%s AND local_ref=%s FOR UPDATE""", (school_id, ref))
        record = cur.fetchone()
        if not record:
            raise HTTPException(404, detail={"error": "staged_student_not_found"})
        if record["student_identity_id"] or record["enrollment_status"] != "staged":
            raise HTTPException(409, detail={"error": "approved_student_use_authorized_roster_deactivation"})
        cur.execute("""DELETE FROM verifyd_safety_managed_students
            WHERE school_id=%s AND local_ref=%s""", (school_id, ref))
        _audit(cur, school_id, str(actor["id"]), "managed_staged_student_discarded", ref)
    return {"ok": True, "staged_record_removed": True,
            "global_verifyd_account_deleted": False,
            "school_managed_enrollment": True}


@router.patch("/schools/{school_id}/managed-roster/students/{local_ref}/tracking-reference")
def update_managed_student_tracking_reference(school_id: str, local_ref: str,
                                              payload: SchoolTrackingReferenceUpdate,
                                              authorization: Optional[str] = Header(default=None)):
    """Change a school record tracking code without altering authorization evidence."""
    _require_enrollment_enabled(school_id)
    actor=_actor(authorization)
    ref=local_ref.strip().upper()
    if not _REF.fullmatch(ref):
        raise HTTPException(422,detail={"error":"invalid_school_local_reference"})
    tracking=payload.tracking_reference.strip().upper()
    if not tracking:
        raise HTTPException(422,detail={"error":"tracking_reference_required"})
    _schema()
    with get_db() as conn:
        cur=conn.cursor()
        _school_admin(cur,school_id,str(actor["id"]))
        cur.execute("SELECT pg_advisory_xact_lock(hashtext(%s))",
                    ("school_tracking_reference:"+school_id,))
        cur.execute("""SELECT 1 FROM verifyd_safety_managed_students
            WHERE school_id=%s AND lower(tracking_reference)=lower(%s) AND local_ref<>%s""",
            (school_id,tracking,ref))
        if cur.fetchone():
            raise HTTPException(409,detail={"error":"tracking_reference_already_in_use"})
        cur.execute("""UPDATE verifyd_safety_managed_students
            SET tracking_reference=%s WHERE school_id=%s AND local_ref=%s
            RETURNING local_ref,tracking_reference""",(tracking,school_id,ref))
        row=cur.fetchone()
        if not row:
            raise HTTPException(404,detail={"error":"school_student_not_found"})
        _audit(cur,school_id,str(actor["id"]),"school_tracking_reference_updated",ref)
    return {"ok":True,"local_ref":row["local_ref"],
            "tracking_reference":row["tracking_reference"],
            "guardian_consent_verified":False,"enrollment_approval_changed":False}


@router.patch("/schools/{school_id}/managed-roster/students/{local_ref}/contact-info")
def update_school_student_contact_fields(school_id:str,local_ref:str,
    payload:SchoolStudentContacts,authorization:Optional[str]=Header(default=None)):
    """School-admin-only PII storage. Contact import does NOT verify guardianship,
    activate communications or authorize any phone/SMS messages.
    """
    _require_enrollment_enabled(school_id)
    actor=_actor(authorization);_schema()
    ref=local_ref.strip().upper()
    if not _REF.fullmatch(ref):
        raise HTTPException(422,detail={"error":"invalid_school_local_reference"})
    with get_db() as conn:
        cur=conn.cursor()
        _school_admin(cur,school_id,str(actor["id"]))
        cur.execute("""SELECT 1 FROM verifyd_safety_managed_students
            WHERE school_id=%s AND local_ref=%s""",(school_id,ref))
        if not cur.fetchone():
            raise HTTPException(404,detail={"error":"school_student_not_found"})
        values=[getattr(payload,k) for k in SchoolStudentContacts.model_fields]
        cur.execute("""INSERT INTO verifyd_safety_managed_student_contacts
           (school_id,local_ref,student_email,student_phone,
            primary_guardian_email,primary_guardian_phone,
            backup_guardian_email,backup_guardian_phone,last_recorded_by)
           VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s)
           ON CONFLICT(school_id,local_ref) DO UPDATE SET
            student_email=EXCLUDED.student_email,
            student_phone=EXCLUDED.student_phone,
            primary_guardian_email=EXCLUDED.primary_guardian_email,
            primary_guardian_phone=EXCLUDED.primary_guardian_phone,
            backup_guardian_email=EXCLUDED.backup_guardian_email,
            backup_guardian_phone=EXCLUDED.backup_guardian_phone,
            last_recorded_by=EXCLUDED.last_recorded_by,updated_at=NOW()""",
           (school_id,ref,*values,str(actor["id"])))
        _audit(cur,school_id,str(actor["id"]),"school_contact_details_recorded",ref)
    return {"ok":True,"local_ref":ref,"contact_details_saved":True,
       "verified_guardian_relationship":False,
       "email_or_sms_sent":False}
