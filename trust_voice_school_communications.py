"""Verified-family school communications, non-emergency training pilot.

School-scoped audiences, scheduled publishing, private small attachments,
per-recipient read receipts, and explicit notification preferences.
No push, email, SMS, or emergency delivery is initiated.
"""
from __future__ import annotations
import base64
import binascii
import re
import threading
import uuid
from datetime import datetime, timezone, timedelta
from typing import Literal, Optional
from fastapi import APIRouter, Header, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field
from database import get_db
from trust_voice_school_safety_training import _actor, _audit, _id_for_handle
from trust_voice_school_safety_registration import _school_admin
from trust_voice_school_family_portal import _family_schema
from trust_voice_school_safety_consents import POLICY_VERSION

router=APIRouter(prefix="/trust-voice/school-safety",tags=["School Communications"])
_lock=threading.Lock()
_ready=False

TEMPLATES={
 "school_event":{"title":"Upcoming school event","body":"Please check the school calendar for event details."},
 "curriculum":{"title":"Curriculum update","body":"A new curriculum update is available for families."},
 "parent_meeting":{"title":"Parent meeting","body":"Please contact the school office for meeting details."},
 "pe_activity":{"title":"Physical education activities","body":"Please review this week's physical education activities."},
 "routine_reminder":{"title":"School reminder","body":"A reminder from your school administration."},
}
_BLOCKED=("911","emergency alert","active shooter","evacuate now","lockdown now")
_ALLOWED_MIME={"application/pdf","text/plain","image/png","image/jpeg"}
_MAX_ATTACHMENT=1024*1024

def _schema():
    global _ready
    _family_schema()
    if _ready: return
    with _lock:
        if _ready: return
        with get_db() as conn:
            cur=conn.cursor()
            cur.execute("""CREATE TABLE IF NOT EXISTS verifyd_safety_comms_student_groups(
                school_id TEXT NOT NULL, student_id TEXT NOT NULL,
                grade_label TEXT NOT NULL DEFAULT '', class_label TEXT NOT NULL DEFAULT '',
                PRIMARY KEY(school_id,student_id))""")
            cur.execute("""CREATE TABLE IF NOT EXISTS verifyd_safety_comms_posts(
                post_id TEXT PRIMARY KEY, school_id TEXT NOT NULL,
                author_id TEXT NOT NULL, title TEXT NOT NULL, body TEXT NOT NULL,
                audience_type TEXT NOT NULL CHECK(audience_type IN ('school','grade','class','families')),
                audience_value TEXT NOT NULL DEFAULT '',
                publish_at TIMESTAMPTZ NOT NULL, created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                archived_at TIMESTAMPTZ,
                CHECK(length(title) BETWEEN 3 AND 160),
                CHECK(length(body) BETWEEN 5 AND 3000))""")
            cur.execute("""CREATE INDEX IF NOT EXISTS verifyd_safety_comms_school_schedule
                ON verifyd_safety_comms_posts(school_id,publish_at DESC)""")
            cur.execute("""CREATE TABLE IF NOT EXISTS verifyd_safety_comms_recipients(
                post_id TEXT NOT NULL REFERENCES verifyd_safety_comms_posts(post_id),
                guardian_id TEXT NOT NULL, read_at TIMESTAMPTZ,
                PRIMARY KEY(post_id,guardian_id))""")
            cur.execute("""CREATE INDEX IF NOT EXISTS verifyd_safety_comms_guardian
                ON verifyd_safety_comms_recipients(guardian_id,post_id)""")
            cur.execute("""CREATE TABLE IF NOT EXISTS verifyd_safety_comms_attachments(
                attachment_id TEXT PRIMARY KEY,
                post_id TEXT NOT NULL REFERENCES verifyd_safety_comms_posts(post_id),
                filename TEXT NOT NULL,mime_type TEXT NOT NULL,file_bytes BYTEA NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW())""")
            cur.execute("""CREATE TABLE IF NOT EXISTS verifyd_safety_comms_preferences(
                guardian_id TEXT PRIMARY KEY,
                in_app_enabled BOOLEAN NOT NULL DEFAULT TRUE,
                push_requested BOOLEAN NOT NULL DEFAULT FALSE)""")
        _ready=True

def _eligible(cur,school_id):
    # Acknowledged, currently linked guardians only. Cross-school data cannot leak.
    cur.execute("""SELECT DISTINCT g.guardian_id, g.child_id
        FROM verifyd_safety_training_guardians g
        JOIN verifyd_safety_training_members m
          ON m.school_id=g.school_id AND m.identity_id=g.guardian_id AND m.role='guardian'
        JOIN verifyd_safety_training_members st
          ON st.school_id=g.school_id AND st.identity_id=g.child_id AND st.role='student'
        JOIN verifyd_safety_training_signoffs x
          ON x.school_id=g.school_id AND x.actor_id=g.guardian_id
         AND x.subject_id=g.child_id AND x.kind='guardian'
         AND x.policy_version=%s AND x.revoked_at IS NULL
        WHERE g.school_id=%s""",(POLICY_VERSION,school_id))
    return [dict(r) for r in cur.fetchall()]

class PostInput(BaseModel):
    title: str=Field(min_length=3,max_length=160)
    body: str=Field(min_length=5,max_length=3000)
    audience_type: Literal["school","grade","class","families"]="school"
    audience_value: str=Field(default="",max_length=80)
    guardian_handles: list[str]=Field(default_factory=list,max_length=50)
    publish_at: Optional[datetime]=None

class GroupInput(BaseModel):
    student_handle: str=Field(min_length=3,max_length=40)
    grade_label: str=Field(default="",max_length=30)
    class_label: str=Field(default="",max_length=50)

class AttachmentInput(BaseModel):
    filename: str=Field(min_length=1,max_length=120)
    mime_type: Literal["application/pdf","text/plain","image/png","image/jpeg"]
    base64_data: str=Field(min_length=4,max_length=1400000)

class PrefInput(BaseModel):
    in_app_enabled: bool=True
    push_requested: bool=False

@router.get("/communications/templates")
def templates(authorization:Optional[str]=Header(default=None)):
    _actor(authorization)
    return {"templates":TEMPLATES,"push_delivery_available":False}

@router.post("/schools/{school_id}/communications/student-group")
def set_student_group(school_id:str,payload:GroupInput,authorization:Optional[str]=Header(default=None)):
    actor=_actor(authorization)
    _schema()
    student_id=_id_for_handle(payload.student_handle)
    with get_db() as conn:
        cur=conn.cursor()
        _school_admin(cur,school_id,str(actor["id"]))
        cur.execute("""SELECT 1 FROM verifyd_safety_training_members
            WHERE school_id=%s AND identity_id=%s AND role='student'""",(school_id,student_id))
        if not cur.fetchone(): raise HTTPException(404,detail={"error":"student_not_on_school_roster"})
        cur.execute("""INSERT INTO verifyd_safety_comms_student_groups
          (school_id,student_id,grade_label,class_label) VALUES(%s,%s,%s,%s)
          ON CONFLICT(school_id,student_id) DO UPDATE
          SET grade_label=EXCLUDED.grade_label,class_label=EXCLUDED.class_label""",
          (school_id,student_id,payload.grade_label.strip(),payload.class_label.strip()))
        _audit(cur,school_id,str(actor["id"]),"comms_student_group_updated",student_id)
    return {"ok":True,"training_only":True}

@router.post("/schools/{school_id}/communications/posts")
def create_post(school_id:str,payload:PostInput,authorization:Optional[str]=Header(default=None)):
    actor=_actor(authorization)
    _schema()
    if any(s in (payload.title+" "+payload.body).lower() for s in _BLOCKED):
        raise HTTPException(422,detail={"error":"use_official_emergency_channels"})
    if payload.audience_type in ("grade","class") and not payload.audience_value.strip():
        raise HTTPException(422,detail={"error":"audience_group_required"})
    if payload.audience_type=="families" and not payload.guardian_handles:
        raise HTTPException(422,detail={"error":"select_guardians"})
    scheduled=payload.publish_at or datetime.now(timezone.utc)
    if scheduled.tzinfo is None: raise HTTPException(422,detail={"error":"timezone_required"})
    scheduled=scheduled.astimezone(timezone.utc)
    if scheduled>datetime.now(timezone.utc)+timedelta(days=365):
        raise HTTPException(422,detail={"error":"schedule_too_far"})
    with get_db() as conn:
        cur=conn.cursor()
        _school_admin(cur,school_id,str(actor["id"]))
        eligible=_eligible(cur,school_id)
        allowed={str(r["guardian_id"]) for r in eligible}
        if payload.audience_type=="school": targets=allowed
        elif payload.audience_type=="families":
            ids=set()
            if len(payload.guardian_handles)!=len(set(h.lower().lstrip("@") for h in payload.guardian_handles)):
                raise HTTPException(422,detail={"error":"duplicate_guardian"})
            for handle in payload.guardian_handles:
                gid=_id_for_handle(handle)
                if gid not in allowed: raise HTTPException(403,detail={"error":"guardian_not_verified_for_school"})
                ids.add(gid)
            targets=ids
        else:
            field="grade_label" if payload.audience_type=="grade" else "class_label"
            cur.execute(f"""SELECT DISTINCT g.guardian_id
                FROM verifyd_safety_comms_student_groups a
                JOIN verifyd_safety_training_guardians g
                  ON g.school_id=a.school_id AND g.child_id=a.student_id
                WHERE a.school_id=%s AND a.{field}=%s""",
                (school_id,payload.audience_value.strip()))
            targets={str(r["guardian_id"]) for r in cur.fetchall()} & allowed
        if not targets: raise HTTPException(422,detail={"error":"no_verified_recipients"})
        pid="sscom_"+uuid.uuid4().hex
        cur.execute("""INSERT INTO verifyd_safety_comms_posts
          (post_id,school_id,author_id,title,body,audience_type,audience_value,publish_at)
          VALUES(%s,%s,%s,%s,%s,%s,%s,%s)""",
          (pid,school_id,str(actor["id"]),payload.title.strip(),payload.body.strip(),
           payload.audience_type,payload.audience_value.strip(),scheduled))
        for gid in sorted(targets):
            cur.execute("""INSERT INTO verifyd_safety_comms_recipients(post_id,guardian_id)
               VALUES(%s,%s)""",(pid,gid))
        _audit(cur,school_id,str(actor["id"]),"school_comms_post_created",pid)
    return {"ok":True,"post_id":pid,"recipients":len(targets),
            "scheduled":scheduled>datetime.now(timezone.utc),
            "in_app_only":True,"push_sent":False,"emergency_alerts_active":False}

@router.get("/schools/{school_id}/communications/posts")
def staff_posts(school_id:str,authorization:Optional[str]=Header(default=None)):
    actor=_actor(authorization);_schema()
    with get_db() as conn:
        cur=conn.cursor();_school_admin(cur,school_id,str(actor["id"]))
        cur.execute("""SELECT p.post_id,p.title,p.body,p.audience_type,p.audience_value,
             p.publish_at,p.created_at,
             count(r.guardian_id)::int AS recipient_count,
             count(r.read_at)::int AS read_count
            FROM verifyd_safety_comms_posts p LEFT JOIN verifyd_safety_comms_recipients r
              ON r.post_id=p.post_id
            WHERE p.school_id=%s AND p.archived_at IS NULL
            GROUP BY p.post_id ORDER BY p.created_at DESC LIMIT 100""",(school_id,))
        rows=[dict(r) for r in cur.fetchall()]
    return {"ok":True,"posts":rows,"training_only":True}

@router.get("/family/communications")
def guardian_posts(authorization:Optional[str]=Header(default=None)):
    actor=_actor(authorization);_schema()
    gid=str(actor["id"])
    with get_db() as conn:
        cur=conn.cursor()
        cur.execute("""SELECT p.post_id,p.title,p.body,p.publish_at,p.school_id,s.school_name,
               r.read_at,p.created_at
            FROM verifyd_safety_comms_posts p
            JOIN verifyd_safety_comms_recipients r ON r.post_id=p.post_id
            JOIN verifyd_safety_training_schools s ON s.school_id=p.school_id
            WHERE r.guardian_id=%s AND p.archived_at IS NULL AND p.publish_at<=NOW()
            ORDER BY p.publish_at DESC LIMIT 100""",(gid,))
        result=[dict(r) for r in cur.fetchall()]
        # Revocation can happen after publication: filter by CURRENT family authority.
        schools={p["school_id"] for p in result}
        approved=set()
        for school in schools:
            if any(str(r["guardian_id"])==gid for r in _eligible(cur,school)):
                approved.add(school)
        result=[p for p in result if p["school_id"] in approved]
        cur.execute("SELECT in_app_enabled,push_requested FROM verifyd_safety_comms_preferences WHERE guardian_id=%s",(gid,))
        pref=cur.fetchone()
    return {"ok":True,"posts":result,"unread_count":sum(1 for p in result if not p["read_at"]),
            "preferences":dict(pref) if pref else {"in_app_enabled":True,"push_requested":False},
            "push_delivery_available":False,"in_app_only":True}

@router.post("/family/communications/{post_id}/read")
def mark_read(post_id:str,authorization:Optional[str]=Header(default=None)):
    actor=_actor(authorization);_schema();gid=str(actor["id"])
    with get_db() as conn:
        cur=conn.cursor()
        cur.execute("""SELECT p.school_id FROM verifyd_safety_comms_posts p
          JOIN verifyd_safety_comms_recipients r ON r.post_id=p.post_id
          WHERE p.post_id=%s AND r.guardian_id=%s AND p.publish_at<=NOW()
            AND p.archived_at IS NULL""",(post_id,gid))
        row=cur.fetchone()
        if not row or not any(str(r["guardian_id"])==gid for r in _eligible(cur,row["school_id"])):
            raise HTTPException(404,detail={"error":"announcement_unavailable"})
        cur.execute("""UPDATE verifyd_safety_comms_recipients SET read_at=COALESCE(read_at,NOW())
            WHERE post_id=%s AND guardian_id=%s""",(post_id,gid))
    return {"ok":True}

@router.post("/family/communications/preferences")
def save_prefs(payload:PrefInput,authorization:Optional[str]=Header(default=None)):
    actor=_actor(authorization);_schema()
    with get_db() as conn:
        conn.cursor().execute("""INSERT INTO verifyd_safety_comms_preferences
         (guardian_id,in_app_enabled,push_requested) VALUES(%s,%s,%s)
         ON CONFLICT(guardian_id) DO UPDATE SET
         in_app_enabled=EXCLUDED.in_app_enabled,push_requested=EXCLUDED.push_requested""",
         (str(actor["id"]),payload.in_app_enabled,payload.push_requested))
    return {"ok":True,"push_delivery_available":False}

@router.post("/schools/{school_id}/communications/posts/{post_id}/attachments")
def attach(school_id:str,post_id:str,payload:AttachmentInput,authorization:Optional[str]=Header(default=None)):
    actor=_actor(authorization);_schema()
    try: blob=base64.b64decode(payload.base64_data,validate=True)
    except (ValueError,binascii.Error): raise HTTPException(422,detail={"error":"invalid_file_data"})
    if not blob or len(blob)>_MAX_ATTACHMENT: raise HTTPException(413,detail={"error":"file_limit_1mb"})
    if payload.mime_type=="application/pdf" and not blob.startswith(b"%PDF-"):
        raise HTTPException(422,detail={"error":"invalid_pdf"})
    if payload.mime_type=="image/png" and not blob.startswith(b"\x89PNG\r\n\x1a\n"):
        raise HTTPException(422,detail={"error":"invalid_png"})
    if payload.mime_type=="image/jpeg" and not blob.startswith(b"\xff\xd8\xff"):
        raise HTTPException(422,detail={"error":"invalid_jpeg"})
    filename=re.sub(r"[^A-Za-z0-9._ -]","_",payload.filename).strip(" .")
    if not filename: raise HTTPException(422,detail={"error":"invalid_filename"})
    with get_db() as conn:
        cur=conn.cursor();_school_admin(cur,school_id,str(actor["id"]))
        cur.execute("SELECT publish_at FROM verifyd_safety_comms_posts WHERE school_id=%s AND post_id=%s AND archived_at IS NULL",(school_id,post_id))
        p=cur.fetchone()
        if not p: raise HTTPException(404,detail={"error":"post_not_found"})
        # Prevent recipient seeing a post before all attachments are complete.
        if p["publish_at"]<=datetime.now(timezone.utc):
            raise HTTPException(409,detail={"error":"attachments_only_before_publish"})
        cur.execute("SELECT count(*) AS n FROM verifyd_safety_comms_attachments WHERE post_id=%s",(post_id,))
        if cur.fetchone()["n"]>=3: raise HTTPException(422,detail={"error":"max_three_attachments"})
        aid="ssatt_"+uuid.uuid4().hex
        cur.execute("""INSERT INTO verifyd_safety_comms_attachments
          (attachment_id,post_id,filename,mime_type,file_bytes) VALUES(%s,%s,%s,%s,%s)""",
          (aid,post_id,filename,payload.mime_type,blob))
        _audit(cur,school_id,str(actor["id"]),"school_comms_attachment_added",aid)
    return {"ok":True,"attachment_id":aid}

@router.get("/family/communications/{post_id}/attachments")
def attachment_list(post_id:str,authorization:Optional[str]=Header(default=None)):
    actor=_actor(authorization);_schema();gid=str(actor["id"])
    with get_db() as conn:
        cur=conn.cursor()
        cur.execute("""SELECT p.school_id FROM verifyd_safety_comms_posts p JOIN verifyd_safety_comms_recipients r
          ON r.post_id=p.post_id WHERE p.post_id=%s AND r.guardian_id=%s
            AND p.publish_at<=NOW() AND p.archived_at IS NULL""",(post_id,gid))
        row=cur.fetchone()
        if not row or not any(str(x["guardian_id"])==gid for x in _eligible(cur,row["school_id"])):
            raise HTTPException(404,detail={"error":"announcement_unavailable"})
        cur.execute("SELECT attachment_id,filename,mime_type,octet_length(file_bytes) AS size FROM verifyd_safety_comms_attachments WHERE post_id=%s",(post_id,))
        files=[dict(r) for r in cur.fetchall()]
    return {"ok":True,"attachments":files}

@router.get("/family/communications/attachments/{attachment_id}")
def download(attachment_id:str,authorization:Optional[str]=Header(default=None)):
    actor=_actor(authorization);_schema();gid=str(actor["id"])
    with get_db() as conn:
        cur=conn.cursor()
        cur.execute("""SELECT a.filename,a.mime_type,a.file_bytes,p.school_id FROM verifyd_safety_comms_attachments a
          JOIN verifyd_safety_comms_posts p ON p.post_id=a.post_id
          JOIN verifyd_safety_comms_recipients r ON r.post_id=p.post_id
          WHERE a.attachment_id=%s AND r.guardian_id=%s AND p.publish_at<=NOW()
          AND p.archived_at IS NULL""",(attachment_id,gid))
        row=cur.fetchone()
        if not row or not any(str(x["guardian_id"])==gid for x in _eligible(cur,row["school_id"])):
            raise HTTPException(404,detail={"error":"attachment_unavailable"})
    return Response(bytes(row["file_bytes"]),media_type=row["mime_type"],headers={
        "Content-Disposition":'attachment; filename="'+row["filename"].replace('"','')+'"',
        "Cache-Control":"no-store","X-Content-Type-Options":"nosniff"})

@router.get("/schools/{school_id}/communications/attachments/{attachment_id}")
def staff_download(school_id:str,attachment_id:str,authorization:Optional[str]=Header(default=None)):
    actor=_actor(authorization);_schema()
    with get_db() as conn:
        cur=conn.cursor();_school_admin(cur,school_id,str(actor["id"]))
        cur.execute("""SELECT a.filename,a.mime_type,a.file_bytes FROM verifyd_safety_comms_attachments a
          JOIN verifyd_safety_comms_posts p ON p.post_id=a.post_id
          WHERE p.school_id=%s AND a.attachment_id=%s""",(school_id,attachment_id))
        row=cur.fetchone()
        if not row: raise HTTPException(404,detail={"error":"attachment_not_found"})
    return Response(bytes(row["file_bytes"]),media_type=row["mime_type"],headers={
        "Content-Disposition":'attachment; filename="'+row["filename"].replace('"','')+'"',
        "Cache-Control":"no-store","X-Content-Type-Options":"nosniff"})
