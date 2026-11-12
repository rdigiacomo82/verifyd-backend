"""Private classroom upload quarantine.

This module does not expose a file download, preview, signed URL, public URL,
AI analysis, conversion, or "clean" status setter. That is intentional:
the current VeriFYD attachment-analysis service is NOT malware scanning.
A separate real scanner and release-to-clean-bucket workflow are required
before any student/teacher can retrieve uploaded bytes.
"""
from __future__ import annotations

import hashlib
import os
import re
import secrets
import threading
from pathlib import PurePosixPath
from typing import Literal, Optional

import boto3
from botocore.client import Config
from fastapi import APIRouter, File, Form, Header, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool

from database import get_db
from trust_voice_classrooms import _identity, _schema as class_schema, _teacher, _member, _assignment

router = APIRouter(prefix="/trust-voice/classroom", tags=["Circle Classroom Quarantine"])
_READY = False
_LOCK = threading.Lock()

MAX_FILE_SIZE = 8 * 1024 * 1024
MAX_FILES_PER_ASSIGNMENT = 12
TYPES = {
    ".pdf": "application/pdf",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".txt": "text/plain",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}


def upload_enabled():
    return (os.getenv("VERIFYD_CIRCLE_CLASSROOM_ENABLED", "0").strip().lower() in ("1","true","yes","on")
            and os.getenv("VERIFYD_CIRCLE_CLASSROOM_QUARANTINE_UPLOAD_ENABLED", "0").strip().lower()
            in ("1","true","yes","on"))


def storage():
    account = os.getenv("R2_ACCOUNT_ID", "").strip()
    key = os.getenv("R2_CLASSROOM_ACCESS_KEY_ID", "").strip()
    secret = os.getenv("R2_CLASSROOM_SECRET_ACCESS_KEY", "").strip()
    bucket = os.getenv("R2_CLASSROOM_QUARANTINE_BUCKET", "").strip()
    existing = os.getenv("R2_TRUST_VOICE_BUCKET", "").strip()
    if not all((account, key, secret, bucket)) or bucket == existing:
        raise HTTPException(503, detail={"error": "isolated_quarantine_storage_not_configured"})
    client = boto3.client(
        "s3", endpoint_url=f"https://{account}.r2.cloudflarestorage.com",
        aws_access_key_id=key, aws_secret_access_key=secret,
        config=Config(signature_version="s3v4"), region_name="auto")
    return client, bucket


def safe_name(filename):
    basename = (filename or "").replace("\\", "/").split("/")[-1][:150]
    value = re.sub(r"[^a-zA-Z0-9._ -]", "_", basename).strip(". ")
    return value or "attachment"


def check_signature(name, body):
    ext = PurePosixPath(name).suffix.lower()
    if ext not in TYPES:
        raise HTTPException(415, detail={"error": "file_type_not_allowed"})
    if ext == ".pdf" and not body.startswith(b"%PDF-"):
        raise HTTPException(415, detail={"error": "invalid_file_signature"})
    if ext == ".png" and not body.startswith(b"\x89PNG\r\n\x1a\n"):
        raise HTTPException(415, detail={"error": "invalid_file_signature"})
    if ext in (".jpg", ".jpeg") and not body.startswith(b"\xff\xd8\xff"):
        raise HTTPException(415, detail={"error": "invalid_file_signature"})
    if ext == ".docx" and not body.startswith(b"PK\x03\x04"):
        raise HTTPException(415, detail={"error": "invalid_file_signature"})
    if ext == ".txt":
        try:
            body.decode("utf-8")
        except UnicodeDecodeError:
            raise HTTPException(415, detail={"error": "invalid_text_encoding"})
        if b"\0" in body:
            raise HTTPException(415, detail={"error": "invalid_text_encoding"})
    return TYPES[ext]


def ensure_upload_schema():
    global _READY
    if _READY:
        return
    with _LOCK:
        if _READY:
            return
        class_schema()
        with get_db() as conn:
            cur = conn.cursor()
            cur.execute("""CREATE TABLE IF NOT EXISTS verifyd_circle_class_uploads(
                upload_id TEXT PRIMARY KEY,
                assignment_id TEXT NOT NULL REFERENCES verifyd_circle_class_assignments(assignment_id) ON DELETE CASCADE,
                owner_id TEXT NOT NULL REFERENCES verifyd_identities(id) ON DELETE CASCADE,
                purpose TEXT NOT NULL CHECK (purpose IN ('teacher_material','student_work')),
                storage_key TEXT NOT NULL UNIQUE,
                original_name TEXT NOT NULL,
                content_type TEXT NOT NULL,
                sha256 TEXT NOT NULL,
                byte_size INTEGER NOT NULL CHECK (byte_size BETWEEN 1 AND 8388608),
                scan_status TEXT NOT NULL DEFAULT 'quarantined' CHECK(scan_status IN ('quarantined','rejected')),
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )""")
            cur.execute("""CREATE INDEX IF NOT EXISTS idx_circle_class_uploads_assignment_owner
                ON verifyd_circle_class_uploads(assignment_id,owner_id)""")
        _READY = True

@router.get("/uploads/health")
def health():
    if not upload_enabled():
        raise HTTPException(404, detail={"error": "classroom_uploads_disabled"})
    return {"enabled":True, "max_bytes":MAX_FILE_SIZE,
            "allowed_extensions":sorted(TYPES),
            "malware_scanner_active":False, "download_available":False,
            "reason":"Uploads are quarantined, not malware scanned."}


@router.post("/classes/{class_id}/assignments/{assignment_id}/uploads")
async def upload(class_id: str, assignment_id: str,
                 purpose: Literal["teacher_material", "student_work"] = Form(...),
                 file: UploadFile = File(...),
                 authorization: Optional[str] = Header(default=None)):
    if not upload_enabled():
        raise HTTPException(404, detail={"error": "classroom_uploads_disabled"})
    user = _identity(authorization)
    ensure_upload_schema()
    with get_db() as conn:
        cur = conn.cursor()
        membership = _member(cur, class_id, user)
        assignment = _assignment(cur, class_id, assignment_id)
        if purpose == "teacher_material":
            if membership["teacher_id"] != user:
                raise HTTPException(404, detail={"error": "assignment_not_found"})
        else:
            if membership["teacher_id"] == user or assignment["status"] != "published":
                raise HTTPException(403, detail={"error": "student_submission_unavailable"})
            cur.execute("""SELECT status FROM verifyd_circle_class_submissions
                WHERE assignment_id=%s AND student_id=%s""", (assignment_id, user))
            record = cur.fetchone()
            if record and record["status"] == "submitted":
                raise HTTPException(409, detail={"error": "already_turned_in"})
        cur.execute("""SELECT COUNT(*) AS count FROM verifyd_circle_class_uploads
            WHERE assignment_id=%s AND owner_id=%s""", (assignment_id,user))
        if cur.fetchone()["count"] >= MAX_FILES_PER_ASSIGNMENT:
            raise HTTPException(429, detail={"error": "attachment_limit"})
    name = safe_name(file.filename)
    if PurePosixPath(name).suffix.lower() not in TYPES:
        raise HTTPException(415, detail={"error": "file_type_not_allowed"})
    # Read a bounded number of bytes. Never place untrusted bytes in a web-
    # accessible directory, database text column, detector worker or preview.
    body = await file.read(MAX_FILE_SIZE + 1)
    if not body:
        raise HTTPException(400, detail={"error": "empty_upload"})
    if len(body) > MAX_FILE_SIZE:
        raise HTTPException(413, detail={"error": "file_too_large"})
    content_type = check_signature(name, body)
    client,bucket = storage()
    upload_id = str(secrets.token_hex(20))
    storage_key = f"unscanned/{secrets.token_hex(24)}"
    digest = hashlib.sha256(body).hexdigest()
    try:
        await run_in_threadpool(client.put_object,
            Bucket=bucket, Key=storage_key, Body=body,
            ContentType="application/octet-stream",
            Metadata={"sha256":digest})
        # Authentication and status are rechecked after network I/O, so a
        # revoked membership does not get a persistent upload record.
        with get_db() as conn:
            cur = conn.cursor()
            membership = _member(cur, class_id, user)
            assignment = _assignment(cur, class_id, assignment_id)
            if purpose == "teacher_material":
                if membership["teacher_id"] != user:
                    raise HTTPException(404, detail={"error": "assignment_not_found"})
            elif membership["teacher_id"] == user or assignment["status"] != "published":
                raise HTTPException(403, detail={"error": "student_submission_unavailable"})
            cur.execute("""INSERT INTO verifyd_circle_class_uploads
                (upload_id,assignment_id,owner_id,purpose,storage_key,original_name,
                content_type,sha256,byte_size) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                (upload_id,assignment_id,user,purpose,storage_key,name,content_type,
                digest,len(body)))
    except Exception:
        try:
            await run_in_threadpool(client.delete_object, Bucket=bucket, Key=storage_key)
        except Exception:
            pass
        raise
    return {
        "upload_id":upload_id, "name":name,"purpose":purpose,
        "size_bytes":len(body),"scan_status":"quarantined",
        "safe_to_open":False,"download_available":False,
        "note":"A real malware scanner must approve the file before it can be used."
    }


@router.get("/classes/{class_id}/assignments/{assignment_id}/uploads")
def list_uploads(class_id: str, assignment_id: str,
                 authorization: Optional[str] = Header(default=None)):
    if not upload_enabled():
        raise HTTPException(404, detail={"error": "classroom_uploads_disabled"})
    user = _identity(authorization)
    ensure_upload_schema()
    with get_db() as conn:
        cur=conn.cursor()
        membership = _member(cur, class_id, user)
        _assignment(cur, class_id, assignment_id)
        cur.execute("""SELECT upload_id,owner_id,purpose,original_name,content_type,
            byte_size,scan_status,created_at
            FROM verifyd_circle_class_uploads
            WHERE assignment_id=%s AND (owner_id=%s OR %s)
            ORDER BY created_at DESC LIMIT 200""",
            (assignment_id,user,membership["teacher_id"] == user))
        rows=[{**dict(x), "download_available":False, "created_at":x["created_at"].isoformat()}
              for x in cur.fetchall()]
    return {"files":rows, "download_available":False}
