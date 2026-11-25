"""Principal-scoped DOCX and text-PDF roster extraction (preview ONLY).

Original document bytes remain in memory for this request, are not persisted
or logged, and NEVER create student identities or send invitations. Only tables
with unambiguous student reference, first-name and last-name columns qualify.
Image-only PDFs cannot be safely interpreted without OCR and are rejected.
"""
from __future__ import annotations

import csv
import io
import re
import unicodedata
import zipfile
from typing import Optional
from fastapi import APIRouter, File, Header, HTTPException, UploadFile
from docx import Document
from pypdf import PdfReader
from database import get_db
from trust_voice_school_safety_training import _actor
from trust_voice_school_safety_registration import _school_admin
from trust_voice_school_managed_roster import _require_enrollment_enabled

router=APIRouter(prefix="/trust-voice/school-safety", tags=["School Roster Document Import"])
_MAX_BYTES=3_000_000
_MAX_UNCOMPRESSED=20_000_000
_MAX_PAGES=50
_MAX_STUDENTS=5000
FIELDS={
 "localref":"local_ref","studentref":"local_ref","studentreference":"local_ref",
 "studentid":"local_ref","studentnumber":"local_ref","studentidentifier":"local_ref",
 "schoolid":"local_ref","osis":"local_ref","osisnumber":"local_ref",
 "firstname":"first_name","givenname":"first_name","studentfirstname":"first_name",
 "lastname":"last_name","surname":"last_name","familyname":"last_name",
 "studentlastname":"last_name",
 "ackmethod":"ack_method","acknowledgmentmethod":"ack_method",
 "guardianname":"guardian_name","signedby":"guardian_name",
 "signeddate":"signed_date","signaturedate":"signed_date",
 "evidenceref":"evidence_ref","evidencereference":"evidence_ref",
 "authorityreference":"authority_reference",
 "primaryguardianname":"primary_guardian_name","primaryparentname":"primary_guardian_name",
 "primaryguardianemail":"primary_guardian_email","primaryparentemail":"primary_guardian_email",
 "backupguardianname":"backup_guardian_name","secondguardianname":"backup_guardian_name",
 "backupguardianemail":"backup_guardian_email","secondguardianemail":"backup_guardian_email",
 "studentemail":"student_email","schoolstudentemail":"student_email",
 "studentphone":"student_phone","studentmobile":"student_phone",
 "primaryguardianphone":"primary_guardian_phone","primaryparentphone":"primary_guardian_phone",
 "backupguardianphone":"backup_guardian_phone","secondguardianphone":"backup_guardian_phone",
}
REQUIRED={"local_ref","first_name","last_name"}
SAFE_REF=re.compile(r"^[A-Za-z0-9_-]{3,64}$")
COLUMNS=[
 "local_ref","first_name","last_name","ack_method","guardian_name",
 "signed_date","evidence_ref","authority_reference","primary_guardian_name",
 "primary_guardian_email","primary_guardian_phone",
 "backup_guardian_name","backup_guardian_email","backup_guardian_phone",
 "student_email","student_phone",
]


def _key(value):
    folded=unicodedata.normalize("NFKD",str(value or ""))
    return re.sub(r"[^a-z0-9]","",folded.lower())


def _map_header(values):
    result={}
    for index,value in enumerate(values):
        canonical=FIELDS.get(_key(value))
        if canonical and canonical in result:
            raise ValueError("repeated_roster_column")
        if canonical:
            result[canonical]=index
    return result if REQUIRED.issubset(result) else None


def _collect_rows(table_rows):
    header=None
    parsed=[]
    for cells in table_rows:
        cells=[str(cell or "").replace("\x00","").strip() for cell in cells]
        if not any(cells):
            continue
        possible=_map_header(cells)
        if possible:
            header=possible
            continue
        if not header:
            continue
        row={name:cells[idx] if idx<len(cells) else "" for name,idx in header.items()}
        if not all(row.get(k) for k in REQUIRED):
            # Footers and incomplete lines are not automatically interpreted.
            continue
        if not SAFE_REF.fullmatch(row["local_ref"]):
            raise ValueError("invalid_school_student_reference")
        if any(len(row[k])>80 for k in ("first_name","last_name")):
            raise ValueError("invalid_student_name_length")
        parsed.append({field:row.get(field,"") for field in COLUMNS})
        if len(parsed)>_MAX_STUDENTS:
            raise ValueError("roster_too_large_split_into_files")
    return parsed


def _docx_rows(raw):
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        if sum(item.file_size for item in archive.infolist())>_MAX_UNCOMPRESSED:
            raise ValueError("word_document_uncompressed_size_too_large")
    document=Document(io.BytesIO(raw))
    rows=[]
    # Use actual Word table boundaries rather than flattening arbitrary prose.
    for table in document.tables:
        parsed=_collect_rows([[cell.text for cell in row.cells] for row in table.rows])
        rows.extend(parsed)
    return rows


def _pdf_rows(raw):
    reader=PdfReader(io.BytesIO(raw), strict=False)
    if reader.is_encrypted:
        raise ValueError("password_protected_pdf_not_supported")
    if len(reader.pages)>_MAX_PAGES:
        raise ValueError("pdf_too_many_pages_export_to_csv")
    result=[]
    for page in reader.pages:
        try:
            text=page.extract_text(extraction_mode="layout") or ""
        except (TypeError,ValueError):
            text=page.extract_text() or ""
        lines=[line.strip() for line in text.splitlines() if line.strip()]
        # PDF extraction loses the cell grid. Only import confidently
        # separated text columns; image/scanned pages fail closed.
        table=[]
        for line in lines:
            if "|" in line:
                cells=[v.strip() for v in line.split("|")]
            elif "\t" in line:
                cells=[v.strip() for v in line.split("\t")]
            elif re.search(r" {2,}",line):
                cells=[v.strip() for v in re.split(r" {2,}",line)]
            elif "," in line:
                cells=next(csv.reader([line]))
            else:
                continue
            table.append(cells)
        result.extend(_collect_rows(table))
    return result


@router.post("/schools/{school_id}/managed-roster/parse-document")
async def parse_school_roster_document(
    school_id:str, file:UploadFile=File(...),
    authorization:Optional[str]=Header(default=None),
):
    """Decode a roster into preview rows only; no DB writes, no email."""
    _require_enrollment_enabled(school_id)
    actor=_actor(authorization)
    with get_db() as conn:
        _school_admin(conn.cursor(),school_id,str(actor["id"]))
    filename=(file.filename or "").strip().lower()
    kind="docx" if filename.endswith(".docx") else "pdf" if filename.endswith(".pdf") else ""
    if not kind:
        raise HTTPException(422,detail={"error":"upload_csv_docx_or_text_pdf_only_legacy_doc_convert_to_docx"})
    try:
        raw=await file.read(_MAX_BYTES+1)
    finally:
        await file.close()
    if not raw or len(raw)>_MAX_BYTES:
        raise HTTPException(413,detail={"error":"document_empty_or_over_3mb"})
    if kind=="pdf" and not raw.startswith(b"%PDF"):
        raise HTTPException(422,detail={"error":"file_is_not_valid_pdf"})
    if kind=="docx" and not raw.startswith(b"PK"):
        raise HTTPException(422,detail={"error":"file_is_not_valid_docx"})
    try:
        rows= _docx_rows(raw) if kind=="docx" else _pdf_rows(raw)
    except ValueError as e:
        raise HTTPException(422,detail={"error":str(e)}) from None
    except Exception:
        # Do not echo document contents or stack trace into logs.
        raise HTTPException(422,detail={"error":"unable_to_read_roster_document"}) from None
    if not rows:
        raise HTTPException(422,detail={"error":
            "no_unambiguous_student_table_found_use_csv_or_structured_docx_scanned_pdf_needs_conversion"})
    ids=[r["local_ref"].lower() for r in rows]
    if len(ids)!=len(set(ids)):
        raise HTTPException(422,detail={"error":"duplicate_school_student_reference_in_document"})
    return {"ok":True,"preview_only":True,"rows":rows,"student_count":len(rows),
        "source_format":kind,"stored":False,"accounts_created":False,
        "messages_sent":False,"warnings":[
            "Confirm column alignment, student IDs, names and guardian emails before staging.",
            "This extraction is not evidence of guardian consent, and does not authorize students."
        ]}
