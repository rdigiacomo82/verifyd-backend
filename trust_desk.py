# ============================================================
#  VeriFYD — trust_desk.py
#
#  Trust Desk ZIP intake helpers.
#  This module intentionally keeps ZIP extraction, file inventory,
#  manifest creation, and return-package assembly separate from the
#  existing video/photo/audio/document detection pipelines.
#
#  Phase 1 skeleton:
#    - Safely extract a submitted ZIP
#    - Inventory and classify supported files
#    - Preserve original folder structure
#    - Build a Trust Desk return ZIP with manifest/report stubs
#
#  Later phases can route each supported file to the existing
#  VeriFYD certification functions and add child certificate IDs.
# ============================================================

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import tempfile
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".webm", ".mkv", ".m4v"}
PHOTO_EXTENSIONS = {".jpg", ".jpeg", ".png", ".heic", ".heif", ".webp", ".gif", ".bmp", ".tif", ".tiff"}
AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".oga", ".opus", ".webm"}
DOCUMENT_EXTENSIONS = {
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    ".odt", ".ods", ".odp", ".txt", ".md", ".csv", ".rtf",
    ".eml", ".msg", ".html", ".htm", ".mhtml", ".mht",
    ".xml", ".json", ".svg", ".vsdx", ".yaml", ".yml",
    ".toml", ".env", ".ini", ".properties", ".conf", ".cfg",
    ".config", ".cnf", ".log", ".sql", ".pst", ".ost", ".dwg", ".dxf",
}

SUPPORTED_EXTENSIONS = VIDEO_EXTENSIONS | PHOTO_EXTENSIONS | AUDIO_EXTENSIONS | DOCUMENT_EXTENSIONS


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def classify_file(filename: str) -> str:
    ext = Path(filename).suffix.lower()
    if ext in VIDEO_EXTENSIONS:
        return "video"
    if ext in PHOTO_EXTENSIONS:
        return "photo"
    if ext in AUDIO_EXTENSIONS:
        return "audio"
    if ext in DOCUMENT_EXTENSIONS:
        return "document"
    return "unsupported"


def safe_extract_zip(zip_path: str, dest_dir: str) -> List[str]:
    """
    Safely extract a ZIP file while preventing zip-slip path traversal.
    Returns the list of extracted file paths.
    """
    extracted: List[str] = []
    dest_root = Path(dest_dir).resolve()
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.infolist():
            if member.is_dir():
                continue
            member_name = member.filename.replace("\\", "/")
            if not member_name or member_name.startswith("/"):
                continue
            target = (dest_root / member_name).resolve()
            if not str(target).startswith(str(dest_root) + os.sep):
                raise ValueError(f"Unsafe ZIP path rejected: {member.filename}")
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member, "r") as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)
            extracted.append(str(target))
    return extracted


def build_inventory(extract_dir: str) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    root = Path(extract_dir).resolve()
    rows: List[Dict[str, Any]] = []
    counts = {"total": 0, "video": 0, "photo": 0, "audio": 0, "document": 0, "unsupported": 0}

    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel_path = path.relative_to(root).as_posix()
        media_type = classify_file(path.name)
        file_hash = sha256_file(str(path))
        size_bytes = path.stat().st_size
        counts["total"] += 1
        counts[media_type] = counts.get(media_type, 0) + 1
        rows.append({
            "relative_path": rel_path,
            "filename": path.name,
            "extension": path.suffix.lower(),
            "media_type": media_type,
            "supported": media_type != "unsupported",
            "size_bytes": size_bytes,
            "sha256": file_hash,
            "processing_status": "inventory_only_pending_certification" if media_type != "unsupported" else "preserved_unsupported",
            "certificate_id": "",
            "certified_file": "",
            "authenticity_score": "",
            "ai_score": "",
        })
    return rows, counts


def write_hash_inventory_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    fieldnames = [
        "relative_path", "filename", "extension", "media_type", "supported",
        "size_bytes", "sha256", "processing_status", "certificate_id",
        "certified_file", "authenticity_score", "ai_score",
    ]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})



def _copy_for_child_enqueue(src_path: Path, job_id: str) -> str:
    """Copy an extracted file to a temp path because enqueue helpers remove the path after upload."""
    suffix = src_path.suffix or ""
    tmp_path = os.path.join(tempfile.gettempdir(), f"trustdesk_child_{job_id}{suffix}")
    shutil.copy2(src_path, tmp_path)
    return tmp_path


def enqueue_child_certification_jobs(
    *,
    inventory: List[Dict[str, Any]],
    extract_dir: str,
    submitter_email: str,
    base_url: str = "",
) -> Dict[str, Any]:
    """
    Trust Desk Phase 2A: enqueue each supported extracted file into the existing
    VeriFYD detection/certification queues.

    This does not wait for child jobs to finish yet. It records each child job ID
    in the manifest so the next phase can collect certified outputs and rebuild
    a final completed Trust Desk package.
    """
    from queue_helper import (
        enqueue_upload,
        enqueue_photo_upload,
        enqueue_audio_upload,
        enqueue_document_upload,
    )

    root = Path(extract_dir).resolve()
    queued = 0
    failed = 0
    failures: List[Dict[str, str]] = []

    for row in inventory:
        media_type = row.get("media_type")
        if row.get("supported") is not True or media_type == "unsupported":
            continue

        rel_path = str(row.get("relative_path") or "")
        src_path = (root / rel_path).resolve()
        if not str(src_path).startswith(str(root) + os.sep) or not src_path.exists() or not src_path.is_file():
            row["processing_status"] = "certification_enqueue_failed"
            row["certification_error"] = "Extracted source file not found"
            failed += 1
            failures.append({"relative_path": rel_path, "error": "Extracted source file not found"})
            continue

        child_job_id = str(uuid.uuid4())
        child_tmp = _copy_for_child_enqueue(src_path, child_job_id)
        filename = src_path.name

        try:
            if media_type == "video":
                enqueue_upload(child_job_id, child_tmp, filename, submitter_email)
                download_url = f"{base_url}/download/{child_job_id}" if base_url else ""
            elif media_type == "photo":
                enqueue_photo_upload(child_job_id, child_tmp, filename, submitter_email)
                download_url = f"{base_url}/download-photo/{child_job_id}" if base_url else ""
            elif media_type == "audio":
                enqueue_audio_upload(child_job_id, child_tmp, filename, submitter_email)
                download_url = f"{base_url}/download-audio/{child_job_id}" if base_url else ""
            elif media_type == "document":
                enqueue_document_upload(child_job_id, child_tmp, filename, submitter_email)
                download_url = f"{base_url}/download-document/{child_job_id}" if base_url else ""
            else:
                row["processing_status"] = "preserved_unsupported"
                continue

            row["processing_status"] = "certification_queued"
            row["certificate_id"] = child_job_id
            row["child_job_id"] = child_job_id
            row["verification_link"] = f"{base_url}/verify-certificate/{child_job_id}" if base_url else ""
            row["download_url"] = download_url
            queued += 1
        except Exception as exc:
            row["processing_status"] = "certification_enqueue_failed"
            row["certification_error"] = str(exc)[:300]
            failed += 1
            failures.append({"relative_path": rel_path, "error": str(exc)[:300]})
            try:
                if child_tmp and os.path.exists(child_tmp):
                    os.remove(child_tmp)
            except Exception:
                pass

    return {"queued": queued, "failed": failed, "failures": failures}


def build_trust_desk_package(
    *,
    job_id: str,
    source_zip_path: str,
    original_zip_filename: str,
    extract_dir: str,
    output_zip_path: str,
    organization: str,
    submitter_name: str,
    submitter_email: str,
    case_number: str,
    notes: str,
    base_url: str = "",
    route_certifications: bool = False,
) -> Dict[str, Any]:
    """
    Build the first Trust Desk return package.
    Phase 1 is inventory + preservation. Individual certification routing
    will be layered on top in a later patch.

    Important: the source ZIP must be extracted before inventory is built.
    Earlier skeleton versions created an empty manifest because they built
    inventory against an empty extraction directory.
    """
    # Start from a clean extraction directory and safely extract the submitted ZIP.
    extract_root = Path(extract_dir)
    if extract_root.exists():
        shutil.rmtree(extract_root, ignore_errors=True)
    extract_root.mkdir(parents=True, exist_ok=True)

    extracted_files = safe_extract_zip(source_zip_path, extract_dir)
    inventory, counts = build_inventory(extract_dir)
    routing_summary = {"queued": 0, "failed": 0, "failures": []}
    if route_certifications:
        routing_summary = enqueue_child_certification_jobs(
            inventory=inventory,
            extract_dir=extract_dir,
            submitter_email=submitter_email,
            base_url=base_url,
        )
    original_zip_hash = sha256_file(source_zip_path)
    generated_at = datetime.now(timezone.utc).isoformat()

    manifest: Dict[str, Any] = {
        "title": "VeriFYD Trust Desk Intake Manifest",
        "phase": "phase_2a_zip_intake_inventory_and_child_certification_routing",
        "trust_desk_job_id": job_id,
        "organization": organization,
        "submitter_name": submitter_name,
        "submitter_email": submitter_email,
        "case_number": case_number,
        "notes": notes,
        "original_zip_filename": original_zip_filename,
        "original_zip_sha256": original_zip_hash,
        "generated_at_utc": generated_at,
        "summary": {
            "extracted_files": len(extracted_files),
            "total_files": counts.get("total", 0),
            "supported_files": counts.get("video", 0) + counts.get("photo", 0) + counts.get("audio", 0) + counts.get("document", 0),
            "video_files": counts.get("video", 0),
            "photo_files": counts.get("photo", 0),
            "audio_files": counts.get("audio", 0),
            "document_files": counts.get("document", 0),
            "unsupported_files": counts.get("unsupported", 0),
            "certification_jobs_queued": routing_summary.get("queued", 0),
            "certification_enqueue_failures": routing_summary.get("failed", 0),
        },
        "certification_routing": routing_summary,
        "items": inventory,
        "next_phase_note": (
            "This Trust Desk package confirms ZIP intake, extraction, file classification, "
            "original hash inventory, package reassembly, and child certification job routing. "
            "The next patch can collect completed child certificates and embed certified outputs in the master ZIP."
        ),
    }

    work_dir = Path(output_zip_path).parent / f"trustdesk_build_{job_id}"
    if work_dir.exists():
        shutil.rmtree(work_dir, ignore_errors=True)
    (work_dir / "00_Master_Report").mkdir(parents=True, exist_ok=True)
    (work_dir / "01_Original_Files").mkdir(parents=True, exist_ok=True)
    (work_dir / "02_Certified_Files").mkdir(parents=True, exist_ok=True)
    (work_dir / "03_Evidence_Packages").mkdir(parents=True, exist_ok=True)
    (work_dir / "04_Unsupported_Files").mkdir(parents=True, exist_ok=True)
    (work_dir / "05_Verification_Links").mkdir(parents=True, exist_ok=True)

    manifest_path = work_dir / "00_Master_Report" / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    hash_csv_path = work_dir / "00_Master_Report" / "hash_inventory.csv"
    write_hash_inventory_csv(str(hash_csv_path), inventory)

    readme = f"""VeriFYD Trust Desk Package\n\nTrust Desk Job ID: {job_id}\nOrganization: {organization or 'Not provided'}\nCase / Claim / Matter Number: {case_number or 'Not provided'}\nSubmitter: {submitter_name or 'Not provided'} <{submitter_email}>\nGenerated UTC: {generated_at}\n\nPhase 1 Status:\nThis package confirms ZIP intake, safe extraction, file classification, SHA-256 hash inventory, and reassembly.\n\nSummary:\n- Extracted files: {manifest['summary'].get('extracted_files', 0)}
- Total files: {manifest['summary']['total_files']}\n- Supported files: {manifest['summary']['supported_files']}\n- Video files: {manifest['summary']['video_files']}\n- Photo files: {manifest['summary']['photo_files']}\n- Audio files: {manifest['summary']['audio_files']}\n- Document files: {manifest['summary']['document_files']}\n- Unsupported files: {manifest['summary']['unsupported_files']}\n\nNext Implementation Phase:\nRoute supported files to VeriFYD video/photo/audio/document certification and insert child certificate IDs into this manifest.\n"""
    (work_dir / "00_Master_Report" / "TrustDesk_ReadMe.txt").write_text(readme, encoding="utf-8")
    (work_dir / "05_Verification_Links" / "verification_links.txt").write_text(
        "Child certificate verification links will be added in the certification-routing phase.\n",
        encoding="utf-8",
    )

    original_root = Path(extract_dir).resolve()
    for src in original_root.rglob("*"):
        if src.is_file():
            rel = src.relative_to(original_root)
            dest = work_dir / "01_Original_Files" / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
            if classify_file(src.name) == "unsupported":
                unsupported_dest = work_dir / "04_Unsupported_Files" / rel
                unsupported_dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, unsupported_dest)

    with zipfile.ZipFile(output_zip_path, "w", compression=zipfile.ZIP_DEFLATED) as out:
        for src in work_dir.rglob("*"):
            if src.is_file():
                out.write(src, src.relative_to(work_dir).as_posix())

    manifest["trust_desk_package_sha256"] = sha256_file(output_zip_path)
    try:
        shutil.rmtree(work_dir, ignore_errors=True)
    except Exception:
        pass
    return manifest
