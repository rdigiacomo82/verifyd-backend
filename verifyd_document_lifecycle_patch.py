"""
VeriFYD document certification lifecycle patch.
Run from repo root:
    py verifyd_document_lifecycle_patch.py

Targeted fixes:
- /job-status/{job_id}: do not return complete while certification_status is processing.
- /verify-certificate/{cid}: expose document PDF/package availability using R2 and Redis fallbacks.
- worker.py: route certification emails through notification_helper compatibility wrapper.
"""
from __future__ import annotations

import pathlib
import re
import textwrap

ROOT = pathlib.Path.cwd()


def read(name: str) -> str:
    p = ROOT / name
    if not p.exists():
        raise FileNotFoundError(f"Missing required file: {name}")
    return p.read_text(encoding="utf-8", errors="replace")


def write(name: str, text: str) -> None:
    (ROOT / name).write_text(text, encoding="utf-8", newline="\n")


JOB_STATUS_BLOCK = r'''
@app.get("/job-status/{job_id}")
def job_status(job_id: str):
    """Poll endpoint for async frontends.

    Important lifecycle rule:
    If a REAL media/document job has stored an early result as job_status=complete
    but certification_status is still processing, keep lifecycle status as processing
    so the frontend continues polling until the certified artifact is ready or failed.

    Backward-compatible fields:
    - status / job_status / job_state = polling lifecycle state
    - result_status / display_status / verdict_status = visible result-card title
    """
    result = get_job_result(job_id)

    if not result or result.get("job_status") == "not_found":
        return JSONResponse({
            "status": "not_found",
            "job_status": "not_found",
            "job_state": "not_found",
        }, status_code=404)

    result_copy = dict(result)

    raw_status = str(result.get("status") or "").strip()
    raw_status_lc = raw_status.lower()
    lifecycle_values = {"queued", "processing", "complete", "error", "not_found"}

    cert_status = str(result.get("certification_status") or "").strip().lower()

    ready_flags = [
        bool(result.get("video_ready") is True),
        bool(result.get("audio_ready") is True),
        bool(result.get("certified_audio_ready") is True),
        bool(result.get("photo_ready") is True),
        bool(result.get("document_ready") is True),
        bool(result.get("certified_file_package_ready") is True),
        bool(result.get("file_package_ready") is True),
    ]

    # If certification is still processing, never tell the frontend the job is complete yet.
    # This prevents document/video/audio pages from stopping polling on early results.
    if cert_status == "processing" and not any(ready_flags):
        job_st = "processing"
    elif cert_status == "failed":
        job_st = "complete"
    else:
        job_st = (
            result.get("job_status")
            or result.get("job_state")
            or ("error" if raw_status_lc == "error" else "")
            or "processing"
        )

    # Preserve the visible verdict/title separately from polling lifecycle.
    if raw_status and raw_status_lc not in lifecycle_values:
        display_status = raw_status
    else:
        label = str(result.get("label") or "").upper()
        media_type = str(result.get("media_type") or "").lower()
        download_type = str(result.get("download_type") or "").lower()

        if media_type == "audio" or download_type == "certified_audio":
            display_status = (
                "REAL AUDIO VERIFIED" if label == "REAL" else
                "AUDIO UNDETERMINED" if label == "UNDETERMINED" else
                "AI AUDIO DETECTED" if label == "AI" else
                "AUDIO ANALYSIS COMPLETE"
            )
        elif media_type == "document" or result.get("document_ready") is not None:
            display_status = (
                "REAL DOCUMENT VERIFIED" if label == "REAL" else
                "DOCUMENT UNDETERMINED" if label == "UNDETERMINED" else
                "AI / TAMPERING DETECTED" if label == "AI" else
                "DOCUMENT ANALYSIS COMPLETE"
            )
        elif media_type == "photo" or result.get("photo_ready") is not None:
            display_status = (
                "REAL PHOTO VERIFIED" if label == "REAL" else
                "PHOTO UNDETERMINED" if label == "UNDETERMINED" else
                "AI DETECTED" if label == "AI" else
                "PHOTO ANALYSIS COMPLETE"
            )
        elif media_type == "video" or result.get("video_ready") is not None:
            display_status = (
                "REAL VIDEO VERIFIED" if label == "REAL" else
                "VIDEO UNDETERMINED" if label == "UNDETERMINED" else
                "AI DETECTED" if label == "AI" else
                "VIDEO ANALYSIS COMPLETE"
            )
        else:
            display_status = raw_status if raw_status and raw_status_lc not in lifecycle_values else "ANALYSIS COMPLETE"

    result_copy["result_status"] = display_status
    result_copy["display_status"] = display_status
    result_copy["verdict_status"] = display_status

    result_copy["status"] = job_st
    result_copy["job_status"] = job_st
    result_copy["job_state"] = job_st

    return JSONResponse(result_copy)
'''.strip() + "\n\n"


VERIFY_CERT_BLOCK = r'''
@app.get("/verify-certificate/{cid}")
def verify_certificate_by_id(cid: str):
    """Verify a certificate ID and expose certified artifact availability.

    For documents, check both stamped PDF and universal evidence-package ZIP
    across R2 and Redis fallback before returning download metadata.
    """
    cid = (cid or "").strip()
    if not cid:
        return JSONResponse({"verified": False, "status": "missing_certificate_id"}, status_code=400)

    cert = get_certificate(cid)
    if not cert:
        return JSONResponse({
            "verified": False,
            "status": "not_found",
            "certificate_id": cid,
            "verification_status": "CERTIFICATE_NOT_FOUND",
        }, status_code=404)

    certified_document_available = False
    certified_file_package_available = False

    # R2 checks.
    try:
        from storage import r2_available, certified_document_exists, certified_file_package_exists
        if r2_available():
            try:
                certified_document_available = bool(certified_document_exists(cid))
            except Exception:
                certified_document_available = False
            try:
                certified_file_package_available = bool(certified_file_package_exists(cid))
            except Exception:
                certified_file_package_available = False
    except Exception as e:
        log.warning("verify-certificate: R2 artifact checks failed for %s: %s", cid, e)

    # Redis fallback checks for local/dev or legacy fallback jobs.
    try:
        r = _get_redis()
        if not certified_document_available:
            certified_document_available = bool(r.exists(f"doccert:{cid}"))
        if not certified_file_package_available:
            certified_file_package_available = bool(r.exists(f"filecert:{cid}"))
    except Exception as e:
        log.warning("verify-certificate: Redis artifact checks failed for %s: %s", cid, e)

    original_file = cert.get("original_file", "") or ""
    original_file_lc = original_file.lower()
    is_document_like = bool(re.search(r"\.(pdf|doc|docx|xls|xlsx|ppt|pptx|txt|rtf|odt|csv|zip)$", original_file_lc))

    media_type = cert.get("media_type") or ("document" if is_document_like or certified_document_available or certified_file_package_available else "")

    download_type = ""
    download_url = ""
    certified_file_package_url = ""

    if certified_file_package_available:
        download_type = "certified_file_package"
        download_url = f"{BASE_URL}/download-certified-file/{cid}"
        certified_file_package_url = download_url
    elif certified_document_available:
        download_type = "certified_document"
        download_url = f"{BASE_URL}/download-document/{cid}"
    elif media_type == "document":
        download_type = "document_unavailable"

    return JSONResponse({
        "verified": True,
        "status": "found",
        "verification_status": "CERTIFICATE_RECORD_FOUND",
        "certificate_id": cid,
        "label": cert.get("label", ""),
        "authenticity": cert.get("authenticity", ""),
        "ai_score": cert.get("ai_score", ""),
        "original_file": original_file,
        "upload_time": cert.get("upload_time", ""),
        "download_count": cert.get("download_count", 0),
        "verified_by": cert.get("verified_by", "VeriFYD"),
        "certified_to": cert.get("email", ""),
        "email": cert.get("email", ""),
        "media_type": media_type,
        "download_type": download_type,
        "download_url": download_url,
        "share_url": download_url,
        "document_ready": bool(certified_document_available or certified_file_package_available),
        "certified_document_available": bool(certified_document_available),
        "certified_file_package_available": bool(certified_file_package_available),
        "certified_file_package_url": certified_file_package_url,
        "verification_report": {
            "title": "VeriFYD Certificate Verification Report",
            "status": "FOUND",
            "certificate_id": cid,
            "database_match": "YES",
            "certified_document_available": "YES" if certified_document_available else "NO",
            "certified_file_package_available": "YES" if certified_file_package_available else "NO",
            "trust_level": "CERTIFIED ARTIFACT FOUND" if (certified_document_available or certified_file_package_available) else "DATABASE RECORD FOUND",
            "message": (
                "This certificate ID exists and a certified downloadable document/package is available."
                if (certified_document_available or certified_file_package_available)
                else "This certificate ID exists, but the certified downloadable artifact is not currently available."
            ),
        },
    })
'''.strip() + "\n\n"


def replace_route_block(text: str, route: str, replacement: str) -> str:
    pattern = re.compile(
        r'@app\.(?:get|post|put|delete)\("' + re.escape(route) + r'"\)\n'
        r'def\s+\w+\([^\n]*\):\n'
        r'.*?'
        r'(?=\n@app\.(?:get|post|put|delete)\(|\Z)',
        re.S,
    )
    new_text, count = pattern.subn(replacement, text, count=1)
    if count != 1:
        raise RuntimeError(f"Could not replace route block for {route}; found {count} matches")
    return new_text


def patch_main() -> None:
    s = read("main.py")
    s = replace_route_block(s, "/verify-certificate/{cid}", VERIFY_CERT_BLOCK)
    s = replace_route_block(s, "/job-status/{job_id}", JOB_STATUS_BLOCK)
    # Ensure re is available for verify_certificate_by_id extension inference.
    if "import re" not in s.split("\n")[:80]:
        s = re.sub(r'(import\s+os\s*\n)', r'\1import re\n', s, count=1)
    write("main.py", s)


def patch_worker() -> None:
    s = read("worker.py")
    s = re.sub(
        r"from\s+emailer\s+import\s+send_certification_email",
        "from notification_helper import send_certification_email_outbox_compat as send_certification_email",
        s,
    )
    write("worker.py", s)


def main() -> None:
    patch_main()
    patch_worker()
    print("Applied VeriFYD document lifecycle/certificate lookup patch.")
    print("Next: py -m py_compile main.py worker.py notification_helper.py emailer.py storage.py")


if __name__ == "__main__":
    main()
