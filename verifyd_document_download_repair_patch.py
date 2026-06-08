from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent


def read(name: str) -> str:
    return (ROOT / name).read_text(encoding="utf-8", errors="replace")


def write(name: str, text: str) -> None:
    (ROOT / name).write_text(text, encoding="utf-8", newline="\n")


VERIFY_CERT_ROUTE = r'''@app.get("/verify-certificate/{cid}")
def verify_certificate_by_id(cid: str):
    """Verify a certificate ID against the VeriFYD certificate database and certified artifacts."""
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

    original_file = str(cert.get("original_file", "") or "")
    original_file_lower = original_file.lower()
    is_document_like = original_file_lower.endswith((
        ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
        ".odt", ".ods", ".odp", ".txt", ".md", ".csv", ".rtf",
        ".eml", ".msg", ".html", ".htm", ".mhtml", ".mht", ".xml",
        ".json", ".svg", ".vsdx", ".yaml", ".yml", ".toml", ".env",
        ".ini", ".properties", ".conf", ".cfg", ".config", ".cnf", ".log",
        ".sql", ".pst", ".ost", ".dwg", ".dxf", ".zip"
    ))
    is_zip_evidence = original_file_lower.endswith(".zip")

    certified_document_available = False
    certified_file_package_available = False

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
    except Exception:
        certified_document_available = False
        certified_file_package_available = False

    try:
        r = _get_redis()
        if not certified_document_available:
            certified_document_available = bool(r.exists(f"doccert:{cid}"))
        if not certified_file_package_available:
            certified_file_package_available = bool(r.exists(f"filecert:{cid}"))
    except Exception:
        pass

    document_url = f"{BASE_URL}/download-document/{cid}" if certified_document_available else ""
    package_url = f"{BASE_URL}/download-certified-file/{cid}" if certified_file_package_available else ""

    # Primary download rule:
    # - normal single-file documents use the certified PDF as primary
    # - ZIP evidence packages use the ZIP package as primary only when the original upload was .zip
    download_url = ""
    download_type = ""
    if is_zip_evidence and certified_file_package_available:
        download_url = package_url
        download_type = "certified_file_package_zip"
    elif certified_document_available:
        download_url = document_url
        download_type = "certified_document"
    elif certified_file_package_available:
        download_url = package_url
        download_type = "certified_file_package"

    media_type = cert.get("media_type") or ("document" if is_document_like or certified_document_available or certified_file_package_available else "")

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
        "certified_to": cert.get("email", ""),
        "email": cert.get("email", ""),
        "media_type": media_type,
        "download_type": download_type,
        "download_url": download_url,
        "document_ready": bool(certified_document_available or certified_file_package_available),
        "certification_status": "ready" if (certified_document_available or certified_file_package_available) else "record_only",
        "certified_document_available": bool(certified_document_available),
        "certified_document_url": document_url,
        "certified_file_package_available": bool(certified_file_package_available),
        "certified_file_package_url": package_url,
        "verification_report": {
            "title": "VeriFYD Certificate Verification Report",
            "status": "FOUND",
            "certificate_id": cid,
            "database_match": "YES",
            "certified_document_available": "YES" if certified_document_available else "NO",
            "certified_file_package_available": "YES" if certified_file_package_available else "NO",
            "trust_level": "CERTIFIED ARTIFACT FOUND" if (certified_document_available or certified_file_package_available) else "DATABASE RECORD FOUND",
            "message": "This certificate ID exists in the VeriFYD certificate database.",
        },
    })
'''


def replace_route_block(text: str, route: str, replacement: str) -> str:
    pattern = re.compile(
        r'@app\.(?:get|post)\(["\']' + re.escape(route) + r'["\']\)\n'
        r'def\s+\w+\([^\)]*\):\n'
        r'(?:    .*\n|\n)*?'
        r'(?=\n@app\.|\ndef\s|\nclass\s|\Z)',
        re.MULTILINE,
    )
    new_text, count = pattern.subn(replacement.rstrip() + "\n", text, count=1)
    if count != 1:
        raise RuntimeError(f"Could not replace route block for {route}; found {count} matches")
    return new_text


def patch_main() -> None:
    s = read("main.py")
    s = replace_route_block(s, "/verify-certificate/{cid}", VERIFY_CERT_ROUTE)
    write("main.py", s)


def patch_worker() -> None:
    s = read("worker.py")

    # Ensure _cert_ttl is available before R2 upload/remove, so we can mirror to Redis even when R2 succeeds.
    marker = '''                    _stored_in_r2 = False
                    _package_stored_in_r2 = False
                    try:
                        from storage import upload_certified_document, upload_certified_file_package, r2_available
'''
    replacement = '''                    _cert_ttl = {"free": 86400, "creator": 259200, "pro": 604800, "enterprise": 2592000}.get(_plan, 86400)
                    _stored_in_r2 = False
                    _package_stored_in_r2 = False
                    try:
                        from storage import upload_certified_document, upload_certified_file_package, r2_available
'''
    if marker in s and replacement not in s:
        s = s.replace(marker, replacement, 1)

    old_doc = '''                            upload_certified_document(job_id, certified_path, _plan)
                            _os.remove(certified_path)
                            _stored_in_r2 = True
                            log.info("Worker: certified document stored in R2: job=%s plan=%s", job_id, _plan)
'''
    new_doc = '''                            # Mirror the certified PDF in Redis even when R2 succeeds.
                            # This protects downloads if the web service cannot HEAD the R2 object immediately.
                            try:
                                if _os.path.exists(certified_path):
                                    with open(certified_path, "rb") as df:
                                        r.setex(f"doccert:{job_id}", _cert_ttl, df.read())
                            except Exception as _mirror_e:
                                log.warning("Worker: certified document Redis mirror failed for %s: %s", job_id, _mirror_e)

                            upload_certified_document(job_id, certified_path, _plan)
                            _os.remove(certified_path)
                            _stored_in_r2 = True
                            log.info("Worker: certified document stored in R2: job=%s plan=%s", job_id, _plan)
'''
    if old_doc in s and "certified document Redis mirror failed" not in s:
        s = s.replace(old_doc, new_doc, 1)

    old_pkg = '''                                package_key = upload_certified_file_package(job_id, package_path, _plan)
                                _os.remove(package_path)
                                _package_stored_in_r2 = True
'''
    new_pkg = '''                                # Mirror the universal package in Redis as a fallback, but do not make it primary for normal PDFs.
                                try:
                                    if _os.path.exists(package_path):
                                        with open(package_path, "rb") as pf:
                                            r.setex(f"filecert:{job_id}", _cert_ttl, pf.read())
                                except Exception as _pkg_mirror_e:
                                    log.warning("Worker: certified file package Redis mirror failed for %s: %s", job_id, _pkg_mirror_e)

                                package_key = upload_certified_file_package(job_id, package_path, _plan)
                                _os.remove(package_path)
                                _package_stored_in_r2 = True
'''
    if old_pkg in s and "certified file package Redis mirror failed" not in s:
        s = s.replace(old_pkg, new_pkg, 1)

    # Avoid duplicate _cert_ttl line after the R2 block if the patch inserted the earlier one.
    s = s.replace(
        '''
                    _cert_ttl = {"free": 86400, "creator": 259200, "pro": 604800, "enterprise": 2592000}.get(_plan, 86400)

                    if not _stored_in_r2 and _os.path.exists(certified_path):
''',
        '''
                    if not _stored_in_r2 and _os.path.exists(certified_path):
''',
        1,
    )

    # Make normal documents explicit. This helps frontend pick Download Certified Document.
    old_result = '''                    else:
                        result["download_url"] = download_url
                        if result.get("universal_certified_file") in ("created", "stored"):
                            result["certified_file_package_url"] = certified_file_package_url
'''
    new_result = '''                    else:
                        result["download_url"] = download_url
                        result["download_type"] = "certified_document"
                        result["primary_download_type"] = "certified_document"
                        if result.get("universal_certified_file") in ("created", "stored"):
                            result["certified_file_package_url"] = certified_file_package_url
'''
    if old_result in s and 'result["primary_download_type"] = "certified_document"' not in s:
        s = s.replace(old_result, new_result, 1)

    write("worker.py", s)


def main() -> None:
    patch_main()
    patch_worker()
    print("Applied VeriFYD document download repair patch.")
    print("Next: py -m py_compile main.py worker.py notification_helper.py emailer.py storage.py")


if __name__ == "__main__":
    main()
