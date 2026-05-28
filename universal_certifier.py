# ============================================================
#  VeriFYD — universal_certifier.py
#
#  Phase 9A-1: Universal Certified File Package
#
#  Creates a signed .zip package for any supported document source file.
#  This complements the existing sealed certified PDF artifact without
#  changing the current PDF verification flow.
# ============================================================

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import zipfile
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Tuple, List

PACKAGE_VERSION = 1
PACKAGE_TYPE = "VERIFYD_UNIVERSAL_CERTIFIED_FILE"
SIGNATURE_ALGORITHM = "HMAC-SHA256"


def _seal_secret() -> str:
    """Return the same private seal secret used by the PDF secure-seal system."""
    return (
        os.environ.get("VERIFYD_SECRET_KEY")
        or os.environ.get("VERIFYD_SEAL_SECRET")
        or os.environ.get("DOCUMENT_SEAL_SECRET")
        or os.environ.get("ADMIN_KEY")
        or "verifyd-dev-seal-change-me"
    )


def _canonical_json(data: Dict[str, Any]) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _safe_name(name: str, fallback: str = "file") -> str:
    name = os.path.basename(str(name or "").replace("\\", "/")).strip()
    if not name or name in (".", ".."):
        return fallback
    return name.replace("\x00", "")[:180]


def _risk_report_from_detail(detail: Dict[str, Any] | None) -> Dict[str, Any]:
    detail = detail or {}
    report = detail.get("document_risk_report") or detail.get("risk_report") or {}
    return report if isinstance(report, dict) else {}


def _sign_package_payload(payload: Dict[str, Any]) -> Tuple[str, str]:
    payload_json = _canonical_json(payload)
    signature = hmac.new(
        _seal_secret().encode("utf-8"),
        payload_json.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    payload_b64 = base64.urlsafe_b64encode(payload_json.encode("utf-8")).decode("ascii")
    return payload_b64, signature


def create_universal_certified_package(
    *,
    original_path: str,
    certified_pdf_path: str,
    package_path: str,
    cert_id: str,
    original_filename: str,
    certified_to: str = "",
    label: str = "",
    authenticity: int | str = 0,
    ai_score: int | str = "",
    original_sha256: str = "",
    detail: Dict[str, Any] | None = None,
    extra_artifacts: Iterable[Dict[str, Any]] | None = None,
) -> str:
    """
    Create a universal VeriFYD certified file package.

    The package intentionally preserves the exact original file bytes and the
    exact issued certified PDF bytes. The signed seal covers both hashes.
    """
    if not os.path.exists(original_path):
        raise FileNotFoundError(f"Original file not found: {original_path}")
    if not os.path.exists(certified_pdf_path):
        raise FileNotFoundError(f"Certified PDF not found: {certified_pdf_path}")

    original_name = _safe_name(original_filename, fallback=f"original_{cert_id}")
    report_name = f"VeriFYD_Certified_Report_{cert_id[:8]}.pdf"

    original_hash = (original_sha256 or "").strip().lower() or _sha256_file(original_path)
    certified_pdf_hash = _sha256_file(certified_pdf_path)
    original_size = os.path.getsize(original_path)
    certified_pdf_size = os.path.getsize(certified_pdf_path)
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    risk_report = _risk_report_from_detail(detail)

    hashes = {
        "algorithm": "SHA-256",
        "original": {
            "path": f"original/{original_name}",
            "filename": original_name,
            "sha256": original_hash,
            "size_bytes": original_size,
        },
        "certified_report": {
            "path": f"certified/{report_name}",
            "filename": report_name,
            "sha256": certified_pdf_hash,
            "size_bytes": certified_pdf_size,
        },
    }

    # Optional Pro/Enterprise expanded ZIP child certification artifacts.
    # Each artifact is included in the package and covered by the signed hash manifest.
    normalized_extra_artifacts: List[Dict[str, Any]] = []
    for idx, artifact in enumerate(list(extra_artifacts or []), start=1):
        try:
            apath = str(artifact.get("path") or "")
            arcname = str(artifact.get("arcname") or "").replace("\", "/").lstrip("/")
            if not apath or not arcname or not os.path.exists(apath) or os.path.isdir(apath):
                continue
            if arcname.startswith("../") or "/../" in arcname:
                continue
            ahash = _sha256_file(apath)
            asize = os.path.getsize(apath)
            entry = {
                "path": arcname,
                "filename": os.path.basename(arcname),
                "sha256": ahash,
                "size_bytes": asize,
                "kind": artifact.get("kind", "extra_artifact"),
                "source_member": artifact.get("source_member", ""),
                "child_certificate_id": artifact.get("child_certificate_id", ""),
            }
            normalized_extra_artifacts.append({"source_path": apath, "arcname": arcname, "hash_entry": entry})
        except Exception:
            continue

    if normalized_extra_artifacts:
        hashes["extra_artifacts"] = [x["hash_entry"] for x in normalized_extra_artifacts]

    certificate = {
        "certificate_id": cert_id,
        "certified_to": certified_to or "",
        "issued_at_utc": now,
        "original_filename": original_name,
        "label": label or "",
        "authenticity": authenticity,
        "ai_score": ai_score,
        "overall_risk": risk_report.get("overall_risk", "UNKNOWN"),
        "risk_score": risk_report.get("risk_score", ""),
        "metadata_integrity": risk_report.get("metadata_integrity", "UNKNOWN"),
        "issuer": "VeriFYD",
        "package_type": PACKAGE_TYPE,
        "package_version": PACKAGE_VERSION,
    }

    seal_payload = {
        "package_type": PACKAGE_TYPE,
        "version": PACKAGE_VERSION,
        "certificate_id": cert_id,
        "issued_at_utc": now,
        "issuer": "VeriFYD",
        "original_filename": original_name,
        "original_sha256": original_hash,
        "certified_report_sha256": certified_pdf_hash,
        "hashes": hashes,
        "certificate": certificate,
    }
    payload_b64, signature = _sign_package_payload(seal_payload)

    seal = {
        "package_type": PACKAGE_TYPE,
        "seal_version": "VERIFYD-PACKAGE-SEAL-V1",
        "algorithm": SIGNATURE_ALGORITHM,
        "payload_b64": payload_b64,
        "signature": signature,
        "payload": seal_payload,
    }

    readme = f"""VeriFYD Universal Certified File Package

Certificate ID: {cert_id}
Issued At UTC: {now}
Original File: {original_name}
Certified To: {certified_to or ''}

This package contains:
- original/                 The exact uploaded source file bytes.
- certified/                The VeriFYD certified PDF report/rendering with secure seal.
- verifyd/                  Signed certificate metadata, hashes, seal, and signature.
- zip_contents_original/    Pro/Enterprise only: extracted original child files from a ZIP evidence package, if applicable.
- zip_contents_certified/   Pro/Enterprise only: certified child reports for supported files inside a ZIP evidence package, if applicable.

The package seal is signed with {SIGNATURE_ALGORITHM}. Verification should compare
all SHA-256 hashes and verify the HMAC signature before trusting the package.
"""

    os.makedirs(os.path.dirname(package_path) or ".", exist_ok=True)
    tmp_path = package_path + ".tmp"
    if os.path.exists(tmp_path):
        os.remove(tmp_path)

    with zipfile.ZipFile(tmp_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        zf.write(original_path, f"original/{original_name}")
        zf.write(certified_pdf_path, f"certified/{report_name}")
        for extra in normalized_extra_artifacts:
            zf.write(extra["source_path"], extra["arcname"])
        zf.writestr("verifyd/certificate.json", json.dumps(certificate, indent=2, sort_keys=True, ensure_ascii=False))
        zf.writestr("verifyd/hashes.json", json.dumps(hashes, indent=2, sort_keys=True, ensure_ascii=False))
        zf.writestr("verifyd/seal.json", json.dumps(seal, indent=2, sort_keys=True, ensure_ascii=False))
        zf.writestr("verifyd/signature.txt", signature + "\n")
        zf.writestr("verifyd/README.txt", readme)

    os.replace(tmp_path, package_path)
    return package_path
