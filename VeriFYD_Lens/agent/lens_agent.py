from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from urllib.parse import urlparse, unquote
from pathlib import Path
from typing import Optional
import hashlib
import ipaddress
import mimetypes
import os
import re
import shutil
import socket
import subprocess
import time
import uuid

import httpx

app = FastAPI(title="VeriFYD Lens Agent", version="0.4.0")

BASE = Path.home() / "VeriFYD" / "Lens"
QUARANTINE = BASE / "Quarantine"
DOWNLOADS = Path.home() / "Downloads"
QUARANTINE.mkdir(parents=True, exist_ok=True)
DOWNLOADS.mkdir(parents=True, exist_ok=True)

MAX_DOWNLOAD_BYTES = 250 * 1024 * 1024
SCAN_STATE: dict[str, dict] = {}

CLOUD_URL = os.environ.get(
    "VERIFYD_LENS_CLOUD_URL",
    "https://verifyd-backend.onrender.com"
).rstrip("/")
CLOUD_KEY = os.environ.get("VERIFYD_LENS_API_KEY", "").strip()
CLOUD_POLL_SECONDS = int(os.environ.get("VERIFYD_LENS_POLL_SECONDS", "3"))
CLOUD_TIMEOUT_SECONDS = int(os.environ.get("VERIFYD_LENS_CLOUD_TIMEOUT", "240"))

class UrlScanRequest(BaseModel):
    url: str

def clamp(v: int) -> int:
    return max(0, min(100, v))

def classify(score: int) -> str:
    if score >= 80:
        return "LOW CONCERN"
    if score >= 50:
        return "REVIEW RECOMMENDED"
    return "HIGH CONCERN"

def safe_filename(name: str) -> str:
    name = unquote(name or "").strip().replace("\\", "_").replace("/", "_")
    name = re.sub(r'[<>:"|?*\x00-\x1f]', "_", name)
    name = name.rstrip(". ")
    return (name[:180] or "download.bin")

def filename_from_url(url: str) -> str:
    p = urlparse(url)
    name = Path(unquote(p.path)).name
    return safe_filename(name or "download.bin")

def resolve_public_host(host: str) -> None:
    if not host:
        raise HTTPException(status_code=400, detail="URL has no hostname.")
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror:
        raise HTTPException(status_code=400, detail="Could not resolve source hostname.")
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if (
            ip.is_private or ip.is_loopback or ip.is_link_local or
            ip.is_multicast or ip.is_reserved or ip.is_unspecified
        ):
            raise HTTPException(status_code=400, detail="Private/local network download targets are blocked.")

def url_findings(url: str) -> tuple[int, list[str]]:
    p = urlparse(url)
    if p.scheme not in {"http", "https"}:
        raise HTTPException(status_code=400, detail="Only HTTP/HTTPS URLs are supported.")
    resolve_public_host(p.hostname or "")
    host = (p.hostname or "").lower()
    path = p.path.lower()
    findings = []
    score = 90
    if p.scheme != "https":
        score -= 12
        findings.append("Source uses unencrypted HTTP")
    if any(host.endswith(t) for t in {".zip", ".mov", ".top", ".click", ".xyz"}):
        score -= 18
        findings.append("Source domain deserves additional review")
    if any(path.endswith(x) for x in [
        ".pdf.exe", ".docx.exe", ".xlsx.exe", ".pptx.exe",
        ".jpg.exe", ".jpeg.exe", ".png.exe", ".txt.exe"
    ]):
        score -= 55
        findings.append("Suspicious double file extension detected")
    if not findings:
        findings.append("No obvious URL-level warning detected")
    return clamp(score), findings

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def find_mpcmdrun() -> Optional[Path]:
    program_data = Path(os.environ.get("ProgramData", r"C:\ProgramData"))
    platform = program_data / "Microsoft" / "Windows Defender" / "Platform"
    if platform.exists():
        candidates = sorted(
            [p / "MpCmdRun.exe" for p in platform.iterdir() if (p / "MpCmdRun.exe").exists()],
            reverse=True
        )
        if candidates:
            return candidates[0]
    legacy = Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Windows Defender" / "MpCmdRun.exe"
    return legacy if legacy.exists() else None

def defender_scan(path: Path) -> dict:
    exe = find_mpcmdrun()
    if not exe:
        return {"status": "UNAVAILABLE", "score_delta": -5,
                "finding": "Microsoft Defender command-line scanner was not found"}
    try:
        cp = subprocess.run(
            [str(exe), "-Scan", "-ScanType", "3", "-File", str(path), "-DisableRemediation", "-ReturnHR"],
            capture_output=True, text=True, timeout=180
        )
        output = ((cp.stdout or "") + "\n" + (cp.stderr or "")).strip()
        low = output.lower()

        if any(x in low for x in ["threat found", "threat detected", "malware detected"]):
            return {"status": "THREAT_OR_WARNING", "score_delta": -75,
                    "finding": "Microsoft Defender reported a threat or security warning",
                    "detail": output[-800:]}
        if cp.returncode == 0:
            return {"status": "NO_KNOWN_THREAT_REPORTED", "score_delta": 0,
                    "finding": "Microsoft Defender reported no known threat in this scan",
                    "detail": output[-800:]}
        return {"status": "SCAN_ERROR", "score_delta": -10,
                "finding": f"Microsoft Defender scan returned HRESULT/exit value {cp.returncode}",
                "detail": output[-800:]}
    except subprocess.TimeoutExpired:
        return {"status": "TIMEOUT", "score_delta": -10, "finding": "Microsoft Defender scan timed out"}
    except Exception as e:
        return {"status": "ERROR", "score_delta": -10,
                "finding": f"Microsoft Defender scan error: {type(e).__name__}"}

def content_type_check(filename: str, header_content_type: str) -> list[str]:
    findings = []
    expected, _ = mimetypes.guess_type(filename)
    actual = (header_content_type or "").split(";")[0].strip().lower()
    if expected and actual and actual not in {"application/octet-stream", expected.lower()}:
        if not (expected.startswith("text/") and actual.startswith("text/")):
            findings.append(f"File type mismatch: extension suggests {expected}, server reported {actual}")
    return findings

def unique_destination(directory: Path, filename: str) -> Path:
    candidate = directory / filename
    if not candidate.exists():
        return candidate
    stem, suffix = candidate.stem, candidate.suffix
    for i in range(1, 1000):
        c = directory / f"{stem} ({i}){suffix}"
        if not c.exists():
            return c
    raise HTTPException(status_code=500, detail="Could not allocate a unique destination filename.")

def cloud_analyze(path: Path, filename: str) -> dict:
    if not CLOUD_KEY:
        return {
            "status": "NOT_CONFIGURED",
            "finding": "VeriFYD cloud authenticity analysis is not configured on this Lens installation.",
            "authenticity_score": None,
            "label": None,
        }

    headers = {"X-VeriFYD-Lens-Key": CLOUD_KEY}
    try:
        with path.open("rb") as fh:
            response = httpx.post(
                f"{CLOUD_URL}/lens/analyze",
                headers=headers,
                files={"file": (filename, fh, "application/octet-stream")},
                timeout=120,
            )
        data = response.json()
        if response.status_code >= 400:
            return {
                "status": "ERROR",
                "finding": f"VeriFYD cloud upload failed ({response.status_code})",
                "authenticity_score": None,
                "label": None,
                "detail": data.get("detail") if isinstance(data, dict) else None,
            }

        job_id = data.get("job_id")
        media_type = data.get("media_type")
        if not job_id:
            return {"status": "ERROR", "finding": "VeriFYD cloud did not return a job ID.",
                    "authenticity_score": None, "label": None}

        deadline = time.time() + CLOUD_TIMEOUT_SECONDS
        while time.time() < deadline:
            time.sleep(CLOUD_POLL_SECONDS)
            r = httpx.get(
                f"{CLOUD_URL}/lens/job/{job_id}",
                headers=headers,
                timeout=30,
            )
            if r.status_code == 404:
                continue
            result = r.json()
            lifecycle = str(result.get("job_status") or "").lower()
            if lifecycle in {"queued", "processing", "started", "busy"}:
                continue
            if lifecycle == "error":
                return {
                    "status": "ERROR",
                    "finding": "VeriFYD cloud analysis failed.",
                    "authenticity_score": None,
                    "label": None,
                    "media_type": media_type,
                }

            authenticity = result.get("authenticity_score")
            label = result.get("label")
            reasoning = result.get("gpt_reasoning") or ""
            return {
                "status": "COMPLETE",
                "finding": f"VeriFYD authenticity analysis complete ({media_type or 'file'})",
                "authenticity_score": authenticity,
                "label": label,
                "media_type": media_type,
                "reasoning": reasoning,
                "gpt_flags": result.get("gpt_flags") or [],
                "ai_score": result.get("ai_score"),
            }

        return {
            "status": "TIMEOUT",
            "finding": "VeriFYD cloud authenticity analysis timed out.",
            "authenticity_score": None,
            "label": None,
            "media_type": media_type,
        }
    except Exception as e:
        return {
            "status": "ERROR",
            "finding": f"VeriFYD cloud connection error: {type(e).__name__}",
            "authenticity_score": None,
            "label": None,
        }

@app.get("/health")
def health():
    return {
        "ok": True,
        "product": "VeriFYD Lens",
        "version": "0.4.0",
        "tagline": "Don't Download Blind.",
        "cloud_configured": bool(CLOUD_KEY),
        "cloud_url": CLOUD_URL,
    }

@app.post("/scan-url")
def scan_url(req: UrlScanRequest):
    score, findings = url_findings(req.url)
    return {"status": "COMPLETE", "summary": classify(score), "security_score": score,
            "authenticity_score": None, "trust_score": score, "findings": findings, "source": req.url}

@app.post("/download-and-scan")
def download_and_scan(req: UrlScanRequest):
    score, findings = url_findings(req.url)
    scan_id = str(uuid.uuid4())
    filename = filename_from_url(req.url)
    quarantine_path = unique_destination(QUARANTINE, f"{scan_id[:8]}_{filename}")

    try:
        with httpx.stream(
            "GET", req.url,
            headers={"User-Agent": "VeriFYD-Lens/0.4", "Accept": "*/*"},
            follow_redirects=True,
            timeout=httpx.Timeout(30.0, read=120.0)
        ) as response:
            response.raise_for_status()
            resolve_public_host(urlparse(str(response.url)).hostname or "")
            content_length = response.headers.get("content-length")
            if content_length and int(content_length) > MAX_DOWNLOAD_BYTES:
                raise HTTPException(status_code=413, detail="File exceeds the 250 MB MVP limit.")
            total = 0
            with quarantine_path.open("wb") as f:
                for chunk in response.iter_bytes(1024 * 1024):
                    total += len(chunk)
                    if total > MAX_DOWNLOAD_BYTES:
                        quarantine_path.unlink(missing_ok=True)
                        raise HTTPException(status_code=413, detail="File exceeds the 250 MB MVP limit.")
                    f.write(chunk)
            findings.extend(content_type_check(filename, response.headers.get("content-type", "")))
            if any("File type mismatch" in x for x in findings):
                score -= 18
    except HTTPException:
        quarantine_path.unlink(missing_ok=True)
        raise
    except Exception as e:
        quarantine_path.unlink(missing_ok=True)
        raise HTTPException(status_code=502, detail=f"Download failed: {type(e).__name__}: {e}")

    sha = sha256_file(quarantine_path)
    findings.append(f"SHA-256 fingerprint created: {sha[:16]}…")

    defender = defender_scan(quarantine_path)
    score += defender["score_delta"]
    findings.append(defender["finding"])

    cloud = cloud_analyze(quarantine_path, filename)
    findings.append(cloud["finding"])

    authenticity = cloud.get("authenticity_score")
    label = cloud.get("label")
    if label:
        findings.append(f"VeriFYD authenticity verdict: {label}")
    if isinstance(authenticity, (int, float)):
        findings.append(f"VeriFYD authenticity score: {int(round(authenticity))}/100")

    score = clamp(score)

    record = {
        "scan_id": scan_id,
        "status": "QUARANTINED",
        "summary": classify(score),
        "security_score": score,
        "authenticity_score": authenticity,
        # Trust remains source/security trust; AI-generated does not automatically mean unsafe.
        "trust_score": score,
        "authenticity_label": label,
        "authenticity_reasoning": cloud.get("reasoning") or "",
        "cloud_status": cloud.get("status"),
        "media_type": cloud.get("media_type"),
        "findings": findings,
        "source": req.url,
        "filename": filename,
        "quarantine_path": str(quarantine_path),
        "sha256": sha,
        "size_bytes": quarantine_path.stat().st_size,
        "defender_status": defender["status"],
        "recommended_action": "release" if score >= 80 else "review" if score >= 50 else "block"
    }
    SCAN_STATE[scan_id] = record
    return record

@app.get("/scan/{scan_id}")
def get_scan(scan_id: str):
    rec = SCAN_STATE.get(scan_id)
    if not rec:
        raise HTTPException(status_code=404, detail="Scan not found in this agent session.")
    return rec

@app.post("/release/{scan_id}")
def release(scan_id: str):
    rec = SCAN_STATE.get(scan_id)
    if not rec:
        raise HTTPException(status_code=404, detail="Scan not found.")
    if rec["status"] != "QUARANTINED":
        raise HTTPException(status_code=409, detail=f"File is not quarantined (status={rec['status']}).")
    src = Path(rec["quarantine_path"])
    if not src.exists():
        raise HTTPException(status_code=404, detail="Quarantined file no longer exists.")
    dest = unique_destination(DOWNLOADS, rec["filename"])
    shutil.move(str(src), str(dest))
    rec["status"] = "RELEASED"
    rec["released_path"] = str(dest)
    return {"ok": True, "status": "RELEASED", "filename": rec["filename"],
            "released_path": str(dest), "sha256": rec["sha256"]}

@app.post("/delete/{scan_id}")
def delete(scan_id: str):
    rec = SCAN_STATE.get(scan_id)
    if not rec:
        raise HTTPException(status_code=404, detail="Scan not found.")
    Path(rec["quarantine_path"]).unlink(missing_ok=True)
    rec["status"] = "DELETED"
    return {"ok": True, "status": "DELETED", "filename": rec["filename"]}
