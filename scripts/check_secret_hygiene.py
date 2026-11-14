#!/usr/bin/env python3
"""Fail CI on high-confidence credentials mistakenly tracked by Git.
Never print credential values. This is not a host-level malware scanner.
"""
from __future__ import annotations
import re
import subprocess
from pathlib import Path

BAD_NAMES = (
    re.compile(r"(^|/)(\.env($|\.)|id_rsa$|id_ed25519$|credentials\.json$|service-account[^/]*\.json$)", re.I),
    re.compile(r"\.(p12|pfx|key|jks|keystore)$", re.I),
)
TOKEN_RULES = {
    "private_key": re.compile(rb"-----BEGIN (?:[A-Z0-9 ]+ )?PRIVATE KEY-----"),
    "aws_access_key": re.compile(rb"(?<![A-Z0-9])(?:AKIA|ASIA)[0-9A-Z]{16}(?![A-Z0-9])"),
    "github_token": re.compile(rb"\bgh(?:p|o|u|s|r)_[A-Za-z0-9_]{30,}\b"),
    "github_fine_grained_token": re.compile(rb"\bgithub_pat_[A-Za-z0-9_]{40,}\b"),
    "stripe_secret_key": re.compile(rb"\bsk_live_[A-Za-z0-9]{20,}\b"),
    "admin_key_in_url": re.compile(rb"/admin[A-Za-z0-9_/-]*\?key=(?!REDACTED|YOUR_KEY|EXAMPLE)[A-Za-z0-9%#_-]{8,}", re.I),
}

def classify(name: str, data: bytes) -> list[str]:
    normalized = name.replace("\\", "/")
    reasons = []
    example = normalized.endswith((".env.example", ".env.sample", ".env.template"))
    if not example and any(rule.search(normalized) for rule in BAD_NAMES):
        reasons.append("sensitive_filename")
    if b"\x00" in data[:1024]:
        return reasons
    for label, rule in TOKEN_RULES.items():
        if rule.search(data):
            reasons.append(label)
    return reasons

def main() -> int:
    root = Path(__file__).resolve().parents[1]
    names = subprocess.run(["git", "ls-files", "-z"], cwd=root,
                           capture_output=True, check=True).stdout.split(b"\x00")
    failures = []
    checked = 0
    for encoded in names:
        if not encoded:
            continue
        name = encoded.decode("utf-8", "replace")
        file = root / name
        if not file.is_file():
            continue
        checked += 1
        if file.stat().st_size > 5 * 1024 * 1024:
            continue
        reasons = classify(name, file.read_bytes())
        if reasons:
            failures.append((name, reasons))
    print(f"Scanned {checked} tracked paths for high-confidence secret leaks.")
    for path, reasons in failures:
        print(f"BLOCK: {path} ({', '.join(reasons)}). Secret values suppressed.")
    if failures:
        print("Remove tracked secrets, inspect Git history and rotate any exposed credentials.")
        return 1
    print("No tracked sensitive filenames or high-confidence key tokens detected.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
