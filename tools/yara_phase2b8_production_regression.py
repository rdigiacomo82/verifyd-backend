from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import yara_x
except Exception as exc:
    raise SystemExit(f"yara_x import failed: {exc}")

from verifyd_engine.yara.loader import compile_rule_packs, discover_rule_packs

RULES_ROOT = ROOT / "rules"
REPORT_DIR = ROOT / "yara_candidates" / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

packs = discover_rule_packs(RULES_ROOT)
enabled_namespaces = [p.namespace for p in packs]

required = {"verifyd_core", "tenable_curated", "intezer_curated"}
missing = sorted(required - set(enabled_namespaces))
if missing:
    raise SystemExit(f"Required enabled production packs missing: {missing}")

bundle = compile_rule_packs(yara_x, RULES_ROOT, version="2026.09.26-phase2b8")
rules = bundle.rules

benign = {
    "plain_text.txt": b"This is a harmless VeriFYD production regression fixture.\r\n",
    "html.html": b"<!doctype html><html><body>Hello VeriFYD.</body></html>",
    "csv.csv": b"name,value\r\nalpha,1\r\n",
    "json.json": b'{"product":"VeriFYD","test":"benign"}',
    "security_terms.txt": (
        b"PowerShell curl cron ransomware rootkit malware security incident response. "
        b"This is explanatory prose only."
    ),
}

# Harmless synthetic byte fixtures that exercise selected signatures without containing malware.
positive = {
    "age_locker_signature.txt": b"agelocker.go\nmain.encrypt\n",
    "venom_signature.txt": b"%%VENOM%AUTHENTICATE%%\n",
}


def scan_file(path: Path) -> list[dict]:
    scanner = yara_x.Scanner(rules)
    scanner.set_timeout(10)
    result = scanner.scan_file(str(path))
    matching_rules = getattr(result, "matching_rules", ())
    return [
        {
            "identifier": getattr(m, "identifier", None),
            "namespace": getattr(m, "namespace", None),
        }
        for m in matching_rules
    ]


benign_results = []
positive_results = []

with tempfile.TemporaryDirectory(prefix="verifyd-phase2b8-") as td:
    base = Path(td)

    for name, data in benign.items():
        p = base / name
        p.write_bytes(data)
        matches = scan_file(p)
        benign_results.append({
            "fixture": name,
            "match_count": len(matches),
            "matches": matches,
        })

    for name, data in positive.items():
        p = base / name
        p.write_bytes(data)
        matches = scan_file(p)
        positive_results.append({
            "fixture": name,
            "match_count": len(matches),
            "matches": matches,
        })

benign_match_count = sum(x["match_count"] for x in benign_results)
positive_match_count = sum(x["match_count"] for x in positive_results)

# Both synthetic fixtures should trigger at least one rule.
positive_ok = all(x["match_count"] >= 1 for x in positive_results)
passed = benign_match_count == 0 and positive_ok

report = {
    "production_enabled": True,
    "enabled_namespaces": enabled_namespaces,
    "ruleset": {
        "version": bundle.info.version,
        "sha256": bundle.info.sha256,
        "rules_loaded": bundle.info.rules_loaded,
        "rule_files_loaded": bundle.info.rule_files_loaded,
        "compile_warnings": list(bundle.info.compile_warnings),
    },
    "benign_results": benign_results,
    "positive_results": positive_results,
    "benign_match_count": benign_match_count,
    "positive_fixture_pass": positive_ok,
    "pass": passed,
}

out = REPORT_DIR / "phase2b8_production_regression.json"
out.write_text(json.dumps(report, indent=2), encoding="utf-8")

print("VeriFYD Phase 2B.8 production ruleset regression")
print(f"Enabled namespaces:    {', '.join(enabled_namespaces)}")
print(f"Rules loaded:          {bundle.info.rules_loaded}")
print(f"Rule files loaded:     {bundle.info.rule_files_loaded}")
print(f"Compile warnings:      {len(bundle.info.compile_warnings)}")
print(f"Benign matches:        {benign_match_count}")
print(f"Positive fixtures:     {'PASS' if positive_ok else 'FAIL'}")
for row in positive_results:
    rendered = ", ".join(
        f"{m.get('namespace')}:{m.get('identifier')}" for m in row["matches"]
    ) or "NO MATCH"
    print(f"  {row['fixture']}: {rendered}")
print(f"Regression result:     {'PASS' if passed else 'FAIL'}")
print(f"Report:                {out}")

if not passed:
    print()
    print("Regression failed. Run:")
    print(r"python tools\yara_phase2b8_disable.py")
    raise SystemExit(1)
