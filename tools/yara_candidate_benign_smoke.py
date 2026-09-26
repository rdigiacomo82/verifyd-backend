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

from verifyd_engine.yara.candidate_triage import combined_compile

CANDIDATE_ROOT = ROOT / "yara_candidates" / "sources"
REPORT_DIR = ROOT / "yara_candidates" / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

sources = {
    p.name: p
    for p in sorted(CANDIDATE_ROOT.iterdir())
    if p.is_dir() and (p / ".git").exists()
} if CANDIDATE_ROOT.exists() else {}

if not sources:
    raise SystemExit("No candidate sources found.")

combined = combined_compile(yara_x, sources)
rules = combined.get("rules_object")
if not combined["ok"] or rules is None:
    raise SystemExit(f"Combined candidate compile failed: {combined.get('error')}")

fixtures = {
    "plain_text.txt": b"This is a harmless VeriFYD benign test document.\r\nNo executable content.\r\n",
    "minimal_html.html": b"<!doctype html><html><body><h1>VeriFYD benign fixture</h1><p>Hello.</p></body></html>",
    "minimal_csv.csv": b"name,value\r\nalpha,1\r\nbeta,2\r\n",
    "json_document.json": b'{"product":"VeriFYD","fixture":"benign","enabled":true}',
    "powershell_word_only.txt": b"This document discusses PowerShell administratively but contains no script commands.",
}

results = []
with tempfile.TemporaryDirectory(prefix="verifyd-yara-benign-") as td:
    base = Path(td)
    for name, data in fixtures.items():
        path = base / name
        path.write_bytes(data)
        scanner = yara_x.Scanner(rules)
        scanner.set_timeout(10)
        scan_results = scanner.scan_file(str(path))
        matching_rules = getattr(scan_results, "matching_rules", ())
        rendered = []
        for match in matching_rules:
            rendered.append({
                "identifier": getattr(match, "identifier", None),
                "namespace": getattr(match, "namespace", None),
            })
        results.append({
            "fixture": name,
            "match_count": len(rendered),
            "matches": rendered,
        })

total_matches = sum(r["match_count"] for r in results)
report = {
    "candidate_only": True,
    "production_enabled": False,
    "fixture_count": len(results),
    "total_matches": total_matches,
    "results": results,
}

path = REPORT_DIR / "phase2b4_benign_smoke_report.json"
path.write_text(json.dumps(report, indent=2), encoding="utf-8")

print("VeriFYD Phase 2B.4 benign smoke test")
print(f"Fixtures:       {len(results)}")
print(f"Total matches:  {total_matches}")
for row in results:
    print(f"  {row['fixture']}: {row['match_count']}")
print(f"Report:         {path}")
print("Production enabled: NO")

raise SystemExit(0)
