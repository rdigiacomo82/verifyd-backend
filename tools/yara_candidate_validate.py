from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import yara_x
except Exception as exc:
    raise SystemExit(f"yara_x import failed: {exc}")

from verifyd_engine.yara.candidate_validator import validate_sources

CANDIDATE_ROOT = ROOT / "yara_candidates" / "sources"
REPORT_DIR = ROOT / "yara_candidates" / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

sources = {
    p.name: p
    for p in sorted(CANDIDATE_ROOT.iterdir())
    if p.is_dir() and (p / ".git").exists()
} if CANDIDATE_ROOT.exists() else {}

if not sources:
    raise SystemExit("No candidate sources found. Run tools\\yara_candidate_fetch.py first.")

report = validate_sources(yara_x, sources)
report_path = REPORT_DIR / "phase2b3_compile_report.json"
report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

print("VeriFYD Phase 2B.3 candidate validation")
print(f"Sources:  {report['source_count']}")
print(f"Files:    {report['file_count']}")
print(f"Compiled: {report['compiled']}")
print(f"Failed:   {report['failed']}")
print(f"Report:   {report_path}")
print("Production enabled: NO")

raise SystemExit(0)
