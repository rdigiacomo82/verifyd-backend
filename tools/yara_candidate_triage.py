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

from verifyd_engine.yara.candidate_triage import combined_compile, triage_sources

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

triage = triage_sources(sources)
combined = combined_compile(yara_x, sources)

report = {
    "triage": triage,
    "combined_compile": {
        k: v for k, v in combined.items() if k != "rules_object"
    },
}

path = REPORT_DIR / "phase2b4_triage_report.json"
path.write_text(json.dumps(report, indent=2), encoding="utf-8")

print("VeriFYD Phase 2B.4 candidate triage")
print(f"Sources:               {triage['source_count']}")
print(f"Rules discovered:      {triage['rule_count']}")
print(f"Duplicate rule names:  {triage['duplicate_rule_name_count']}")
print(f"Missing author:         {triage['metadata_quality']['missing_author']}")
print(f"Missing description:    {triage['metadata_quality']['missing_description']}")
print(f"Missing reference:      {triage['metadata_quality']['missing_reference']}")
print(f"Missing date:           {triage['metadata_quality']['missing_date']}")
print(f"Combined compile:       {'PASS' if combined['ok'] else 'FAIL'}")
print(f"Combined files:         {combined['files']}")
print(f"Compile warnings:       {len(combined['warnings'])}")
print(f"Report:                 {path}")
print("Production enabled:     NO")

raise SystemExit(0)
