from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from verifyd_engine.yara.promotion import rank_candidates

REPORT_DIR = ROOT / "yara_candidates" / "reports"
triage_path = REPORT_DIR / "phase2b4_triage_report.json"
compile_path = REPORT_DIR / "phase2b3_compile_report.json"

if not triage_path.exists() or not compile_path.exists():
    raise SystemExit("Required Phase 2B.3/2B.4 reports not found.")

triage = json.loads(triage_path.read_text(encoding="utf-8"))
compile_report = json.loads(compile_path.read_text(encoding="utf-8"))

ranked = rank_candidates(triage, compile_report)

summary = {
    "candidate_only": True,
    "production_enabled": False,
    "rule_count": len(ranked),
    "tier_counts": {
        tier: sum(1 for r in ranked if r.tier == tier)
        for tier in ("A", "B", "C", "HOLD")
    },
    "top_candidates": [r.to_dict() for r in ranked[:25]],
    "all_candidates": [r.to_dict() for r in ranked],
}

out = REPORT_DIR / "phase2b5_promotion_rank.json"
out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

print("VeriFYD Phase 2B.5 promotion ranking")
print(f"Rules ranked: {len(ranked)}")
for tier in ("A", "B", "C", "HOLD"):
    print(f"Tier {tier}: {summary['tier_counts'][tier]}")
print("Top 10 candidates:")
for item in ranked[:10]:
    print(f"  {item.score:3d}  {item.tier:4s}  {item.source_id}  {item.rule}")
print(f"Report: {out}")
print("Production enabled: NO")
