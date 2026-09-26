from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "yara_candidates" / "reports"

triage_path = REPORT_DIR / "phase2b4_triage_report.json"
compile_path = REPORT_DIR / "phase2b3_compile_report.json"
rank_path = REPORT_DIR / "phase2b5_promotion_rank.json"

for p in (triage_path, compile_path, rank_path):
    if not p.exists():
        raise SystemExit(f"Missing required report: {p}")

triage = json.loads(triage_path.read_text(encoding="utf-8"))
compile_report = json.loads(compile_path.read_text(encoding="utf-8"))
rank = json.loads(rank_path.read_text(encoding="utf-8"))

compile_by_path = {
    str(x.get("path")): x
    for x in (compile_report.get("results") or [])
}
scores_by_file = defaultdict(list)
for item in rank.get("all_candidates") or []:
    scores_by_file[str(item.get("file"))].append(item)

rows = []

for file_path, items in scores_by_file.items():
    path = Path(file_path)
    source_id = str(items[0].get("source_id") or "")
    text = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""

    rule_names = [str(x.get("rule") or "") for x in items]
    compile_info = compile_by_path.get(file_path) or {}
    warnings = compile_info.get("warnings") or []

    private_rules = len(re.findall(r"(?m)^\s*private\s+rule\s+", text))
    global_rules = len(re.findall(r"(?m)^\s*global\s+rule\s+", text))
    vaccine_rules = [n for n in rule_names if "vaccine" in n.lower()]
    helper_like = [n for n in rule_names if any(k in n.lower() for k in ("helper", "generic", "common"))]

    avg_score = round(sum(int(x.get("score") or 0) for x in items) / max(1, len(items)), 1)
    min_score = min(int(x.get("score") or 0) for x in items)
    max_score = max(int(x.get("score") or 0) for x in items)

    score = int(round(avg_score))
    reasons = []
    flags = []

    if compile_info.get("compiled") is True:
        reasons.append("file compiles with YARA-X")
        score += 5
    else:
        flags.append("BLOCK: compile failed")
        score = 0

    if not warnings:
        reasons.append("no file compile warnings")
        score += 3
    else:
        flags.append(f"compile warnings: {len(warnings)}")
        score -= min(10, 2 * len(warnings))

    if private_rules:
        reasons.append(f"contains {private_rules} private helper rule(s); promote whole file")
    if global_rules:
        reasons.append(f"contains {global_rules} global rule(s); requires careful regression review")
        score -= 5

    if vaccine_rules:
        flags.append(f"automated Vaccine rules present: {len(vaccine_rules)}")
        score -= 12

    if len(items) > 20:
        flags.append(f"large rule file: {len(items)} rules")
        score -= 5

    score = max(0, min(100, score))

    if any(x.startswith("BLOCK:") for x in flags):
        recommendation = "HOLD"
    elif vaccine_rules:
        recommendation = "REVIEW"
    elif score >= 90 and len(warnings) == 0:
        recommendation = "SHORTLIST"
    elif score >= 80:
        recommendation = "REVIEW"
    else:
        recommendation = "HOLD"

    rows.append({
        "source_id": source_id,
        "file": file_path,
        "filename": path.name,
        "rule_count": len(items),
        "private_rule_count": private_rules,
        "global_rule_count": global_rules,
        "vaccine_rule_count": len(vaccine_rules),
        "average_rule_score": avg_score,
        "min_rule_score": min_score,
        "max_rule_score": max_score,
        "file_score": score,
        "recommendation": recommendation,
        "warnings": warnings,
        "flags": flags,
        "reasons": reasons,
        "rule_names": rule_names,
    })

rows.sort(key=lambda x: (
    {"SHORTLIST": 0, "REVIEW": 1, "HOLD": 2}.get(x["recommendation"], 3),
    -x["file_score"],
    x["source_id"],
    x["filename"].lower(),
))

report = {
    "candidate_only": True,
    "production_enabled": False,
    "file_count": len(rows),
    "shortlist_count": sum(1 for x in rows if x["recommendation"] == "SHORTLIST"),
    "review_count": sum(1 for x in rows if x["recommendation"] == "REVIEW"),
    "hold_count": sum(1 for x in rows if x["recommendation"] == "HOLD"),
    "files": rows,
}

out = REPORT_DIR / "phase2b6_file_promotion_rank.json"
out.write_text(json.dumps(report, indent=2), encoding="utf-8")

print("VeriFYD Phase 2B.6 file-level promotion review")
print(f"Files ranked: {report['file_count']}")
print(f"SHORTLIST:    {report['shortlist_count']}")
print(f"REVIEW:       {report['review_count']}")
print(f"HOLD:         {report['hold_count']}")
print("Top file candidates:")
for x in rows[:12]:
    print(
        f"  {x['file_score']:3d}  {x['recommendation']:9s}  "
        f"{x['source_id']:22s}  {x['filename']}  "
        f"rules={x['rule_count']} private={x['private_rule_count']} vaccine={x['vaccine_rule_count']}"
    )
print(f"Report: {out}")
print("Production enabled: NO")
