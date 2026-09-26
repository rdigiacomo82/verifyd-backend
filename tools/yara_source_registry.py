from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "rules" / "sources.json"

if not REGISTRY.exists():
    raise SystemExit("rules/sources.json not found")

data = json.loads(REGISTRY.read_text(encoding="utf-8"))
sources = data.get("sources") or []

allowed = [s for s in sources if s.get("allowed_for_candidate_ingestion")]
hold = [s for s in sources if not s.get("allowed_for_candidate_ingestion")]

print("VeriFYD YARA source registry")
print(f"Candidate-approved sources: {len(allowed)}")
for s in allowed:
    print(f"  ALLOW  {s['id']}: {s['license']} - {s['repository']}")

print(f"Hold sources: {len(hold)}")
for s in hold:
    print(f"  HOLD   {s['id']}: {s['license']} - {s['repository']}")

bad = []
for s in sources:
    for field in ("id", "name", "repository", "license", "status"):
        if not str(s.get(field) or "").strip():
            bad.append(f"{s.get('id','<unknown>')}: missing {field}")

if bad:
    print("Errors:")
    for item in bad:
        print(" -", item)
    raise SystemExit(1)

raise SystemExit(0)
