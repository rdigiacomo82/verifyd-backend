from __future__ import annotations

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "rules" / "sources.json"
CANDIDATE_ROOT = ROOT / "yara_candidates" / "sources"

if not REGISTRY.exists():
    raise SystemExit("rules/sources.json not found")

data = json.loads(REGISTRY.read_text(encoding="utf-8"))
sources = [
    s for s in (data.get("sources") or [])
    if s.get("allowed_for_candidate_ingestion") is True
]

CANDIDATE_ROOT.mkdir(parents=True, exist_ok=True)

for src in sources:
    source_id = src["id"]
    url = src["repository"]
    target = CANDIDATE_ROOT / source_id

    if target.exists():
        print(f"Refreshing {source_id}...")
        subprocess.run(["git", "-C", str(target), "fetch", "--depth", "1", "origin"], check=True)
        subprocess.run(["git", "-C", str(target), "reset", "--hard", "origin/HEAD"], check=True)
    else:
        print(f"Cloning {source_id}...")
        subprocess.run(["git", "clone", "--depth", "1", url, str(target)], check=True)

print("Candidate source fetch complete.")
print(f"Staging root: {CANDIDATE_ROOT}")
print("These files are outside rules/ and are NOT loaded by production Lens.")
