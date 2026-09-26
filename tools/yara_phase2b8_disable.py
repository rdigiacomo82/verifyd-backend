from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKS = [
    ROOT / "rules" / "tenable_curated" / "pack.json",
    ROOT / "rules" / "intezer_curated" / "pack.json",
]

for path in PACKS:
    if not path.exists():
        continue
    data = json.loads(path.read_text(encoding="utf-8"))
    data["enabled"] = False
    data["trusted"] = False
    data["promotion_status"] = "disabled_rollback"
    data["promotion_note"] = "Disabled by Phase 2B.8 rollback utility."
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(f"Disabled {data.get('id', path.parent.name)}")

print("Phase 2B.8 rollback complete.")
