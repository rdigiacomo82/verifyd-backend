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
        raise SystemExit(f"Missing pack manifest: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    data["enabled"] = True
    # Third-party rules remain explicitly non-first-party even when enabled.
    data["trusted"] = False
    data["version"] = "2026.09.26"
    data["promotion_status"] = "production_enabled"
    data["promotion_note"] = "Enabled after Phase 2B.7 compile and benign regression PASS."
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(f"Enabled {data['id']}")

print("Phase 2B.8 manifests updated.")
print("External packs are enabled but remain trusted=false.")
