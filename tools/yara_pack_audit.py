from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from verifyd_engine.yara.pack_manager import inventory_rule_packs

report = inventory_rule_packs(ROOT / "rules")
print(json.dumps(report, indent=2))
raise SystemExit(0 if report.get("ok") else 1)
