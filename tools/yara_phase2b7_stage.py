from __future__ import annotations

import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CAND = ROOT / "yara_candidates" / "sources"
RULES = ROOT / "rules"

selection = {
    "tenable_curated": {
        "source_id": "tenable_yara_rules",
        "source_name": "Tenable YARA Rules",
        "license": "BSD-3-Clause",
        "repo": "https://github.com/tenable/yara-rules.git",
        "files": [
            ("malware/cerber3.yar", "cerber3.yar"),
            ("malware/venom.yar", "venom.yar"),
        ],
    },
    "intezer_curated": {
        "source_id": "intezer_yara_rules",
        "source_name": "Intezer YARA Rules",
        "license": "MIT",
        "repo": "https://github.com/intezer/yara-rules.git",
        "files": [
            ("AgeLocker.yar", "AgeLocker.yar"),
            ("Doki_Attack.yar", "Doki_Attack.yar"),
        ],
    },
}

for pack_id, cfg in selection.items():
    src_root = CAND / cfg["source_id"]
    if not src_root.exists():
        raise SystemExit(f"Candidate source missing: {src_root}")

    dst = RULES / pack_id
    dst.mkdir(parents=True, exist_ok=True)

    for src_rel, dst_name in cfg["files"]:
        src = src_root / src_rel
        if not src.exists():
            raise SystemExit(f"Selected rule missing: {src}")
        shutil.copy2(src, dst / dst_name)

    # Preserve upstream license text in each source-specific pack.
    license_candidates = [
        src_root / "LICENSE",
        src_root / "LICENSE.md",
        src_root / "LICENSE.txt",
    ]
    license_src = next((p for p in license_candidates if p.exists()), None)
    if license_src:
        shutil.copy2(license_src, dst / "UPSTREAM_LICENSE.txt")

    manifest = {
        "schema_version": 1,
        "id": pack_id,
        "namespace": pack_id,
        "enabled": False,
        "trusted": False,
        "allow_includes": False,
        "version": "2026.09.26-rc1",
        "source": {
            "name": cfg["source_name"],
            "type": "curated_third_party",
            "url": cfg["repo"],
        },
        "license": {
            "id": cfg["license"],
            "redistribution": "Upstream license retained in UPSTREAM_LICENSE.txt",
        },
        "description": "Phase 2B.7 curated external YARA subset. Disabled pending final regression and promotion.",
        "categories": ["malware"],
        "default_severity": "high",
        "last_reviewed": "2026-09-26",
    }
    (dst / "pack.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

print("Phase 2B.7 curated packs staged.")
print("tenable_curated: cerber3.yar, venom.yar")
print("intezer_curated: AgeLocker.yar, Doki_Attack.yar")
print("Both packs remain ENABLED = false.")
