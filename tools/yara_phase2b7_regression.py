from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import yara_x
except Exception as exc:
    raise SystemExit(f"yara_x import failed: {exc}")

PACKS = [
    ROOT / "rules" / "tenable_curated",
    ROOT / "rules" / "intezer_curated",
]

compiler = yara_x.Compiler()
if hasattr(compiler, "max_warnings"):
    compiler.max_warnings(100)
compiler.enable_includes(False)

file_count = 0
for pack in PACKS:
    if not pack.exists():
        raise SystemExit(f"Missing staged pack: {pack}")
    compiler.new_namespace(pack.name)
    for path in sorted(set(pack.glob("*.yar")) | set(pack.glob("*.yara"))):
        compiler.add_source(path.read_text(encoding="utf-8", errors="replace"), origin=str(path))
        file_count += 1

warnings = tuple(str(w) for w in (compiler.warnings() if hasattr(compiler, "warnings") else ()))
rules = compiler.build()

fixtures = {
    "plain_text.txt": b"This is a harmless VeriFYD benign regression fixture.\r\n",
    "html.html": b"<!doctype html><html><body>Hello VeriFYD.</body></html>",
    "csv.csv": b"name,value\r\nalpha,1\r\n",
    "json.json": b'{"product":"VeriFYD","test":"benign"}',
    "admin_terms.txt": b"PowerShell curl cron ransomware rootkit security analysis only.",
}

results = []
with tempfile.TemporaryDirectory(prefix="verifyd-phase2b7-") as td:
    base = Path(td)
    for name, data in fixtures.items():
        p = base / name
        p.write_bytes(data)
        scanner = yara_x.Scanner(rules)
        scanner.set_timeout(10)
        scan_results = scanner.scan_file(str(p))
        matching_rules = getattr(scan_results, "matching_rules", ())
        matches = [
            {
                "identifier": getattr(m, "identifier", None),
                "namespace": getattr(m, "namespace", None),
            }
            for m in matching_rules
        ]
        results.append({
            "fixture": name,
            "match_count": len(matches),
            "matches": matches,
        })

total_matches = sum(x["match_count"] for x in results)
report = {
    "candidate_only": True,
    "production_enabled": False,
    "staged_pack_count": len(PACKS),
    "file_count": file_count,
    "compile_warnings": list(warnings),
    "benign_fixture_count": len(results),
    "benign_match_count": total_matches,
    "benign_results": results,
    "pass": total_matches == 0,
}

out = ROOT / "yara_candidates" / "reports" / "phase2b7_final_regression.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(report, indent=2), encoding="utf-8")

print("VeriFYD Phase 2B.7 final regression")
print(f"Staged packs:       {len(PACKS)}")
print(f"Rule files:         {file_count}")
print(f"Compile warnings:   {len(warnings)}")
print(f"Benign fixtures:    {len(results)}")
print(f"Benign matches:     {total_matches}")
print(f"Regression result:  {'PASS' if report['pass'] else 'FAIL'}")
print(f"Report:             {out}")
print("Production enabled: NO")
