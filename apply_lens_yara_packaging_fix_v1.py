#!/usr/bin/env python3
"""
VeriFYD Lens Phase 2A.3 — YARA packaged-rule path + Agent startup fix

Purpose:
  - Fix lens_yara.py so a PyInstaller onefile EXE can find bundled rules at sys._MEIPASS/rules.
  - Ensure lens_agent.py has a valid local uvicorn startup block.

This patch is surgical and safe to rerun.
It does not change endpoint AV, static security, cloud authenticity, release handling,
backend APIs, extension files, or installer delivery.
"""
from __future__ import annotations

from pathlib import Path
import shutil
import time

MARKER = "VERIFYD_LENS_YARA_PACKAGING_FIX_V1"
ROOT = Path.cwd()
AGENT = ROOT / "VeriFYD_Lens" / "agent" / "lens_agent.py"
YARA = ROOT / "VeriFYD_Lens" / "agent" / "lens_yara.py"

print(f"{MARKER}: project root = {ROOT}")

if not AGENT.exists():
    raise SystemExit(f"ERROR: lens_agent.py not found: {AGENT}")
if not YARA.exists():
    raise SystemExit(f"ERROR: lens_yara.py not found: {YARA}")

def backup(path: Path, label: str) -> Path:
    b = path.with_name(f"{path.name}.{label}_{int(time.time())}.bak")
    shutil.copy2(path, b)
    print(f"Backed up: {b}")
    return b

changed = []

# ---------------------------------------------------------------------------
# 1) Patch lens_yara.py for PyInstaller onefile bundled rules.
# ---------------------------------------------------------------------------
yt = YARA.read_text(encoding="utf-8")

if MARKER not in yt:
    y_new = yt

    if "import sys\n" not in y_new:
        if "import os\nimport time\n" in y_new:
            y_new = y_new.replace("import os\nimport time\n", "import os\nimport sys\nimport time\n", 1)
        elif "import os\r\nimport time\r\n" in y_new:
            y_new = y_new.replace("import os\r\nimport time\r\n", "import os\r\nimport sys\r\nimport time\r\n", 1)
        else:
            raise SystemExit("ERROR: Could not find import block in lens_yara.py. Aborting.")

    old_func = """def _default_rules_root() -> Path:
    # lens_yara.py lives in VeriFYD_Lens/agent, so parent is VeriFYD_Lens.
    return _agent_dir().parent / "rules"
"""
    new_func = """def _default_rules_root() -> Path:
    # VERIFYD_LENS_YARA_PACKAGING_FIX_V1
    # Source layout:
    #   VeriFYD_Lens/agent/lens_yara.py -> VeriFYD_Lens/rules
    # PyInstaller onefile layout with --add-data "..\\\\rules;rules":
    #   sys._MEIPASS/rules
    frozen_root = getattr(sys, "_MEIPASS", None)
    if frozen_root:
        candidate = Path(frozen_root) / "rules"
        if candidate.exists():
            return candidate
    return _agent_dir().parent / "rules"
"""
    if old_func not in y_new:
        if "def _default_rules_root() -> Path:" in y_new and "sys._MEIPASS" in y_new:
            print("lens_yara.py already appears to contain a frozen rules path fix.")
        else:
            raise SystemExit("ERROR: Could not find _default_rules_root block in lens_yara.py. Aborting.")
    else:
        y_new = y_new.replace(old_func, new_func, 1)

    if MARKER not in y_new:
        marker_anchor = "# VERIFYD_LENS_YARA_ENGINE_V1\n"
        if marker_anchor in y_new:
            y_new = y_new.replace(marker_anchor, marker_anchor + f"# {MARKER}\n", 1)
        else:
            y_new = f"# {MARKER}\n" + y_new

    if y_new != yt:
        backup(YARA, "before_yara_packaging_fix_v1")
        YARA.write_text(y_new, encoding="utf-8", newline="")
        changed.append(str(YARA))
        print(f"Wrote: {YARA}")
    else:
        print("Unchanged: lens_yara.py")
else:
    print("Already patched: lens_yara.py contains packaging fix marker.")

# ---------------------------------------------------------------------------
# 2) Ensure lens_agent.py has one valid startup block.
# ---------------------------------------------------------------------------
at = AGENT.read_text(encoding="utf-8")
startup_block = "\n\nif __name__ == '__main__':\n    import uvicorn\n    uvicorn.run(app, host='127.0.0.1', port=8765)\n"

idx = at.rfind("\nif __name__")
if idx == -1:
    a_new = at.rstrip() + startup_block
else:
    tail = at[idx:]
    if "uvicorn.run(app" in tail or "__main__" in tail:
        a_new = at[:idx].rstrip() + startup_block
    else:
        a_new = at.rstrip() + startup_block

if a_new != at:
    backup(AGENT, "before_startup_fix_v1")
    AGENT.write_text(a_new, encoding="utf-8", newline="")
    changed.append(str(AGENT))
    print(f"Wrote: {AGENT}")
else:
    print("Unchanged: lens_agent.py")

print("")
if changed:
    print("Patch complete. Changed files:")
    for item in changed:
        print(f"  - {item}")
else:
    print("Patch complete. No changes were needed.")

print("")
print("Next validation commands:")
print("  py -m py_compile VeriFYD_Lens\\agent\\lens_agent.py VeriFYD_Lens\\agent\\lens_yara.py")
print("  cd VeriFYD_Lens\\agent")
print("  py lens_agent.py")
