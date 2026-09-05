from pathlib import Path
import shutil
import time

MARKER = "VERIFYD_LENS_YARA_INTEGRATION_V1"
ENGINE_MARKER = "VERIFYD_LENS_YARA_ENGINE_V1"

ROOT = Path.cwd()
AGENT = ROOT / "VeriFYD_Lens" / "agent" / "lens_agent.py"
YARA_MODULE = ROOT / "VeriFYD_Lens" / "agent" / "lens_yara.py"
RULES = ROOT / "VeriFYD_Lens" / "rules" / "builtin" / "verifyd_builtin.yar"

print(f"{MARKER}: project root = {ROOT}")

if not AGENT.exists():
    raise SystemExit(f"ERROR: lens_agent.py not found: {AGENT}")
if not YARA_MODULE.exists():
    raise SystemExit(f"ERROR: Phase 2A.1 module missing: {YARA_MODULE}")
if not RULES.exists():
    raise SystemExit(f"ERROR: Phase 2A.1 rules missing: {RULES}")

text = AGENT.read_text(encoding="utf-8")
if MARKER in text:
    print("Already integrated: lens_agent.py already contains Phase 2A YARA integration marker.")
    print("No changes made.")
    raise SystemExit(0)

backup = AGENT.with_name(f"lens_agent.py.before_yara_integration_v1_{int(time.time())}.bak")
shutil.copy2(AGENT, backup)
print(f"Backed up: {backup}")

# 1) Import the standalone YARA engine in the same fail-open style as static security.
static_import = '''# VERIFYD_LENS_STATIC_SECURITY_V1
try:
    from lens_security import scan_static_security
except Exception:
    scan_static_security = None
'''
import_block = '''# VERIFYD_LENS_STATIC_SECURITY_V1
try:
    from lens_security import scan_static_security
except Exception:
    scan_static_security = None

# VERIFYD_LENS_YARA_INTEGRATION_V1
try:
    from lens_yara import scan_file as scan_yara_security
except Exception:
    scan_yara_security = None
'''
if static_import not in text:
    raise SystemExit("ERROR: Could not find static security import block. Aborting without modifying lens_agent.py.")
text = text.replace(static_import, import_block, 1)

# 2) Add YARA scan immediately after SHA/security scanning state and before static inspection.
anchor = '''        SCAN_STATE[scan_id].update(status="SECURITY_SCANNING",summary="SECURITY SCANNING",sha256=sha,size_bytes=q.stat().st_size,quarantine_path=str(q),findings=findings)
        # VERIFYD_LENS_STATIC_SECURITY_V1 — additive; authenticity pipeline untouched.
'''
yara_block = '''        SCAN_STATE[scan_id].update(status="SECURITY_SCANNING",summary="SECURITY SCANNING",sha256=sha,size_bytes=q.stat().st_size,quarantine_path=str(q),findings=findings)
        # VERIFYD_LENS_YARA_INTEGRATION_V1 — additive; fail-open; authenticity pipeline untouched.
        if scan_yara_security is not None:
            yara_security=scan_yara_security(q)
        else:
            yara_security={"engine":"verifyd_yara_x_v1","engine_label":"YARA-X unavailable","status":"UNAVAILABLE","score_delta":0,"hard_block":False,"matches":[],"match_count":0,"rule_count":0,"rule_files":[],"finding":"YARA-X security inspection was unavailable; existing security checks continued.","details":{}}
        score+=int(yara_security.get("score_delta",0) or 0)
        if yara_security.get("finding"):
            findings.append(yara_security.get("finding"))
        # VERIFYD_LENS_STATIC_SECURITY_V1 — additive; authenticity pipeline untouched.
'''
if anchor not in text:
    raise SystemExit("ERROR: Could not find SECURITY_SCANNING/static-security anchor. Aborting without modifying lens_agent.py.")
text = text.replace(anchor, yara_block, 1)

# 3) Add YARA fields to every scan-state response that already includes static security details.
old_fields = 'static_security_status=static_security.get("status"),static_security_engine=static_security.get("engine"),static_security_details=static_security.get("details"),findings='
new_fields = 'static_security_status=static_security.get("status"),static_security_engine=static_security.get("engine"),static_security_details=static_security.get("details"),yara_status=yara_security.get("status"),yara_engine=yara_security.get("engine"),yara_rule_count=yara_security.get("rule_count"),yara_match_count=yara_security.get("match_count"),yara_matches=yara_security.get("matches"),yara_details=yara_security.get("details"),findings='
count = text.count(old_fields)
if count < 3:
    raise SystemExit(f"ERROR: Expected at least 3 static-security field groups, found {count}. Aborting without modifying lens_agent.py.")
text = text.replace(old_fields, new_fields)
print(f"Updated scan-state response fields: {count} locations")

# 4) Allow high-confidence YARA hard blocks to use the existing BLOCKED path.
old_hard_block = '        if static_security.get("hard_block"):\n'
new_hard_block = '        if static_security.get("hard_block") or yara_security.get("hard_block"):\n'
if old_hard_block not in text:
    raise SystemExit("ERROR: Could not find static hard-block branch. Aborting without modifying lens_agent.py.")
text = text.replace(old_hard_block, new_hard_block, 1)

AGENT.write_text(text, encoding="utf-8", newline="")
print(f"Wrote: {AGENT}")
print("Patch complete. Only lens_agent.py was modified.")
print("Next: run py -m py_compile VeriFYD_Lens\\agent\\lens_agent.py VeriFYD_Lens\\agent\\lens_yara.py")
