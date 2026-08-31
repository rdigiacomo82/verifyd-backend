from pathlib import Path

p = Path("main.py")
if not p.exists():
    raise SystemExit("Run this script from the VeriFYD repository root where main.py exists.")

text = p.read_text(encoding="utf-8")
marker = "# VERIFYD_LENS_CLOUD_ROUTER_V04"

block = """
# VERIFYD_LENS_CLOUD_ROUTER_V04
try:
    from lens_cloud import router as lens_cloud_router
    app.include_router(lens_cloud_router)
    print("[verifyd-lens] cloud routes enabled")
except Exception as exc:
    print(f"[verifyd-lens] cloud routes not enabled: {exc}")
"""

if marker in text:
    print("Lens cloud router already registered; no main.py change needed.")
else:
    backup = p.with_name("main.py.before_verifyd_lens_cloud_v04.bak")
    backup.write_text(text, encoding="utf-8")
    p.write_text(text.rstrip() + "\n\n" + block + "\n", encoding="utf-8")
    print(f"Updated main.py. Backup: {backup}")
