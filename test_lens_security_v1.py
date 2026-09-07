from pathlib import Path
import json
import tempfile
import zipfile
import sys

ROOT = Path(__file__).resolve().parent
AGENT_DIR = ROOT / "VeriFYD_Lens" / "agent"
sys.path.insert(0, str(AGENT_DIR))

from lens_security import scan_static_security

def show(name, result):
    print(f"\n=== {name} ===")
    print(json.dumps(result, indent=2))
    return result

failures = []

with tempfile.TemporaryDirectory(prefix="verifyd_lens_security_test_") as td:
    d = Path(td)

    # 1. Benign minimal PDF-like file: should NOT hard block.
    p1 = d / "benign.pdf"
    p1.write_bytes(b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog >>\nendobj\n%%EOF\n")
    r1 = show("BENIGN PDF", scan_static_security(p1, p1.name))
    if r1.get("hard_block"):
        failures.append("Benign PDF incorrectly hard-blocked")

    # 2. Inert text containing suspicious PDF markers: should hard block
    # because /Launch + JavaScript appear together. This is NOT executable malware.
    p2 = d / "suspicious_markers.pdf"
    p2.write_bytes(
        b"%PDF-1.4\n"
        b"1 0 obj << /OpenAction 2 0 R /Launch /JavaScript /JS (test) >> endobj\n"
        b"%%EOF\n"
    )
    r2 = show("PDF LAUNCH + SCRIPT MARKERS", scan_static_security(p2, p2.name))
    if not r2.get("hard_block"):
        failures.append("PDF Launch+script markers were not hard-blocked")

    # 3. Double-extension decoy. Plain text only; no executable payload.
    p3 = d / "invoice.pdf.exe"
    p3.write_text("This is an inert Lens security test file.", encoding="utf-8")
    r3 = show("DOUBLE EXTENSION", scan_static_security(p3, p3.name))
    if not r3.get("hard_block"):
        failures.append("Double-extension decoy was not hard-blocked")

    # 4. ZIP path traversal entry. Harmless archive; do not extract it.
    p4 = d / "traversal.zip"
    with zipfile.ZipFile(p4, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("../safe_test.txt", "inert test content")
    r4 = show("ZIP PATH TRAVERSAL", scan_static_security(p4, p4.name))
    if not r4.get("hard_block"):
        failures.append("ZIP path-traversal structure was not hard-blocked")

    # 5. DOCX-style ZIP with macro marker in a normally macro-free extension.
    p5 = d / "macro_marker.docx"
    with zipfile.ZipFile(p5, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", "<Types></Types>")
        zf.writestr("word/document.xml", "<document>test</document>")
        zf.writestr("word/vbaProject.bin", b"INERT_TEST_MARKER")
    r5 = show("DOCX MACRO MARKER", scan_static_security(p5, p5.name))
    if not r5.get("hard_block"):
        failures.append("Macro marker inside .docx was not hard-blocked")

print("\n==============================")
if failures:
    print("TEST RESULT: FAILED")
    for item in failures:
        print(" -", item)
    raise SystemExit(1)

print("TEST RESULT: PASSED")
print("All tests are inert structural samples; no live malware was created or executed.")
