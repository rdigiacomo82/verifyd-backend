"""
VeriFYD email deliverability patch.
Run from repo root:
    py verifyd_email_deliverability_patch.py
"""
from __future__ import annotations
import pathlib, re

ROOT = pathlib.Path.cwd()

def read(name: str) -> str:
    p = ROOT / name
    if not p.exists():
        raise FileNotFoundError(f"Missing required file: {name}")
    return p.read_text(encoding="utf-8", errors="replace")

def write(name: str, text: str) -> None:
    (ROOT / name).write_text(text, encoding="utf-8", newline="\n")

def patch_emailer() -> None:
    s = read("emailer.py")

    s = re.sub(r'FROM_ADDRESS\s*=\s*"[^"]*"', 'FROM_ADDRESS   = os.environ.get("FROM_ADDRESS", "verify@vfvid.com")', s, count=1)
    s = re.sub(r'FROM_NAME\s*=\s*"[^"]*"', 'FROM_NAME      = os.environ.get("FROM_NAME", "VeriFYD")', s, count=1)
    s = re.sub(r'SITE_URL\s*=\s*"[^"]*"', 'SITE_URL       = os.environ.get("SITE_URL", "https://vfvid.com").rstrip("/")', s, count=1)
    s = re.sub(r'BACKEND_URL\s*=\s*"[^"]*"', 'BACKEND_URL    = os.environ.get("BACKEND_URL", "https://verifyd-backend.onrender.com").rstrip("/")', s, count=1)

    s = re.sub(
        r'cert_url\s*=\s*f"\{SITE_URL\}/v/\{certificate_id\}"',
        'cert_url      = f"{SITE_URL}/verify-certificate/{certificate_id}"',
        s,
        count=1,
    )
    if 'cert_url      = f"{SITE_URL}/verify-certificate/{certificate_id}"' not in s:
        s = re.sub(
            r'cert_url\s*=.*certificate_id.*\n',
            '    cert_url      = f"{SITE_URL}/verify-certificate/{certificate_id}"\n',
            s,
            count=1,
        )

    # Keep email-click URLs aligned with vfvid.com. Backend download_url remains intact for app/outbox.
    s = s.replace('href="{download_url}"', 'href="{cert_url}"')
    s = s.replace('                  {download_url}\n', '                  {cert_url}\n')

    s = re.sub(
        r'"subject":\s*f"&#10003; Your \{_media\} has been certified.*?VeriFYD #\{short_id\}"',
        '"subject": f"VeriFYD Certification Ready - {_Media} #{short_id}"',
        s,
        count=1,
    )

    marker = """    return _send({
        "from":    f"{FROM_NAME} <{FROM_ADDRESS}>",
        "to":      [to_email],
        "subject": f"{code} is your VeriFYD verification code",
        "html":    html,
    })"""
    replacement = """    text = (
        f"Your VeriFYD verification code is: {code}\\n\\n"
        "This code expires shortly. If you did not request this code, you can ignore this email.\\n\\n"
        "VeriFYD\\n"
        f"{SITE_URL}"
    )

    return _send({
        "from":    f"{FROM_NAME} <{FROM_ADDRESS}>",
        "to":      [to_email],
        "subject": "Your VeriFYD verification code",
        "html":    html,
        "text":    text,
    })"""
    if marker in s:
        s = s.replace(marker, replacement, 1)
    else:
        s = s.replace('"subject": f"{code} is your VeriFYD verification code",', '"subject": "Your VeriFYD verification code",', 1)

    s = s.replace('Download link: {download_url}', 'Certificate page: {cert_url}')
    s = s.replace('Download URL: {download_url}', 'Certificate page: {cert_url}')

    write("emailer.py", s)

def main() -> None:
    patch_emailer()
    print("Applied VeriFYD deliverability domain-link patch.")
    print("Next: py -m py_compile emailer.py main.py worker.py notification_helper.py storage.py")
    print("Render env recommended: FROM_ADDRESS=verify@vfvid.com, FROM_NAME=VeriFYD, SITE_URL=https://vfvid.com")

if __name__ == "__main__":
    main()
