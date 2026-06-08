from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent

def read(name: str) -> str:
    return (ROOT / name).read_text(encoding="utf-8", errors="replace")

def write(name: str, text: str) -> None:
    (ROOT / name).write_text(text, encoding="utf-8", newline="\n")

def patch_database() -> None:
    path = "database.py"
    s = read(path)

    if "def normalize_email(" not in s:
        helper = '''
def normalize_email(email: str) -> str:
    """
    Canonicalize user emails before storing or looking them up.

    This intentionally only repairs VeriFYD's own internal domain typo:
      @vfvid.co -> @vfvid.com

    It does not rewrite general .co addresses because .co is a valid TLD.
    """
    value = (email or "").strip().lower()

    if value.endswith("@vfvid.co"):
        value = value[:-len("@vfvid.co")] + "@vfvid.com"

    return value

'''
        marker = "def is_valid_email(email: str) -> bool:"
        if marker not in s:
            raise RuntimeError("Could not find is_valid_email marker in database.py")
        s = s.replace(marker, helper + marker, 1)

    replacements = [
        ("email_lower = email.strip().lower()", "email_lower = normalize_email(email)"),
        ("email_lower = email.lower().strip()", "email_lower = normalize_email(email)"),
        ("(email.lower().strip(),)", "(normalize_email(email),)"),
        ("email.strip()", "normalize_email(email)"),
    ]

    for old, new in replacements:
        s = s.replace(old, new)

    # Avoid over-normalizing display-only values inside logs/comments is fine,
    # but ensure inserts store canonical email as the visible email too.
    s = s.replace(
        "(email, email_lower, now.isoformat(), now.isoformat(), now.isoformat())",
        "(email_lower, email_lower, now.isoformat(), now.isoformat(), now.isoformat())"
    )

    s = s.replace(
        "(email, email_lower, plan, total_uses, period_uses,",
        "(email_lower, email_lower, plan, total_uses, period_uses,"
    )

    s = s.replace(
        '"email":          normalize_email(email),',
        '"email":          email_lower,'
    )

    write(path, s)

def patch_main() -> None:
    path = "main.py"
    s = read(path)

    if "normalize_email" not in s.split("\n", 80)[0:80]:
        s = s.replace(
            "is_valid_email, FREE_USES, get_certificate,",
            "is_valid_email, normalize_email, FREE_USES, get_certificate,",
            1
        )

    # Register email endpoint
    s = s.replace(
        'email = body.get("email", "").strip()',
        'email = normalize_email(body.get("email", ""))'
    )

    # Common FastAPI form/query endpoints. Insert normalization immediately after the def line.
    def add_after_def(text: str, signature: str, line: str) -> str:
        if signature not in text:
            return text
        block_start = text.find(signature)
        next_chunk = text[block_start:block_start + 300]
        if line.strip() in next_chunk:
            return text
        return text.replace(signature, signature + line, 1)

    s = add_after_def(
        s,
        "async def upload(file: UploadFile = File(...), email: str = Form(...)):\n",
        "    email = normalize_email(email)\n"
    )

    s = add_after_def(
        s,
        "async def upload_photo(file: UploadFile = File(...), email: str = Form(...)):\n",
        "    email = normalize_email(email)\n"
    )

    s = add_after_def(
        s,
        "async def upload_audio(file: UploadFile = File(...), email: str = Form(...)):\n",
        "    email = normalize_email(email)\n"
    )

    s = add_after_def(
        s,
        "async def upload_document(file: UploadFile = File(...), email: str = Form(...)):\n",
        "    email = normalize_email(email)\n"
    )

    s = add_after_def(
        s,
        "def upload_limits(email: str = \"\"):\n",
        "    email = normalize_email(email) if email else \"\"\n"
    )

    s = add_after_def(
        s,
        "def user_status(email: str = \"\"):\n",
        "    email = normalize_email(email) if email else \"\"\n"
    )

    s = add_after_def(
        s,
        "async def send_otp(email: str = Form(...)):\n",
        "    email = normalize_email(email)\n"
    )

    s = add_after_def(
        s,
        "async def verify_otp_route(email: str = Form(...), code: str = Form(...)):\n",
        "    email = normalize_email(email)\n"
    )

    s = add_after_def(
        s,
        "def admin_reset_user(email: str = \"\", key: str = \"\"):\n",
        "    email = normalize_email(email) if email else \"\"\n"
    )

    s = add_after_def(
        s,
        "def admin_upgrade_user(email: str = \"\", plan: str = \"enterprise\", key: str = \"\"):\n",
        "    email = normalize_email(email) if email else \"\"\n"
    )

    s = add_after_def(
        s,
        "def admin_delete_user(email: str = \"\", key: str = \"\"):\n",
        "    email = normalize_email(email) if email else \"\"\n"
    )

    write(path, s)

def main() -> None:
    patch_database()
    patch_main()
    print("Applied VeriFYD email canonicalization patch.")
    print("Next: py -m py_compile database.py main.py worker.py notification_helper.py emailer.py storage.py")

if __name__ == "__main__":
    main()