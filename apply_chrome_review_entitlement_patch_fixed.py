from pathlib import Path
import shutil
import sys
from datetime import datetime

MAIN = Path("main.py")
MARKER = "# VERIFYD_CHROME_REVIEW_ENTITLEMENT_V1"

BLOCK = """
# VERIFYD_CHROME_REVIEW_ENTITLEMENT_V1
@app.post("/admin/lens/create-review-entitlement")
async def admin_lens_create_review_entitlement(
    key: str = Form(...),
    reviewer_email: str = Form("chrome-review@vfvid.com"),
):
    if not _is_admin(key):
        return JSONResponse({"error": "unauthorized"}, status_code=401)

    reviewer_email = (reviewer_email or "").strip().lower()
    if not reviewer_email or not is_valid_email(reviewer_email):
        return JSONResponse({"error": "valid_email_required"}, status_code=400)

    import secrets
    from datetime import datetime, timezone
    from database import get_db

    review_order_id = "chrome_web_store_review"
    now = datetime.now(timezone.utc).isoformat()

    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT buyer_email, status, entitlement_token FROM lens_purchases WHERE paypal_order_id = %s",
            (review_order_id,),
        )
        existing = cur.fetchone()

        if existing:
            row = dict(existing)
            token = (row.get("entitlement_token") or "").strip()

            if not token:
                token = "vfyd_lens_" + secrets.token_urlsafe(24)
                cur.execute(
                    "UPDATE lens_purchases SET buyer_email=%s, product_id='verifyd_lens_beta', amount='0.00', currency='USD', status='COMPLETED', entitlement_token=%s, completed_at=COALESCE(completed_at,%s) WHERE paypal_order_id=%s",
                    (reviewer_email, token, now, review_order_id),
                )
            elif row.get("status") != "COMPLETED" or row.get("buyer_email") != reviewer_email:
                cur.execute(
                    "UPDATE lens_purchases SET buyer_email=%s, product_id='verifyd_lens_beta', amount='0.00', currency='USD', status='COMPLETED', completed_at=COALESCE(completed_at,%s) WHERE paypal_order_id=%s",
                    (reviewer_email, now, review_order_id),
                )

            return {
                "status": "existing",
                "reviewer_email": reviewer_email,
                "entitlement_token": token,
                "product_id": "verifyd_lens_beta",
                "amount": "0.00",
                "currency": "USD",
            }

        token = "vfyd_lens_" + secrets.token_urlsafe(24)
        capture_id = "chrome_review_capture_" + secrets.token_hex(12)

        cur.execute(
            "INSERT INTO lens_purchases (paypal_order_id,paypal_capture_id,buyer_email,product_id,amount,currency,status,entitlement_token,created_at,completed_at) VALUES (%s,%s,%s,%s,%s,%s,'COMPLETED',%s,%s,%s)",
            (review_order_id, capture_id, reviewer_email, "verifyd_lens_beta", "0.00", "USD", token, now, now),
        )

    return {
        "status": "created",
        "reviewer_email": reviewer_email,
        "entitlement_token": token,
        "product_id": "verifyd_lens_beta",
        "amount": "0.00",
        "currency": "USD",
    }


"""

def main():
    if not MAIN.exists():
        print("ERROR: main.py was not found in the current folder.")
        print("Run this script from C:\\Users\\RDigiacomo2\\VeriFYD")
        sys.exit(1)

    text = MAIN.read_text(encoding="utf-8")

    if MARKER in text:
        print("Patch already present. No changes made.")
        return

    anchors = [
        '@app.post("/admin/lens/upload-installer")',
        "@app.post('/admin/lens/upload-installer')",
    ]

    pos = -1
    for anchor in anchors:
        pos = text.find(anchor)
        if pos != -1:
            break

    if pos == -1:
        print("ERROR: Could not find the Lens admin installer route anchor.")
        sys.exit(1)

    stamp = datetime.now().strftime("%Y%m%d%d_%H%M%S")
    backup = MAIN.with_name(f"main.py.before_chrome_review_{stamp}.bak")
    shutil.copy2(MAIN, backup)

    MAIN.write_text(text[:pos] + BLOCK + text[pos:], encoding="utf-8")

    print("SUCCESS: Chrome reviewer entitlement endpoint added.")
    print(f"Backup: {backup}")
    print("Route: POST /admin/lens/create-review-entitlement")
    print("Next: run py -m py_compile main.py")

if __name__ == "__main__":
    main()
