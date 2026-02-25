import os
import logging

log = logging.getLogger("verifyd.emailer")

RESEND_API_KEY = os.environ.get("RESEND_API_KEY", "")
FROM_ADDRESS   = "onboarding@resend.dev"
FROM_NAME      = "VeriFYD"


def send_otp_email(to_email: str, code: str) -> bool:
    if not RESEND_API_KEY:
        log.error("RESEND_API_KEY not set")
        return False
    try:
        import resend
        resend.api_key = RESEND_API_KEY
        payload = {
            "from": f"{FROM_NAME} <{FROM_ADDRESS}>",
            "to": [to_email],
            "subject": f"Your VeriFYD verification code",
            "html": f"<div style='font-family:sans-serif;background:#111;color:#fff;padding:40px;border-radius:12px;'><h1>Veri<span style='color:#f59e0b;'>FYD</span></h1><p>Your verification code is:</p><h2 style='font-size:48px;color:#f59e0b;letter-spacing:12px;font-family:monospace;'>{code}</h2><p style='color:#888;'>Expires in 10 minutes. If you did not request this, ignore this email.</p></div>",
        }
        result = resend.Emails.send(payload)
        log.info("OTP sent to %s id=%s", to_email, result.get("id"))
        return True
    except Exception as e:
        log.error("Failed to send OTP to %s: %s", to_email, e)
        return False