# ============================================================
#  VeriFYD — emailer.py
#  Sends OTP verification emails via Resend API
# ============================================================

import os
import logging

log = logging.getLogger("verifyd.emailer")

RESEND_API_KEY = os.environ.get("RESEND_API_KEY", "")
FROM_ADDRESS   = "noreply@vfvid.com"   # TODO: switch to noreply@vfvid.com once domain verified
FROM_NAME      = "VeriFYD"


def send_otp_email(to_email: str, code: str) -> bool:
    """
    Send OTP verification email via Resend.
    Returns True on success, False on failure.
    """
    if not RESEND_API_KEY:
        log.error("RESEND_API_KEY not set — cannot send OTP email")
        return False

    html = f"""
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body style="margin:0;padding:0;background-color:#0a0a0a;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;">
  <table width="100%" cellpadding="0" cellspacing="0" style="background-color:#0a0a0a;padding:40px 20px;">
    <tr>
      <td align="center">
        <table width="500" cellpadding="0" cellspacing="0" style="background-color:#111111;border-radius:12px;border:1px solid #222222;overflow:hidden;max-width:500px;width:100%;">

          <!-- Header -->
          <tr>
            <td style="background:linear-gradient(135deg,#1a1a1a 0%,#0f0f0f 100%);padding:32px;text-align:center;border-bottom:1px solid #222;">
              <h1 style="margin:0;font-size:28px;font-weight:800;color:#ffffff;letter-spacing:-0.5px;">
                Veri<span style="color:#f59e0b;">FYD</span>
              </h1>
              <p style="margin:8px 0 0;color:#666;font-size:13px;letter-spacing:2px;text-transform:uppercase;">
                AI Video Detection
              </p>
            </td>
          </tr>

          <!-- Body -->
          <tr>
            <td style="padding:40px 32px;">
              <h2 style="margin:0 0 12px;font-size:22px;font-weight:700;color:#ffffff;">
                Verify your email
              </h2>
              <p style="margin:0 0 32px;color:#888888;font-size:15px;line-height:1.6;">
                Enter the code below to verify your email address and start using VeriFYD.
                This code expires in <strong style="color:#f59e0b;">10 minutes</strong>.
              </p>

              <!-- OTP Code Box -->
              <div style="background:#1a1a1a;border:2px solid #f59e0b;border-radius:12px;padding:28px;text-align:center;margin:0 0 32px;">
                <p style="margin:0 0 8px;color:#888;font-size:12px;letter-spacing:2px;text-transform:uppercase;">
                  Your verification code
                </p>
                <p style="margin:0;font-size:48px;font-weight:800;color:#f59e0b;letter-spacing:12px;font-family:'Courier New',monospace;">
                  {code}
                </p>
              </div>

              <p style="margin:0;color:#555555;font-size:13px;line-height:1.6;">
                If you didn't request this code, you can safely ignore this email.
                Someone may have entered your email address by mistake.
              </p>
            </td>
          </tr>

          <!-- Footer -->
          <tr>
            <td style="padding:20px 32px;border-top:1px solid #1a1a1a;text-align:center;">
              <p style="margin:0;color:#444444;font-size:12px;">
                © 2025 VeriFYD · <a href="https://vfvid.com" style="color:#f59e0b;text-decoration:none;">vfvid.com</a>
              </p>
            </td>
          </tr>

        </table>
      </td>
    </tr>
  </table>
</body>
</html>
"""

    payload = {
        "from":    f"{FROM_NAME} <{FROM_ADDRESS}>",
        "to":      [to_email],
        "subject": f"{code} is your VeriFYD verification code",
        "html":    html,
    }

    try:
        import resend
        resend.api_key = RESEND_API_KEY
        result = resend.Emails.send(payload)
        log.info("OTP email sent to %s — id: %s", to_email, result.get("id"))
        return True

    except Exception as e:
        log.error("Failed to send OTP email to %s: %s", to_email, e)
        return False