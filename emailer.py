# ============================================================
#  VeriFYD — emailer.py
#  Sends transactional emails via Resend API
#  - OTP verification emails
#  - Post-certification emails (REAL videos)
# ============================================================

import os
import logging

log = logging.getLogger("verifyd.emailer")

RESEND_API_KEY = os.environ.get("RESEND_API_KEY", "")
FROM_ADDRESS   = "noreply@vfvid.com"
FROM_NAME      = "VeriFYD"
SITE_URL       = "https://vfvid.com"
BACKEND_URL    = "https://verifyd-backend.onrender.com"


def _send(payload: dict) -> bool:
    """Internal helper — sends email via Resend SDK."""
    if not RESEND_API_KEY:
        log.error("RESEND_API_KEY not set — cannot send email")
        return False
    try:
        import resend
        resend.api_key = RESEND_API_KEY
        result = resend.Emails.send(payload)
        log.info("Email sent to %s — id: %s", payload.get("to"), result.get("id"))
        return True
    except Exception as e:
        log.error("Failed to send email: %s", e)
        return False


def _header_html() -> str:
    return """
      <tr>
        <td style="background:linear-gradient(135deg,#1a1a1a 0%,#0f0f0f 100%);padding:32px;text-align:center;border-bottom:1px solid #222;">
          <h1 style="margin:0;font-size:28px;font-weight:800;color:#ffffff;letter-spacing:-0.5px;">
            Veri<span style="color:#f59e0b;">FYD</span>
          </h1>
          <p style="margin:8px 0 0;color:#666;font-size:13px;letter-spacing:2px;text-transform:uppercase;">
            AI Video Detection
          </p>
        </td>
      </tr>"""


def _footer_html() -> str:
    return f"""
      <tr>
        <td style="padding:20px 32px;border-top:1px solid #1a1a1a;text-align:center;">
          <p style="margin:0 0 8px;color:#444444;font-size:12px;">
            <a href="{SITE_URL}/privacy" style="color:#555;text-decoration:none;">Privacy Policy</a>
            &nbsp;·&nbsp;
            <a href="{SITE_URL}/terms" style="color:#555;text-decoration:none;">Terms of Service</a>
            &nbsp;·&nbsp;
            <a href="{SITE_URL}" style="color:#f59e0b;text-decoration:none;">vfvid.com</a>
          </p>
          <p style="margin:0;color:#333333;font-size:11px;">
            © 2026 VeriFYD · You are receiving this because you used VeriFYD to verify a video.
          </p>
        </td>
      </tr>"""


def send_otp_email(to_email: str, code: str) -> bool:
    """Send OTP verification email via Resend."""
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
          {_header_html()}
          <tr>
            <td style="padding:40px 32px;">
              <h2 style="margin:0 0 12px;font-size:22px;font-weight:700;color:#ffffff;">
                Verify your email
              </h2>
              <p style="margin:0 0 32px;color:#888888;font-size:15px;line-height:1.6;">
                Enter the code below to verify your email address and start using VeriFYD.
                This code expires in <strong style="color:#f59e0b;">10 minutes</strong>.
              </p>
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
              </p>
            </td>
          </tr>
          {_footer_html()}
        </table>
      </td>
    </tr>
  </table>
</body>
</html>"""

    return _send({
        "from":    f"{FROM_NAME} <{FROM_ADDRESS}>",
        "to":      [to_email],
        "subject": f"{code} is your VeriFYD verification code",
        "html":    html,
    })


def send_certification_email(
    to_email: str,
    certificate_id: str,
    authenticity: int,
    original_filename: str,
    download_url: str,
) -> bool:
    """
    Send post-certification email when a video is certified REAL.
    Includes download link, certificate link, and score.
    """
    cert_url      = f"{SITE_URL}/v/{certificate_id}"
    short_id      = certificate_id[:8].upper()
    safe_filename = original_filename[:40] + ("..." if len(original_filename) > 40 else "")

    # Score color — green for high, yellow for moderate
    score_color = "#22c55e" if authenticity >= 75 else "#f59e0b"

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
          {_header_html()}

          <!-- Hero -->
          <tr>
            <td style="padding:40px 32px 24px;text-align:center;">
              <div style="display:inline-block;background:#052e16;border:1px solid #22c55e;border-radius:100px;padding:8px 20px;margin-bottom:24px;">
                <span style="color:#22c55e;font-size:13px;font-weight:700;letter-spacing:1px;text-transform:uppercase;">
                  &#10003; &nbsp;Real Video Verified
                </span>
              </div>
              <h2 style="margin:0 0 8px;font-size:24px;font-weight:800;color:#ffffff;">
                Your video has been certified
              </h2>
              <p style="margin:0;color:#888888;font-size:15px;line-height:1.6;">
                VeriFYD has analyzed <strong style="color:#ccc;">{safe_filename}</strong>
                and issued an official authenticity certificate.
              </p>
            </td>
          </tr>

          <!-- Score Card -->
          <tr>
            <td style="padding:0 32px 32px;">
              <table width="100%" cellpadding="0" cellspacing="0"
                     style="background:#1a1a1a;border:1px solid #2a2a2a;border-radius:12px;overflow:hidden;">
                <tr>
                  <td width="50%" style="padding:24px;text-align:center;border-right:1px solid #2a2a2a;">
                    <p style="margin:0 0 6px;color:#666;font-size:11px;letter-spacing:2px;text-transform:uppercase;">
                      Authenticity Score
                    </p>
                    <p style="margin:0;font-size:48px;font-weight:800;color:{score_color};line-height:1;">
                      {authenticity}%
                    </p>
                  </td>
                  <td width="50%" style="padding:24px;text-align:center;">
                    <p style="margin:0 0 6px;color:#666;font-size:11px;letter-spacing:2px;text-transform:uppercase;">
                      Certificate ID
                    </p>
                    <p style="margin:0;font-size:18px;font-weight:700;color:#f59e0b;font-family:'Courier New',monospace;letter-spacing:1px;">
                      {short_id}...
                    </p>
                  </td>
                </tr>
              </table>
            </td>
          </tr>

          <!-- CTA Buttons -->
          <tr>
            <td style="padding:0 32px 32px;">
              <!-- Download — primary action -->
              <a href="{download_url}"
                 style="display:block;background:#22c55e;color:#000000;text-decoration:none;
                        text-align:center;padding:16px;border-radius:10px;
                        font-size:15px;font-weight:700;margin-bottom:12px;">
                &#11015;&#65039; &nbsp;Download Certified Video
              </a>
              <!-- View Certificate — secondary -->
              <a href="{cert_url}"
                 style="display:block;background:#1a1a1a;color:#f59e0b;text-decoration:none;
                        text-align:center;padding:16px;border-radius:10px;
                        font-size:15px;font-weight:700;border:1px solid #333;">
                &#127942; &nbsp;View Official Certificate
              </a>
            </td>
          </tr>

          <!-- Share certified video link -->
          <tr>
            <td style="padding:0 32px 32px;">
              <div style="background:#0f0f0f;border:1px solid #1e1e1e;border-radius:10px;padding:20px;text-align:center;">
                <p style="margin:0 0 8px;color:#888;font-size:13px;line-height:1.6;">
                  Share your certified video link
                </p>
                <a href="{download_url}"
                   style="display:block;font-size:12px;color:#f59e0b;font-family:'Courier New',monospace;
                          word-break:break-all;text-decoration:none;">
                  {download_url}
                </a>
              </div>
            </td>
          </tr>

          <!-- What this means -->
          <tr>
            <td style="padding:0 32px 32px;">
              <p style="margin:0 0 12px;color:#555;font-size:12px;line-height:1.7;">
                <strong style="color:#777;">What does this mean?</strong><br>
                VeriFYD's dual-engine analysis (signal detection + GPT-4o vision AI) assessed
                your video and found no significant indicators of AI generation. Your certified
                video includes the VeriFYD watermark as proof of authenticity.
              </p>
              <p style="margin:0;color:#444;font-size:11px;line-height:1.6;">
                Note: VeriFYD results represent analytical guidance. See our
                <a href="{SITE_URL}/terms" style="color:#555;">Terms of Service</a> for full details.
              </p>
            </td>
          </tr>

          {_footer_html()}
        </table>
      </td>
    </tr>
  </table>
</body>
</html>"""

    return _send({
        "from":    f"{FROM_NAME} <{FROM_ADDRESS}>",
        "to":      [to_email],
        "subject": f"&#10003; Your video has been certified — VeriFYD #{short_id}",
        "html":    html,
    })
