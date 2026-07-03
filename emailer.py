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
            © 2026 VeriFYD · You are receiving this because a VeriFYD certified record was securely sent to you.
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
    is_photo: bool = False,
    is_document: bool = False,
    is_audio: bool = False,
) -> bool:
    """
    Send post-certification email when a video, photo, document, or audio file is certified REAL.
    Includes download link, certificate link, and score.
    Pass is_photo=True, is_document=True, or is_audio=True for correct language.
    """
    cert_url      = f"{SITE_URL}/v/{certificate_id}"
    short_id      = certificate_id[:8].upper()
    safe_filename = original_filename[:40] + ("..." if len(original_filename) > 40 else "")

    # Media-type strings — keep existing video/photo behavior, add document wording safely.
    if is_document:
        _media = "document"
        _Media = "Document"
        _verified_lbl = "Real Document Verified"
        _dl_label = "Download Certified Document"
        _share_label = "Share Your Certified Document"
        _share_desc = "Share this link with anyone to verify your certified document:"
        _share_tap = "Tap the link above to open or download your certified document directly."
        _what_body = (
            "VeriFYD analyzed your document for metadata, content, and authenticity indicators. "
            "Your certified document includes a small VeriFYD mark and a server-side certificate record."
        )
        if "/download-certified-file/" in str(download_url):
            _dl_label = "Download All Certified Files"
            _share_label = "Share Certified File Package"
            _share_desc = "Share this link with authorized recipients to download the certified file package:"
            _share_tap = "Tap the link above to download the full certified ZIP package, including individually certified internal files when available."
            _what_body = (
                "VeriFYD certified your ZIP evidence package and, for eligible Pro/Enterprise accounts, "
                "created certified reports for supported files inside the ZIP. Download the full certified "
                "package to access the parent report, child certified reports, original source files, and signed manifest."
            )
    elif is_audio:
        _media = "audio"
        _Media = "Audio"
        _verified_lbl = "Real Audio Verified"
        _dl_label = "Download Certified Audio"
        _share_label = "Share Your Certified Audio"
        _share_desc = "Share this link with authorized recipients to download your certified audio:"
        _share_tap = "Tap the link above to open or download your certified audio directly."
        _what_body = (
            "VeriFYD analyzed your audio for synthetic-generation indicators, metadata concerns, "
            "spectral consistency, noise-floor behavior, dynamic range, and audio-forensic authenticity signals. "
            "Your certified audio preserves the audible content and includes a server-side certificate record."
        )
    elif is_photo:
        _media = "photo"
        _Media = "Photo"
        _verified_lbl = "Real Photo Verified"
        _dl_label = "Download Certified Photo"
        _share_label = "Share Your Certified Photo"
        _share_desc = f"Share this link with anyone to prove your {_media} is real:"
        _share_tap = f"Tap the link above to stream or share your certified {_media} directly."
        _what_body = (
            f"VeriFYD's dual-engine analysis (signal detection + GPT-4o vision AI) assessed "
            f"your {_media} and found no significant indicators of AI generation. Your certified "
            f"{_media} includes the VeriFYD watermark as proof of authenticity."
        )
    else:
        _media = "video"
        _Media = "Video"
        _verified_lbl = "Real Video Verified"
        _dl_label = "Download Certified Video"
        _share_label = "Share Your Certified Video"
        _share_desc = f"Share this link with anyone to prove your {_media} is real:"
        _share_tap = f"Tap the link above to stream or share your certified {_media} directly."
        _what_body = (
            f"VeriFYD's dual-engine analysis (signal detection + GPT-4o vision AI) assessed "
            f"your {_media} and found no significant indicators of AI generation. Your certified "
            f"{_media} includes the VeriFYD watermark as proof of authenticity."
        )
    _subject      = f"&#10003; Your {_media} has been certified — VeriFYD #{short_id}"

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
                  &#10003; &nbsp;{_verified_lbl}
                </span>
              </div>
              <h2 style="margin:0 0 8px;font-size:24px;font-weight:800;color:#ffffff;">
                Your {_media} has been certified
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
                &#11015;&#65039; &nbsp;{_dl_label}
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
              <div style="background:#0f0f0f;border:1px solid #7c3aed;border-radius:10px;padding:20px;text-align:center;">
                <p style="margin:0 0 10px;color:#c4b5fd;font-size:13px;font-weight:700;letter-spacing:1px;text-transform:uppercase;">
                  🔗 {_share_label}
                </p>
                <p style="margin:0 0 12px;color:#888;font-size:12px;line-height:1.6;">
                  {_share_desc}
                </p>
                <a href="{download_url}"
                   style="display:block;background:#1a1a1a;border:1px solid #7c3aed;border-radius:8px;
                          font-size:13px;color:#c4b5fd;font-family:'Courier New',monospace;
                          word-break:break-all;text-decoration:none;padding:12px;">
                  {download_url}
                </a>
                <p style="margin:10px 0 0;color:#555;font-size:11px;">
                  {_share_tap}
                </p>
              </div>
            </td>
          </tr>

          <!-- What this means -->
          <tr>
            <td style="padding:0 32px 32px;">
              <p style="margin:0 0 12px;color:#555;font-size:12px;line-height:1.7;">
                <strong style="color:#777;">What does this mean?</strong><br>
                {_what_body}
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
        "subject": f"&#10003; Your {_media} has been certified — VeriFYD #{short_id}",
        "html":    html,
    })




def send_trust_mail_ready_email(
    to_email: str,
    certificate_id: str,
    subject: str = "",
    sender: str = "",
    recipient: str = "",
    message_date: str = "",
    report_url: str = "",
    package_url: str = "",
    original_sha256: str = "",
    attachment_count: int = 0,
) -> bool:
    # Send Trust Mail completion email. Trust Mail is evidence preservation, not AI detection.
    short_id = (certificate_id or "")[:8].upper()
    report_url = report_url or f"{BACKEND_URL}/download-trust-mail/{certificate_id}"
    package_url = package_url or f"{BACKEND_URL}/download-trust-mail-package/{certificate_id}"
    verify_url = f"{SITE_URL}/verify-certificate"
    safe_subject = (subject or "Certified email record")[:120]
    safe_sender = (sender or "Not available")[:160]
    safe_recipient = (recipient or "Not available")[:160]
    html = f"""
<!DOCTYPE html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1.0"></head>
<body style="margin:0;padding:0;background-color:#0a0a0a;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;">
<table width="100%" cellpadding="0" cellspacing="0" style="background-color:#0a0a0a;padding:40px 20px;"><tr><td align="center">
<table width="560" cellpadding="0" cellspacing="0" style="background-color:#111111;border-radius:12px;border:1px solid #222222;overflow:hidden;max-width:560px;width:100%;">
{_header_html()}
<tr><td style="padding:40px 32px 24px;text-align:center;"><div style="display:inline-block;background:#1f1600;border:1px solid #f59e0b;border-radius:100px;padding:8px 20px;margin-bottom:24px;"><span style="color:#f59e0b;font-size:13px;font-weight:700;letter-spacing:1px;text-transform:uppercase;">Trust Mail Certified</span></div><h2 style="margin:0 0 8px;font-size:24px;font-weight:800;color:#ffffff;">Your certified email record is ready</h2><p style="margin:0;color:#888888;font-size:15px;line-height:1.6;">VeriFYD preserved this email file, extracted headers, inventoried attachments, and generated certified evidence hashes.</p></td></tr>
<tr><td style="padding:0 32px 24px;"><div style="background:#0f0f0f;border:1px solid #2a2a2a;border-radius:12px;padding:20px;color:#d1d5db;font-size:14px;line-height:1.8;"><strong style="color:#ffffff;">Subject:</strong> {safe_subject}<br><strong style="color:#ffffff;">From:</strong> {safe_sender}<br><strong style="color:#ffffff;">To:</strong> {safe_recipient}<br><strong style="color:#ffffff;">Date:</strong> {message_date or 'Recorded in VeriFYD certificate'}<br><strong style="color:#ffffff;">Attachments:</strong> {attachment_count}<br><strong style="color:#ffffff;">Certificate ID:</strong> <span style="color:#f59e0b;font-family:'Courier New',monospace;">{certificate_id}</span></div></td></tr>
<tr><td style="padding:0 32px 28px;"><a href="{report_url}" style="display:block;background:#f59e0b;color:#000;text-decoration:none;text-align:center;padding:16px;border-radius:10px;font-size:15px;font-weight:800;margin-bottom:12px;">Download Certified Trust Mail Report</a><a href="{package_url}" style="display:block;background:#1a1a1a;color:#f59e0b;text-decoration:none;text-align:center;padding:16px;border-radius:10px;font-size:15px;font-weight:700;border:1px solid #333;margin-bottom:12px;">Download Trust Mail Evidence Package</a><a href="{verify_url}" style="display:block;background:#111;color:#c4b5fd;text-decoration:none;text-align:center;padding:16px;border-radius:10px;font-size:15px;font-weight:700;border:1px solid #7c3aed;">Verify Certificate ID</a></td></tr>
<tr><td style="padding:0 32px 32px;"><p style="margin:0;color:#555;font-size:12px;line-height:1.7;">Trust Mail preserves email evidence and metadata. It does not claim the sender identity is legally proven; it records the submitted file, headers, content, attachments, and hashes for later verification.</p></td></tr>
{_footer_html()}
</table></td></tr></table></body></html>"""
    return _send({"from": f"{FROM_NAME} <{FROM_ADDRESS}>", "to": [to_email], "subject": f"Trust Mail Certified — VeriFYD #{short_id}", "html": html})

def send_web_capture_ready_email(
    to_email: str,
    certificate_id: str,
    captured_url: str,
    final_url: str = "",
    page_title: str = "",
    report_url: str = "",
    package_url: str = "",
    captured_at: str = "",
    screenshot_sha256: str = "",
    html_sha256: str = "",
) -> bool:
    """Send Certified Web Capture completion email.

    Web capture is evidence preservation, not AI/deepfake detection, so this
    intentionally avoids authenticity-score and AI-risk wording.
    """
    short_id = (certificate_id or "")[:8].upper()
    captured_url = captured_url or final_url or ""
    final_url = final_url or captured_url
    page_title = page_title or "Captured webpage"
    report_url = report_url or f"{BACKEND_URL}/download-web-capture/{certificate_id}"
    package_url = package_url or f"{BACKEND_URL}/download-web-capture-package/{certificate_id}"
    verify_url = f"{SITE_URL}/verify-certificate"
    safe_title = page_title[:80] + ("..." if len(page_title) > 80 else "")
    safe_url = captured_url[:120] + ("..." if len(captured_url) > 120 else "")

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
        <table width="560" cellpadding="0" cellspacing="0" style="background-color:#111111;border-radius:12px;border:1px solid #222222;overflow:hidden;max-width:560px;width:100%;">
          {_header_html()}

          <tr>
            <td style="padding:40px 32px 24px;text-align:center;">
              <div style="display:inline-block;background:#1f1600;border:1px solid #f59e0b;border-radius:100px;padding:8px 20px;margin-bottom:24px;">
                <span style="color:#f59e0b;font-size:13px;font-weight:700;letter-spacing:1px;text-transform:uppercase;">
                  Certified Web Capture Created
                </span>
              </div>
              <h2 style="margin:0 0 8px;font-size:24px;font-weight:800;color:#ffffff;">
                Your certified web capture is ready
              </h2>
              <p style="margin:0;color:#888888;font-size:15px;line-height:1.6;">
                VeriFYD captured and preserved this public webpage as timestamped evidence.
              </p>
            </td>
          </tr>

          <tr>
            <td style="padding:0 32px 24px;">
              <div style="background:#0f0f0f;border:1px solid #2a2a2a;border-radius:12px;padding:20px;color:#d1d5db;font-size:14px;line-height:1.8;">
                <strong style="color:#ffffff;">Page Title:</strong> {safe_title}<br>
                <strong style="color:#ffffff;">Captured URL:</strong> <span style="word-break:break-all;">{safe_url}</span><br>
                <strong style="color:#ffffff;">Certificate ID:</strong> <span style="color:#f59e0b;font-family:'Courier New',monospace;">{certificate_id}</span><br>
                <strong style="color:#ffffff;">Captured At:</strong> {captured_at or "Recorded in VeriFYD certificate"}
              </div>
            </td>
          </tr>

          <tr>
            <td style="padding:0 32px 28px;">
              <a href="{report_url}"
                 style="display:block;background:#f59e0b;color:#000000;text-decoration:none;text-align:center;padding:16px;border-radius:10px;font-size:15px;font-weight:800;margin-bottom:12px;">
                Download Certified Web Capture Report
              </a>
              <a href="{package_url}"
                 style="display:block;background:#1a1a1a;color:#f59e0b;text-decoration:none;text-align:center;padding:16px;border-radius:10px;font-size:15px;font-weight:700;border:1px solid #333;margin-bottom:12px;">
                Download Web Capture Evidence Package
              </a>
              <a href="{verify_url}"
                 style="display:block;background:#111111;color:#c4b5fd;text-decoration:none;text-align:center;padding:16px;border-radius:10px;font-size:15px;font-weight:700;border:1px solid #7c3aed;">
                Verify Certificate ID
              </a>
            </td>
          </tr>

          <tr>
            <td style="padding:0 32px 32px;">
              <p style="margin:0 0 12px;color:#555;font-size:12px;line-height:1.7;">
                <strong style="color:#777;">What is included?</strong><br>
                The evidence package includes the captured screenshot, HTML snapshot, metadata, SHA-256 hash records,
                certified PDF report, certificate ID, and verification details.
              </p>
              <p style="margin:0;color:#444;font-size:11px;line-height:1.6;">
                Note: Certified Web Capture preserves what a public webpage displayed at capture time. It does not certify
                that the webpage's claims are true or perform AI/deepfake analysis.
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
        "from": f"{FROM_NAME} <{FROM_ADDRESS}>",
        "to": [to_email],
        "subject": f"Certified Web Capture Ready — VeriFYD #{short_id}",
        "html": html,
    })


def send_enterprise_welcome_email(
    to_email:     str,
    company_name: str,
    api_key:      str,
    brand_color:  str = "#f59e0b",
) -> bool:
    """
    Sent automatically when an Enterprise PayPal subscription activates.
    Contains the embed code, API key, and quick-start instructions.
    """
    embed_script = (
        f'&lt;div id="verifyd-widget"&gt;&lt;/div&gt;\n'
        f'&lt;script src="{BACKEND_URL}/widget.js?key={api_key}"&gt;&lt;/script&gt;'
    )

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
        <table width="560" cellpadding="0" cellspacing="0"
               style="background-color:#111111;border-radius:12px;border:1px solid #222222;
                      overflow:hidden;max-width:560px;width:100%;">
          {_header_html()}

          <tr>
            <td style="padding:40px 32px 24px;text-align:center;">
              <div style="display:inline-block;background:#0c1a07;border:1px solid #f59e0b;
                          border-radius:100px;padding:8px 20px;margin-bottom:24px;">
                <span style="color:#f59e0b;font-size:13px;font-weight:700;
                             letter-spacing:1px;text-transform:uppercase;">
                  &#9889;&nbsp; Enterprise Plan Active
                </span>
              </div>
              <h2 style="margin:0 0 10px;font-size:24px;font-weight:800;color:#ffffff;">
                Welcome to VeriFYD Enterprise
              </h2>
              <p style="margin:0;color:#888888;font-size:15px;line-height:1.7;">
                Your subscription is active. Below is everything you need to embed
                the VeriFYD detection widget on <strong style="color:#ccc;">{company_name}</strong>.
              </p>
            </td>
          </tr>

          <tr>
            <td style="padding:0 32px 28px;">
              <p style="margin:0 0 10px;color:#9ca3af;font-size:12px;
                        letter-spacing:2px;text-transform:uppercase;font-weight:600;">
                Your API Key &mdash; keep this private
              </p>
              <div style="background:#0f0f0f;border:1px solid #374151;border-radius:10px;
                          padding:16px 20px;">
                <code style="color:#f59e0b;font-size:14px;font-family:'Courier New',monospace;
                             word-break:break-all;letter-spacing:0.5px;">
                  {api_key}
                </code>
              </div>
            </td>
          </tr>

          <tr>
            <td style="padding:0 32px 28px;">
              <p style="margin:0 0 10px;color:#9ca3af;font-size:12px;
                        letter-spacing:2px;text-transform:uppercase;font-weight:600;">
                Embed Code &mdash; paste into your website
              </p>
              <div style="background:#0f0f0f;border:1px solid #374151;border-radius:10px;
                          padding:16px 20px;">
                <code style="color:#a78bfa;font-size:12px;font-family:'Courier New',monospace;
                             white-space:pre-wrap;word-break:break-all;line-height:1.8;
                             display:block;">
{embed_script}
                </code>
              </div>
            </td>
          </tr>

          <tr>
            <td style="padding:0 32px 32px;text-align:center;">
              <a href="mailto:support@vfvid.com"
                 style="display:inline-block;background:#1a1a1a;color:#f59e0b;
                        text-decoration:none;padding:14px 28px;border-radius:8px;
                        font-size:14px;font-weight:700;border:1px solid #333;">
                Contact Enterprise Support
              </a>
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
        "subject": f"\U0001f680 Your VeriFYD Enterprise account is ready \u2014 {company_name}",
        "html":    html,
    })




def send_trust_desk_ready_email(
    to_email: str,
    trust_desk_job_id: str,
    organization: str,
    case_number: str,
    download_url: str,
    summary: dict | None = None,
) -> bool:
    """Send Trust Desk package-ready email."""
    summary = summary or {}
    short_id = (trust_desk_job_id or "")[:8].upper()
    org = organization or "Not provided"
    case = case_number or "Not provided"
    total = summary.get("total_files", 0)
    supported = summary.get("supported_files", 0)
    unsupported = summary.get("unsupported_files", 0)
    videos = summary.get("video_files", 0)
    photos = summary.get("photo_files", 0)
    audio = summary.get("audio_files", 0)
    docs = summary.get("document_files", 0)

    html = f"""
<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1.0"></head>
<body style="margin:0;padding:0;background-color:#0a0a0a;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;">
  <table width="100%" cellpadding="0" cellspacing="0" style="background-color:#0a0a0a;padding:40px 20px;">
    <tr><td align="center">
      <table width="560" cellpadding="0" cellspacing="0" style="background-color:#111111;border-radius:12px;border:1px solid #222;max-width:560px;width:100%;overflow:hidden;">
        {_header_html()}
        <tr><td style="padding:36px 32px 20px;text-align:center;">
          <div style="display:inline-block;background:#1f1600;border:1px solid #f59e0b;border-radius:100px;padding:8px 18px;margin-bottom:22px;">
            <span style="color:#f59e0b;font-size:13px;font-weight:700;letter-spacing:1px;text-transform:uppercase;">Trust Desk Package Ready</span>
          </div>
          <h2 style="margin:0 0 8px;font-size:24px;font-weight:800;color:#fff;">Your Trust Desk package is ready</h2>
          <p style="margin:0;color:#9ca3af;font-size:15px;line-height:1.6;">VeriFYD completed ZIP intake, file inventory, hash preservation, and package assembly.</p>
        </td></tr>
        <tr><td style="padding:8px 32px 24px;">
          <div style="background:#0f0f0f;border:1px solid #2a2a2a;border-radius:10px;padding:18px 20px;color:#d1d5db;font-size:14px;line-height:1.8;">
            <strong style="color:#fff;">Trust Desk Job:</strong> {trust_desk_job_id}<br>
            <strong style="color:#fff;">Organization:</strong> {org}<br>
            <strong style="color:#fff;">Case / Claim / Matter:</strong> {case}<br>
            <strong style="color:#fff;">Total Files:</strong> {total}<br>
            <strong style="color:#fff;">Supported Files:</strong> {supported}<br>
            <strong style="color:#fff;">Videos:</strong> {videos} &nbsp; <strong style="color:#fff;">Photos:</strong> {photos} &nbsp; <strong style="color:#fff;">Audio:</strong> {audio} &nbsp; <strong style="color:#fff;">Documents:</strong> {docs}<br>
            <strong style="color:#fff;">Unsupported Preserved:</strong> {unsupported}
          </div>
        </td></tr>
        <tr><td style="padding:0 32px 34px;text-align:center;">
          <a href="{download_url}" style="display:inline-block;background:#f59e0b;color:#000;text-decoration:none;padding:14px 26px;border-radius:8px;font-size:14px;font-weight:800;">Download Trust Desk Package</a>
        </td></tr>
        {_footer_html()}
      </table>
    </td></tr>
  </table>
</body>
</html>"""
    return _send({
        "from": f"{FROM_NAME} <{FROM_ADDRESS}>",
        "to": [to_email],
        "subject": f"VeriFYD Trust Desk Package Ready - #{short_id}",
        "html": html,
    })

# ─────────────────────────────────────────────
# Certified Send / VeriFYD Trust Mail — branded delivery email
# ─────────────────────────────────────────────
def send_certified_delivery_email(
    *,
    recipient_email: str,
    recipient_name: str = "",
    sender_email: str = "",
    message: str = "",
    certificate_id: str,
    original_filename: str = "",
    certified_to: str = "",
    authenticity: str | int = "",
    ai_score: str | int = "",
    upload_time: str = "",
    original_sha256: str = "",
    certified_document_sha256: str = "",
    certified_file_package_sha256: str = "",
    report_url: str = "",
    package_url: str = "",
    verify_url: str = "",
    include_report: bool = True,
    include_package: bool = True,
    include_verify_link: bool = True,
    attachment_status: list | None = None,
    attachments: list | None = None,
) -> bool:
    """
    Send an official VeriFYD Trust Mail delivery email to a third-party recipient.

    This is intentionally separate from the normal uploader notification email.
    It tells the recipient the evidence was certified through VeriFYD before delivery,
    includes the certificate ID, hashes, verification link, and optional attachments.
    """
    import html as _html

    def esc(value, limit: int = 1200) -> str:
        text = str(value or "")
        if len(text) > limit:
            text = text[: limit - 1] + "…"
        return _html.escape(text)

    def _clean_percent(value) -> str:
        text = str(value or "").strip()
        if not text:
            return ""
        return text if text.endswith("%") else f"{text}%"

    short_id = str(certificate_id or "")[:8].upper()
    recipient_label = esc(recipient_name or recipient_email, 160)
    sender_label = esc(sender_email or certified_to or "VeriFYD user", 200)
    safe_filename = esc(original_filename or "Certified evidence", 240)
    verify_url = verify_url or f"{SITE_URL}/verify-certificate"

    authenticity_line = ""
    if str(authenticity).strip() != "":
        authenticity_line = f"<p style='margin:0 0 7px;color:#d1d5db;font-size:14px;'><strong>Certification Score:</strong> {_clean_percent(esc(authenticity,80))}</p>"
    ai_line = ""
    if str(ai_score).strip() != "":
        ai_line = f"<p style='margin:0;color:#d1d5db;font-size:14px;'><strong>AI / Manipulation Indicators:</strong> {_clean_percent(esc(ai_score,80))}</p>"

    note_html = ""
    if message:
        note_html = f"""
          <div style="margin:22px 0 0;padding:16px;border-radius:10px;background:#101827;border:1px solid #263246;">
            <p style="margin:0 0 8px;color:#9ca3af;font-size:12px;letter-spacing:1px;text-transform:uppercase;">Sender Note</p>
            <p style="margin:0;color:#e5e7eb;font-size:14px;line-height:1.55;">{esc(message, 700)}</p>
          </div>
        """

    attach_lines = []
    for item in list(attachment_status or []):
        try:
            name = esc(item.get("filename", "file"), 180)
            status = esc(item.get("status", ""), 260)
            attach_lines.append(f"<li style='margin:4px 0;color:#9ca3af;font-size:13px;'>{name}: {status}</li>")
        except Exception:
            continue
    attachment_html = ""
    if attach_lines:
        attachment_html = f"""
          <div style="margin:18px 0 0;">
            <p style="margin:0 0 6px;color:#9ca3af;font-size:12px;letter-spacing:1px;text-transform:uppercase;">Attachment Status</p>
            <ul style="margin:0;padding-left:18px;">{''.join(attach_lines)}</ul>
          </div>
        """

    # Email-client-safe button table. Keep inline CSS; do not rely on Tailwind/classes.
    button_cells = []
    if include_report and report_url:
        button_cells.append(f"""
          <td align="center" valign="top" style="padding:0 8px 10px 0;">
            <a href="{esc(report_url, 2000)}" style="display:inline-block;width:178px;min-width:178px;padding:14px 10px;background:#2563eb;color:#ffffff;text-decoration:none;border-radius:10px;font-weight:800;font-size:14px;line-height:1.25;text-align:center;box-shadow:0 8px 18px rgba(37,99,235,0.28);">
              Download Certified Report
            </a>
          </td>
        """)
    if include_package and package_url:
        button_cells.append(f"""
          <td align="center" valign="top" style="padding:0 8px 10px 0;">
            <a href="{esc(package_url, 2000)}" style="display:inline-block;width:178px;min-width:178px;padding:14px 10px;background:#9333ea;color:#ffffff;text-decoration:none;border-radius:10px;font-weight:800;font-size:14px;line-height:1.25;text-align:center;box-shadow:0 8px 18px rgba(147,51,234,0.28);">
              Download Evidence Package
            </a>
          </td>
        """)
    if include_verify_link and verify_url:
        button_cells.append(f"""
          <td align="center" valign="top" style="padding:0 0 10px 0;">
            <a href="{esc(verify_url, 2000)}" style="display:inline-block;width:178px;min-width:178px;padding:14px 10px;background:#f59e0b;color:#111827;text-decoration:none;border-radius:10px;font-weight:900;font-size:14px;line-height:1.25;text-align:center;box-shadow:0 8px 18px rgba(245,158,11,0.30);">
              Verify Certificate
            </a>
          </td>
        """)

    buttons_html = ""
    if button_cells:
        buttons_html = f"""
          <table role="presentation" cellpadding="0" cellspacing="0" border="0" width="100%" style="margin:24px 0 4px;border-collapse:collapse;">
            <tr>
              {''.join(button_cells)}
            </tr>
          </table>
        """

    html = f"""
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body style="margin:0;padding:0;background-color:#050505;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;">
  <table width="100%" cellpadding="0" cellspacing="0" style="background-color:#050505;padding:34px 16px;">
    <tr>
      <td align="center">
        <table width="680" cellpadding="0" cellspacing="0" style="max-width:680px;width:100%;background:#0b0f14;border:1px solid #1f2937;border-radius:16px;overflow:hidden;">
          <tr>
            <td style="padding:28px 28px 24px;text-align:center;background:linear-gradient(135deg,#050505 0%,#111827 58%,#1e1b4b 100%);border-bottom:1px solid #263246;">
              <div style="display:inline-block;width:70px;height:70px;border-radius:50%;border:2px solid #38bdf8;box-shadow:0 0 22px rgba(56,189,248,.35);line-height:70px;text-align:center;color:#22c55e;font-size:38px;font-weight:900;margin-bottom:10px;">✓</div>
              <h1 style="margin:0;font-size:30px;font-weight:900;color:#ffffff;letter-spacing:-0.7px;">
                Veri<span style="color:#f59e0b;">FYD</span> Trust Mail
              </h1>
              <p style="margin:8px 0 0;color:#93c5fd;font-size:12px;letter-spacing:2px;text-transform:uppercase;font-weight:700;">
                Certified evidence delivered with VeriFYD seal
              </p>
            </td>
          </tr>
          <tr>
            <td style="padding:30px 30px 26px;">
              <h2 style="margin:0 0 10px;font-size:22px;color:#f9fafb;">You received VeriFYD Trust Mail</h2>
              <p style="margin:0 0 18px;color:#d1d5db;font-size:15px;line-height:1.65;">
                {recipient_label}, <strong style="color:#ffffff;">{sender_label}</strong> sent you a VeriFYD-certified record.
                This file was certified through VeriFYD before delivery and includes a Certificate ID, SHA-256 hash record, verification link, and certified downloads when available.
              </p>

              <div style="background:#111827;border:1px solid #334155;border-radius:12px;padding:18px;margin:20px 0;">
                <p style="margin:0 0 8px;color:#9ca3af;font-size:12px;letter-spacing:1px;text-transform:uppercase;">Certificate ID</p>
                <p style="margin:0;color:#ffffff;font-size:17px;font-family:'Courier New',monospace;font-weight:700;word-break:break-all;">{esc(certificate_id,120)}</p>
                <div style="height:1px;background:#243244;margin:16px 0;"></div>
                <p style="margin:0 0 7px;color:#d1d5db;font-size:14px;"><strong>Certified item:</strong> {safe_filename}</p>
                <p style="margin:0 0 7px;color:#d1d5db;font-size:14px;"><strong>Certified to:</strong> {esc(certified_to or sender_email,220)}</p>
                <p style="margin:0 0 7px;color:#d1d5db;font-size:14px;"><strong>Certified on:</strong> {esc(upload_time,160)}</p>
                {authenticity_line}
                {ai_line}
              </div>

              <div style="background:#07111f;border:1px solid #1e3a5f;border-radius:12px;padding:16px;margin:20px 0;">
                <p style="margin:0 0 8px;color:#93c5fd;font-size:12px;letter-spacing:1px;text-transform:uppercase;">SHA-256 Hashes</p>
                <p style="margin:0 0 8px;color:#dbeafe;font-size:12px;line-height:1.55;word-break:break-all;"><strong>Original:</strong> {esc(original_sha256 or 'Not available',160)}</p>
                <p style="margin:0 0 8px;color:#dbeafe;font-size:12px;line-height:1.55;word-break:break-all;"><strong>Certified Report:</strong> {esc(certified_document_sha256 or 'Not available',160)}</p>
                <p style="margin:0;color:#dbeafe;font-size:12px;line-height:1.55;word-break:break-all;"><strong>Evidence Package:</strong> {esc(certified_file_package_sha256 or 'Not available',160)}</p>
              </div>

              {note_html}

              {buttons_html}

              {attachment_html}

              <p style="margin:22px 0 0;color:#6b7280;font-size:12px;line-height:1.55;">
                VeriFYD certification records can be independently verified using the Certificate ID above. Links may expire based on plan retention and security settings; the certificate record remains independently verifiable.
              </p>
            </td>
          </tr>
          {_footer_html()}
        </table>
      </td>
    </tr>
  </table>
</body>
</html>
"""

    payload = {
        "from": f"{FROM_NAME} <{FROM_ADDRESS}>",
        "to": [recipient_email],
        "subject": f"VeriFYD Trust Mail — Certificate #{short_id}",
        "html": html,
    }
    if attachments:
        payload["attachments"] = attachments
    return _send(payload)
