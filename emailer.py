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
