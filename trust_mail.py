# ============================================================
#  VeriFYD — trust_mail.py
#  Trust Mail Phase 1: .eml/.msg preservation, header/body extraction,
#  attachment inventory, PDF report, and evidence ZIP package.
# ============================================================
from __future__ import annotations
import hashlib, json, os, re, shutil, zipfile
from datetime import datetime, timezone
from email import policy
from email.parser import BytesParser
from email.header import decode_header, make_header
from pathlib import Path
from typing import Any, Dict, List


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data or b"").hexdigest()


def _safe_filename(name: str, fallback: str = "file") -> str:
    name = os.path.basename(str(name or "").replace("\\", "/")).strip().replace("\x00", "")
    if not name or name in (".", ".."):
        name = fallback
    return re.sub(r"[^A-Za-z0-9._ -]+", "_", name)[:160] or fallback


def _decode_header_value(value: Any) -> str:
    value = "" if value is None else str(value)
    try:
        return str(make_header(decode_header(value))).strip()
    except Exception:
        return value.strip()


def _clean_text(value: Any, max_len: int | None = None) -> str:
    if value is None:
        text = ""
    elif isinstance(value, bytes):
        text = value.decode("utf-8", errors="replace")
    else:
        text = str(value)
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{4,}", "\n\n\n", text).strip()
    if max_len and len(text) > max_len:
        text = text[: max_len - 1] + "…"
    return text


def _html_to_text(html: str) -> str:
    text = str(html or "")
    text = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", text)
    text = re.sub(r"(?i)<\s*br\s*/?\s*>", "\n", text)
    text = re.sub(r"(?i)</\s*(p|div|tr|li|table|blockquote|h[1-6])\s*>", "\n", text)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    import html as _html
    return _clean_text(_html.unescape(text))


def _headers_to_dict(msg) -> Dict[str, str]:
    important = ["From", "To", "Cc", "Bcc", "Reply-To", "Return-Path", "Subject", "Date", "Message-ID", "In-Reply-To", "References", "Delivered-To", "Authentication-Results", "Received-SPF", "DKIM-Signature", "DMARC-Filter", "X-Originating-IP", "X-Mailer", "User-Agent", "MIME-Version", "Content-Type"]
    out: Dict[str, str] = {}
    for name in important:
        vals = msg.get_all(name, [])
        if vals:
            out[name] = " | ".join(_decode_header_value(v) for v in vals)
    for name, value in list(msg.items())[:200]:
        if str(name) not in out:
            out[str(name)] = _decode_header_value(value)
    return out


def parse_eml(src_path: str, attachments_dir: str) -> Dict[str, Any]:
    with open(src_path, "rb") as fh:
        msg = BytesParser(policy=policy.default).parse(fh)
    headers = _headers_to_dict(msg)
    body_plain: List[str] = []
    body_html_raw: List[str] = []
    attachments: List[Dict[str, Any]] = []
    Path(attachments_dir).mkdir(parents=True, exist_ok=True)
    parts = msg.walk() if msg.is_multipart() else [msg]
    for idx, part in enumerate(parts, start=1):
        ctype = str(part.get_content_type() or "").lower()
        disp = str(part.get_content_disposition() or "").lower()
        filename = part.get_filename()
        decoded_filename = _decode_header_value(filename) if filename else ""
        if disp == "attachment" or decoded_filename:
            payload = part.get_payload(decode=True) or b""
            safe = _safe_filename(decoded_filename or f"attachment_{idx}")
            final = os.path.join(attachments_dir, safe)
            base, ext = os.path.splitext(final)
            n = 1
            while os.path.exists(final):
                n += 1
                final = f"{base}_{n}{ext}"
            with open(final, "wb") as af:
                af.write(payload)
            attachments.append({"filename": os.path.basename(final), "content_type": ctype, "size_bytes": len(payload), "sha256": sha256_bytes(payload)})
            continue
        if ctype in ("text/plain", "text/html"):
            try:
                content = part.get_content()
            except Exception:
                content = _clean_text(part.get_payload(decode=True) or b"")
            if ctype == "text/plain":
                body_plain.append(_clean_text(content))
            else:
                body_html_raw.append(str(content or ""))
    html_body = "\n\n".join(x for x in body_html_raw if x).strip()
    plain_body = "\n\n".join(x for x in body_plain if x).strip()
    return {"source_type": "EML", "headers": headers, "from": headers.get("From", ""), "to": headers.get("To", ""), "cc": headers.get("Cc", ""), "bcc": headers.get("Bcc", ""), "reply_to": headers.get("Reply-To", ""), "return_path": headers.get("Return-Path", ""), "subject": headers.get("Subject", ""), "message_date": headers.get("Date", ""), "message_id": headers.get("Message-ID", ""), "body_text": plain_body or _html_to_text(html_body), "body_html": html_body, "attachments": attachments}


def parse_msg(src_path: str, attachments_dir: str) -> Dict[str, Any]:
    try:
        import extract_msg
    except Exception as exc:
        raise RuntimeError("Missing dependency extract-msg for .msg Trust Mail parsing") from exc
    Path(attachments_dir).mkdir(parents=True, exist_ok=True)
    msg = extract_msg.Message(src_path)
    try:
        sender = _clean_text(getattr(msg, "sender", "") or getattr(msg, "sender_email", ""))
        to = _clean_text(getattr(msg, "to", ""))
        cc = _clean_text(getattr(msg, "cc", ""))
        bcc = _clean_text(getattr(msg, "bcc", ""))
        subject = _clean_text(getattr(msg, "subject", ""))
        date = _clean_text(getattr(msg, "date", ""))
        message_id = _clean_text(getattr(msg, "messageId", "") or getattr(msg, "message_id", ""))
        headers = {"From": sender, "To": to, "Cc": cc, "Bcc": bcc, "Subject": subject, "Date": date, "Message-ID": message_id}
        attachments: List[Dict[str, Any]] = []
        for idx, att in enumerate(list(getattr(msg, "attachments", []) or [])[:100], start=1):
            name = getattr(att, "longFilename", "") or getattr(att, "shortFilename", "") or getattr(att, "displayName", "") or f"attachment_{idx}"
            safe = _safe_filename(name, f"attachment_{idx}")
            data = getattr(att, "data", b"") or b""
            if isinstance(data, str):
                data = data.encode("utf-8", errors="replace")
            if data:
                final = os.path.join(attachments_dir, safe)
                base, ext = os.path.splitext(final)
                n = 1
                while os.path.exists(final):
                    n += 1
                    final = f"{base}_{n}{ext}"
                with open(final, "wb") as af:
                    af.write(data)
                attachments.append({"filename": os.path.basename(final), "content_type": "", "size_bytes": len(data), "sha256": sha256_bytes(data)})
            else:
                attachments.append({"filename": safe, "content_type": "", "size_bytes": 0, "sha256": "", "note": "Attachment name preserved; binary extraction unavailable in runtime."})
        html_body = str(getattr(msg, "htmlBody", "") or getattr(msg, "html_body", "") or "")
        plain_body = _clean_text(getattr(msg, "body", "") or "")
        return {"source_type": "MSG", "headers": headers, "from": sender, "to": to, "cc": cc, "bcc": bcc, "reply_to": "", "return_path": "", "subject": subject, "message_date": date, "message_id": message_id, "body_text": plain_body or _html_to_text(html_body), "body_html": html_body, "attachments": attachments}
    finally:
        try:
            msg.close()
        except Exception:
            pass


def parse_trust_mail(src_path: str, attachments_dir: str) -> Dict[str, Any]:
    ext = Path(src_path).suffix.lower()
    if ext == ".eml":
        return parse_eml(src_path, attachments_dir)
    if ext == ".msg":
        return parse_msg(src_path, attachments_dir)
    raise ValueError("Trust Mail Phase 1 accepts .eml and .msg files only.")


def create_trust_mail_report_pdf(record: Dict[str, Any], report_path: str, cert_id: str, filename: str, original_sha256: str) -> str:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    from reportlab.lib import colors
    import textwrap
    width, height = letter
    c = canvas.Canvas(report_path, pagesize=letter)
    c.setAuthor("VeriFYD")
    c.setTitle(f"VeriFYD Certified Trust Mail Report {cert_id}")
    margin, y, page = 42, height - 48, 1
    def mark():
        c.setFont("Helvetica-Bold", 8)
        x_right = width - 34
        x = x_right - c.stringWidth("VERIFYD", "Helvetica-Bold", 8)
        c.setFillGray(0.15); c.drawString(x, 24, "VERI")
        c.setFillColor(colors.HexColor("#f59e0b")); c.drawString(x + c.stringWidth("VERI", "Helvetica-Bold", 8), 24, "FYD")
        c.setFont("Helvetica", 5.5); c.setFillGray(0.35); c.drawRightString(x_right, 17, "trust mail certified")
    def footer():
        c.setFont("Helvetica", 6.4); c.setFillGray(0.45)
        c.drawString(margin, 24, f"VeriFYD Certified Trust Mail Report • Page {page}")
        c.drawRightString(width - margin, 24, f"Certificate: {cert_id[:18]}")
        mark()
    def new_page():
        nonlocal y, page
        footer(); c.showPage(); page += 1; y = height - 48
        c.setFont("Helvetica-Bold", 12); c.setFillGray(0.05); c.drawString(margin, y, "VeriFYD Certified Trust Mail Report"); y -= 22
    c.setFont("Helvetica-Bold", 18); c.setFillGray(0.05); c.drawString(margin, y, "VeriFYD Certified Trust Mail Report"); y -= 20
    c.setFont("Helvetica", 8.5); c.setFillGray(0.35); c.drawString(margin, y, "Email preservation, header extraction, attachment inventory, and SHA-256 evidence hashes."); y -= 22
    rows = [("Certificate ID", cert_id), ("Source Type", record.get("source_type", "")), ("Original File", filename), ("Subject", record.get("subject", "")), ("From", record.get("from", "")), ("To", record.get("to", "")), ("Cc", record.get("cc", "")), ("Date", record.get("message_date", "")), ("Message-ID", record.get("message_id", "")), ("Attachments", str(len(record.get("attachments") or []))), ("Original SHA-256", original_sha256)]
    label_w, row_h = 105, 18
    for label, value in rows:
        if y < 80: new_page()
        c.setFillGray(0.94); c.rect(margin, y-row_h+4, label_w, row_h, fill=1, stroke=1)
        c.setFillGray(1); c.rect(margin+label_w, y-row_h+4, width-2*margin-label_w, row_h, fill=1, stroke=1)
        c.setFont("Helvetica-Bold", 7.5); c.setFillGray(0.05); c.drawString(margin+4, y-row_h+9, label)
        c.setFont("Helvetica", 7.2); c.drawString(margin+label_w+4, y-row_h+9, _clean_text(value, 118))
        y -= row_h
    y -= 16
    if record.get("attachments"):
        c.setFont("Helvetica-Bold", 10); c.setFillGray(0.05); c.drawString(margin, y, "Attachment Inventory"); y -= 14
        for att in record.get("attachments", [])[:100]:
            if y < 70: new_page()
            line = f"{att.get('filename','')} | {att.get('size_bytes',0)} bytes | SHA-256: {att.get('sha256','')}"
            c.setFont("Helvetica", 7.2); c.drawString(margin, y, _clean_text(line, 125)); y -= 10
        y -= 8
    c.setFont("Helvetica-Bold", 10); c.setFillGray(0.05); c.drawString(margin, y, "Message Body Preview"); y -= 14
    for raw in (record.get("body_text") or "[No readable body text found.]").splitlines()[:1200]:
        for line in textwrap.wrap(raw, width=100) or [""]:
            if y < 56: new_page()
            c.setFont("Helvetica", 7.5); c.setFillGray(0.08); c.drawString(margin, y, line[:120]); y -= 9.2
    footer(); c.save(); return report_path


def build_trust_mail_evidence_package(record: Dict[str, Any], *, original_path: str, report_path: str, package_path: str, cert_id: str, filename: str, original_sha256: str, report_sha256: str, attachments_dir: str) -> str:
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    metadata = {"title": "VeriFYD Certified Trust Mail Metadata", "record_type": "trust_mail", "certificate_id": cert_id, "issued_at_utc": now, "source_type": record.get("source_type", ""), "original_filename": filename, "subject": record.get("subject", ""), "from": record.get("from", ""), "to": record.get("to", ""), "cc": record.get("cc", ""), "bcc": record.get("bcc", ""), "reply_to": record.get("reply_to", ""), "return_path": record.get("return_path", ""), "message_date": record.get("message_date", ""), "message_id": record.get("message_id", ""), "attachment_count": len(record.get("attachments") or []), "issuer": "VeriFYD"}
    hashes = {"algorithm": "SHA-256", "original_email": original_sha256, "certified_trust_mail_report_pdf": report_sha256, "attachments": record.get("attachments") or []}
    tmp = package_path + ".tmp"
    if os.path.exists(tmp): os.remove(tmp)
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        zf.write(original_path, f"original/{_safe_filename(filename, 'original_email')}")
        zf.write(report_path, f"certified/VeriFYD_Certified_Trust_Mail_Report_{cert_id[:8]}.pdf")
        zf.writestr("verifyd/metadata.json", json.dumps(metadata, indent=2, ensure_ascii=False, sort_keys=True))
        zf.writestr("verifyd/headers.json", json.dumps(record.get("headers") or {}, indent=2, ensure_ascii=False, sort_keys=True))
        zf.writestr("verifyd/hashes.json", json.dumps(hashes, indent=2, ensure_ascii=False, sort_keys=True))
        zf.writestr("body/body.txt", record.get("body_text") or "")
        if record.get("body_html"):
            zf.writestr("body/body.html", record.get("body_html") or "")
        if os.path.isdir(attachments_dir):
            for p in Path(attachments_dir).rglob("*"):
                if p.is_file():
                    zf.write(str(p), f"attachments/{p.name}")
        zf.writestr("README.txt", f"VeriFYD Certified Trust Mail Evidence Package\n\nCertificate ID: {cert_id}\nIssued At UTC: {now}\nOriginal File: {filename}\n")
    os.replace(tmp, package_path)
    return package_path


def create_certified_trust_mail(src_path: str, cert_id: str, email: str = "", filename: str = "") -> Dict[str, Any]:
    filename = filename or os.path.basename(src_path)
    work_dir = Path(os.path.join(os.path.dirname(src_path) or ".", f"trust_mail_{cert_id}"))
    if work_dir.exists(): shutil.rmtree(work_dir, ignore_errors=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    attachments_dir = str(work_dir / "attachments")
    record = parse_trust_mail(src_path, attachments_dir)
    original_sha256 = sha256_file(src_path)
    report_path = str(work_dir / f"VeriFYD_Certified_Trust_Mail_Report_{cert_id[:8]}.pdf")
    create_trust_mail_report_pdf(record, report_path, cert_id, filename, original_sha256)
    report_sha256 = sha256_file(report_path)
    package_path = str(work_dir / f"VeriFYD_Trust_Mail_Evidence_Package_{cert_id[:8]}.zip")
    build_trust_mail_evidence_package(record, original_path=src_path, report_path=report_path, package_path=package_path, cert_id=cert_id, filename=filename, original_sha256=original_sha256, report_sha256=report_sha256, attachments_dir=attachments_dir)
    package_sha256 = sha256_file(package_path)
    return {**record, "work_dir": str(work_dir), "report_path": report_path, "package_path": package_path, "original_sha256": original_sha256, "report_sha256": report_sha256, "package_sha256": package_sha256}
