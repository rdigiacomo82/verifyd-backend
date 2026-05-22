# ============================================================
#  VeriFYD — doc_certifier.py
#
#  Creates certified/stamped PDF documents for VeriFYD Docs.
#  PDF input: overlays a VeriFYD verification footer on every page
#             and appends a certificate summary page.
#  DOCX/TXT/CSV/MD input: creates a certificate summary PDF.
# ============================================================

from __future__ import annotations

import os
import tempfile
from datetime import datetime, timezone
from typing import Dict, Any


def _short(text: str, limit: int = 90) -> str:
    text = str(text or "")
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _draw_certificate_page(c, width, height, cert_id: str, authenticity: int,
                           label: str, filename: str, sha256: str,
                           detail: Dict[str, Any] | None = None) -> None:
    detail = detail or {}
    margin = 54
    y = height - 64

    c.setFont("Helvetica-Bold", 24)
    c.drawString(margin, y, "VeriFYD Document Certificate")
    y -= 26

    c.setFont("Helvetica", 10)
    c.drawString(margin, y, f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    y -= 34

    # Status box
    c.setLineWidth(1)
    c.roundRect(margin, y - 92, width - margin * 2, 92, 10, stroke=1, fill=0)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(margin + 18, y - 28, f"Status: {label}")
    c.setFont("Helvetica-Bold", 15)
    c.drawString(margin + 18, y - 56, f"Authenticity Score: {authenticity}%")
    c.setFont("Helvetica", 9)
    c.drawString(margin + 18, y - 78, f"Certificate ID: {cert_id}")
    y -= 125

    rows = [
        ("Original file", _short(filename, 95)),
        ("SHA-256", sha256 or detail.get("sha256", "")),
        ("Document type", detail.get("document_type", "unknown")),
        ("Pages", str(detail.get("pages", 0))),
        ("Embedded images", str(detail.get("embedded_images", 0))),
        ("AI score", str(detail.get("ai_score", ""))),
        ("GPT score", str(detail.get("gpt_ai_score", ""))),
        ("Metadata score", str(detail.get("metadata_score", ""))),
        ("Text score", str(detail.get("text_score", ""))),
    ]

    c.setFont("Helvetica-Bold", 13)
    c.drawString(margin, y, "Verification Details")
    y -= 22

    for key, value in rows:
        if not value:
            continue
        c.setFont("Helvetica-Bold", 9)
        c.drawString(margin, y, f"{key}:")
        c.setFont("Helvetica", 8)
        if key == "SHA-256" and len(value) > 48:
            c.drawString(margin + 110, y, value[:64])
            y -= 12
            c.drawString(margin + 110, y, value[64:])
        else:
            c.drawString(margin + 110, y, _short(value, 82))
        y -= 18

    reasoning = str(detail.get("gpt_reasoning", "") or "")
    if reasoning:
        y -= 8
        c.setFont("Helvetica-Bold", 13)
        c.drawString(margin, y, "Analysis Summary")
        y -= 18
        c.setFont("Helvetica", 8)
        # crude wrapping without pulling in extra deps
        words = reasoning.replace("\n", " ").split()
        line = ""
        for word in words[:260]:
            candidate = (line + " " + word).strip()
            if len(candidate) > 95:
                c.drawString(margin, y, line)
                y -= 11
                line = word
                if y < 72:
                    c.drawString(margin, y, "[summary truncated]")
                    break
            else:
                line = candidate
        else:
            if line and y >= 72:
                c.drawString(margin, y, line)

    c.setFont("Helvetica-Oblique", 8)
    c.drawString(margin, 34, "VeriFYD provides analytical guidance, not legal notarization. Verify source and context independently.")


def _make_overlay(width, height, cert_id: str, authenticity: int, label: str) -> str:
    from reportlab.pdfgen import canvas

    fd, overlay_path = tempfile.mkstemp(suffix="_verifyd_overlay.pdf")
    os.close(fd)
    c = canvas.Canvas(overlay_path, pagesize=(width, height))

    # Footer band
    c.setFillGray(0.08)
    c.rect(0, 0, width, 38, fill=1, stroke=0)
    c.setFillGray(1)
    c.setFont("Helvetica-Bold", 9)
    c.drawString(32, 23, "VERI")
    c.setFillColorRGB(0.96, 0.62, 0.04)
    c.drawString(56, 23, "FYD")
    c.setFillGray(1)
    c.setFont("Helvetica", 8)
    c.drawString(92, 23, f"Certified Document | {label} | Authenticity {authenticity}% | ID {cert_id[:8].upper()}")
    c.setFont("Helvetica", 6.5)
    c.drawRightString(width - 32, 11, "verify at vfvid.com")
    c.save()
    return overlay_path


def stamp_document(src_path: str, dest_path: str, cert_id: str,
                   authenticity: int, label: str, filename: str,
                   sha256: str = "", detail: Dict[str, Any] | None = None) -> str:
    """
    Create a certified PDF at dest_path. Always outputs PDF.
    """
    ext = os.path.splitext(src_path)[1].lower()
    detail = detail or {}

    if ext == ".pdf":
        from pypdf import PdfReader, PdfWriter
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import letter

        reader = PdfReader(src_path)
        writer = PdfWriter()
        overlay_paths = []
        try:
            for page in reader.pages:
                width = float(page.mediabox.width)
                height = float(page.mediabox.height)
                overlay = _make_overlay(width, height, cert_id, authenticity, label)
                overlay_paths.append(overlay)
                overlay_page = PdfReader(overlay).pages[0]
                page.merge_page(overlay_page)
                writer.add_page(page)

            # Append certificate page
            fd, cert_page = tempfile.mkstemp(suffix="_verifyd_certpage.pdf")
            os.close(fd)
            overlay_paths.append(cert_page)
            c = canvas.Canvas(cert_page, pagesize=letter)
            _draw_certificate_page(c, letter[0], letter[1], cert_id, authenticity, label, filename, sha256, detail)
            c.save()
            writer.add_page(PdfReader(cert_page).pages[0])

            with open(dest_path, "wb") as out:
                writer.write(out)
        finally:
            for p in overlay_paths:
                try:
                    if os.path.exists(p):
                        os.remove(p)
                except Exception:
                    pass
        return dest_path

    # Non-PDF MVP: create a certificate PDF. Original document hash is preserved.
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import letter

    c = canvas.Canvas(dest_path, pagesize=letter)
    _draw_certificate_page(c, letter[0], letter[1], cert_id, authenticity, label, filename, sha256, detail)
    c.save()
    return dest_path
