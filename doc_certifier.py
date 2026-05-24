# ============================================================
#  VeriFYD — doc_certifier.py
#
#  Creates certified/stamped PDF documents for VeriFYD Docs.
#
#  PDF input:
#    - Preserves the original PDF pages.
#    - Adds only a small lower-right VeriFYD mark.
#    - Does NOT append a full certificate-result page.
#    - Does NOT add a footer bar.
#
#  Image input:
#    - Converts JPG/JPEG/PNG into a marked certified PDF.
#
#  XLSX input:
#    - Renders workbook sheets/cell values into a certified PDF.
#    - Adds only the small VeriFYD lower-right mark on each page.
#
#  PPTX input:
#    - Renders slide text/tables into a certified PDF.
#    - Adds only the small VeriFYD lower-right mark on each page.
#
#  Other non-PDF inputs:
#    - Creates a simple certified PDF summary fallback.
#
#  The official certificate details remain in the database, email,
#  and /certificate endpoint.
# ============================================================

from __future__ import annotations

import os
import tempfile
from typing import Dict, Any, Iterable, List, Tuple


def _draw_verifyd_mark(c, width: float, y: float = 24, x_right_pad: float = 34) -> None:
    """Draw a subtle lower-right VeriFYD mark on the active ReportLab canvas."""
    x_right = width - x_right_pad

    c.saveState()
    try:
        c.setFillAlpha(0.72)
    except Exception:
        pass

    c.setFont("Helvetica-Bold", 8)
    full = "VERIFYD"
    total_w = c.stringWidth(full, "Helvetica-Bold", 8)
    x = x_right - total_w

    c.setFillGray(0.15)
    c.drawString(x, y, "VERI")
    veri_w = c.stringWidth("VERI", "Helvetica-Bold", 8)

    c.setFillColorRGB(0.96, 0.62, 0.04)
    c.drawString(x + veri_w, y, "FYD")

    c.setFont("Helvetica", 5.5)
    c.setFillGray(0.35)
    c.drawRightString(x_right, y - 7, "certified")
    c.restoreState()


def _make_logo_overlay(width: float, height: float) -> str:
    """Create a one-page transparent overlay with a small lower-right VeriFYD mark."""
    from reportlab.pdfgen import canvas

    fd, overlay_path = tempfile.mkstemp(suffix="_verifyd_logo_overlay.pdf")
    os.close(fd)

    c = canvas.Canvas(overlay_path, pagesize=(width, height))
    _draw_verifyd_mark(c, width)
    c.save()
    return overlay_path


def _safe_text(value: Any, max_len: int = 80) -> str:
    """Convert a spreadsheet/document value into safe single-line text."""
    if value is None:
        return ""
    text = str(value).replace("\r", " ").replace("\n", " ").strip()
    if len(text) > max_len:
        return text[: max_len - 1] + "…"
    return text


def _chunked_rows(rows: List[List[str]], rows_per_page: int) -> Iterable[List[List[str]]]:
    for i in range(0, len(rows), rows_per_page):
        yield rows[i : i + rows_per_page]


def _create_image_pdf(src_path: str, dest_path: str, cert_id: str, authenticity: int,
                      label: str, filename: str, sha256: str = "") -> str:
    """Create a certified PDF that preserves the uploaded image visually and adds only the small VeriFYD mark."""
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.utils import ImageReader
    from PIL import Image

    width, height = letter
    c = canvas.Canvas(dest_path, pagesize=letter)

    with Image.open(src_path) as img:
        img_w, img_h = img.size

    margin = 36
    max_w = width - (margin * 2)
    max_h = height - (margin * 2)
    scale = min(max_w / max(1, img_w), max_h / max(1, img_h))
    draw_w = img_w * scale
    draw_h = img_h * scale
    x = (width - draw_w) / 2
    y = (height - draw_h) / 2

    c.drawImage(
        ImageReader(src_path),
        x,
        y,
        width=draw_w,
        height=draw_h,
        preserveAspectRatio=True,
        mask="auto",
    )

    _draw_verifyd_mark(c, width)

    c.setAuthor("VeriFYD")
    c.setTitle(f"VeriFYD Certified Document {cert_id}")
    c.setSubject(f"{label} | Authenticity {authenticity}% | {filename}")
    c.save()
    return dest_path


def _extract_xlsx_rows(src_path: str, max_columns: int = 12, max_rows_per_sheet: int = 500) -> List[Tuple[str, List[List[str]]]]:
    """
    Extract visible workbook cell values for certified rendering.

    This is not intended to be a perfect Excel clone. It creates a readable,
    certified PDF rendering of the workbook contents so the returned document
    is useful instead of only showing a certificate summary page.
    """
    try:
        from openpyxl import load_workbook
    except Exception as e:
        raise RuntimeError("Missing dependency: openpyxl. Add openpyxl>=3.1.0 to requirements.txt") from e

    wb = load_workbook(src_path, data_only=True, read_only=True)
    rendered: List[Tuple[str, List[List[str]]]] = []

    for ws in wb.worksheets:
        rows: List[List[str]] = []
        row_count = 0

        for row in ws.iter_rows(values_only=True):
            values = [_safe_text(v) for v in list(row)[:max_columns]]

            # Trim empty trailing cells so sparse rows don't waste space.
            while values and values[-1] == "":
                values.pop()

            if values:
                rows.append(values)

            row_count += 1
            if row_count >= max_rows_per_sheet:
                rows.append([f"[VeriFYD note: sheet truncated after {max_rows_per_sheet} scanned rows]"])
                break

        if not rows:
            rows = [["[No visible cell values found on this sheet]"]]

        rendered.append((ws.title or "Sheet", rows))

    try:
        wb.close()
    except Exception:
        pass

    return rendered


def _draw_table_page(c, width: float, height: float, title: str, subtitle: str,
                     rows: List[List[str]], page_num: int, total_hint: str = "") -> None:
    """Draw one page of an XLSX worksheet table."""
    margin_x = 34
    top = height - 34
    bottom = 42

    c.setFont("Helvetica-Bold", 13)
    c.setFillGray(0.05)
    c.drawString(margin_x, top, title[:95])

    c.setFont("Helvetica", 7.5)
    c.setFillGray(0.35)
    c.drawString(margin_x, top - 14, subtitle[:130])

    if total_hint:
        c.drawRightString(width - margin_x, top - 14, total_hint[:60])

    y = top - 34
    row_h = 18
    available_w = width - (margin_x * 2)

    max_cols = max((len(r) for r in rows), default=1)
    max_cols = max(1, min(max_cols, 12))
    col_w = available_w / max_cols

    for ridx, row in enumerate(rows):
        if y < bottom + row_h:
            break

        if ridx == 0:
            c.setFillGray(0.90)
            c.rect(margin_x, y - 4, available_w, row_h, fill=1, stroke=0)
            font_name = "Helvetica-Bold"
        elif ridx % 2 == 0:
            c.setFillGray(0.975)
            c.rect(margin_x, y - 4, available_w, row_h, fill=1, stroke=0)
            font_name = "Helvetica"
        else:
            font_name = "Helvetica"

        c.setStrokeGray(0.80)
        c.setLineWidth(0.25)

        for cidx in range(max_cols):
            x = margin_x + (cidx * col_w)
            c.rect(x, y - 4, col_w, row_h, fill=0, stroke=1)

            value = row[cidx] if cidx < len(row) else ""
            c.setFillGray(0.08)
            c.setFont(font_name, 6.7)

            approx_chars = max(8, int(col_w / 4.0))
            display = _safe_text(value, approx_chars)
            c.drawString(x + 3, y + 2, display)

        y -= row_h

    c.setFont("Helvetica", 6.5)
    c.setFillGray(0.45)
    c.drawString(margin_x, 24, f"VeriFYD certified workbook rendering • Page {page_num}")
    _draw_verifyd_mark(c, width, y=24, x_right_pad=34)


def _create_xlsx_pdf(src_path: str, dest_path: str, cert_id: str, authenticity: int,
                     label: str, filename: str, sha256: str = "") -> str:
    """Render an XLSX workbook into a readable certified PDF with a small VeriFYD mark."""
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import landscape, letter

    width, height = landscape(letter)
    c = canvas.Canvas(dest_path, pagesize=(width, height))

    sheets = _extract_xlsx_rows(src_path)
    page_num = 1

    c.setAuthor("VeriFYD")
    c.setTitle(f"VeriFYD Certified Spreadsheet {cert_id}")
    c.setSubject(f"{label} | Authenticity {authenticity}% | {filename}")

    c.setFont("Helvetica-Bold", 20)
    c.drawString(54, height - 72, "VeriFYD Certified Spreadsheet")
    c.setFont("Helvetica", 10)
    c.drawString(54, height - 105, f"Status: {label}")
    c.drawString(54, height - 123, f"Authenticity Score: {authenticity}%")
    c.drawString(54, height - 141, f"Certificate ID: {cert_id}")
    c.drawString(54, height - 159, f"Original file: {filename}")
    c.drawString(54, height - 177, f"Workbook sheets: {len(sheets)}")
    if sha256:
        c.drawString(54, height - 198, "SHA-256:")
        c.setFont("Helvetica", 7.5)
        c.drawString(54, height - 212, sha256[:120])
    c.setFont("Helvetica-Oblique", 8)
    c.setFillGray(0.35)
    c.drawString(54, 42, "The following pages are a VeriFYD-rendered PDF view of the uploaded XLSX workbook.")
    _draw_verifyd_mark(c, width, y=24, x_right_pad=34)
    c.showPage()
    page_num += 1

    rows_per_page = 28
    for sheet_name, rows in sheets:
        chunks = list(_chunked_rows(rows, rows_per_page))
        for idx, chunk in enumerate(chunks, start=1):
            subtitle = f"File: {filename} • Sheet: {sheet_name}"
            total_hint = f"Sheet page {idx}/{len(chunks)}"
            _draw_table_page(
                c,
                width,
                height,
                title=f"Worksheet: {sheet_name}",
                subtitle=subtitle,
                rows=chunk,
                page_num=page_num,
                total_hint=total_hint,
            )
            c.showPage()
            page_num += 1

    c.save()
    return dest_path



def _extract_pptx_slides(src_path: str, max_slides: int = 80) -> List[Tuple[str, List[str]]]:
    """
    Extract readable slide text/table values for certified rendering.

    This is not intended to be a perfect PowerPoint visual clone. It creates a
    readable certified PDF rendering of the deck contents so the returned
    document is useful instead of only showing a certificate summary page.
    """
    try:
        from pptx import Presentation
    except Exception as e:
        raise RuntimeError("Missing dependency: python-pptx. Add python-pptx>=0.6.23 to requirements.txt") from e

    prs = Presentation(src_path)
    slides: List[Tuple[str, List[str]]] = []

    for idx, slide in enumerate(prs.slides, start=1):
        if idx > max_slides:
            slides.append(("VeriFYD note", [f"Presentation truncated after {max_slides} slides."]))
            break

        lines: List[str] = []
        for shape in slide.shapes:
            try:
                if getattr(shape, "has_text_frame", False) and shape.text and shape.text.strip():
                    for line in str(shape.text).splitlines():
                        clean = _safe_text(line, 180)
                        if clean:
                            lines.append(clean)
            except Exception:
                pass

            try:
                if getattr(shape, "has_table", False):
                    table = shape.table
                    for row in table.rows:
                        vals = []
                        for cell in row.cells:
                            clean = _safe_text(cell.text, 90)
                            if clean:
                                vals.append(clean)
                        if vals:
                            lines.append(" | ".join(vals))
            except Exception:
                pass

        if not lines:
            lines = ["[No extractable slide text found]"]

        title = f"Slide {idx}"
        if lines and not lines[0].startswith("["):
            title = f"Slide {idx}: {_safe_text(lines[0], 70)}"

        slides.append((title, lines))

    if not slides:
        slides = [("Presentation", ["[No slides found]"])]

    return slides


def _draw_pptx_slide_page(c, width: float, height: float, title: str, subtitle: str,
                          lines: List[str], page_num: int) -> None:
    """Draw one readable page for a PPTX slide."""
    margin_x = 54
    top = height - 54
    bottom = 48

    c.setFont("Helvetica-Bold", 16)
    c.setFillGray(0.05)
    c.drawString(margin_x, top, title[:95])

    c.setFont("Helvetica", 8)
    c.setFillGray(0.35)
    c.drawString(margin_x, top - 16, subtitle[:130])

    y = top - 48
    line_h = 14

    for idx, line in enumerate(lines[:34]):
        if y < bottom + line_h:
            c.setFont("Helvetica-Oblique", 8)
            c.setFillGray(0.40)
            c.drawString(margin_x, y, "[Additional slide text truncated on this certified rendering page]")
            break

        is_first = idx == 0 and not line.startswith("[")
        c.setFont("Helvetica-Bold" if is_first else "Helvetica", 9 if is_first else 8.5)
        c.setFillGray(0.08)

        wrapped = []
        current = ""
        for word in str(line).split():
            test = (current + " " + word).strip()
            if c.stringWidth(test, "Helvetica", 8.5) > (width - margin_x * 2):
                if current:
                    wrapped.append(current)
                current = word
            else:
                current = test
        if current:
            wrapped.append(current)

        for wline in wrapped[:3]:
            if y < bottom + line_h:
                break
            prefix = "" if is_first else "• "
            c.drawString(margin_x, y, prefix + wline)
            y -= line_h
        y -= 2

    c.setFont("Helvetica", 6.5)
    c.setFillGray(0.45)
    c.drawString(margin_x, 24, f"VeriFYD certified presentation rendering • Page {page_num}")
    _draw_verifyd_mark(c, width, y=24, x_right_pad=34)


def _create_pptx_pdf(src_path: str, dest_path: str, cert_id: str, authenticity: int,
                     label: str, filename: str, sha256: str = "") -> str:
    """Render a PPTX presentation into a readable certified PDF with a small VeriFYD mark."""
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import landscape, letter

    width, height = landscape(letter)
    c = canvas.Canvas(dest_path, pagesize=(width, height))

    slides = _extract_pptx_slides(src_path)
    page_num = 1

    c.setAuthor("VeriFYD")
    c.setTitle(f"VeriFYD Certified Presentation {cert_id}")
    c.setSubject(f"{label} | Authenticity {authenticity}% | {filename}")

    c.setFont("Helvetica-Bold", 20)
    c.drawString(54, height - 72, "VeriFYD Certified Presentation")
    c.setFont("Helvetica", 10)
    c.drawString(54, height - 105, f"Status: {label}")
    c.drawString(54, height - 123, f"Authenticity Score: {authenticity}%")
    c.drawString(54, height - 141, f"Certificate ID: {cert_id}")
    c.drawString(54, height - 159, f"Original file: {filename}")
    c.drawString(54, height - 177, f"Slides: {len(slides)}")
    if sha256:
        c.drawString(54, height - 198, "SHA-256:")
        c.setFont("Helvetica", 7.5)
        c.drawString(54, height - 212, sha256[:120])
    c.setFont("Helvetica-Oblique", 8)
    c.setFillGray(0.35)
    c.drawString(54, 42, "The following pages are a VeriFYD-rendered PDF view of the uploaded PPTX presentation.")
    _draw_verifyd_mark(c, width, y=24, x_right_pad=34)
    c.showPage()
    page_num += 1

    for slide_title, lines in slides:
        _draw_pptx_slide_page(
            c,
            width,
            height,
            title=slide_title,
            subtitle=f"File: {filename}",
            lines=lines,
            page_num=page_num,
        )
        c.showPage()
        page_num += 1

    c.save()
    return dest_path

def _create_non_pdf_certificate(dest_path: str, cert_id: str, authenticity: int,
                                label: str, filename: str, sha256: str = "") -> str:
    """Fallback for non-PDF docs: create a simple marked PDF summary."""
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import letter

    width, height = letter
    c = canvas.Canvas(dest_path, pagesize=letter)
    c.setFont("Helvetica-Bold", 20)
    c.drawString(54, height - 72, "VeriFYD Certified Document")
    c.setFont("Helvetica", 11)
    c.drawString(54, height - 105, f"Status: {label}")
    c.drawString(54, height - 123, f"Authenticity Score: {authenticity}%")
    c.drawString(54, height - 141, f"Certificate ID: {cert_id}")
    c.drawString(54, height - 159, f"Original file: {filename}")
    if sha256:
        c.drawString(54, height - 177, "SHA-256:")
        c.setFont("Helvetica", 8)
        c.drawString(54, height - 191, sha256[:96])
    c.setFont("Helvetica-Oblique", 8)
    c.setFillGray(0.35)
    c.drawString(54, 42, "VeriFYD provides analytical guidance, not legal notarization.")

    _draw_verifyd_mark(c, width)
    c.save()
    return dest_path


def stamp_document(src_path: str, dest_path: str, cert_id: str,
                   authenticity: int, label: str, filename: str,
                   sha256: str = "", detail: Dict[str, Any] | None = None) -> str:
    """
    Create a certified PDF at dest_path.

    For PDFs: preserve original pages and add only a small lower-right VeriFYD mark.
    For JPG/JPEG/PNG: create a marked PDF containing the image.
    For XLSX: render workbook sheets into a marked PDF.
    For PPTX: render presentation slides into a marked PDF.
    For other non-PDFs: create a simple PDF summary fallback.
    """
    ext = os.path.splitext(src_path)[1].lower()

    if ext in (".jpg", ".jpeg", ".png"):
        return _create_image_pdf(src_path, dest_path, cert_id, authenticity, label, filename, sha256)

    if ext == ".xlsx":
        return _create_xlsx_pdf(src_path, dest_path, cert_id, authenticity, label, filename, sha256)

    if ext == ".pptx":
        return _create_pptx_pdf(src_path, dest_path, cert_id, authenticity, label, filename, sha256)

    if ext == ".pdf":
        from pypdf import PdfReader, PdfWriter

        reader = PdfReader(src_path)
        writer = PdfWriter()
        overlay_paths: list[str] = []

        try:
            for page in reader.pages:
                width = float(page.mediabox.width)
                height = float(page.mediabox.height)
                overlay = _make_logo_overlay(width, height)
                overlay_paths.append(overlay)
                overlay_page = PdfReader(overlay).pages[0]
                page.merge_page(overlay_page)
                writer.add_page(page)

            try:
                existing_meta = reader.metadata or {}
                metadata = {str(k): str(v) for k, v in existing_meta.items() if v is not None}
                metadata.update({
                    "/VeriFYD": "Certified",
                    "/VeriFYD_Certificate_ID": cert_id,
                    "/VeriFYD_Label": label,
                    "/VeriFYD_Authenticity": str(authenticity),
                    "/VeriFYD_SHA256_Original": sha256 or "",
                })
                writer.add_metadata(metadata)
            except Exception:
                pass

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

    return _create_non_pdf_certificate(dest_path, cert_id, authenticity, label, filename, sha256)
