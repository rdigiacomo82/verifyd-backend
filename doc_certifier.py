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
#    - Renders a visual slide approximation into a certified PDF.
#    - Preserves slide dimensions, positioned text boxes, tables, images,
#      and basic filled shapes where possible.
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
from typing import Any, Dict, Iterable, List, Tuple


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


# ─────────────────────────────────────────────
# XLSX rendering
# ─────────────────────────────────────────────

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


# ─────────────────────────────────────────────
# PPTX rendering
# ─────────────────────────────────────────────

def _pptx_emu_to_pt(value: Any) -> float:
    """Convert PowerPoint EMUs to PDF points."""
    try:
        return float(value) / 914400.0 * 72.0
    except Exception:
        return 0.0


def _pptx_rgb_tuple(color_obj: Any, default: Tuple[float, float, float] = (1, 1, 1)) -> Tuple[float, float, float]:
    """Return ReportLab RGB floats from a python-pptx color object."""
    try:
        rgb = color_obj.rgb
        if rgb:
            return (rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0)
    except Exception:
        pass
    return default


def _pptx_text_runs(shape: Any) -> List[Tuple[str, bool, float]]:
    """Extract text with basic bold/font-size hints from a PPTX text shape."""
    runs: List[Tuple[str, bool, float]] = []
    try:
        tf = shape.text_frame
        for para in tf.paragraphs:
            parts: List[str] = []
            bold = False
            font_size = 11.0
            for run in para.runs:
                txt = (run.text or "").strip()
                if not txt:
                    continue
                parts.append(txt)
                try:
                    bold = bool(run.font.bold) or bold
                except Exception:
                    pass
                try:
                    if run.font.size:
                        font_size = max(6.0, min(30.0, float(run.font.size.pt)))
                except Exception:
                    pass
            if parts:
                runs.append((" ".join(parts), bold, font_size))
            else:
                txt = (getattr(para, "text", "") or "").strip()
                if txt:
                    runs.append((txt, False, font_size))
    except Exception:
        try:
            txt = (shape.text or "").strip()
            if txt:
                for line in txt.splitlines():
                    if line.strip():
                        runs.append((line.strip(), False, 11.0))
        except Exception:
            pass
    return runs


def _wrap_reportlab_text(c, text: str, font_name: str, font_size: float, max_width: float) -> List[str]:
    """Simple word wrapping for ReportLab text drawing."""
    text = str(text or "").replace("\t", " ").strip()
    if not text:
        return []

    lines: List[str] = []
    current = ""
    for word in text.split():
        trial = (current + " " + word).strip()
        if c.stringWidth(trial, font_name, font_size) <= max_width:
            current = trial
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines or [text]


def _draw_pptx_visual_slide(c, slide: Any, slide_idx: int, prs_width: float, prs_height: float,
                            page_w: float, page_h: float, filename: str, page_num: int,
                            temp_files: List[str]) -> None:
    """Draw one PPTX slide as a visual approximation with positioned shapes/text/images."""
    from reportlab.lib.utils import ImageReader

    margin = 34.0
    footer_h = 28.0
    max_w = page_w - (margin * 2)
    max_h = page_h - (margin * 2) - footer_h
    scale = min(max_w / max(1.0, prs_width), max_h / max(1.0, prs_height))
    slide_w = prs_width * scale
    slide_h = prs_height * scale
    origin_x = (page_w - slide_w) / 2.0
    origin_y = footer_h + margin + ((max_h - slide_h) / 2.0)

    # Slide canvas.
    c.saveState()
    c.setFillColorRGB(1, 1, 1)
    c.rect(origin_x, origin_y, slide_w, slide_h, fill=1, stroke=0)
    c.setStrokeGray(0.70)
    c.setLineWidth(0.8)
    c.rect(origin_x, origin_y, slide_w, slide_h, fill=0, stroke=1)
    c.restoreState()

    # Best-effort visual rendering of slide contents.
    for shape in slide.shapes:
        try:
            left = origin_x + _pptx_emu_to_pt(shape.left) * scale
            top_from_slide = _pptx_emu_to_pt(shape.top) * scale
            width = max(1.0, _pptx_emu_to_pt(shape.width) * scale)
            height = max(1.0, _pptx_emu_to_pt(shape.height) * scale)
            bottom = origin_y + slide_h - top_from_slide - height
        except Exception:
            continue

        # Pictures: preserve embedded image content where possible.
        try:
            if hasattr(shape, "image") and getattr(shape, "image", None):
                img_blob = shape.image.blob
                ext = shape.image.ext or "png"
                fd, img_path = tempfile.mkstemp(suffix=f"_pptx_img.{ext}")
                os.close(fd)
                with open(img_path, "wb") as fh:
                    fh.write(img_blob)
                temp_files.append(img_path)
                c.drawImage(
                    ImageReader(img_path),
                    left,
                    bottom,
                    width=width,
                    height=height,
                    preserveAspectRatio=True,
                    anchor="c",
                    mask="auto",
                )
                continue
        except Exception:
            pass

        # Tables: draw an approximate positioned grid.
        try:
            if getattr(shape, "has_table", False):
                table = shape.table
                rows = list(table.rows)
                cols = len(table.columns) if table.columns else 1
                row_h = height / max(1, len(rows))
                col_w = width / max(1, cols)
                for r_idx, row in enumerate(rows):
                    y = bottom + height - ((r_idx + 1) * row_h)
                    for c_idx, cell in enumerate(row.cells):
                        x = left + (c_idx * col_w)
                        c.setStrokeGray(0.70)
                        c.setLineWidth(0.35)
                        c.rect(x, y, col_w, row_h, fill=0, stroke=1)
                        txt = _safe_text(cell.text, 80)
                        if txt:
                            c.setFont("Helvetica", max(4.5, min(8, row_h * 0.38)))
                            c.setFillGray(0.05)
                            c.drawString(x + 2, y + max(2, row_h - 9), _safe_text(txt, int(max(8, col_w / 4.2))))
                continue
        except Exception:
            pass

        # Basic filled auto-shapes.
        try:
            fill = getattr(shape, "fill", None)
            if fill and getattr(fill, "type", None):
                rgb = _pptx_rgb_tuple(fill.fore_color, default=(0.94, 0.94, 0.94))
                c.saveState()
                c.setFillColorRGB(*rgb)
                c.setStrokeGray(0.82)
                c.setLineWidth(0.25)
                c.rect(left, bottom, width, height, fill=1, stroke=1)
                c.restoreState()
        except Exception:
            pass

        # Text boxes and shape text in approximate original position.
        try:
            if getattr(shape, "has_text_frame", False):
                runs = _pptx_text_runs(shape)
                if runs:
                    c.saveState()
                    c.setFillGray(0.05)
                    y = bottom + height - 7
                    for text, bold, font_size in runs:
                        scaled_font = max(5.0, min(22.0, font_size * scale * 1.15))
                        font_name = "Helvetica-Bold" if bold else "Helvetica"
                        wrapped = _wrap_reportlab_text(c, text, font_name, scaled_font, max(10.0, width - 8))
                        for line in wrapped[:8]:
                            if y < bottom + 4:
                                break
                            c.setFont(font_name, scaled_font)
                            c.drawString(left + 4, y, line)
                            y -= scaled_font * 1.25
                        y -= scaled_font * 0.25
                    c.restoreState()
        except Exception:
            pass

    c.setFont("Helvetica", 6.5)
    c.setFillGray(0.45)
    c.drawString(margin, 24, f"VeriFYD certified presentation rendering • Slide {slide_idx} • Page {page_num} • {filename[:80]}")
    _draw_verifyd_mark(c, page_w, y=24, x_right_pad=34)


def _create_pptx_pdf(src_path: str, dest_path: str, cert_id: str, authenticity: int,
                     label: str, filename: str, sha256: str = "") -> str:
    """
    Render a PPTX presentation into a certified PDF with a visual slide approximation.

    Note: this avoids LibreOffice/system conversion dependencies, so it is safe for
    Render. It preserves slide text, tables, pictures, and basic shape placement.
    """
    try:
        from pptx import Presentation
    except Exception as e:
        raise RuntimeError("Missing dependency: python-pptx. Add python-pptx>=0.6.23 to requirements.txt") from e

    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import landscape, letter

    prs = Presentation(src_path)
    prs_width = _pptx_emu_to_pt(prs.slide_width)
    prs_height = _pptx_emu_to_pt(prs.slide_height)
    page_w, page_h = landscape(letter)
    c = canvas.Canvas(dest_path, pagesize=(page_w, page_h))
    temp_files: List[str] = []

    c.setAuthor("VeriFYD")
    c.setTitle(f"VeriFYD Certified Presentation {cert_id}")
    c.setSubject(f"{label} | Authenticity {authenticity}% | {filename}")

    # Cover/verification page.
    c.setFont("Helvetica-Bold", 20)
    c.drawString(54, page_h - 72, "VeriFYD Certified Presentation")
    c.setFont("Helvetica", 10)
    c.drawString(54, page_h - 105, f"Status: {label}")
    c.drawString(54, page_h - 123, f"Authenticity Score: {authenticity}%")
    c.drawString(54, page_h - 141, f"Certificate ID: {cert_id}")
    c.drawString(54, page_h - 159, f"Original file: {filename}")
    c.drawString(54, page_h - 177, f"Slides: {len(prs.slides)}")
    if sha256:
        c.drawString(54, page_h - 198, "SHA-256:")
        c.setFont("Helvetica", 7.5)
        c.drawString(54, page_h - 212, sha256[:120])
    c.setFont("Helvetica-Oblique", 8)
    c.setFillGray(0.35)
    c.drawString(54, 42, "The following pages are a VeriFYD-rendered PDF view of the uploaded PPTX presentation.")
    _draw_verifyd_mark(c, page_w, y=24, x_right_pad=34)
    c.showPage()

    try:
        if len(prs.slides) == 0:
            c.setFont("Helvetica", 12)
            c.drawString(54, page_h - 72, "[No slides found in uploaded presentation]")
            _draw_verifyd_mark(c, page_w, y=24, x_right_pad=34)
            c.showPage()
        else:
            for idx, slide in enumerate(prs.slides, start=1):
                _draw_pptx_visual_slide(
                    c=c,
                    slide=slide,
                    slide_idx=idx,
                    prs_width=prs_width,
                    prs_height=prs_height,
                    page_w=page_w,
                    page_h=page_h,
                    filename=filename,
                    page_num=idx + 1,
                    temp_files=temp_files,
                )
                c.showPage()
    finally:
        for path in temp_files:
            try:
                if os.path.exists(path):
                    os.remove(path)
            except Exception:
                pass

    c.save()
    return dest_path


# ─────────────────────────────────────────────
# Fallback and public entry point
# ─────────────────────────────────────────────

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
    For PPTX: render presentation slides into a visual marked PDF approximation.
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
