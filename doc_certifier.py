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


def _convert_heic_to_renderable_image(src_path: str) -> str:
    """
    Convert HEIC/HEIF to a temporary JPEG/PNG that ReportLab/Pillow can render.

    The photo detector already handles HEIC by converting with ffmpeg. This helper
    mirrors that behavior for document certification so the analysis result does
    not succeed while the certified PDF fails to render/upload.
    """
    import subprocess
    import shutil

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        fd, tmp_jpg = tempfile.mkstemp(suffix="_verifyd_heic_render.jpg")
        os.close(fd)
        try:
            cmd = [ffmpeg, "-y", "-i", src_path, "-frames:v", "1", "-q:v", "2", tmp_jpg]
            result = subprocess.run(cmd, capture_output=True, timeout=60)
            if result.returncode == 0 and os.path.exists(tmp_jpg) and os.path.getsize(tmp_jpg) > 1000:
                return tmp_jpg
        except Exception:
            pass
        try:
            if os.path.exists(tmp_jpg):
                os.remove(tmp_jpg)
        except Exception:
            pass

    try:
        from pillow_heif import register_heif_opener
        register_heif_opener()
        from PIL import Image

        fd, tmp_png = tempfile.mkstemp(suffix="_verifyd_heic_render.png")
        os.close(fd)
        with Image.open(src_path) as img:
            if img.mode not in ("RGB", "RGBA"):
                img = img.convert("RGB")
            img.save(tmp_png, format="PNG")
        if os.path.exists(tmp_png) and os.path.getsize(tmp_png) > 1000:
            return tmp_png
        try:
            os.remove(tmp_png)
        except Exception:
            pass
    except Exception:
        pass

    raise RuntimeError("HEIC/HEIF certification rendering failed. ffmpeg or pillow-heif could not convert the image.")


def _create_image_pdf(src_path: str, dest_path: str, cert_id: str, authenticity: int,
                      label: str, filename: str, sha256: str = "") -> str:
    """Create a certified PDF that preserves uploaded image documents visually."""
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.utils import ImageReader
    from PIL import Image, ImageSequence

    ext = os.path.splitext(src_path)[1].lower()
    render_src = src_path
    temp_images: List[str] = []

    if ext in (".heic", ".heif"):
        render_src = _convert_heic_to_renderable_image(src_path)
        temp_images.append(render_src)

    width, height = letter
    c = canvas.Canvas(dest_path, pagesize=letter)
    c.setAuthor("VeriFYD")
    c.setTitle(f"VeriFYD Certified Document {cert_id}")
    c.setSubject(f"{label} | Authenticity {authenticity}% | {filename}")

    margin = 36
    max_w = width - (margin * 2)
    max_h = height - (margin * 2)

    try:
        with Image.open(render_src) as img:
            frames = []
            try:
                for frame in ImageSequence.Iterator(img):
                    frames.append(frame.copy())
                    if len(frames) >= 50:
                        break
            except Exception:
                frames = [img.copy()]

        if not frames:
            with Image.open(render_src) as img:
                frames = [img.copy()]

        for idx, frame in enumerate(frames):
            if frame.mode not in ("RGB", "RGBA"):
                frame = frame.convert("RGB")

            draw_source = render_src
            if len(frames) > 1 or ext in (".tif", ".tiff", ".webp", ".heic", ".heif"):
                fd, tmp_img = tempfile.mkstemp(suffix=f"_verifyd_frame_{idx}.png")
                os.close(fd)
                frame.save(tmp_img, format="PNG")
                temp_images.append(tmp_img)
                draw_source = tmp_img

            img_w, img_h = frame.size
            scale = min(max_w / max(1, img_w), max_h / max(1, img_h))
            draw_w = img_w * scale
            draw_h = img_h * scale
            x = (width - draw_w) / 2
            y = (height - draw_h) / 2

            c.drawImage(
                ImageReader(draw_source),
                x,
                y,
                width=draw_w,
                height=draw_h,
                preserveAspectRatio=True,
                mask="auto",
            )
            _draw_verifyd_mark(c, width)
            if idx < len(frames) - 1:
                c.showPage()
    finally:
        for t in temp_images:
            try:
                if os.path.exists(t):
                    os.remove(t)
            except Exception:
                pass

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




def _attach_original_file_to_pdf(pdf_path: str, original_path: str, original_filename: str) -> str:
    """
    Embed the original uploaded file inside the certified PDF as an attachment.

    This is especially important for Office files such as PPTX where a pure Python
    renderer cannot perfectly reproduce every visual detail, animation, theme,
    background, or SmartArt element. The PDF remains the certified viewing copy,
    while the exact original file is preserved inside the PDF for later review.
    """
    try:
        from pypdf import PdfReader, PdfWriter
    except Exception:
        # pypdf is already required for PDF certification, but do not fail the
        # whole certification if attachment support is unexpectedly unavailable.
        return pdf_path

    if not os.path.exists(pdf_path) or not os.path.exists(original_path):
        return pdf_path

    tmp_out = pdf_path + ".attached.tmp.pdf"
    try:
        reader = PdfReader(pdf_path)
        writer = PdfWriter()
        for page in reader.pages:
            writer.add_page(page)

        try:
            existing_meta = reader.metadata or {}
            metadata = {str(k): str(v) for k, v in existing_meta.items() if v is not None}
            metadata.update({
                "/VeriFYD_Embedded_Original": original_filename,
            })
            writer.add_metadata(metadata)
        except Exception:
            pass

        with open(original_path, "rb") as fh:
            original_bytes = fh.read()

        # pypdf supports embedded file attachments using add_attachment.
        writer.add_attachment(original_filename or os.path.basename(original_path), original_bytes)

        with open(tmp_out, "wb") as out:
            writer.write(out)

        os.replace(tmp_out, pdf_path)
    except Exception:
        try:
            if os.path.exists(tmp_out):
                os.remove(tmp_out)
        except Exception:
            pass
    return pdf_path


def _stamp_existing_pdf_file(src_pdf_path: str, dest_path: str, cert_id: str,
                             authenticity: int, label: str, sha256: str = "") -> str:
    """Preserve an existing PDF's pages and add only the small lower-right VeriFYD mark."""
    from pypdf import PdfReader, PdfWriter

    reader = PdfReader(src_pdf_path)
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


def _try_convert_pptx_to_pdf_with_libreoffice(src_path: str) -> str | None:
    """
    Try to use LibreOffice/soffice for a true PPTX-to-PDF conversion.

    Many Render Python environments do not include LibreOffice. When it is not
    available, return None and let the pure-Python renderer handle the fallback.
    """
    try:
        import shutil
        import subprocess
    except Exception:
        return None

    soffice = shutil.which("soffice") or shutil.which("libreoffice")
    if not soffice:
        return None

    out_dir = tempfile.mkdtemp(prefix="verifyd_pptx_convert_")
    try:
        cmd = [
            soffice,
            "--headless",
            "--convert-to",
            "pdf",
            "--outdir",
            out_dir,
            src_path,
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=90)
        if result.returncode != 0:
            return None

        base = os.path.splitext(os.path.basename(src_path))[0] + ".pdf"
        converted = os.path.join(out_dir, base)
        if os.path.exists(converted) and os.path.getsize(converted) > 1000:
            # Caller is responsible for deleting the returned file and its folder.
            return converted
        return None
    except Exception:
        return None


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
    Render a PPTX presentation into a certified PDF.

    Preferred path:
      1. If LibreOffice/soffice is available on the server, convert the PPTX to a
         true PDF rendering, stamp each page, and embed the original PPTX.

    Safe fallback:
      2. If LibreOffice is unavailable, use a pure-Python visual approximation
         and embed the original PPTX as a PDF attachment so the exact source file
         travels with the certified PDF.
    """
    try:
        from pptx import Presentation
    except Exception as e:
        raise RuntimeError("Missing dependency: python-pptx. Add python-pptx>=0.6.23 to requirements.txt") from e

    # First try a true Office conversion. This preserves the original slide
    # visuals far better than python-pptx can. It will silently fall back if
    # LibreOffice is not installed in Render.
    converted_pdf = _try_convert_pptx_to_pdf_with_libreoffice(src_path)
    if converted_pdf:
        converted_dir = os.path.dirname(converted_pdf)
        try:
            _stamp_existing_pdf_file(
                src_pdf_path=converted_pdf,
                dest_path=dest_path,
                cert_id=cert_id,
                authenticity=authenticity,
                label=label,
                sha256=sha256,
            )
            _attach_original_file_to_pdf(dest_path, src_path, filename or os.path.basename(src_path))
            return dest_path
        finally:
            try:
                if os.path.exists(converted_pdf):
                    os.remove(converted_pdf)
            except Exception:
                pass
            try:
                if converted_dir and os.path.isdir(converted_dir):
                    os.rmdir(converted_dir)
            except Exception:
                pass

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
    c.drawString(54, 58, "The following pages are a VeriFYD-rendered PDF view of the uploaded PPTX presentation.")
    c.drawString(54, 44, "The original PPTX file is embedded inside this certified PDF as an attachment for exact-source review.")
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

    # Embed the original deck so the certified PDF includes the exact source
    # presentation even when pure-Python visual rendering cannot perfectly
    # reproduce every PowerPoint feature.
    _attach_original_file_to_pdf(dest_path, src_path, filename or os.path.basename(src_path))
    return dest_path



def _rtf_regex_fallback_for_render(raw: str) -> str:
    """Table-preserving RTF cleanup for certified PDF rendering."""
    import re
    raw = raw or ""
    # Remove only obvious embedded binary/image/object groups. Do not remove table
    # structure because many legal/government RTFs are form-style tables.
    raw = re.sub(r"{\\\*?\\pict[^{}]*(?:{[^{}]*}[^{}]*)*}", " ", raw, flags=re.DOTALL)
    raw = re.sub(r"{\\object[^{}]*(?:{[^{}]*}[^{}]*)*}", " ", raw, flags=re.DOTALL)

    replacements = {
        "\\par": "\n",
        "\\line": "\n",
        "\\tab": "\t",
        "\\cell": "|",
        "\\row": "\n",
    }
    for src, dst in replacements.items():
        raw = raw.replace(src, dst)

    def _hex_to_char(match):
        try:
            return bytes.fromhex(match.group(1)).decode("cp1252", errors="replace")
        except Exception:
            return " "

    raw = re.sub(r"\\'([0-9a-fA-F]{2})", _hex_to_char, raw)
    raw = re.sub(r"\\u(-?\d+)\??", lambda m: chr(int(m.group(1)) % 65536), raw)
    raw = re.sub(r"\\[a-zA-Z]+-?\d* ?", "", raw)
    raw = raw.replace("{", " ").replace("}", " ")
    raw = re.sub(r"\x00+", " ", raw)
    raw = re.sub(r"[ \t]+", " ", raw)
    raw = re.sub(r" *\| *", "|", raw)
    raw = re.sub(r"\n\s*\n\s*\n+", "\n\n", raw)
    return raw.strip()


def _strip_rtf_for_render(raw: str) -> str:
    """RTF-to-text cleanup for certified rendering with table-preserving fallback."""
    import re
    regex_text = _rtf_regex_fallback_for_render(raw)

    try:
        from striprtf.striprtf import rtf_to_text
        text = rtf_to_text(raw or "")
        text = re.sub(r"\x00+", " ", text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
        text = text.strip()
        # striprtf is usually cleaner, but it can over-strip table-based forms.
        # Keep the fallback when it preserves substantially more readable content.
        if len(text) >= 500 or len(text) >= max(120, len(regex_text) * 0.35):
            return text
    except Exception:
        pass

    return regex_text



def _plain_text_from_binary_chunks(data: bytes, limit: int = 2_000_000) -> str:
    """Best-effort readable text extraction from legacy Office binary streams."""
    import re
    fragments: List[str] = []
    sample = data[:limit]
    for enc in ("utf-16-le", "latin-1"):
        try:
            decoded = sample.decode(enc, errors="ignore")
        except Exception:
            continue
        decoded = decoded.replace("\x00", " ")
        decoded = re.sub(r"[\x01-\x08\x0b\x0c\x0e-\x1f]+", " ", decoded)
        decoded = re.sub(r"[ \t]+", " ", decoded)
        pieces = re.findall(r"[A-Za-z0-9][A-Za-z0-9\s\.,;:\-_/@$%#&()\[\]{}'\"!?]{4,}", decoded)
        fragments.extend(p.strip() for p in pieces if p and len(p.strip()) >= 5)
    seen = set()
    cleaned: List[str] = []
    for frag in fragments:
        compact = re.sub(r"\s+", " ", frag).strip()
        if len(compact) < 5:
            continue
        key = compact[:160].lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(compact)
        if sum(len(x) for x in cleaned) > limit:
            break
    return "\n".join(cleaned)


def _read_ole_text_for_render(src_path: str) -> str:
    """Best-effort legacy DOC/PPT text rendering helper."""
    try:
        import olefile
        parts: List[str] = []
        with olefile.OleFileIO(src_path) as ole:
            for stream in ole.listdir()[:80]:
                try:
                    raw = ole.openstream(stream).read(750_000)
                    text = _plain_text_from_binary_chunks(raw, limit=750_000)
                    if text:
                        parts.append(text)
                except Exception:
                    continue
        if parts:
            return "\n".join(parts)
    except Exception:
        pass
    try:
        with open(src_path, "rb") as fh:
            return _plain_text_from_binary_chunks(fh.read(2_000_000))
    except Exception:
        return ""


def _read_xls_for_render(src_path: str) -> str:
    """Render legacy XLS cell values into plain text for certified PDF output."""
    try:
        import xlrd
        book = xlrd.open_workbook(src_path, on_demand=True)
        lines: List[str] = []
        rows_seen = 0
        for sheet_name in book.sheet_names():
            sh = book.sheet_by_name(sheet_name)
            lines.append(f"Worksheet: {sheet_name}")
            for r_idx in range(sh.nrows):
                vals: List[str] = []
                for c_idx in range(min(sh.ncols, 30)):
                    try:
                        val = sh.cell_value(r_idx, c_idx)
                    except Exception:
                        continue
                    if val in (None, ""):
                        continue
                    if isinstance(val, float) and val.is_integer():
                        val = int(val)
                    text = str(val).strip()
                    if text:
                        vals.append(text[:300])
                if vals:
                    lines.append(" | ".join(vals))
                rows_seen += 1
                if rows_seen >= 5000:
                    lines.append("[VeriFYD note: legacy workbook rendering truncated after 5000 scanned rows]")
                    break
            if rows_seen >= 5000:
                break
        try:
            book.release_resources()
        except Exception:
            pass
        return "\n".join(lines)
    except Exception:
        return _read_ole_text_for_render(src_path)


def _read_msg_for_render(src_path: str) -> str:
    """Render Outlook MSG headers/body into plain text for certified PDF output."""
    try:
        import extract_msg
        msg = extract_msg.Message(src_path)
        try:
            lines: List[str] = []
            header_map = [
                ("From", getattr(msg, "sender", "") or ""),
                ("To", getattr(msg, "to", "") or ""),
                ("Cc", getattr(msg, "cc", "") or ""),
                ("Subject", getattr(msg, "subject", "") or ""),
                ("Date", getattr(msg, "date", "") or ""),
                ("Message-ID", getattr(msg, "messageId", "") or ""),
            ]
            for label, value in header_map:
                value = str(value or "").strip()
                if value:
                    lines.append(f"{label}: {value}")
            attachments = list(getattr(msg, "attachments", []) or [])
            if attachments:
                names = []
                for att in attachments[:30]:
                    name = str(getattr(att, "longFilename", "") or getattr(att, "shortFilename", "") or "")
                    if name:
                        names.append(name)
                lines.append(f"Attachments: {len(attachments)}" + (f" ({', '.join(names[:10])})" if names else ""))
            lines.append("")
            body = str(getattr(msg, "body", "") or getattr(msg, "htmlBody", "") or "")
            if body and "<" in body and ">" in body:
                import re
                body = re.sub(r"<[^>]+>", " ", body)
                body = re.sub(r"\s+", " ", body).strip()
            lines.append(body)
            return "\n".join(lines).strip()
        finally:
            try:
                msg.close()
            except Exception:
                pass
    except Exception:
        return _read_ole_text_for_render(src_path)


def _read_odf_for_render(src_path: str) -> str:
    """Extract readable text/table content from ODT/ODS/ODP for certified PDF rendering."""
    import re
    import zipfile
    import xml.etree.ElementTree as ET

    ext = os.path.splitext(src_path)[1].lower()
    kind = {".odt": "OpenDocument Text", ".ods": "OpenDocument Spreadsheet", ".odp": "OpenDocument Presentation"}.get(ext, "OpenDocument")
    lines: List[str] = [kind]

    def _local(tag: str) -> str:
        return str(tag).split("}")[-1].lower()

    def _text(elem: Any) -> str:
        try:
            return re.sub(r"\s+", " ", " ".join(t.strip() for t in elem.itertext() if t and t.strip())).strip()
        except Exception:
            return ""

    try:
        with zipfile.ZipFile(src_path) as zf:
            names = set(zf.namelist())
            if "content.xml" not in names:
                return "[No OpenDocument content.xml found]"
            root = ET.fromstring(zf.read("content.xml"))

            for elem in root.iter():
                name = _local(elem.tag)
                if ext == ".ods" and name == "table":
                    table_name = elem.attrib.get("{urn:oasis:names:tc:opendocument:xmlns:table:1.0}name", "Sheet")
                    lines.append(f"\nWorksheet: {table_name}")
                elif ext == ".odp" and name == "page":
                    page_name = elem.attrib.get("{urn:oasis:names:tc:opendocument:xmlns:drawing:1.0}name", "Slide")
                    lines.append(f"\nSlide: {page_name}")
                elif ext == ".ods" and name == "table-row":
                    cells: List[str] = []
                    for child in list(elem):
                        if _local(child.tag) == "table-cell":
                            value = _text(child)
                            if value:
                                cells.append(value[:300])
                    if cells:
                        lines.append(" | ".join(cells))
                elif name in ("h", "p"):
                    value = _text(elem)
                    if value:
                        lines.append(value[:1000])
    except Exception as e:
        return f"[OpenDocument rendering failed: {str(e)[:120]}]"

    cleaned = [line for line in lines if line and line.strip()]
    return "\n".join(cleaned) if cleaned else "[No readable OpenDocument text could be extracted]"


def _read_text_for_certified_render(src_path: str, ext: str) -> str:
    """Read RTF/EML/text-family documents for certified PDF rendering."""
    data = b""
    with open(src_path, "rb") as fh:
        data = fh.read(2_500_000)

    if ext == ".eml":
        try:
            from email import policy
            from email.parser import BytesParser
            msg = BytesParser(policy=policy.default).parsebytes(data)
            lines: List[str] = []
            for h in ("From", "To", "Cc", "Subject", "Date", "Message-ID", "Reply-To", "Return-Path"):
                v = str(msg.get(h, "") or "").strip()
                if v:
                    lines.append(f"{h}: {v}")
            lines.append("")
            body_parts: List[str] = []
            if msg.is_multipart():
                for part in msg.walk():
                    ctype = part.get_content_type()
                    disp = str(part.get_content_disposition() or "")
                    if ctype == "text/plain" and disp != "attachment":
                        try:
                            body_parts.append(str(part.get_content()))
                        except Exception:
                            pass
                    elif ctype == "text/html" and disp != "attachment" and not body_parts:
                        try:
                            import re
                            body_parts.append(re.sub(r"<[^>]+>", " ", str(part.get_content())))
                        except Exception:
                            pass
            else:
                try:
                    body_parts.append(str(msg.get_content()))
                except Exception:
                    pass
            lines.append("\n\n".join(x.strip() for x in body_parts if x and x.strip()))
            return "\n".join(lines).strip()
        except Exception:
            pass

    if ext == ".msg":
        return _read_msg_for_render(src_path)

    if ext in (".odt", ".ods", ".odp"):
        return _read_odf_for_render(src_path)

    if ext in (".doc", ".ppt"):
        return _read_ole_text_for_render(src_path)

    if ext == ".xls":
        return _read_xls_for_render(src_path)

    text = ""
    for enc in ("utf-8", "utf-16", "latin-1"):
        try:
            text = data.decode(enc, errors="replace")
            break
        except Exception:
            continue
    if not text:
        text = data.decode("latin-1", errors="replace")
    if ext == ".rtf":
        return _strip_rtf_for_render(text)
    return text.strip()


def _create_text_render_pdf(src_path: str, dest_path: str, cert_id: str, authenticity: int,
                            label: str, filename: str, sha256: str = "") -> str:
    """Create a readable certified PDF rendering for EML/RTF/TXT-like documents."""
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import letter

    width, height = letter
    c = canvas.Canvas(dest_path, pagesize=letter)
    c.setAuthor("VeriFYD")
    c.setTitle(f"VeriFYD Certified Document {cert_id}")
    c.setSubject(f"{label} | Authenticity {authenticity}% | {filename}")

    ext = os.path.splitext(src_path)[1].lower()
    text = _read_text_for_certified_render(src_path, ext)
    if not text:
        text = "[No readable text could be extracted from this document.]"

    margin_x = 54
    y = height - 54
    line_h = 11
    page_num = 1

    def header() -> None:
        nonlocal y
        c.setFont("Helvetica-Bold", 14)
        c.setFillGray(0.05)
        c.drawString(margin_x, y, "VeriFYD Certified Document Rendering")
        y -= 18
        c.setFont("Helvetica", 7.5)
        c.setFillGray(0.35)
        c.drawString(margin_x, y, f"File: {filename} • Status: {label} • Authenticity: {authenticity}%")
        y -= 22

    def footer() -> None:
        c.setFont("Helvetica", 6.5)
        c.setFillGray(0.45)
        c.drawString(margin_x, 24, f"VeriFYD certified text rendering • Page {page_num}")
        _draw_verifyd_mark(c, width, y=24, x_right_pad=34)

    header()
    max_w = width - (margin_x * 2)
    for paragraph in text.splitlines():
        if not paragraph.strip():
            y -= line_h
            if y < 50:
                footer(); c.showPage(); page_num += 1; y = height - 54; header()
            continue
        words = paragraph.split()
        current = ""
        for word in words:
            trial = (current + " " + word).strip()
            if c.stringWidth(trial, "Helvetica", 8.5) <= max_w:
                current = trial
            else:
                if y < 50:
                    footer(); c.showPage(); page_num += 1; y = height - 54; header()
                c.setFont("Helvetica", 8.5)
                c.setFillGray(0.08)
                c.drawString(margin_x, y, current[:160])
                y -= line_h
                current = word
        if current:
            if y < 50:
                footer(); c.showPage(); page_num += 1; y = height - 54; header()
            c.setFont("Helvetica", 8.5)
            c.setFillGray(0.08)
            c.drawString(margin_x, y, current[:160])
            y -= line_h
    footer()
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

    if ext in (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".heic", ".heif"):
        _create_image_pdf(src_path, dest_path, cert_id, authenticity, label, filename, sha256)
        if ext in (".heic", ".heif"):
            _attach_original_file_to_pdf(dest_path, src_path, filename or os.path.basename(src_path))
        return dest_path

    if ext == ".xlsx":
        return _create_xlsx_pdf(src_path, dest_path, cert_id, authenticity, label, filename, sha256)

    if ext == ".pptx":
        return _create_pptx_pdf(src_path, dest_path, cert_id, authenticity, label, filename, sha256)

    if ext in (".rtf", ".eml", ".msg", ".doc", ".ppt", ".xls", ".odt", ".ods", ".odp"):
        _create_text_render_pdf(src_path, dest_path, cert_id, authenticity, label, filename, sha256)
        # Preserve exact source file for legal/forensic review, especially when
        # the certified PDF is a readable text rendering rather than a perfect
        # native-layout clone.
        _attach_original_file_to_pdf(dest_path, src_path, filename or os.path.basename(src_path))
        return dest_path

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


