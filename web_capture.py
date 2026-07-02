# ============================================================
#  VeriFYD — web_capture.py
#
#  Phase 3B-1: Certified Web & Social Capture MVP
#  Captures a public URL screenshot, hashes the evidence, creates
#  a certified PDF report, and builds an evidence package ZIP.
# ============================================================

from __future__ import annotations

import hashlib
import ipaddress
import json
import os
import socket
import tempfile
import zipfile
from datetime import datetime, timezone
from typing import Any, Dict
from urllib.parse import urlparse, urlunparse


BLOCKED_HOSTS = {"localhost", "127.0.0.1", "0.0.0.0", "::1"}


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _json_safe(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except Exception:
        return str(value)


def normalize_public_url(raw_url: str) -> str:
    """Normalize and validate a public http(s) URL with a basic SSRF guard."""
    raw_url = (raw_url or "").strip()
    if not raw_url:
        raise ValueError("URL is required.")
    if not raw_url.lower().startswith(("http://", "https://")):
        raw_url = "https://" + raw_url

    parsed = urlparse(raw_url)
    if parsed.scheme.lower() not in ("http", "https"):
        raise ValueError("Only http and https URLs can be captured.")
    if not parsed.hostname:
        raise ValueError("URL must include a valid hostname.")

    host = parsed.hostname.strip().lower().rstrip(".")
    if host in BLOCKED_HOSTS or host.endswith(".local"):
        raise ValueError("Private or local URLs cannot be captured.")

    # Block direct private IPs and hostnames resolving only to private/internal IPs.
    try:
        ip = ipaddress.ip_address(host)
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast or ip.is_reserved:
            raise ValueError("Private or local IP addresses cannot be captured.")
    except ValueError as exc:
        # If this was our explicit rejection, re-raise. Otherwise host is a name.
        if "cannot be captured" in str(exc):
            raise
        try:
            infos = socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)
            addrs = {info[4][0] for info in infos if info and info[4]}
            if not addrs:
                raise ValueError("URL hostname could not be resolved.")
            for addr in addrs:
                try:
                    ip = ipaddress.ip_address(addr)
                    if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast or ip.is_reserved:
                        raise ValueError("Private or local URLs cannot be captured.")
                except ValueError as inner:
                    if "cannot be captured" in str(inner):
                        raise
        except socket.gaierror:
            raise ValueError("URL hostname could not be resolved.")

    # Drop URL fragments; they are browser-local and do not change server content.
    normalized = parsed._replace(fragment="")
    return urlunparse(normalized)


def capture_public_page(url: str, out_dir: str, *, timeout_ms: int = 30000) -> Dict[str, Any]:
    """Capture a visible-page screenshot and basic page metadata using Playwright."""
    url = normalize_public_url(url)
    os.makedirs(out_dir, exist_ok=True)

    screenshot_path = os.path.join(out_dir, "web_capture_screenshot.png")
    html_path = os.path.join(out_dir, "captured_page.html")

    captured_at = datetime.now(timezone.utc).isoformat()
    final_url = url
    title = ""
    viewport = {"width": 1365, "height": 1800}
    html_saved = False

    try:
        from playwright.sync_api import sync_playwright
    except Exception as exc:
        raise RuntimeError(
            "Playwright is not installed or Chromium is unavailable. Add playwright to requirements.txt "
            "and run: python -m playwright install chromium"
        ) from exc

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=["--no-sandbox", "--disable-dev-shm-usage", "--disable-gpu"],
        )
        try:
            context = browser.new_context(
                viewport=viewport,
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/126.0.0.0 Safari/537.36 VeriFYD-WebCapture/1.0"
                ),
                locale="en-US",
            )
            page = context.new_page()
            page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
            try:
                page.wait_for_load_state("networkidle", timeout=8000)
            except Exception:
                pass
            try:
                title = (page.title() or "").strip()[:300]
            except Exception:
                title = ""
            try:
                final_url = page.url or url
            except Exception:
                final_url = url
            page.screenshot(path=screenshot_path, full_page=False)
            try:
                html = page.content() or ""
                with open(html_path, "w", encoding="utf-8", errors="replace") as fh:
                    fh.write(html[:2_000_000])
                html_saved = True
            except Exception:
                html_saved = False
            try:
                context.close()
            except Exception:
                pass
        finally:
            browser.close()

    screenshot_sha256 = sha256_file(screenshot_path)
    html_sha256 = sha256_file(html_path) if html_saved and os.path.exists(html_path) else ""

    return {
        "captured_url": url,
        "final_url": final_url,
        "page_title": title,
        "captured_at": captured_at,
        "viewport": viewport,
        "screenshot_path": screenshot_path,
        "screenshot_sha256": screenshot_sha256,
        "html_path": html_path if html_saved else "",
        "html_sha256": html_sha256,
    }


def create_web_capture_report_pdf(capture: Dict[str, Any], report_path: str, cert_id: str, email: str = "") -> str:
    """Create a certified PDF report for the captured public web page."""
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.units import inch
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfgen import canvas
    from PIL import Image

    width, height = letter
    margin = 0.55 * inch
    c = canvas.Canvas(report_path, pagesize=letter)
    c.setAuthor("VeriFYD")
    c.setTitle(f"VeriFYD Certified Web Capture {cert_id}")
    c.setSubject(capture.get("captured_url", ""))

    def line(label: str, value: str, y: float) -> float:
        c.setFont("Helvetica-Bold", 8.5)
        c.setFillGray(0.18)
        c.drawString(margin, y, label)
        c.setFont("Helvetica", 8.2)
        c.setFillGray(0.08)
        max_chars = 105
        text = str(value or "")
        if len(text) <= max_chars:
            c.drawString(margin + 110, y, text)
            return y - 13
        chunks = [text[i:i + max_chars] for i in range(0, len(text), max_chars)][:4]
        for idx, chunk in enumerate(chunks):
            c.drawString(margin + 110, y - (idx * 11), chunk)
        return y - (len(chunks) * 11) - 3

    c.setFont("Helvetica-Bold", 21)
    c.setFillGray(0.05)
    c.drawString(margin, height - margin, "VeriFYD Certified Web Capture")

    c.setFont("Helvetica", 9)
    c.setFillGray(0.35)
    c.drawString(margin, height - margin - 18, "Public web page preservation report")

    y = height - margin - 46
    y = line("Certificate ID", cert_id, y)
    y = line("Captured URL", capture.get("captured_url", ""), y)
    y = line("Final URL", capture.get("final_url", ""), y)
    y = line("Page Title", capture.get("page_title", ""), y)
    y = line("Captured At", capture.get("captured_at", ""), y)
    if email:
        y = line("Certified To", email, y)
    y = line("Screenshot SHA-256", capture.get("screenshot_sha256", ""), y)
    if capture.get("html_sha256"):
        y = line("HTML Snapshot SHA-256", capture.get("html_sha256", ""), y)

    c.setFont("Helvetica", 8)
    c.setFillGray(0.30)
    disclaimer = (
        "This report preserves the visible public web page at the time of capture. "
        "Screenshots and hashes are evidence records; VeriFYD does not claim control over the original website "
        "or guarantee that dynamic content will remain available after capture."
    )
    for chunk in [disclaimer[i:i+105] for i in range(0, len(disclaimer), 105)]:
        c.drawString(margin, y, chunk)
        y -= 10

    screenshot_path = capture.get("screenshot_path") or ""
    if os.path.exists(screenshot_path):
        y -= 8
        c.setFont("Helvetica-Bold", 10)
        c.setFillGray(0.05)
        c.drawString(margin, y, "Captured visible page")
        y -= 8
        try:
            with Image.open(screenshot_path) as img:
                img_w, img_h = img.size
            max_w = width - (2 * margin)
            max_h = y - margin - 20
            scale = min(max_w / max(1, img_w), max_h / max(1, img_h))
            draw_w = img_w * scale
            draw_h = img_h * scale
            x = (width - draw_w) / 2
            c.drawImage(ImageReader(screenshot_path), x, margin + 20, width=draw_w, height=draw_h, preserveAspectRatio=True, mask="auto")
        except Exception as exc:
            c.setFont("Helvetica", 9)
            c.drawString(margin, y - 18, f"Screenshot preview could not be embedded: {str(exc)[:120]}")

    c.setFont("Helvetica-Bold", 8)
    c.setFillGray(0.16)
    c.drawRightString(width - margin, 20, "VeriFYD certified web capture")
    c.save()
    return report_path


def create_web_capture_package(capture: Dict[str, Any], report_path: str, package_path: str, cert_id: str, email: str = "") -> str:
    """Create a ZIP evidence package for a web capture."""
    metadata = {
        "title": "VeriFYD Certified Web Capture Metadata",
        "certificate_id": cert_id,
        "certified_to": email or "",
        "captured_url": capture.get("captured_url", ""),
        "final_url": capture.get("final_url", ""),
        "page_title": capture.get("page_title", ""),
        "captured_at": capture.get("captured_at", ""),
        "viewport": capture.get("viewport", {}),
        "screenshot_sha256": capture.get("screenshot_sha256", ""),
        "html_sha256": capture.get("html_sha256", ""),
        "report_sha256": sha256_file(report_path) if os.path.exists(report_path) else "",
    }

    hashes = {
        "screenshot_png": capture.get("screenshot_sha256", ""),
        "captured_html": capture.get("html_sha256", ""),
        "certified_web_capture_report_pdf": metadata["report_sha256"],
    }

    with zipfile.ZipFile(package_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("metadata.json", json.dumps(metadata, indent=2, ensure_ascii=False))
        zf.writestr("hashes.json", json.dumps(hashes, indent=2, ensure_ascii=False))
        if os.path.exists(report_path):
            zf.write(report_path, "VeriFYD_Certified_Web_Capture_Report.pdf")
        screenshot_path = capture.get("screenshot_path") or ""
        if os.path.exists(screenshot_path):
            zf.write(screenshot_path, "web_capture_screenshot.png")
        html_path = capture.get("html_path") or ""
        if html_path and os.path.exists(html_path):
            zf.write(html_path, "captured_page.html")
    return package_path


def create_certified_web_capture(url: str, cert_id: str, email: str = "") -> Dict[str, Any]:
    """End-to-end capture helper used by the RQ worker."""
    out_dir = tempfile.mkdtemp(prefix=f"verifyd_web_capture_{cert_id[:8]}_")
    capture = capture_public_page(url, out_dir)
    report_path = os.path.join(out_dir, f"VeriFYD_Web_Capture_{cert_id}.pdf")
    package_path = os.path.join(out_dir, f"VeriFYD_Web_Capture_Package_{cert_id}.zip")
    create_web_capture_report_pdf(capture, report_path, cert_id, email=email)
    create_web_capture_package(capture, report_path, package_path, cert_id, email=email)

    capture.update({
        "work_dir": out_dir,
        "report_path": report_path,
        "package_path": package_path,
        "report_sha256": sha256_file(report_path),
        "package_sha256": sha256_file(package_path),
    })
    return capture
