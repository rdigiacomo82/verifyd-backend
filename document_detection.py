# ============================================================
#  VeriFYD — document_detection.py  v1
#
#  MVP document authenticity engine for PDF / DOCX / TXT.
#  Returns the same tuple signature used by video/photo detection:
#      (authenticity:int, label:str, detail:dict)
#
#  Engines:
#    1. File/hash + metadata inspection
#    2. Text extraction + statistical writing signals
#    3. GPT semantic document analysis, if OPENAI_API_KEY is set
#    4. PDF embedded-image inventory / suspicious producer flags
# ============================================================

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
from collections import Counter
from datetime import datetime
from typing import Any, Dict, List, Tuple

log = logging.getLogger("verifyd.document_detection")

THRESHOLD_REAL = 55
THRESHOLD_UNDETERMINED = 40
MAX_GPT_CHARS = 12000

AI_PRODUCER_KEYWORDS = (
    "chatgpt", "openai", "sora", "dall-e", "dalle", "midjourney",
    "stable diffusion", "stability ai", "runway", "kling", "pika",
    "firefly", "adobe firefly", "gemini", "bard", "claude", "anthropic",
    "canva", "copilot", "microsoft designer", "gamma", "beautiful.ai",
)

COMMON_GENERATOR_KEYWORDS = (
    "microsoft word", "word", "pages", "google docs", "adobe acrobat",
    "preview", "libreoffice", "powerpoint", "excel", "quartz pdfcontext",
)

AI_TEXT_PATTERNS = (
    "in today's", "ever-evolving", "delve into", "it's important to note",
    "in conclusion", "as an ai", "unlock the potential", "seamlessly",
    "robust solution", "cutting-edge", "game-changer", "leverage",
    "streamline", "comprehensive approach", "dynamic landscape",
)


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _clip_text(text: str, limit: int = MAX_GPT_CHARS) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    if len(text) <= limit:
        return text
    head = text[: limit // 2]
    tail = text[-limit // 2 :]
    return head + "\n\n[...middle omitted for analysis length...]\n\n" + tail


def _parse_pdf_date(value: Any) -> str:
    """Convert a PDF D:YYYYMMDDHHmmSS date to a readable ISO-like string."""
    if not value:
        return ""
    s = str(value).strip()
    if s.startswith("D:"):
        raw = s[2:16]
        try:
            return datetime.strptime(raw, "%Y%m%d%H%M%S").isoformat()
        except Exception:
            return s
    return s


def _read_pdf(path: str) -> Tuple[str, Dict[str, Any]]:
    meta: Dict[str, Any] = {"type": "pdf", "pages": 0, "embedded_images": 0, "metadata": {}}
    text_parts: List[str] = []
    try:
        from pypdf import PdfReader
    except Exception as e:
        raise RuntimeError("Missing dependency: pypdf. Add pypdf>=4.0.0 to requirements.txt") from e

    reader = PdfReader(path)
    meta["pages"] = len(reader.pages)

    raw_meta = reader.metadata or {}
    parsed_meta = {}
    for k, v in raw_meta.items():
        key = str(k).lstrip("/")
        parsed_meta[key] = _parse_pdf_date(v) if "date" in key.lower() else str(v)
    meta["metadata"] = parsed_meta

    # Text extraction
    for i, page in enumerate(reader.pages[:50]):
        try:
            page_text = page.extract_text() or ""
            if page_text.strip():
                text_parts.append(page_text)
        except Exception as e:
            log.debug("PDF text extraction failed page=%d: %s", i, e)

    # Embedded-image count, best-effort only
    image_count = 0
    for page in reader.pages[:50]:
        try:
            resources = page.get("/Resources", {}) or {}
            xobj = resources.get("/XObject", {}) or {}
            if hasattr(xobj, "get_object"):
                xobj = xobj.get_object()
            for _, obj in xobj.items():
                try:
                    resolved = obj.get_object() if hasattr(obj, "get_object") else obj
                    if resolved.get("/Subtype") == "/Image":
                        image_count += 1
                except Exception:
                    continue
        except Exception:
            continue
    meta["embedded_images"] = image_count
    return "\n".join(text_parts), meta


def _read_docx(path: str) -> Tuple[str, Dict[str, Any]]:
    meta: Dict[str, Any] = {"type": "docx", "pages": 0, "embedded_images": 0, "metadata": {}}
    try:
        from docx import Document
    except Exception as e:
        raise RuntimeError("Missing dependency: python-docx. Add python-docx>=1.1.0 to requirements.txt") from e

    doc = Document(path)
    text_parts = [p.text for p in doc.paragraphs if p.text and p.text.strip()]
    # Tables often contain invoice/resume values
    for table in doc.tables:
        for row in table.rows:
            vals = [cell.text.strip() for cell in row.cells if cell.text and cell.text.strip()]
            if vals:
                text_parts.append(" | ".join(vals))

    props = doc.core_properties
    meta["metadata"] = {
        "author": props.author or "",
        "last_modified_by": props.last_modified_by or "",
        "created": props.created.isoformat() if props.created else "",
        "modified": props.modified.isoformat() if props.modified else "",
        "title": props.title or "",
        "subject": props.subject or "",
        "keywords": props.keywords or "",
        "comments": props.comments or "",
        "revision": str(props.revision or ""),
    }
    # Inline shapes gives a decent embedded image count
    try:
        meta["embedded_images"] = len(doc.inline_shapes)
    except Exception:
        meta["embedded_images"] = 0
    return "\n".join(text_parts), meta


def _read_txt(path: str) -> Tuple[str, Dict[str, Any]]:
    meta = {"type": "txt", "pages": 0, "embedded_images": 0, "metadata": {}}
    with open(path, "rb") as f:
        data = f.read(2_000_000)
    for enc in ("utf-8", "utf-16", "latin-1"):
        try:
            return data.decode(enc, errors="replace"), meta
        except Exception:
            continue
    return data.decode("latin-1", errors="replace"), meta


def _read_image(path: str) -> Tuple[str, Dict[str, Any]]:
    """Best-effort metadata extraction for JPG/JPEG/PNG document images."""
    meta: Dict[str, Any] = {"type": "image", "pages": 1, "embedded_images": 1, "metadata": {}}
    try:
        from PIL import Image, ExifTags
    except Exception as e:
        raise RuntimeError("Missing dependency: Pillow. Add Pillow>=10.0.0 to requirements.txt") from e

    with Image.open(path) as img:
        md: Dict[str, Any] = {
            "format": img.format or "",
            "mode": img.mode or "",
            "width": str(img.width),
            "height": str(img.height),
        }

        # PNG text/info chunks and software fields.
        try:
            for k, v in (img.info or {}).items():
                if isinstance(v, (str, int, float)):
                    md[str(k)] = str(v)[:500]
        except Exception:
            pass

        # JPEG EXIF fields.
        try:
            exif = img.getexif()
            tag_map = getattr(ExifTags, "TAGS", {})
            for tag_id, value in exif.items():
                tag_name = tag_map.get(tag_id, str(tag_id))
                if isinstance(value, bytes):
                    value = value[:80].hex()
                md[str(tag_name)] = str(value)[:500]
        except Exception:
            pass

    meta["metadata"] = md
    return "", meta


def _extract_document(path: str) -> Tuple[str, Dict[str, Any]]:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return _read_pdf(path)
    if ext == ".docx":
        return _read_docx(path)
    if ext in (".txt", ".md", ".csv"):
        return _read_txt(path)
    if ext in (".jpg", ".jpeg", ".png"):
        return _read_image(path)
    raise RuntimeError(f"Unsupported document format: {ext}")


def _metadata_score(meta: Dict[str, Any]) -> Tuple[int, List[str]]:
    score = 0
    flags: List[str] = []
    md = {str(k).lower(): str(v).lower() for k, v in (meta.get("metadata") or {}).items()}
    joined = " ".join(md.values())

    for kw in AI_PRODUCER_KEYWORDS:
        if kw in joined:
            score += 35
            flags.append(f"metadata references possible AI tool: {kw}")
            break

    producer = md.get("producer", "") or md.get("creator", "")
    if producer and not any(k in producer for k in COMMON_GENERATOR_KEYWORDS):
        if any(k in producer for k in ("pdfkit", "wkhtml", "weasyprint", "reportlab", "chromium", "headless")):
            score += 10
            flags.append("document was generated by an automated PDF/rendering pipeline")

    created = md.get("creationdate", "") or md.get("created", "")
    modified = md.get("moddate", "") or md.get("modified", "")
    if created and modified and created != modified:
        score += 5
        flags.append("creation and modification timestamps differ")

    if meta.get("type") == "pdf" and meta.get("pages", 0) > 0 and meta.get("embedded_images", 0) >= meta.get("pages", 0):
        score += 8
        flags.append("PDF appears image-heavy or scanned; visual tamper review recommended")

    return min(score, 60), flags


def _text_stats_score(text: str) -> Tuple[int, List[str], Dict[str, Any]]:
    flags: List[str] = []
    clean = re.sub(r"\s+", " ", text or "").strip()
    words = re.findall(r"[A-Za-z']+", clean.lower())
    sentences = [s.strip() for s in re.split(r"[.!?]+", clean) if len(s.strip()) > 2]
    stats = {
        "word_count": len(words),
        "sentence_count": len(sentences),
        "unique_word_ratio": 0.0,
        "avg_sentence_len": 0.0,
        "sentence_len_cv": 0.0,
        "ai_phrase_hits": [],
    }

    if not words:
        return 20, ["no extractable text; document may be scanned or image-only"], stats

    counts = Counter(words)
    unique_ratio = len(counts) / max(1, len(words))
    stats["unique_word_ratio"] = round(unique_ratio, 3)

    sent_lens = [len(re.findall(r"[A-Za-z']+", s)) for s in sentences]
    if sent_lens:
        avg = sum(sent_lens) / len(sent_lens)
        var = sum((x - avg) ** 2 for x in sent_lens) / len(sent_lens)
        cv = math.sqrt(var) / avg if avg else 0
        stats["avg_sentence_len"] = round(avg, 2)
        stats["sentence_len_cv"] = round(cv, 3)
    else:
        cv = 0

    score = 0
    lower = clean.lower()
    phrase_hits = [p for p in AI_TEXT_PATTERNS if p in lower]
    stats["ai_phrase_hits"] = phrase_hits[:8]
    if len(phrase_hits) >= 3:
        score += 16
        flags.append(f"multiple AI-style phrases detected ({len(phrase_hits)})")
    elif len(phrase_hits) >= 1:
        score += 6
        flags.append("some AI-style phrasing detected")

    if len(words) >= 250 and unique_ratio < 0.34:
        score += 10
        flags.append("low vocabulary variation for document length")

    if len(sent_lens) >= 8 and cv < 0.38:
        score += 10
        flags.append("unusually uniform sentence rhythm")

    # Very short docs are difficult to authenticate using writing style.
    if len(words) < 80:
        score = min(score, 10)
        flags.append("short document; text-style AI confidence reduced")

    return min(score, 40), flags, stats


def _gpt_document_score(text: str, meta: Dict[str, Any], ext: str) -> Tuple[int, str, List[str], bool]:
    """Use OpenAI if configured. Safe fallback returns unavailable."""
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        return 50, "GPT document analysis unavailable: OPENAI_API_KEY not set.", [], False

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        prompt = {
            "task": "Document authenticity analysis",
            "instructions": (
                "Return JSON only. Estimate probability this document text/metadata was AI-generated, "
                "synthetically assembled, or materially tampered. Do not judge whether claims are true unless internal inconsistencies are present. "
                "Use 0=confident authentic human/original, 100=confident AI/generated/tampered."
            ),
            "file_type": ext,
            "metadata": meta.get("metadata", {}),
            "pages": meta.get("pages", 0),
            "embedded_images": meta.get("embedded_images", 0),
            "text_excerpt": _clip_text(text),
            "schema": {"ai_probability": "integer 0-100", "flags": ["short strings"], "reasoning": "one concise paragraph"},
        }
        resp = client.chat.completions.create(
            model=os.environ.get("OPENAI_TEXT_MODEL", "gpt-4o-mini"),
            temperature=0,
            messages=[
                {"role": "system", "content": "You are a forensic document authentication analyst. Return strict JSON."},
                {"role": "user", "content": json.dumps(prompt)},
            ],
        )
        raw = resp.choices[0].message.content or "{}"
        raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        data = json.loads(raw)
        score = max(0, min(100, _safe_int(data.get("ai_probability", 50), 50)))
        flags = [str(x) for x in data.get("flags", [])][:8]
        reasoning = str(data.get("reasoning", ""))[:1200]
        return score, reasoning, flags, True
    except Exception as e:
        log.warning("GPT document analysis failed: %s", e)
        return 50, f"GPT document analysis failed: {str(e)[:120]}", [], False


def _label_from_ai_score(ai_score: int) -> Tuple[int, str]:
    authenticity = 100 - int(round(ai_score))
    if authenticity >= THRESHOLD_REAL:
        label = "REAL"
    elif authenticity >= THRESHOLD_UNDETERMINED:
        label = "UNDETERMINED"
    else:
        label = "AI"
    return authenticity, label


def _build_reasoning(label: str, authenticity: int, flags: List[str], gpt_reasoning: str, meta: Dict[str, Any], text_stats: Dict[str, Any]) -> str:
    doc_type = meta.get("type", "document").upper()
    if label == "REAL":
        lead = f"This {doc_type} document shows no strong indicators of AI generation or material tampering."
    elif label == "UNDETERMINED":
        lead = f"This {doc_type} document contains mixed or incomplete authenticity signals."
    else:
        lead = f"This {doc_type} document shows indicators consistent with AI generation, automated assembly, or tampering."

    evidence = "; ".join(flags[:4]) if flags else "metadata and text signals were reviewed"
    if gpt_reasoning and "unavailable" not in gpt_reasoning.lower() and "failed" not in gpt_reasoning.lower():
        return f"{lead} Key evidence: {evidence}. GPT review: {gpt_reasoning} Authenticity score: {authenticity}%."
    return f"{lead} Key evidence: {evidence}. Authenticity score: {authenticity}%."


def run_document_detection(path: str) -> Tuple[int, str, Dict[str, Any]]:
    """Run document authentication and return (authenticity, label, detail)."""
    ext = os.path.splitext(path)[1].lower()
    sha256 = _sha256_file(path)
    size_bytes = os.path.getsize(path)

    text, meta = _extract_document(path)
    meta_score, meta_flags = _metadata_score(meta)
    text_score, text_flags, text_stats = _text_stats_score(text)
    gpt_score, gpt_reasoning, gpt_flags, gpt_available = _gpt_document_score(text, meta, ext)

    # Weighting: metadata/text are stable and cheap; GPT is semantic but not final authority.
    # If GPT unavailable, use metadata/text only and pull toward uncertain when evidence is weak.
    if gpt_available:
        combined = meta_score * 0.30 + text_score * 0.25 + gpt_score * 0.45
        w_gpt = 0.45
    else:
        base = meta_score * 0.55 + text_score * 0.45
        # Avoid over-certifying text-only docs without GPT review.
        combined = max(12.0, base) if base < 35 else base
        w_gpt = 0.0

    # Strong metadata AI flag should be respected even if writing looks ordinary.
    if meta_score >= 35:
        combined = max(combined, 62.0)

    # No extractable text + image-heavy PDF should not be certified REAL in MVP.
    if not text.strip() and meta.get("type") == "pdf":
        combined = max(combined, 45.0)

    combined = max(0, min(100, int(round(combined))))
    authenticity, label = _label_from_ai_score(combined)

    flags = meta_flags + text_flags + gpt_flags
    reasoning = _build_reasoning(label, authenticity, flags, gpt_reasoning, meta, text_stats)

    detail = {
        "ai_score": combined,
        "authenticity": authenticity,
        "label": label,
        "signal_ai_score": int(round(meta_score * 0.55 + text_score * 0.45)),
        "gpt_ai_score": gpt_score,
        "gpt_available": gpt_available,
        "gpt_reasoning": reasoning,
        "gpt_flags": flags[:10],
        "metadata_score": meta_score,
        "text_score": text_score,
        "weight_signal": 1.0 - w_gpt,
        "weight_gpt": w_gpt,
        "blend_mode": "document metadata+text+gpt" if gpt_available else "document metadata+text only",
        "content_type": "document",
        "document_type": meta.get("type", ext.lstrip(".")),
        "sha256": sha256,
        "file_size_bytes": size_bytes,
        "pages": meta.get("pages", 0),
        "embedded_images": meta.get("embedded_images", 0),
        "text_stats": text_stats,
        "metadata": meta.get("metadata", {}),
        "threshold_real": THRESHOLD_REAL,
        "threshold_undet": THRESHOLD_UNDETERMINED,
    }
    log.info(
        "Document detection complete | type=%s meta=%d text=%d gpt=%d combined=%d auth=%d label=%s",
        meta.get("type"), meta_score, text_score, gpt_score, combined, authenticity, label,
    )
    return authenticity, label, detail
