# ============================================================
# VeriFYD Trust Voice — Attachment Content Validation
# VERIFYD_TRUST_VOICE_ATTACHMENT_VALIDATION_V1
#
# Validates actual attachment bytes before private R2 upload.
# This is routing hardening only; it does not prove safety/authenticity.
# ============================================================

from __future__ import annotations

import os
import re
import zipfile
from typing import Optional

VALIDATION_MODE = "signature_and_container_v1"

PHOTO_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif"}
VIDEO_EXTENSIONS = {
    ".mp4", ".mov", ".m4v", ".webm", ".avi", ".mkv",
    ".mpg", ".mpeg", ".3gp", ".3g2", ".mts", ".m2ts",
    ".ts", ".ogv", ".flv", ".wmv",
}
AUDIO_EXTENSIONS = {
    ".mp3", ".wav", ".m4a", ".aac", ".flac",
    ".ogg", ".oga", ".opus", ".webm",
}
DOCUMENT_EXTENSIONS = {
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    ".odt", ".ods", ".odp", ".txt", ".md", ".csv", ".rtf",
    ".eml", ".msg", ".html", ".htm", ".mhtml", ".mht",
    ".xml", ".json", ".svg", ".vsdx", ".yaml", ".yml",
    ".toml", ".env", ".ini", ".properties", ".conf", ".cfg",
    ".config", ".cnf", ".log", ".sql",
}

TEXT_DOCUMENT_EXTENSIONS = {
    ".txt", ".md", ".csv", ".eml", ".html", ".htm", ".mhtml", ".mht",
    ".xml", ".json", ".svg", ".yaml", ".yml", ".toml", ".env", ".ini",
    ".properties", ".conf", ".cfg", ".config", ".cnf", ".log", ".sql",
}
OLE_DOCUMENT_EXTENSIONS = {".doc", ".xls", ".ppt", ".msg"}
ZIP_DOCUMENT_EXTENSIONS = {".docx", ".xlsx", ".pptx", ".vsdx", ".odt", ".ods", ".odp"}

CANONICAL_MIME = {
    ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
    ".webp": "image/webp", ".heic": "image/heic", ".heif": "image/heif",
    ".mp4": "video/mp4", ".mov": "video/quicktime", ".m4v": "video/x-m4v",
    ".webm": "video/webm", ".avi": "video/x-msvideo", ".mkv": "video/x-matroska",
    ".mpg": "video/mpeg", ".mpeg": "video/mpeg", ".3gp": "video/3gpp",
    ".3g2": "video/3gpp2", ".mts": "video/mp2t", ".m2ts": "video/mp2t",
    ".ts": "video/mp2t", ".ogv": "video/ogg", ".flv": "video/x-flv",
    ".wmv": "video/x-ms-wmv", ".mp3": "audio/mpeg", ".wav": "audio/wav",
    ".m4a": "audio/mp4", ".aac": "audio/aac", ".flac": "audio/flac",
    ".ogg": "audio/ogg", ".oga": "audio/ogg", ".opus": "audio/ogg",
    ".pdf": "application/pdf", ".doc": "application/msword",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".xls": "application/vnd.ms-excel",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".ppt": "application/vnd.ms-powerpoint",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ".odt": "application/vnd.oasis.opendocument.text",
    ".ods": "application/vnd.oasis.opendocument.spreadsheet",
    ".odp": "application/vnd.oasis.opendocument.presentation",
    ".txt": "text/plain", ".md": "text/markdown", ".csv": "text/csv",
    ".rtf": "application/rtf", ".eml": "message/rfc822",
    ".msg": "application/vnd.ms-outlook", ".html": "text/html", ".htm": "text/html",
    ".mhtml": "multipart/related", ".mht": "multipart/related", ".xml": "application/xml",
    ".json": "application/json", ".svg": "image/svg+xml", ".vsdx": "application/vnd.visio",
    ".yaml": "application/yaml", ".yml": "application/yaml", ".toml": "application/toml",
    ".env": "text/plain", ".ini": "text/plain", ".properties": "text/plain",
    ".conf": "text/plain", ".cfg": "text/plain", ".config": "text/plain",
    ".cnf": "text/plain", ".log": "text/plain", ".sql": "application/sql",
}

OLE_MAGIC = bytes.fromhex("D0CF11E0A1B11AE1")
ASF_MAGIC = bytes.fromhex("3026B2758E66CF11A6D900AA0062CE6C")
EBML_MAGIC = bytes.fromhex("1A45DFA3")
_HEIF_BRANDS = {b"heic", b"heix", b"hevc", b"hevx", b"heim", b"heis", b"hevm", b"hevs", b"mif1", b"msf1"}
_SVG_RE = re.compile(br"<\s*svg(?:\s|>)", re.IGNORECASE)


class AttachmentContentError(Exception):
    def __init__(
        self,
        code: str = "attachment_content_mismatch",
        message: str = "The file contents do not match the selected file type.",
    ):
        super().__init__(message)
        self.code = code
        self.message = message


def _fail() -> None:
    raise AttachmentContentError()


def _read_sample(path: str, limit: int = 262144) -> bytes:
    with open(path, "rb") as fh:
        return fh.read(limit)


def _looks_like_text(sample: bytes) -> bool:
    if not sample:
        return False
    if sample.startswith((b"\xff\xfe", b"\xfe\xff")):
        return True
    bad = 0
    for value in sample:
        if value == 0:
            bad += 4
        elif value < 32 and value not in (9, 10, 12, 13):
            bad += 1
    return (bad / max(len(sample), 1)) <= 0.02


def _is_riff(sample: bytes, form: bytes) -> bool:
    return len(sample) >= 12 and sample[:4] == b"RIFF" and sample[8:12] == form


def _is_mp3(sample: bytes) -> bool:
    if sample.startswith(b"ID3"):
        return True

    if len(sample) < 4 or sample[0] != 0xFF or (sample[1] & 0xE0) != 0xE0:
        return False

    version_id = (sample[1] >> 3) & 0x03
    layer_id = (sample[1] >> 1) & 0x03
    bitrate_index = (sample[2] >> 4) & 0x0F
    sample_rate_index = (sample[2] >> 2) & 0x03

    if version_id == 0x01 or layer_id == 0x00:
        return False
    if bitrate_index in {0x00, 0x0F} or sample_rate_index == 0x03:
        return False

    return True


def _is_aac_adts(sample: bytes) -> bool:
    return len(sample) >= 2 and sample[0] == 0xFF and (sample[1] & 0xF6) == 0xF0


def _is_mpeg_video(sample: bytes) -> bool:
    return sample.startswith((b"\x00\x00\x01\xba", b"\x00\x00\x01\xb3"))


def _ogg_codec(sample: bytes) -> str:
    head = sample[:262144]
    if b"OpusHead" in head:
        return "opus"
    if b"\x01vorbis" in head:
        return "vorbis"
    if b"\x80theora" in head:
        return "theora"
    if b"\x7fFLAC" in head:
        return "flac"
    return ""


def _ebml_doctype(sample: bytes) -> str:
    head = sample[:65536].lower()
    if b"webm" in head:
        return "webm"
    if b"matroska" in head:
        return "matroska"
    return ""


def _bmff_handlers(path: str) -> set[bytes]:
    # Locate ordinary ISO-BMFF 'hdlr' boxes without loading the whole
    # attachment into memory. Handler type is 12 bytes after the box type.
    handlers: set[bytes] = set()
    tail = b""

    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(1024 * 1024)
            if not chunk:
                break

            data = tail + chunk
            start = 0

            while True:
                idx = data.find(b"hdlr", start)
                if idx < 0:
                    break

                if idx >= 4 and idx + 16 <= len(data):
                    size = int.from_bytes(data[idx - 4:idx], "big", signed=False)
                    if size >= 20:
                        handler = data[idx + 12:idx + 16]
                        if handler in {b"vide", b"soun"}:
                            handlers.add(handler)

                start = idx + 4

            tail = data[-32:]

    return handlers


def _is_transport_stream(sample: bytes, m2ts_ok: bool = False) -> bool:
    for offset in (0, 1, 2, 3):
        if (
            len(sample) > offset + 376
            and sample[offset] == 0x47
            and sample[offset + 188] == 0x47
            and sample[offset + 376] == 0x47
        ):
            return True
    if m2ts_ok:
        for offset in (4, 5, 6, 7):
            if (
                len(sample) > offset + 384
                and sample[offset] == 0x47
                and sample[offset + 192] == 0x47
                and sample[offset + 384] == 0x47
            ):
                return True
    return False


def _bmff_info(sample: bytes) -> tuple[bool, set[bytes]]:
    brands: set[bytes] = set()
    if len(sample) >= 12 and sample[4:8] == b"ftyp":
        brands.add(sample[8:12])
        for index in range(16, min(len(sample), 128), 4):
            brand = sample[index:index + 4]
            if len(brand) == 4:
                brands.add(brand)
        return True, brands
    if len(sample) >= 8 and sample[4:8] in {b"moov", b"mdat", b"wide", b"free", b"skip"}:
        return True, brands
    return False, brands


def _zip_document_kind(path: str) -> Optional[str]:
    try:
        with zipfile.ZipFile(path, "r") as zf:
            infos = zf.infolist()
            if len(infos) > 10000:
                return None

            names = {info.filename.replace("\\", "/").lower() for info in infos}

            has_ooxml_root = (
                "[content_types].xml" in names
                and "_rels/.rels" in names
            )

            if has_ooxml_root and any(name.startswith("word/") for name in names):
                return "docx"
            if has_ooxml_root and any(name.startswith("xl/") for name in names):
                return "xlsx"
            if has_ooxml_root and any(name.startswith("ppt/") for name in names):
                return "pptx"
            if has_ooxml_root and any(name.startswith("visio/") for name in names):
                return "vsdx"

            if "mimetype" in names:
                try:
                    info = zf.getinfo("mimetype")
                    if info.file_size > 512:
                        return None
                    with zf.open(info, "r") as member:
                        raw = member.read(256)
                except Exception:
                    raw = b""

                if raw == b"application/vnd.oasis.opendocument.text":
                    return "odt"
                if raw == b"application/vnd.oasis.opendocument.spreadsheet":
                    return "ods"
                if raw == b"application/vnd.oasis.opendocument.presentation":
                    return "odp"
    except Exception:
        return None

    return None


def _media_category(extension: str, sample: bytes) -> str:
    if extension in PHOTO_EXTENSIONS:
        return "photo"
    if extension == ".webm":
        lower = sample.lower()
        has_video = any(marker in lower for marker in (b"v_vp8", b"v_vp9", b"v_av1"))
        has_audio = any(marker in lower for marker in (b"a_opus", b"a_vorbis"))
        if has_video:
            return "video"
        if has_audio and not has_video:
            return "audio"
        return "unknown"
    if extension in AUDIO_EXTENSIONS:
        return "audio"
    if extension in VIDEO_EXTENSIONS:
        return "video"
    if extension in DOCUMENT_EXTENSIONS:
        return "document"
    return "unknown"


def _canonical_content_type(extension: str, category: str) -> str:
    if extension == ".webm":
        return "audio/webm" if category == "audio" else "video/webm"
    return CANONICAL_MIME.get(extension, "application/octet-stream")


def validate_attachment_content(
    path: str,
    *,
    extension: str,
    supplied_content_type: str = "",
) -> dict:
    """Validate actual bytes before Trust Voice storage/analysis routing."""
    ext = (extension or "").strip().lower()
    if not ext or not os.path.isfile(path) or os.path.getsize(path) <= 0:
        _fail()

    sample = _read_sample(path)

    if ext in {".jpg", ".jpeg"}:
        if not sample.startswith(b"\xff\xd8\xff"):
            _fail()
    elif ext == ".png":
        if not sample.startswith(b"\x89PNG\r\n\x1a\n"):
            _fail()
    elif ext == ".webp":
        if not _is_riff(sample, b"WEBP"):
            _fail()
    elif ext in {".heic", ".heif"}:
        is_bmff, brands = _bmff_info(sample)
        major_brand = sample[8:12] if len(sample) >= 12 else b""
        if major_brand in {b"avif", b"avis"}:
            _fail()
        if not is_bmff or not (brands & _HEIF_BRANDS):
            _fail()
    elif ext in {".mp4", ".mov", ".m4v", ".m4a", ".3gp", ".3g2"}:
        is_bmff, _brands = _bmff_info(sample)
        if not is_bmff:
            _fail()

        handlers = _bmff_handlers(path)
        if ext == ".m4a":
            if b"soun" not in handlers or b"vide" in handlers:
                _fail()
        else:
            if b"vide" not in handlers:
                _fail()
    elif ext == ".webm":
        if not sample.startswith(EBML_MAGIC) or _ebml_doctype(sample) != "webm":
            _fail()
        category = _media_category(ext, sample)
        if category == "unknown":
            _fail()
    elif ext == ".mkv":
        if not sample.startswith(EBML_MAGIC):
            _fail()
        if _ebml_doctype(sample) not in {"matroska", "webm"}:
            _fail()
        lower = sample.lower()
        if not any(marker in lower for marker in (
            b"v_vp8", b"v_vp9", b"v_av1", b"v_mpeg4",
            b"v_mpeg2", b"v_mpeg1", b"v_theora", b"v_ms/vfw/fourcc"
        )):
            _fail()
    elif ext == ".avi":
        if not _is_riff(sample, b"AVI "):
            _fail()
    elif ext in {".mpg", ".mpeg"}:
        if not _is_mpeg_video(sample):
            _fail()
    elif ext == ".ts":
        if not _is_transport_stream(sample, m2ts_ok=False):
            _fail()
    elif ext in {".mts", ".m2ts"}:
        if not _is_transport_stream(sample, m2ts_ok=True):
            _fail()
    elif ext in {".ogg", ".oga", ".opus", ".ogv"}:
        if not sample.startswith(b"OggS"):
            _fail()

        codec = _ogg_codec(sample)
        if ext == ".ogv":
            if codec != "theora":
                _fail()
        elif ext == ".opus":
            if codec != "opus":
                _fail()
        else:
            if codec not in {"opus", "vorbis", "flac"}:
                _fail()
    elif ext == ".flv":
        if not sample.startswith(b"FLV"):
            _fail()
    elif ext == ".wmv":
        if not sample.startswith(ASF_MAGIC):
            _fail()
    elif ext == ".mp3":
        if not _is_mp3(sample):
            _fail()
    elif ext == ".wav":
        if not _is_riff(sample, b"WAVE"):
            _fail()
    elif ext == ".aac":
        if not _is_aac_adts(sample):
            _fail()
    elif ext == ".flac":
        if not sample.startswith(b"fLaC"):
            _fail()
    elif ext == ".pdf":
        if not sample.lstrip().startswith(b"%PDF-"):
            _fail()
    elif ext in OLE_DOCUMENT_EXTENSIONS:
        if not sample.startswith(OLE_MAGIC):
            _fail()
    elif ext in ZIP_DOCUMENT_EXTENSIONS:
        if _zip_document_kind(path) != ext.lstrip("."):
            _fail()
    elif ext == ".rtf":
        if not _looks_like_text(sample) or not sample.lstrip().lower().startswith(b"{\\rtf"):
            _fail()
    elif ext == ".svg":
        if not _looks_like_text(sample) or not _SVG_RE.search(sample[:65536]):
            _fail()
    elif ext in TEXT_DOCUMENT_EXTENSIONS:
        if not _looks_like_text(sample):
            _fail()
    else:
        raise AttachmentContentError(
            code="attachment_validation_unavailable",
            message="This file type cannot currently be validated for Trust Voice attachments.",
        )

    category = _media_category(ext, sample)
    if category == "unknown":
        _fail()

    return {
        "validation": VALIDATION_MODE,
        "content_type": _canonical_content_type(ext, category),
        "media_category": category,
    }
