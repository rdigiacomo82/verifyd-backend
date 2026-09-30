# ============================================================
# VeriFYD Trust Voice — Phase 2B/3 Profile Media
# VERIFYD_TRUST_VOICE_PROFILE_MEDIA_V1
#
# Adds authenticated profile image/logo upload and public display.
#
# Design:
#   - Personal users may upload a profile photo.
#   - Businesses may upload a company logo.
#   - Emoji remains the fallback identity decoration.
#   - Uploaded media NEVER implies VeriFYD verification.
#   - Uses an isolated PostgreSQL table; no existing table altered.
#   - Images are validated and normalized with Pillow.
#
# IMPORTANT:
#   Disabled unless VERIFYD_TRUST_VOICE_PROFILE_MEDIA_ENABLED=1
# ============================================================

from __future__ import annotations

import io
import logging
import os
from datetime import datetime, timezone
from typing import Literal, Optional

from fastapi import (
    APIRouter,
    File,
    Form,
    Header,
    HTTPException,
    Response,
    UploadFile,
)
from PIL import Image, ImageOps, UnidentifiedImageError

from database import get_db
from trust_voice_signaling import _identity_by_id, _verify_token

log = logging.getLogger("verifyd.trust_voice.profile_media")

router = APIRouter(prefix="/trust-voice", tags=["Trust Voice Profile Media"])

FEATURE_VERSION = "0.3.0"

MAX_UPLOAD_BYTES = 3 * 1024 * 1024
OUTPUT_SIZE = 512
ALLOWED_INPUT_MIME = {
    "image/jpeg",
    "image/png",
    "image/webp",
}
ALLOWED_MEDIA_KIND = {"personal_photo", "business_logo"}


def _enabled() -> bool:
    value = (
        os.environ.get("VERIFYD_TRUST_VOICE_PROFILE_MEDIA_ENABLED", "") or ""
    ).strip().lower()
    return value in {"1", "true", "yes", "on"}


def _require_enabled() -> None:
    if not _enabled():
        raise HTTPException(status_code=404, detail="not_found")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_schema() -> None:
    """Create only the isolated Trust Voice profile media table."""
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS verifyd_identity_media (
                identity_id      TEXT PRIMARY KEY,
                media_kind       TEXT NOT NULL,
                mime_type        TEXT NOT NULL,
                image_bytes      BYTEA NOT NULL,
                width            INTEGER NOT NULL,
                height           INTEGER NOT NULL,
                byte_size        INTEGER NOT NULL,
                created_at       TEXT NOT NULL,
                updated_at       TEXT NOT NULL
            )
            """
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_verifyd_identity_media_kind
            ON verifyd_identity_media(media_kind)
            """
        )


def _identity_from_bearer(authorization: Optional[str]) -> dict:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail={"error": "session_required"})

    token = authorization.split(" ", 1)[1].strip()
    if not token:
        raise HTTPException(status_code=401, detail={"error": "session_required"})

    payload = _verify_token(token)
    identity = _identity_by_id(payload.get("identity_id", ""))
    if not identity:
        raise HTTPException(status_code=401, detail={"error": "identity_not_found"})
    return identity


def _normalize_image(raw: bytes) -> tuple[bytes, int, int]:
    if not raw:
        raise HTTPException(status_code=400, detail={"error": "empty_image"})

    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail={
                "error": "image_too_large",
                "message": "Profile image must be 3 MB or smaller.",
            },
        )

    try:
        with Image.open(io.BytesIO(raw)) as opened:
            image = ImageOps.exif_transpose(opened)
            image.load()

            if image.width < 64 or image.height < 64:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "image_too_small",
                        "message": "Profile image must be at least 64×64 pixels.",
                    },
                )

            if image.width * image.height > 25_000_000:
                raise HTTPException(
                    status_code=413,
                    detail={
                        "error": "image_dimensions_too_large",
                        "message": "Profile image dimensions are too large.",
                    },
                )

            if image.mode not in ("RGB", "RGBA"):
                image = image.convert("RGBA")

            contained = ImageOps.contain(
                image,
                (OUTPUT_SIZE, OUTPUT_SIZE),
                method=Image.Resampling.LANCZOS,
            )

            canvas = Image.new("RGBA", (OUTPUT_SIZE, OUTPUT_SIZE), (0, 0, 0, 0))
            x = (OUTPUT_SIZE - contained.width) // 2
            y = (OUTPUT_SIZE - contained.height) // 2
            canvas.paste(
                contained,
                (x, y),
                contained if contained.mode == "RGBA" else None,
            )

            out = io.BytesIO()
            canvas.save(out, format="WEBP", quality=88, method=6)
            data = out.getvalue()

    except HTTPException:
        raise
    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(
            status_code=400,
            detail={
                "error": "invalid_image",
                "message": "Upload a valid JPEG, PNG, or WebP image.",
            },
        )

    return data, OUTPUT_SIZE, OUTPUT_SIZE


def _media_metadata(identity_id: str) -> Optional[dict]:
    _ensure_schema()
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT identity_id, media_kind, mime_type, width, height,
                   byte_size, created_at, updated_at
            FROM verifyd_identity_media
            WHERE identity_id = %s
            LIMIT 1
            """,
            (identity_id,),
        )
        row = cur.fetchone()

    if not row:
        return None

    row = dict(row)
    return {
        "identity_id": row["identity_id"],
        "media_kind": row["media_kind"],
        "mime_type": row["mime_type"],
        "width": row["width"],
        "height": row["height"],
        "byte_size": row["byte_size"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "image_url": f"/trust-voice/profile-media/{row['identity_id']}",
        "trust_note": "Profile media is user-supplied and does not itself indicate VeriFYD verification.",
    }


@router.get("/profile-media/health")
def profile_media_health():
    _require_enabled()
    return {
        "status": "ok",
        "feature": "trust_voice_profile_media",
        "version": FEATURE_VERSION,
        "max_upload_mb": 3,
        "output_format": "webp",
        "output_size": OUTPUT_SIZE,
    }


@router.post("/profile-media")
async def upload_profile_media(
    media_kind: Literal["personal_photo", "business_logo"] = Form(...),
    file: UploadFile = File(...),
    authorization: Optional[str] = Header(default=None),
):
    _require_enabled()
    identity = _identity_from_bearer(authorization)

    declared_type = (file.content_type or "").lower().strip()
    if declared_type not in ALLOWED_INPUT_MIME:
        raise HTTPException(
            status_code=415,
            detail={
                "error": "unsupported_image_type",
                "message": "Use JPEG, PNG, or WebP.",
            },
        )

    raw = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail={
                "error": "image_too_large",
                "message": "Profile image must be 3 MB or smaller.",
            },
        )

    normalized, width, height = _normalize_image(raw)
    _ensure_schema()
    now = _now_iso()

    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO verifyd_identity_media (
                identity_id,
                media_kind,
                mime_type,
                image_bytes,
                width,
                height,
                byte_size,
                created_at,
                updated_at
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (identity_id)
            DO UPDATE SET
                media_kind = EXCLUDED.media_kind,
                mime_type = EXCLUDED.mime_type,
                image_bytes = EXCLUDED.image_bytes,
                width = EXCLUDED.width,
                height = EXCLUDED.height,
                byte_size = EXCLUDED.byte_size,
                updated_at = EXCLUDED.updated_at
            """,
            (
                identity["id"],
                media_kind,
                "image/webp",
                normalized,
                width,
                height,
                len(normalized),
                now,
                now,
            ),
        )

    return {"ok": True, "media": _media_metadata(identity["id"])}


@router.get("/profile-media/{identity_id}")
def get_profile_media(identity_id: str):
    _require_enabled()
    _ensure_schema()

    identity = _identity_by_id(identity_id)
    if not identity:
        raise HTTPException(status_code=404, detail={"error": "identity_not_found"})

    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT mime_type, image_bytes
            FROM verifyd_identity_media
            WHERE identity_id = %s
            LIMIT 1
            """,
            (identity_id,),
        )
        row = cur.fetchone()

    if not row:
        raise HTTPException(
            status_code=404,
            detail={"error": "profile_media_not_found"},
        )

    row = dict(row)
    return Response(
        content=bytes(row["image_bytes"]),
        media_type=row["mime_type"],
        headers={
            "Cache-Control": "public, max-age=300",
            "X-Content-Type-Options": "nosniff",
        },
    )


@router.get("/profile-media/{identity_id}/metadata")
def get_profile_media_metadata(identity_id: str):
    _require_enabled()

    identity = _identity_by_id(identity_id)
    if not identity:
        raise HTTPException(status_code=404, detail={"error": "identity_not_found"})

    media = _media_metadata(identity_id)
    if not media:
        raise HTTPException(
            status_code=404,
            detail={"error": "profile_media_not_found"},
        )

    return {"media": media}


@router.delete("/profile-media")
def delete_profile_media(
    authorization: Optional[str] = Header(default=None),
):
    _require_enabled()
    identity = _identity_from_bearer(authorization)
    _ensure_schema()

    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            "DELETE FROM verifyd_identity_media WHERE identity_id = %s",
            (identity["id"],),
        )

    return {
        "ok": True,
        "identity_id": identity["id"],
        "fallback": "display_emoji",
    }
