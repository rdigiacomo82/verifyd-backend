# ============================================================
#  VeriFYD — database.py  (PostgreSQL edition)
#
#  Drop-in replacement for the SQLite version.
#  All function signatures are identical — no changes needed
#  in main.py, worker.py, or anywhere else.
#
#  Requires env var:
#    DATABASE_URL = postgresql://user:pass@host/dbname
# ============================================================

import os
import json
import logging
import re
import random
import string
from datetime import datetime, timezone, timedelta
from contextlib import contextmanager
from typing import Optional

import psycopg2
import psycopg2.extras

log = logging.getLogger("verifyd.db")

DATABASE_URL = os.environ.get("DATABASE_URL", "")

# ── Free tier limit ──────────────────────────────────────────
FREE_USES = 10

# ── Plan limits (uses per billing period) ───────────────────
PLAN_LIMITS = {
    "free":         10,
    "creator":      100,
    "pro":          500,
    "enterprise":   999999,
}

OTP_EXPIRY_MINUTES = 10
MAX_OTP_ATTEMPTS   = 5


@contextmanager
def get_db():
    """Yield a psycopg2 connection with RealDictCursor (rows behave like dicts)."""
    conn = psycopg2.connect(DATABASE_URL, cursor_factory=psycopg2.extras.RealDictCursor)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    """Create all tables if they don't exist. Safe to call on every startup."""
    with get_db() as conn:
        cur = conn.cursor()

        # ── Certificates ─────────────────────────────────────
        cur.execute("""
            CREATE TABLE IF NOT EXISTS certificates (
                id              SERIAL PRIMARY KEY,
                cert_id         TEXT    UNIQUE NOT NULL,
                email           TEXT    NOT NULL,
                original_file   TEXT,
                upload_time     TEXT    NOT NULL,
                label           TEXT    NOT NULL DEFAULT 'UNDETERMINED',
                authenticity    INTEGER,
                ai_score        INTEGER,
                sha256          TEXT,
                original_sha256 TEXT,
                certified_document_sha256 TEXT,
                certified_file_package_sha256 TEXT,
                certified_audio_sha256 TEXT,
                certified_photo_sha256 TEXT,
                certified_file_hash TEXT,
                original_hash TEXT,
                download_count  INTEGER DEFAULT 0
            )
        """)

        # ── Certificate hash columns (safe for existing Postgres DBs) ─────────
        for _column_name in (
            "original_sha256",
            "certified_document_sha256",
            "certified_file_package_sha256",
            "certified_audio_sha256",
            "certified_photo_sha256",
            "certified_file_hash",
            "original_hash",
        ):
            try:
                cur.execute(f"ALTER TABLE certificates ADD COLUMN IF NOT EXISTS {_column_name} TEXT")
            except Exception as e:
                log.warning("Could not ensure certificates.%s column: %s", _column_name, e)

        # ── Users ────────────────────────────────────────────
        cur.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id              SERIAL PRIMARY KEY,
                email           TEXT    UNIQUE NOT NULL,
                email_lower     TEXT    UNIQUE NOT NULL,
                plan            TEXT    NOT NULL DEFAULT 'free',
                total_uses      INTEGER NOT NULL DEFAULT 0,
                period_uses     INTEGER NOT NULL DEFAULT 0,
                period_start    TEXT    NOT NULL,
                created_at      TEXT    NOT NULL,
                last_seen       TEXT    NOT NULL,
                paypal_sub_id   TEXT,
                notes           TEXT,
                email_verified  INTEGER NOT NULL DEFAULT 0
            )
        """)

        # ── OTP Verification ─────────────────────────────────
        cur.execute("""
            CREATE TABLE IF NOT EXISTS email_otp (
                id          SERIAL PRIMARY KEY,
                email_lower TEXT    NOT NULL,
                code        TEXT    NOT NULL,
                created_at  TEXT    NOT NULL,
                expires_at  TEXT    NOT NULL,
                verified    INTEGER NOT NULL DEFAULT 0,
                attempts    INTEGER NOT NULL DEFAULT 0
            )
        """)

        # ── Enterprise API Keys ───────────────────────────────
        cur.execute("""
            CREATE TABLE IF NOT EXISTS api_keys (
                id              SERIAL PRIMARY KEY,
                api_key         TEXT    UNIQUE NOT NULL,
                owner_email     TEXT    NOT NULL,
                company_name    TEXT    NOT NULL DEFAULT '',
                logo_url        TEXT    NOT NULL DEFAULT '',
                brand_color     TEXT    NOT NULL DEFAULT '#f59e0b',
                widget_domains  TEXT    NOT NULL DEFAULT '*',
                plan            TEXT    NOT NULL DEFAULT 'enterprise',
                active          INTEGER NOT NULL DEFAULT 1,
                total_uses      INTEGER NOT NULL DEFAULT 0,
                created_at      TEXT    NOT NULL,
                last_used       TEXT
            )
        """)


        # ── VeriFYD Vault Records ─────────────────────────────
        cur.execute("""
            CREATE TABLE IF NOT EXISTS vault_records (
                id              SERIAL PRIMARY KEY,
                vault_key       TEXT    UNIQUE NOT NULL,
                cert_id         TEXT    UNIQUE NOT NULL,
                email           TEXT    NOT NULL DEFAULT '',
                vault_status    TEXT    NOT NULL DEFAULT 'stored',
                created_at      TEXT    NOT NULL,
                updated_at      TEXT    NOT NULL,
                original_file   TEXT    NOT NULL DEFAULT '',
                media_type      TEXT    NOT NULL DEFAULT '',
                original_sha256 TEXT    NOT NULL DEFAULT '',
                certified_document_sha256 TEXT NOT NULL DEFAULT '',
                certified_file_package_sha256 TEXT NOT NULL DEFAULT '',
                certified_audio_sha256 TEXT NOT NULL DEFAULT '',
                certified_photo_sha256 TEXT NOT NULL DEFAULT '',
                certified_file_hash TEXT NOT NULL DEFAULT '',
                stored_evidence_json TEXT NOT NULL DEFAULT '{}',
                evidence_timeline_json TEXT NOT NULL DEFAULT '[]',
                verification_report_json TEXT NOT NULL DEFAULT '{}',
                notes           TEXT    NOT NULL DEFAULT ''
            )
        """)

        # ── Vault columns (safe for existing Postgres DBs) ───────────────
        for _column_name, _column_type in (
            ("email", "TEXT NOT NULL DEFAULT ''"),
            ("vault_status", "TEXT NOT NULL DEFAULT 'stored'"),
            ("created_at", "TEXT NOT NULL DEFAULT ''"),
            ("updated_at", "TEXT NOT NULL DEFAULT ''"),
            ("original_file", "TEXT NOT NULL DEFAULT ''"),
            ("media_type", "TEXT NOT NULL DEFAULT ''"),
            ("original_sha256", "TEXT NOT NULL DEFAULT ''"),
            ("certified_document_sha256", "TEXT NOT NULL DEFAULT ''"),
            ("certified_file_package_sha256", "TEXT NOT NULL DEFAULT ''"),
            ("certified_audio_sha256", "TEXT NOT NULL DEFAULT ''"),
            ("certified_photo_sha256", "TEXT NOT NULL DEFAULT ''"),
            ("certified_file_hash", "TEXT NOT NULL DEFAULT ''"),
            ("stored_evidence_json", "TEXT NOT NULL DEFAULT '{}'"),
            ("evidence_timeline_json", "TEXT NOT NULL DEFAULT '[]'"),
            ("verification_report_json", "TEXT NOT NULL DEFAULT '{}'"),
            ("notes", "TEXT NOT NULL DEFAULT ''"),
        ):
            try:
                cur.execute(f"ALTER TABLE vault_records ADD COLUMN IF NOT EXISTS {_column_name} {_column_type}")
            except Exception as e:
                log.warning("Could not ensure vault_records.%s column: %s", _column_name, e)


        log.info("PostgreSQL database initialized")


# ─────────────────────────────────────────────────────────────
#  Email validation
# ─────────────────────────────────────────────────────────────

def is_valid_email(email: str) -> bool:
    pattern = r'^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email.strip()))


# ─────────────────────────────────────────────────────────────
#  Email normalization / typo guard
# ─────────────────────────────────────────────────────────────

COMMON_EMAIL_DOMAIN_TYPOS = {
    "gmail.co": "gmail.com",
    "googlemail.co": "googlemail.com",
    "yahoo.co": "yahoo.com",
    "outlook.co": "outlook.com",
    "hotmail.co": "hotmail.com",
    "icloud.co": "icloud.com",
    "aol.co": "aol.com",
    "protonmail.co": "protonmail.com",
    "proton.co": "proton.me",
}


def normalize_email_value(email: str) -> str:
    """Normalize an email value without changing aliases or truncating."""
    return (email or "").strip().lower()


def get_email_typo_suggestion(email: str) -> Optional[str]:
    """Return a suggested correction for common consumer-domain typos.

    This intentionally does not block all .co domains. It only catches common
    accidental entries like gmail.co when the user most likely meant gmail.com.
    """
    normalized = normalize_email_value(email)
    if "@" not in normalized:
        return None
    local, domain = normalized.rsplit("@", 1)
    suggested_domain = COMMON_EMAIL_DOMAIN_TYPOS.get(domain)
    if not suggested_domain:
        return None
    return f"{local}@{suggested_domain}"


def get_user_by_email(email: str) -> Optional[dict]:
    """Read-only user lookup. Does not create or update a user row."""
    email_lower = normalize_email_value(email)
    if not email_lower:
        return None
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT * FROM users WHERE email_lower = %s",
            (email_lower,)
        )
        row = cur.fetchone()
        return dict(row) if row else None


# ─────────────────────────────────────────────────────────────
#  User management
# ─────────────────────────────────────────────────────────────

def get_or_create_user(email: str) -> dict:
    email_lower = email.strip().lower()
    suggestion = get_email_typo_suggestion(email_lower)
    if suggestion:
        raise ValueError(f"possible_email_typo:{email_lower}:{suggestion}")
    now = datetime.now(timezone.utc).isoformat()

    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT * FROM users WHERE email_lower = %s",
            (email_lower,)
        )
        row = cur.fetchone()

        if row:
            cur.execute(
                "UPDATE users SET last_seen = %s WHERE email_lower = %s",
                (now, email_lower)
            )
            return dict(row)

        # New user
        cur.execute(
            """
            INSERT INTO users
                (email, email_lower, plan, total_uses, period_uses,
                 period_start, created_at, last_seen, email_verified)
            VALUES (%s, %s, 'free', 0, 0, %s, %s, %s, 0)
            """,
            (email.strip(), email_lower, now, now, now)
        )
        return {
            "email":          email.strip(),
            "email_lower":    email_lower,
            "plan":           "free",
            "total_uses":     0,
            "period_uses":    0,
            "period_start":   now,
            "created_at":     now,
            "last_seen":      now,
            "paypal_sub_id":  None,
            "email_verified": 0,
        }


def get_user_status(email: str, create: bool = True) -> dict:
    """Return usage status.

    create=True preserves the original behavior for upload/payment flows.
    create=False is for read-only UI checks so partially typed valid-looking
    addresses, such as gmail.co, do not create accidental admin rows.
    """
    user = get_or_create_user(email) if create else get_user_by_email(email)
    if not user:
        return {
            "allowed":     True,
            "uses_left":   FREE_USES,
            "plan":        "free",
            "total_uses":  0,
            "period_uses": 0,
            "limit":       FREE_USES,
            "over_limit":  False,
            "user_exists": False,
        }

    plan  = user.get("plan", "free")
    limit = PLAN_LIMITS.get(plan, FREE_USES)

    if plan != "free":
        period_start_str = user.get("period_start", "")
        try:
            period_start = datetime.fromisoformat(period_start_str)
            if period_start.tzinfo is None:
                period_start = period_start.replace(tzinfo=timezone.utc)
            if (datetime.now(timezone.utc) - period_start).days >= 30:
                reset_period_uses(email)
                user["period_uses"] = 0
        except Exception:
            pass

    period_uses = user.get("period_uses", 0)
    over_limit  = period_uses >= limit

    return {
        "allowed":     not over_limit,
        "uses_left":   max(0, limit - period_uses),
        "plan":        plan,
        "total_uses":  user.get("total_uses", 0),
        "period_uses": period_uses,
        "limit":       limit,
        "over_limit":  over_limit,
        "user_exists": True,
    }


def increment_user_uses(email: str) -> dict:
    email_lower = email.strip().lower()
    now = datetime.now(timezone.utc).isoformat()

    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            UPDATE users
            SET total_uses  = total_uses  + 1,
                period_uses = period_uses + 1,
                last_seen   = %s
            WHERE email_lower = %s
            """,
            (now, email_lower)
        )

    return get_user_status(email)


def upgrade_user_plan(email: str, plan: str, paypal_sub_id: str = None) -> None:
    email_lower = email.strip().lower()
    now = datetime.now(timezone.utc).isoformat()

    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            UPDATE users
            SET plan          = %s,
                period_uses   = 0,
                period_start  = %s,
                paypal_sub_id = COALESCE(%s, paypal_sub_id),
                last_seen     = %s
            WHERE email_lower = %s
            """,
            (plan, now, paypal_sub_id, now, email_lower)
        )
    log.info("User %s upgraded to plan: %s", email_lower, plan)


def reset_period_uses(email: str) -> None:
    email_lower = email.strip().lower()
    now = datetime.now(timezone.utc).isoformat()
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            "UPDATE users SET period_uses = 0, period_start = %s WHERE email_lower = %s",
            (now, email_lower)
        )


# ─────────────────────────────────────────────────────────────
#  Certificate operations
# ─────────────────────────────────────────────────────────────

def insert_certificate(
    cert_id:       str,
    email:         str,
    original_file: str,
    label:         str,
    authenticity:  int,
    ai_score:      int,
    sha256:        Optional[str] = None,
    original_sha256: Optional[str] = None,
    certified_document_sha256: Optional[str] = None,
    certified_file_package_sha256: Optional[str] = None,
    certified_audio_sha256: Optional[str] = None,
    certified_photo_sha256: Optional[str] = None,
    certified_file_hash: Optional[str] = None,
    original_hash: Optional[str] = None,
) -> None:
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO certificates
                (cert_id, email, original_file, upload_time,
                 label, authenticity, ai_score, sha256,
                 original_sha256, certified_document_sha256,
                 certified_file_package_sha256, certified_audio_sha256, certified_photo_sha256, certified_file_hash, original_hash)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (cert_id) DO UPDATE SET
                email = EXCLUDED.email,
                original_file = EXCLUDED.original_file,
                label = EXCLUDED.label,
                authenticity = EXCLUDED.authenticity,
                ai_score = EXCLUDED.ai_score,
                sha256 = COALESCE(NULLIF(EXCLUDED.sha256, ''), certificates.sha256),
                original_sha256 = COALESCE(NULLIF(EXCLUDED.original_sha256, ''), certificates.original_sha256),
                certified_document_sha256 = COALESCE(NULLIF(EXCLUDED.certified_document_sha256, ''), certificates.certified_document_sha256),
                certified_file_package_sha256 = COALESCE(NULLIF(EXCLUDED.certified_file_package_sha256, ''), certificates.certified_file_package_sha256),
                certified_audio_sha256 = COALESCE(NULLIF(EXCLUDED.certified_audio_sha256, ''), certificates.certified_audio_sha256),
                certified_photo_sha256 = COALESCE(NULLIF(EXCLUDED.certified_photo_sha256, ''), certificates.certified_photo_sha256),
                certified_file_hash = COALESCE(NULLIF(EXCLUDED.certified_file_hash, ''), certificates.certified_file_hash),
                original_hash = COALESCE(NULLIF(EXCLUDED.original_hash, ''), certificates.original_hash)
            """,
            (
                cert_id,
                email,
                original_file,
                datetime.now(timezone.utc).isoformat(),
                label,
                authenticity,
                ai_score,
                sha256,
                original_sha256 or sha256,
                certified_document_sha256,
                certified_file_package_sha256,
                certified_audio_sha256,
                certified_photo_sha256,
                certified_file_hash or certified_document_sha256 or certified_audio_sha256 or certified_photo_sha256,
                original_hash or original_sha256 or sha256,
            ),
        )
    log.info("Stored certificate %s  label=%s  authenticity=%d", cert_id, label, authenticity)


def update_certificate_hashes(
    cert_id: str,
    original_sha256: Optional[str] = None,
    certified_document_sha256: Optional[str] = None,
    certified_file_package_sha256: Optional[str] = None,
    certified_audio_sha256: Optional[str] = None,
    certified_photo_sha256: Optional[str] = None,
    certified_file_hash: Optional[str] = None,
    original_hash: Optional[str] = None,
) -> None:
    """Persist artifact hashes after certified PDF/ZIP artifacts are created.

    This function is intentionally additive: blank/None inputs do not wipe
    existing stored hashes. It is safe to call more than once for a cert_id.
    """
    updates = []
    values = []

    def add(column: str, value: Optional[str]):
        if value is not None and str(value).strip():
            updates.append(f"{column} = %s")
            values.append(str(value).strip().lower())

    add("original_sha256", original_sha256)
    add("certified_document_sha256", certified_document_sha256)
    add("certified_file_package_sha256", certified_file_package_sha256)
    add("certified_audio_sha256", certified_audio_sha256)
    add("certified_photo_sha256", certified_photo_sha256)
    add("certified_file_hash", certified_file_hash or certified_document_sha256 or certified_audio_sha256 or certified_photo_sha256)
    add("original_hash", original_hash or original_sha256)

    if original_sha256:
        updates.append("sha256 = COALESCE(NULLIF(sha256, ''), %s)")
        values.append(str(original_sha256).strip().lower())

    if not updates:
        return

    values.append(cert_id)
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            f"UPDATE certificates SET {', '.join(updates)} WHERE cert_id = %s",
            tuple(values),
        )
    log.info(
        "Updated certificate hashes cert_id=%s original=%s document=%s package=%s audio=%s photo=%s",
        cert_id,
        bool(original_sha256),
        bool(certified_document_sha256),
        bool(certified_file_package_sha256),
        bool(certified_audio_sha256),
        bool(certified_photo_sha256),
    )


def get_certificate(cert_id: str) -> Optional[dict]:
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT * FROM certificates WHERE cert_id = %s",
            (cert_id,),
        )
        row = cur.fetchone()
        return dict(row) if row else None


def increment_downloads(cert_id: str) -> None:
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            "UPDATE certificates SET download_count = download_count + 1 WHERE cert_id = %s",
            (cert_id,),
        )



# ─────────────────────────────────────────────────────────────
#  VeriFYD Vault operations
# ─────────────────────────────────────────────────────────────

def generate_vault_key(cert_id: str) -> str:
    """Generate a stable user-facing Vault Key from a certificate ID.

    This is a reference/access key, not an encryption key.
    """
    clean = (cert_id or "").strip().replace("-", "").upper()
    return f"VFYD-VAULT-{(clean or 'UNKNOWN')[:8]}"


def _vault_media_type(cert: dict) -> str:
    if cert.get("certified_document_sha256"):
        return "document"
    if cert.get("certified_audio_sha256"):
        return "audio"
    if cert.get("certified_photo_sha256"):
        return "photo"
    return "video"


def _vault_bool(value) -> bool:
    return bool(value is not None and str(value).strip())


def _vault_evidence_from_cert(cert: dict) -> dict:
    return {
        "original_file_hash": _vault_bool(cert.get("original_sha256") or cert.get("original_hash") or cert.get("sha256")),
        "certified_document": _vault_bool(cert.get("certified_document_sha256")),
        "certified_evidence_package": _vault_bool(cert.get("certified_file_package_sha256")),
        "certified_audio": _vault_bool(cert.get("certified_audio_sha256")),
        "certified_photo": _vault_bool(cert.get("certified_photo_sha256")),
        "evidence_timeline": True,
        "verification_report": True,
    }


def _vault_timeline_from_cert(cert: dict) -> list:
    original_hash = cert.get("original_sha256") or cert.get("original_hash") or cert.get("sha256") or ""
    doc_hash = cert.get("certified_document_sha256") or ""
    package_hash = cert.get("certified_file_package_sha256") or ""
    audio_hash = cert.get("certified_audio_sha256") or ""
    photo_hash = cert.get("certified_photo_sha256") or ""
    label = cert.get("label") or ""
    authenticity = cert.get("authenticity")
    ai_score = cert.get("ai_score")

    timeline = []
    timeline.append({
        "event": "File Uploaded / Certificate Created",
        "timestamp": cert.get("upload_time") or "",
        "detail": cert.get("original_file") or "",
        "status": "complete",
    })
    if original_hash:
        timeline.append({
            "event": "Original File Hash Generated",
            "timestamp": cert.get("upload_time") or "",
            "detail": f"SHA-256: {original_hash}",
            "status": "complete",
        })
    if label:
        timeline.append({
            "event": "Authenticity Analysis Completed",
            "timestamp": cert.get("upload_time") or "",
            "detail": f"{label} — {authenticity}% authenticity / {ai_score}% AI risk",
            "status": "complete",
        })
    if doc_hash:
        timeline.append({
            "event": "Certified Document Created",
            "timestamp": cert.get("upload_time") or "",
            "detail": f"SHA-256: {doc_hash}",
            "status": "complete",
        })
    if package_hash:
        timeline.append({
            "event": "Certified Evidence Package Created",
            "timestamp": cert.get("upload_time") or "",
            "detail": f"SHA-256: {package_hash}",
            "status": "complete",
        })
    if audio_hash:
        timeline.append({
            "event": "Certified Audio Created",
            "timestamp": cert.get("upload_time") or "",
            "detail": f"SHA-256: {audio_hash}",
            "status": "complete",
        })
    if photo_hash:
        timeline.append({
            "event": "Certified Photo Created",
            "timestamp": cert.get("upload_time") or "",
            "detail": f"SHA-256: {photo_hash}",
            "status": "complete",
        })
    if doc_hash or package_hash or audio_hash or photo_hash:
        timeline.append({
            "event": "Certified Files Stored",
            "timestamp": cert.get("upload_time") or "",
            "detail": "Certified files available for secure preservation",
            "status": "complete",
        })
    timeline.append({
        "event": "Certificate Record Verified",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "detail": "Verified in VeriFYD Vault",
        "status": "complete",
    })
    return timeline


def _vault_report_from_cert(cert: dict, vault_key: str) -> dict:
    return {
        "title": "VeriFYD Vault Record",
        "status": "STORED",
        "vault_key": vault_key,
        "certificate_id": cert.get("cert_id") or "",
        "vault_status": "Stored in VeriFYD Vault",
        "message": "This certificate record has been saved as a VeriFYD Vault reference for future lookup and preservation workflow.",
        "original_sha256": cert.get("original_sha256") or cert.get("original_hash") or cert.get("sha256") or "",
        "certified_document_sha256": cert.get("certified_document_sha256") or "",
        "certified_file_package_sha256": cert.get("certified_file_package_sha256") or "",
        "certified_audio_sha256": cert.get("certified_audio_sha256") or "",
        "certified_photo_sha256": cert.get("certified_photo_sha256") or "",
    }


def save_certificate_to_vault(cert_id: str, email: str = "", notes: str = "") -> dict:
    """Create or update a VeriFYD Vault record for an existing certificate."""
    cert_id = (cert_id or "").strip()
    if not cert_id:
        raise ValueError("missing_certificate_id")

    cert = get_certificate(cert_id)
    if not cert:
        raise ValueError("certificate_not_found")

    now = datetime.now(timezone.utc).isoformat()
    vault_key = generate_vault_key(cert_id)
    cert_email = (email or cert.get("email") or "").strip().lower()
    original_sha256 = cert.get("original_sha256") or cert.get("original_hash") or cert.get("sha256") or ""
    certified_file_hash = (
        cert.get("certified_file_hash")
        or cert.get("certified_file_package_sha256")
        or cert.get("certified_document_sha256")
        or cert.get("certified_audio_sha256")
        or cert.get("certified_photo_sha256")
        or ""
    )
    media_type = _vault_media_type(cert)
    stored_evidence = _vault_evidence_from_cert(cert)
    timeline = _vault_timeline_from_cert(cert)
    report = _vault_report_from_cert(cert, vault_key)

    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO vault_records (
                vault_key, cert_id, email, vault_status, created_at, updated_at,
                original_file, media_type, original_sha256,
                certified_document_sha256, certified_file_package_sha256,
                certified_audio_sha256, certified_photo_sha256, certified_file_hash,
                stored_evidence_json, evidence_timeline_json, verification_report_json, notes
            )
            VALUES (%s, %s, %s, 'stored', %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (cert_id) DO UPDATE SET
                email = EXCLUDED.email,
                vault_status = 'stored',
                updated_at = EXCLUDED.updated_at,
                original_file = EXCLUDED.original_file,
                media_type = EXCLUDED.media_type,
                original_sha256 = EXCLUDED.original_sha256,
                certified_document_sha256 = EXCLUDED.certified_document_sha256,
                certified_file_package_sha256 = EXCLUDED.certified_file_package_sha256,
                certified_audio_sha256 = EXCLUDED.certified_audio_sha256,
                certified_photo_sha256 = EXCLUDED.certified_photo_sha256,
                certified_file_hash = EXCLUDED.certified_file_hash,
                stored_evidence_json = EXCLUDED.stored_evidence_json,
                evidence_timeline_json = EXCLUDED.evidence_timeline_json,
                verification_report_json = EXCLUDED.verification_report_json,
                notes = COALESCE(NULLIF(EXCLUDED.notes, ''), vault_records.notes)
            RETURNING *
            """,
            (
                vault_key, cert_id, cert_email, now, now,
                cert.get("original_file") or "", media_type, original_sha256,
                cert.get("certified_document_sha256") or "",
                cert.get("certified_file_package_sha256") or "",
                cert.get("certified_audio_sha256") or "",
                cert.get("certified_photo_sha256") or "",
                certified_file_hash,
                json.dumps(stored_evidence, sort_keys=True),
                json.dumps(timeline, sort_keys=True),
                json.dumps(report, sort_keys=True),
                notes or "",
            ),
        )
        row = cur.fetchone()

    log.info("Vault record stored cert_id=%s vault_key=%s", cert_id, vault_key)
    return _hydrate_vault_record(dict(row))


def _hydrate_vault_record(row: dict) -> dict:
    if not row:
        return row
    for key, default in (
        ("stored_evidence_json", {}),
        ("evidence_timeline_json", []),
        ("verification_report_json", {}),
    ):
        try:
            row[key.replace("_json", "")] = json.loads(row.get(key) or json.dumps(default))
        except Exception:
            row[key.replace("_json", "")] = default
    return row


def get_vault_record(vault_key: str) -> Optional[dict]:
    vault_key = (vault_key or "").strip().upper()
    if not vault_key:
        return None
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("SELECT * FROM vault_records WHERE UPPER(vault_key) = %s", (vault_key,))
        row = cur.fetchone()
        return _hydrate_vault_record(dict(row)) if row else None


def get_vault_record_by_cert_id(cert_id: str) -> Optional[dict]:
    cert_id = (cert_id or "").strip()
    if not cert_id:
        return None
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("SELECT * FROM vault_records WHERE cert_id = %s", (cert_id,))
        row = cur.fetchone()
        return _hydrate_vault_record(dict(row)) if row else None



def list_certificates(limit: int = 50, label: Optional[str] = None) -> list:
    with get_db() as conn:
        cur = conn.cursor()
        if label:
            cur.execute(
                "SELECT * FROM certificates WHERE label = %s ORDER BY upload_time DESC LIMIT %s",
                (label, limit),
            )
        else:
            cur.execute(
                "SELECT * FROM certificates ORDER BY upload_time DESC LIMIT %s",
                (limit,),
            )
        return [dict(r) for r in cur.fetchall()]


# ─────────────────────────────────────────────────────────────
#  OTP Verification
# ─────────────────────────────────────────────────────────────

def is_email_verified(email: str) -> bool:
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT email_verified FROM users WHERE email_lower = %s",
            (email.lower().strip(),)
        )
        row = cur.fetchone()
        return bool(row and row["email_verified"])


def create_otp(email: str) -> str:
    email_lower = email.lower().strip()
    code        = "".join(random.choices(string.digits, k=6))
    now         = datetime.now(timezone.utc)
    expires     = now + timedelta(minutes=OTP_EXPIRY_MINUTES)

    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("DELETE FROM email_otp WHERE email_lower = %s", (email_lower,))
        cur.execute(
            """INSERT INTO email_otp (email_lower, code, created_at, expires_at, verified, attempts)
               VALUES (%s, %s, %s, %s, 0, 0)""",
            (email_lower, code, now.isoformat(), expires.isoformat())
        )

    log.info("OTP created for %s (expires %s)", email_lower, expires.isoformat())
    return code


def verify_otp(email: str, code: str) -> tuple:
    email_lower = email.lower().strip()
    now         = datetime.now(timezone.utc)

    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT * FROM email_otp WHERE email_lower = %s ORDER BY id DESC LIMIT 1",
            (email_lower,)
        )
        row = cur.fetchone()

        if not row:
            return False, "No verification code found. Please request a new code."

        expires_at = datetime.fromisoformat(row["expires_at"])
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=timezone.utc)
        if now > expires_at:
            cur.execute("DELETE FROM email_otp WHERE email_lower = %s", (email_lower,))
            return False, "Verification code expired. Please request a new code."

        if row["attempts"] >= MAX_OTP_ATTEMPTS:
            cur.execute("DELETE FROM email_otp WHERE email_lower = %s", (email_lower,))
            return False, "Too many attempts. Please request a new code."

        cur.execute(
            "UPDATE email_otp SET attempts = attempts + 1 WHERE email_lower = %s",
            (email_lower,)
        )

        if row["code"] != code.strip():
            remaining = MAX_OTP_ATTEMPTS - row["attempts"] - 1
            return False, f"Incorrect code. {remaining} attempts remaining."

        # Success
        cur.execute("DELETE FROM email_otp WHERE email_lower = %s", (email_lower,))
        cur.execute(
            "UPDATE users SET email_verified = 1 WHERE email_lower = %s",
            (email_lower,)
        )
        # Create verified user if they don't exist yet
        cur.execute(
            """INSERT INTO users
               (email, email_lower, plan, total_uses, period_uses,
                period_start, created_at, last_seen, email_verified)
               VALUES (%s, %s, 'free', 0, 0, %s, %s, %s, 1)
               ON CONFLICT (email_lower) DO NOTHING""",
            (email, email_lower, now.isoformat(), now.isoformat(), now.isoformat())
        )

    log.info("Email verified successfully: %s", email_lower)

    status = get_user_status(email)
    if status["over_limit"]:
        log.warning("Verified user %s is already at limit (%d/%d uses)",
                    email_lower, status["period_uses"], status["limit"])
        return True, "limit_reached"

    return True, "Email verified successfully."

# ─────────────────────────────────────────────────────────────
#  Enterprise API Key management
# ─────────────────────────────────────────────────────────────

def _generate_api_key() -> str:
    """Generate a secure API key in format: vfyd_live_<32 hex chars>"""
    import secrets
    return "vfyd_live_" + secrets.token_hex(16)


def create_api_key(
    owner_email:   str,
    company_name:  str = "",
    logo_url:      str = "",
    brand_color:   str = "#f59e0b",
    widget_domains: str = "*",
) -> dict:
    """Create a new Enterprise API key. Returns the full key record."""
    key = _generate_api_key()
    now = datetime.now(timezone.utc).isoformat()

    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO api_keys
                (api_key, owner_email, company_name, logo_url,
                 brand_color, widget_domains, plan, active, total_uses, created_at)
            VALUES (%s, %s, %s, %s, %s, %s, 'enterprise', 1, 0, %s)
            RETURNING *
            """,
            (key, owner_email.lower().strip(), company_name,
             logo_url, brand_color, widget_domains, now)
        )
        row = cur.fetchone()
    log.info("API key created for %s: %s...", owner_email, key[:20])
    return dict(row)


def get_api_key(api_key: str) -> Optional[dict]:
    """Look up an API key. Returns None if not found or inactive."""
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT * FROM api_keys WHERE api_key = %s AND active = 1",
            (api_key,)
        )
        row = cur.fetchone()
        return dict(row) if row else None


def increment_api_key_uses(api_key: str) -> None:
    """Increment usage counter and update last_used timestamp."""
    now = datetime.now(timezone.utc).isoformat()
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            UPDATE api_keys
            SET total_uses = total_uses + 1,
                last_used  = %s
            WHERE api_key = %s
            """,
            (now, api_key)
        )


def revoke_api_key(api_key: str) -> None:
    """Deactivate an API key without deleting it (preserves usage history)."""
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            "UPDATE api_keys SET active = 0 WHERE api_key = %s",
            (api_key,)
        )
    log.info("API key revoked: %s...", api_key[:20])


def list_api_keys(owner_email: Optional[str] = None) -> list:
    """List all API keys, optionally filtered by owner."""
    with get_db() as conn:
        cur = conn.cursor()
        if owner_email:
            cur.execute(
                "SELECT * FROM api_keys WHERE owner_email = %s ORDER BY created_at DESC",
                (owner_email.lower().strip(),)
            )
        else:
            cur.execute("SELECT * FROM api_keys ORDER BY created_at DESC")
        return [dict(r) for r in cur.fetchall()]


def update_api_key_branding(
    api_key:       str,
    company_name:  Optional[str] = None,
    logo_url:      Optional[str] = None,
    brand_color:   Optional[str] = None,
    widget_domains: Optional[str] = None,
) -> Optional[dict]:
    """Update branding config for an existing API key."""
    with get_db() as conn:
        cur = conn.cursor()
        if company_name  is not None:
            cur.execute("UPDATE api_keys SET company_name  = %s WHERE api_key = %s", (company_name,  api_key))
        if logo_url      is not None:
            cur.execute("UPDATE api_keys SET logo_url      = %s WHERE api_key = %s", (logo_url,      api_key))
        if brand_color   is not None:
            cur.execute("UPDATE api_keys SET brand_color   = %s WHERE api_key = %s", (brand_color,   api_key))
        if widget_domains is not None:
            cur.execute("UPDATE api_keys SET widget_domains = %s WHERE api_key = %s", (widget_domains, api_key))
    return get_api_key(api_key)
