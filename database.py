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
                download_count  INTEGER DEFAULT 0
            )
        """)

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

        log.info("PostgreSQL database initialized")


# ─────────────────────────────────────────────────────────────
#  Email validation
# ─────────────────────────────────────────────────────────────

def is_valid_email(email: str) -> bool:
    pattern = r'^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email.strip()))


# ─────────────────────────────────────────────────────────────
#  User management
# ─────────────────────────────────────────────────────────────

def get_or_create_user(email: str) -> dict:
    email_lower = email.strip().lower()
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


def get_user_status(email: str) -> dict:
    user  = get_or_create_user(email)
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
) -> None:
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO certificates
                (cert_id, email, original_file, upload_time,
                 label, authenticity, ai_score, sha256)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (cert_id) DO NOTHING
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
            ),
        )
    log.info("Stored certificate %s  label=%s  authenticity=%d", cert_id, label, authenticity)


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
