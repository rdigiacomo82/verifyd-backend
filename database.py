# ============================================================
#  VeriFYD — database.py  (SQLite certificate + user store)
# ============================================================

import sqlite3
import os
import logging
import re
from datetime import datetime, timezone
from contextlib import contextmanager
from typing import Optional

log = logging.getLogger("verifyd.db")

# Use persistent disk on Render (/data), fallback to local for development
_DEFAULT_DB = "/data/verifyd.db" if os.path.isdir("/data") else "verifyd.db"
DB_PATH = os.getenv("DB_PATH", _DEFAULT_DB)

# ── Free tier limit ──────────────────────────────────────────
FREE_USES = 10

# ── Plan limits (uses per billing period) ───────────────────
PLAN_LIMITS = {
    "free":         10,      # Starter — $0 forever
    "creator":      100,     # Creator — $19/month
    "pro":          500,     # Pro AI  — $39/month
    "enterprise":   999999,  # Enterprise — custom/unlimited
}


@contextmanager
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    """Create all tables. Safe to call on every startup."""
    with get_db() as conn:
        # ── Certificates ─────────────────────────────────────
        conn.execute("""
            CREATE TABLE IF NOT EXISTS certificates (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
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
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                email           TEXT    UNIQUE NOT NULL,
                email_lower     TEXT    UNIQUE NOT NULL,
                plan            TEXT    NOT NULL DEFAULT 'free',
                total_uses      INTEGER NOT NULL DEFAULT 0,
                period_uses     INTEGER NOT NULL DEFAULT 0,
                period_start    TEXT    NOT NULL,
                created_at      TEXT    NOT NULL,
                last_seen       TEXT    NOT NULL,
                paypal_sub_id   TEXT,
                notes           TEXT
            )
        """)

        log.info("Database initialized: %s", DB_PATH)


# ─────────────────────────────────────────────────────────────
#  Email validation
# ─────────────────────────────────────────────────────────────

def is_valid_email(email: str) -> bool:
    """Basic RFC-5322 email format check."""
    pattern = r'^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email.strip()))


# ─────────────────────────────────────────────────────────────
#  User management
# ─────────────────────────────────────────────────────────────

def get_or_create_user(email: str) -> dict:
    """
    Return existing user record or create a new free-tier user.
    Always normalizes email to lowercase for lookup.
    """
    email_lower = email.strip().lower()
    now = datetime.now(timezone.utc).isoformat()

    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE email_lower = ?",
            (email_lower,)
        ).fetchone()

        if row:
            # Update last_seen
            conn.execute(
                "UPDATE users SET last_seen = ? WHERE email_lower = ?",
                (now, email_lower)
            )
            return dict(row)

        # New user
        conn.execute(
            """
            INSERT INTO users
                (email, email_lower, plan, total_uses, period_uses,
                 period_start, created_at, last_seen)
            VALUES (?, ?, 'free', 0, 0, ?, ?, ?)
            """,
            (email.strip(), email_lower, now, now, now)
        )
        return {
            "email":        email.strip(),
            "email_lower":  email_lower,
            "plan":         "free",
            "total_uses":   0,
            "period_uses":  0,
            "period_start": now,
            "created_at":   now,
            "last_seen":    now,
            "paypal_sub_id": None,
        }


def get_user_status(email: str) -> dict:
    """
    Returns user info including whether they are within their usage limit.

    Returns dict with:
        allowed      : bool   — True if they can perform an analysis
        uses_left    : int    — analyses remaining this period
        plan         : str    — current plan name
        total_uses   : int    — lifetime total
        period_uses  : int    — uses this billing period
        limit        : int    — max uses for their plan
        over_limit   : bool   — True if they've exceeded their limit
    """
    user = get_or_create_user(email)
    plan  = user.get("plan", "free")
    limit = PLAN_LIMITS.get(plan, FREE_USES)
    period_uses = user.get("period_uses", 0)
    over_limit  = period_uses >= limit

    return {
        "allowed":      not over_limit,
        "uses_left":    max(0, limit - period_uses),
        "plan":         plan,
        "total_uses":   user.get("total_uses", 0),
        "period_uses":  period_uses,
        "limit":        limit,
        "over_limit":   over_limit,
    }


def increment_user_uses(email: str) -> dict:
    """
    Increment both total_uses and period_uses for the user.
    Returns updated status dict.
    """
    email_lower = email.strip().lower()
    now = datetime.now(timezone.utc).isoformat()

    with get_db() as conn:
        conn.execute(
            """
            UPDATE users
            SET total_uses  = total_uses  + 1,
                period_uses = period_uses + 1,
                last_seen   = ?
            WHERE email_lower = ?
            """,
            (now, email_lower)
        )

    return get_user_status(email)


def upgrade_user_plan(email: str, plan: str, paypal_sub_id: str = None) -> None:
    """
    Upgrade a user to a paid plan and reset their period uses.
    Called after successful PayPal subscription confirmation.
    """
    email_lower = email.strip().lower()
    now = datetime.now(timezone.utc).isoformat()

    with get_db() as conn:
        conn.execute(
            """
            UPDATE users
            SET plan         = ?,
                period_uses  = 0,
                period_start = ?,
                paypal_sub_id = COALESCE(?, paypal_sub_id),
                last_seen    = ?
            WHERE email_lower = ?
            """,
            (plan, now, paypal_sub_id, now, email_lower)
        )
    log.info("User %s upgraded to plan: %s", email_lower, plan)


def reset_period_uses(email: str) -> None:
    """Reset period_uses to 0 for a new billing cycle."""
    email_lower = email.strip().lower()
    now = datetime.now(timezone.utc).isoformat()
    with get_db() as conn:
        conn.execute(
            "UPDATE users SET period_uses = 0, period_start = ? WHERE email_lower = ?",
            (now, email_lower)
        )


# ─────────────────────────────────────────────────────────────
#  Certificate operations (unchanged)
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
        conn.execute(
            """
            INSERT INTO certificates
                (cert_id, email, original_file, upload_time,
                 label, authenticity, ai_score, sha256)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
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
        row = conn.execute(
            "SELECT * FROM certificates WHERE cert_id = ?",
            (cert_id,),
        ).fetchone()
        return dict(row) if row else None


def increment_downloads(cert_id: str) -> None:
    with get_db() as conn:
        conn.execute(
            "UPDATE certificates SET download_count = download_count + 1 WHERE cert_id = ?",
            (cert_id,),
        )


def list_certificates(limit: int = 50, label: Optional[str] = None) -> list:
    with get_db() as conn:
        if label:
            rows = conn.execute(
                "SELECT * FROM certificates WHERE label = ? ORDER BY upload_time DESC LIMIT ?",
                (label, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM certificates ORDER BY upload_time DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

