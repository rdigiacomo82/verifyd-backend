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

        # ── Detection Results (per-engine scores, for feedback/tuning) ──
        cur.execute("""
            CREATE TABLE IF NOT EXISTS detection_results (
                id                  SERIAL PRIMARY KEY,
                cert_id             TEXT    UNIQUE NOT NULL,
                job_id              TEXT,
                created_at          TEXT    NOT NULL,
                -- Final verdict
                label               TEXT    NOT NULL,
                authenticity        INTEGER,
                ai_score            INTEGER,
                blend_mode          TEXT,
                content_type        TEXT,
                -- Engine 1: Signal
                signal_ai_score     INTEGER,
                -- Engine 2: GPT
                gpt_ai_score        INTEGER,
                gpt_available       INTEGER DEFAULT 1,
                gpt_reasoning       TEXT,
                gpt_flags           TEXT,     -- JSON array
                gpt_scores          TEXT,     -- JSON object (12 dimensions)
                generator_guess     TEXT,
                -- Engine 3: Metadata
                metadata_ai_score   INTEGER,
                metadata_confidence TEXT,
                metadata_evidence   TEXT,     -- JSON array
                metadata_adjustment REAL,
                -- Engine 4: Audio
                audio_ai_score      INTEGER,
                audio_confidence    TEXT,
                audio_evidence      TEXT,     -- JSON array
                audio_adjustment    REAL,
                -- Feedback (filled in post-hoc by admin or user)
                correct_label       TEXT,     -- NULL = unreviewed, else "REAL"|"AI"|"UNDETERMINED"
                is_false_positive   INTEGER,  -- 1 = we said AI but it was real
                is_false_negative   INTEGER,  -- 1 = we said REAL but it was AI
                reviewed_at         TEXT,
                reviewer_note       TEXT
            )
        """)

        # ── Corrections log (admin manual overrides) ─────────
        cur.execute("""
            CREATE TABLE IF NOT EXISTS corrections (
                id              SERIAL PRIMARY KEY,
                cert_id         TEXT    NOT NULL,
                submitted_at    TEXT    NOT NULL,
                original_label  TEXT    NOT NULL,
                correct_label   TEXT    NOT NULL,
                error_type      TEXT,   -- "false_positive" | "false_negative" | "uncertain"
                reviewer        TEXT,   -- "admin" | "user" | "auto"
                note            TEXT,
                signal_ai_score INTEGER,
                gpt_ai_score    INTEGER,
                content_type    TEXT
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
#  Detection results — per-engine score storage
# ─────────────────────────────────────────────────────────────

def insert_detection_result(cert_id: str, job_id: str, detail: dict) -> None:
    """Store full per-engine detection scores for a completed job."""
    import json as _json
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).isoformat()

    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO detection_results (
                cert_id, job_id, created_at,
                label, authenticity, ai_score, blend_mode, content_type,
                signal_ai_score,
                gpt_ai_score, gpt_available, gpt_reasoning, gpt_flags,
                gpt_scores, generator_guess,
                metadata_ai_score, metadata_confidence, metadata_evidence, metadata_adjustment,
                audio_ai_score, audio_confidence, audio_evidence, audio_adjustment
            ) VALUES (
                %s, %s, %s,
                %s, %s, %s, %s, %s,
                %s,
                %s, %s, %s, %s,
                %s, %s,
                %s, %s, %s, %s,
                %s, %s, %s, %s
            )
            ON CONFLICT (cert_id) DO NOTHING
        """, (
            cert_id, job_id, now,
            detail.get("label"),
            detail.get("authenticity"),
            detail.get("ai_score"),
            detail.get("blend_mode"),
            detail.get("content_type"),
            detail.get("signal_ai_score"),
            detail.get("gpt_ai_score"),
            int(detail.get("gpt_available", True)),
            detail.get("gpt_reasoning", "")[:500],
            _json.dumps(detail.get("gpt_flags", [])),
            _json.dumps(detail.get("gpt_scores", {})),
            detail.get("generator_guess", "Unknown"),
            detail.get("metadata_ai_score"),
            detail.get("metadata_confidence"),
            _json.dumps(detail.get("metadata_evidence", [])),
            detail.get("metadata_adjustment"),
            detail.get("audio_ai_score"),
            detail.get("audio_confidence"),
            _json.dumps(detail.get("audio_evidence", [])),
            detail.get("audio_adjustment"),
        ))
    log.info("Stored detection result for cert_id=%s", cert_id)


def insert_correction(cert_id: str, original_label: str, correct_label: str,
                       error_type: str, reviewer: str = "admin",
                       note: str = "", detail: dict = None) -> None:
    """Log a manual correction for a misclassified video."""
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).isoformat()
    detail = detail or {}

    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO corrections (
                cert_id, submitted_at, original_label, correct_label,
                error_type, reviewer, note,
                signal_ai_score, gpt_ai_score, content_type
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            cert_id, now, original_label, correct_label,
            error_type, reviewer, note,
            detail.get("signal_ai_score"),
            detail.get("gpt_ai_score"),
            detail.get("content_type"),
        ))

        # Also update detection_results with the correction
        is_fp = 1 if error_type == "false_positive" else 0
        is_fn = 1 if error_type == "false_negative" else 0
        cur.execute("""
            UPDATE detection_results
            SET correct_label = %s,
                is_false_positive = %s,
                is_false_negative = %s,
                reviewed_at = %s,
                reviewer_note = %s
            WHERE cert_id = %s
        """, (correct_label, is_fp, is_fn, now, note, cert_id))

    log.info("Correction logged: cert=%s  %s→%s  type=%s",
             cert_id, original_label, correct_label, error_type)


def get_false_positive_rate(days: int = 30) -> dict:
    """Return false positive/negative rates over the last N days."""
    from datetime import datetime, timezone, timedelta
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()

    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT
                COUNT(*)                                    AS total_reviewed,
                SUM(CASE WHEN is_false_positive = 1 THEN 1 ELSE 0 END) AS false_positives,
                SUM(CASE WHEN is_false_negative = 1 THEN 1 ELSE 0 END) AS false_negatives
            FROM detection_results
            WHERE reviewed_at IS NOT NULL AND reviewed_at >= %s
        """, (cutoff,))
        row = dict(cur.fetchone() or {})

    total = row.get("total_reviewed") or 0
    fp    = row.get("false_positives") or 0
    fn    = row.get("false_negatives") or 0
    return {
        "total_reviewed":      total,
        "false_positives":     fp,
        "false_negatives":     fn,
        "fp_rate":             round(fp / total, 3) if total else None,
        "fn_rate":             round(fn / total, 3) if total else None,
        "period_days":         days,
    }


def list_detection_results(limit: int = 100, label: str = None,
                            unreviewed_only: bool = False) -> list:
    """List recent detection results, optionally filtered."""
    with get_db() as conn:
        cur = conn.cursor()
        conditions = []
        params = []
        if label:
            conditions.append("label = %s")
            params.append(label)
        if unreviewed_only:
            conditions.append("reviewed_at IS NULL")
        where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
        cur.execute(f"""
            SELECT cert_id, created_at, label, authenticity, ai_score,
                   content_type, signal_ai_score, gpt_ai_score,
                   generator_guess, metadata_ai_score, audio_ai_score,
                   correct_label, is_false_positive, is_false_negative,
                   reviewed_at
            FROM detection_results
            {where}
            ORDER BY created_at DESC
            LIMIT %s
        """, params + [limit])
        rows = cur.fetchall()
        return [dict(r) for r in rows] if rows else []