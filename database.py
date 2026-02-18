# ============================================================
#  VeriFYD — database.py  (SQLite certificate store)
#
#  Column mapping to detection pipeline:
#    authenticity   ← authenticity  (100 - ai_score, 0–100)
#    ai_score       ← detail["ai_score"]  (raw engine output, 0–100)
#    label          ← label  ("REAL" | "UNDETERMINED" | "AI")
#    sha256         ← hex digest of raw uploaded file
#    download_count ← incremented by /download/ route
#
#  Removed: primary_score / secondary_score — no secondary
#  detector exists yet. Add back when external_detector.py
#  is wired to a real model.
# ============================================================

import sqlite3
import os
import logging
from datetime import datetime, timezone
from contextlib import contextmanager
from typing import Optional

log = logging.getLogger("verifyd.db")

DB_PATH = os.getenv("DB_PATH", "certificates.db")


# ─────────────────────────────────────────────
#  Connection helpers
# ─────────────────────────────────────────────

def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


@contextmanager
def get_db():
    conn = _connect()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ─────────────────────────────────────────────
#  Schema
# ─────────────────────────────────────────────

def init_db() -> None:
    """Create tables if they don't exist. Safe to call on every startup."""
    with get_db() as conn:
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
        log.info("Database initialized: %s", DB_PATH)


# ─────────────────────────────────────────────
#  Write operations
# ─────────────────────────────────────────────

def insert_certificate(
    cert_id:       str,
    email:         str,
    original_file: str,
    label:         str,
    authenticity:  int,
    ai_score:      int,
    sha256:        Optional[str] = None,
) -> None:
    """
    Store a detection result.

    Parameters match the return values of run_detection() plus
    the cert_id and email collected by the /upload/ route.
    """
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


# ─────────────────────────────────────────────
#  Read operations
# ─────────────────────────────────────────────

def get_certificate(cert_id: str) -> Optional[dict]:
    """Return a single certificate record or None if not found."""
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM certificates WHERE cert_id = ?",
            (cert_id,),
        ).fetchone()
        return dict(row) if row else None


def increment_downloads(cert_id: str) -> None:
    """Bump download_count by 1 each time a certified video is downloaded."""
    with get_db() as conn:
        conn.execute(
            "UPDATE certificates SET download_count = download_count + 1 WHERE cert_id = ?",
            (cert_id,),
        )


def list_certificates(limit: int = 50, label: Optional[str] = None) -> list:
    """
    Return the most recent certificates, optionally filtered by label.

    Parameters
    ----------
    limit : int
        Maximum number of rows to return (default 50).
    label : str, optional
        Filter to "REAL", "UNDETERMINED", or "AI".
    """
    with get_db() as conn:
        if label:
            rows = conn.execute(
                """
                SELECT * FROM certificates
                WHERE label = ?
                ORDER BY upload_time DESC
                LIMIT ?
                """,
                (label, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT * FROM certificates
                ORDER BY upload_time DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

