# ============================================================
#  VeriFYD – database.py  (SQLite certificate store)
# ============================================================

import sqlite3
import os
import logging
from datetime import datetime, timezone
from contextlib import contextmanager

log = logging.getLogger("verifyd.db")

DB_PATH = os.getenv("DB_PATH", "certificates.db")


def _connect():
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
    finally:
        conn.close()


def init_db():
    """Create tables if they don't exist."""
    with get_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS certificates (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                cert_id         TEXT UNIQUE NOT NULL,
                email           TEXT NOT NULL,
                original_file   TEXT,
                upload_time     TEXT NOT NULL,
                status          TEXT NOT NULL DEFAULT 'PENDING',
                authenticity    INTEGER,
                ai_likelihood   INTEGER,
                primary_score   INTEGER,
                secondary_score INTEGER,
                sha256          TEXT,
                download_count  INTEGER DEFAULT 0
            )
        """)
        log.info("📦 Database initialized: %s", DB_PATH)


def insert_certificate(
    cert_id: str,
    email: str,
    original_file: str,
    status: str,
    authenticity: int,
    ai_likelihood: int,
    primary_score: int,
    secondary_score: int,
    sha256: str | None = None,
):
    with get_db() as conn:
        conn.execute(
            """INSERT INTO certificates
               (cert_id, email, original_file, upload_time, status,
                authenticity, ai_likelihood, primary_score, secondary_score, sha256)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                cert_id,
                email,
                original_file,
                datetime.now(timezone.utc).isoformat(),
                status,
                authenticity,
                ai_likelihood,
                primary_score,
                secondary_score,
                sha256,
            ),
        )
    log.info("📦 Stored certificate %s (%s)", cert_id, status)


def get_certificate(cert_id: str) -> dict | None:
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM certificates WHERE cert_id = ?", (cert_id,)
        ).fetchone()
        return dict(row) if row else None


def increment_downloads(cert_id: str):
    with get_db() as conn:
        conn.execute(
            "UPDATE certificates SET download_count = download_count + 1 WHERE cert_id = ?",
            (cert_id,),
        )


def list_certificates(limit: int = 50, status: str | None = None) -> list[dict]:
    with get_db() as conn:
        if status:
            rows = conn.execute(
                "SELECT * FROM certificates WHERE status = ? ORDER BY upload_time DESC LIMIT ?",
                (status, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM certificates ORDER BY upload_time DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]
