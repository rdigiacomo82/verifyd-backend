"""Blocking permission and lifecycle verification on disposable PostgreSQL only."""
import ast
import os
import pathlib
import threading
import unittest
from contextlib import contextmanager

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
except ImportError:
    psycopg2 = None

ROOT = pathlib.Path(__file__).resolve().parent.parent


@contextmanager
def _db():
    conn = psycopg2.connect(os.environ["VERIFYD_BLOCKING_TEST_PG_URL"],
                            cursor_factory=RealDictCursor)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


@unittest.skipUnless(psycopg2 and os.environ.get("VERIFYD_BLOCKING_TEST_PG_URL"),
                     "Disposable blocking CI database required")
class BlockingDatabaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with _db() as conn:
            cur = conn.cursor()
            cur.execute("SELECT current_database() AS name")
            if cur.fetchone()["name"] != "verifyd_blocking_ci":
                raise RuntimeError("Refusing non-CI database")
            cur.execute("DROP TABLE IF EXISTS verifyd_voice_blocks CASCADE")
            cur.execute("DROP TABLE IF EXISTS verifyd_identities CASCADE")
            cur.execute("CREATE TABLE verifyd_identities (id TEXT PRIMARY KEY)")
            cur.execute("INSERT INTO verifyd_identities (id) VALUES ('sender'),('recipient')")
        source = (ROOT / "trust_voice_blocking.py").read_text()
        code = ast.parse(source)
        funcs = [n for n in code.body if isinstance(n, ast.FunctionDef)
                 and n.name in {"ensure_block_schema", "identities_blocked"}]
        cls.namespace = {
            "get_db": _db, "threading": threading, "_block_schema_ready": False,
            "_block_schema_lock": threading.Lock(), "blocking_enabled": lambda: True,
        }
        exec(compile(ast.Module(body=funcs, type_ignores=[]), "<isolated-blocking>", "exec"),
             cls.namespace)
        cls.namespace["ensure_block_schema"]()

    def test_block_prevents_contact_both_directions(self):
        with _db() as conn:
            conn.cursor().execute(
                "INSERT INTO verifyd_voice_blocks "
                "(blocker_identity_id,blocked_identity_id,created_at) "
                "VALUES ('recipient','sender','now')"
            )
        self.assertTrue(self.namespace["identities_blocked"]("sender","recipient"))
        self.assertTrue(self.namespace["identities_blocked"]("recipient","sender"))
        with _db() as conn:
            conn.cursor().execute(
                "DELETE FROM verifyd_voice_blocks WHERE blocker_identity_id='recipient'"
            )
        self.assertFalse(self.namespace["identities_blocked"]("sender","recipient"))

    def test_repeated_permission_checks_do_not_repeat_table_creation(self):
        self.assertTrue(self.namespace["_block_schema_ready"])
        self.namespace["get_db"] = lambda: (_ for _ in ()).throw(AssertionError("DDL repeated"))
        self.namespace["ensure_block_schema"]()
        self.namespace["get_db"] = _db

    def test_identity_deletion_cascades_blocks(self):
        with _db() as conn:
            conn.cursor().execute(
                "INSERT INTO verifyd_voice_blocks "
                "(blocker_identity_id,blocked_identity_id,created_at) "
                "VALUES ('recipient','sender','now')"
            )
        with _db() as conn:
            cur = conn.cursor()
            cur.execute("DELETE FROM verifyd_identities WHERE id='sender'")
            cur.execute("SELECT count(*) AS n FROM verifyd_voice_blocks")
            self.assertEqual(cur.fetchone()["n"], 0)

    def test_call_and_message_gates_reuse_blocking_decision(self):
        for file in ("trust_voice_signaling.py","trust_voice_messages.py","trust_voice_permissions.py"):
            self.assertIn("identities_blocked(", (ROOT / file).read_text())
