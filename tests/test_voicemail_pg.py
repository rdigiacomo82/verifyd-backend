"""Real SQL checks using disposable GitHub Actions PostgreSQL, NEVER production."""
import ast
import json
import os
import pathlib
import unittest
from contextlib import contextmanager
from datetime import datetime, timezone

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
except ImportError:
    psycopg2 = None

ROOT = pathlib.Path(__file__).resolve().parent.parent


@contextmanager
def _db():
    conn = psycopg2.connect(os.environ["VERIFYD_VOICEMAIL_TEST_PG_URL"],
                            cursor_factory=RealDictCursor)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


@unittest.skipUnless(psycopg2 and os.environ.get("VERIFYD_VOICEMAIL_TEST_PG_URL"),
                     "Disposable voicemail CI PostgreSQL required")
class PostgresVoicemailTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with _db() as conn:
            cur=conn.cursor()
            cur.execute("SELECT current_database() AS name")
            if cur.fetchone()["name"] != "verifyd_voicemail_ci":
                raise RuntimeError("Refusing any non-test database")
            for table in ("verifyd_voice_voicemail_consents", "verifyd_voice_voicemail_reviews",
                          "verifyd_voice_attachments", "verifyd_identities"):
                cur.execute(f'DROP TABLE IF EXISTS "{table}" CASCADE')
            cur.execute("CREATE TABLE verifyd_identities (id TEXT PRIMARY KEY)")
            cur.execute("CREATE TABLE verifyd_voice_attachments (attachment_id TEXT PRIMARY KEY)")
            cur.execute("INSERT INTO verifyd_identities(id) VALUES ('alice'),('bob')")
            cur.execute("INSERT INTO verifyd_voice_attachments(attachment_id) VALUES ('att1'),('att2')")

        source = (ROOT/"trust_voice_voicemail_intelligence.py").read_text()
        parsed = ast.parse(source)
        names = {"_schema", "_review_state", "_set_state", "_public", "_now",
                 "_consent_row", "_consent_active", "_consent_public",
                 "update_voicemail_account_consent"}
        definitions = [node for node in parsed.body if isinstance(node, ast.FunctionDef)
                       and node.name in names]
        for node in definitions:
            node.decorator_list = []
        compiled = "from __future__ import annotations\n" + "\n\n".join(ast.unparse(n) for n in definitions)
        class ExpectedDenial(Exception):
            def __init__(self, status_code, detail):
                self.status_code, self.detail = status_code, detail
        space = {
            "get_db": _db, "json": json, "datetime": datetime, "timezone": timezone,
            "Header": lambda default=None: default,
            "CONSENT_VERSION": "voicemail-intelligence-2026-11-v1",
            "HTTPException": ExpectedDenial,
            "_require_enabled": lambda: None, "_require_pilot": lambda owner: None,
            "_owner": lambda _: {"id": "alice"},
        }
        cls.denial = ExpectedDenial
        exec(compile(compiled, "<voicemail schema and review>", "exec"), space)
        cls.functions = space
        space["_schema"]()

    def setUp(self):
        with _db() as conn:
            cur=conn.cursor()
            cur.execute("DELETE FROM verifyd_voice_voicemail_reviews")
            cur.execute("DELETE FROM verifyd_voice_voicemail_consents")
            cur.execute("INSERT INTO verifyd_voice_voicemail_reviews "
                        "(attachment_id,recipient_identity_id,status,created_at,updated_at) "
                        "VALUES ('att1','alice','queued','now','now')")

    def test_only_recipient_report_is_returned(self):
        get = self.functions["_review_state"]
        self.assertEqual(get("att1","alice")["status"], "queued")
        self.assertIsNone(get("att1","bob"))

    def test_result_is_private_and_json_decoded(self):
        self.assertTrue(self.functions["_set_state"](
            "att1", "alice", "completed", {"transcript":"Private call", "risk":{"level":"blue"}}
        ))
        row=self.functions["_review_state"]("att1","alice")
        self.assertEqual(self.functions["_public"](row)["analysis"]["transcript"],"Private call")
        self.assertIsNone(self.functions["_review_state"]("att1","bob"))

    def test_account_consent_is_opt_in_and_persists_across_sessions(self):
        from types import SimpleNamespace
        self.assertFalse(self.functions["_consent_active"]("alice"))
        with self.assertRaises(self.denial) as denied:
            self.functions["update_voicemail_account_consent"](
                SimpleNamespace(enabled=True, acknowledge_external_processing=False),
                authorization="session-a",
            )
        self.assertEqual(denied.exception.status_code, 400)
        accepted = self.functions["update_voicemail_account_consent"](
            SimpleNamespace(enabled=True, acknowledge_external_processing=True),
            authorization="session-a",
        )
        self.assertTrue(accepted["consent_granted"])
        # The next session/device reads the same server-stored account record.
        self.assertTrue(self.functions["_consent_active"]("alice"))
        self.assertFalse(self.functions["_consent_active"]("bob"))
        self.assertEqual(accepted["transcription_provider"], "OpenAI")
        self.assertTrue(accepted["manual_analysis_only"])

    def test_consent_revocation_cancels_queued_analysis_and_blocks_late_worker(self):
        from types import SimpleNamespace
        update = self.functions["update_voicemail_account_consent"]
        update(SimpleNamespace(enabled=True, acknowledge_external_processing=True), "session")
        with _db() as conn:
            conn.cursor().execute(
                "UPDATE verifyd_voice_voicemail_reviews SET consent_mode='account' "
                "WHERE attachment_id='att1' AND recipient_identity_id='alice'"
            )
        rejected = update(SimpleNamespace(enabled=False, acknowledge_external_processing=False), "session")
        self.assertFalse(rejected["consent_granted"])
        self.assertFalse(self.functions["_consent_active"]("alice"))
        state = self.functions["_review_state"]("att1","alice")
        self.assertEqual(state["status"], "consent_revoked")
        self.assertFalse(self.functions["_set_state"](
            "att1","alice","completed",{"transcript":"Must not be saved after revocation"}
        ))

    def test_old_per_message_consent_not_silently_saved_as_account_preference(self):
        self.assertFalse(self.functions["_consent_active"]("alice"))
        with _db() as conn:
            cur = conn.cursor()
            cur.execute("SELECT count(*) AS total FROM verifyd_voice_voicemail_consents")
            self.assertEqual(cur.fetchone()["total"], 0)

    def test_deleted_inbox_entry_cannot_be_repopulated_by_worker(self):
        with _db() as conn:
            conn.cursor().execute("UPDATE verifyd_voice_voicemail_reviews "
                                  "SET deleted_at='now', result_json=NULL WHERE attachment_id='att1'")
        self.assertFalse(self.functions["_set_state"](
            "att1","alice","completed",{"transcript":"Should never be restored"}
        ))
        self.assertIsNone(self.functions["_review_state"]("att1","alice")["result_json"])

    def test_deleting_account_deletes_private_transcript(self):
        with _db() as conn:
            conn.cursor().execute("INSERT INTO verifyd_voice_voicemail_consents "
                "(identity_id,consent_version,granted,consented_at,updated_at) "
                "VALUES ('alice','voicemail-intelligence-2026-11-v1',TRUE,'now','now')")
            conn.cursor().execute("DELETE FROM verifyd_identities WHERE id='alice'")
            cur=conn.cursor()
            cur.execute("SELECT count(*) AS total FROM verifyd_voice_voicemail_reviews")
            self.assertEqual(cur.fetchone()["total"],0)
            cur.execute("SELECT count(*) AS total FROM verifyd_voice_voicemail_consents")
            self.assertEqual(cur.fetchone()["total"], 0)
        with _db() as conn:
            conn.cursor().execute("INSERT INTO verifyd_identities(id) VALUES ('alice')")

    def test_deleting_attachment_cascades_private_transcript(self):
        with _db() as conn:
            conn.cursor().execute("DELETE FROM verifyd_voice_attachments WHERE attachment_id='att1'")
            cur=conn.cursor()
            cur.execute("SELECT count(*) AS total FROM verifyd_voice_voicemail_reviews")
            self.assertEqual(cur.fetchone()["total"],0)
        with _db() as conn:
            conn.cursor().execute("INSERT INTO verifyd_voice_attachments(attachment_id) VALUES ('att1')")
