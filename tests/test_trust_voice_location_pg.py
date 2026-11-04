"""Integration checks against *ephemeral* GitHub Actions PostgreSQL only.

Never point DATABASE_URL here at a production database. The test destroys tables.
"""
import os
import time
import types
import unittest
from contextlib import contextmanager

from test_trust_voice_location import load_location, Response

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
except ImportError:
    psycopg2 = None


@contextmanager
def _db():
    conn = psycopg2.connect(os.environ["VERIFYD_LOCATION_TEST_PG_URL"], cursor_factory=RealDictCursor)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


@unittest.skipUnless(psycopg2 and os.environ.get("VERIFYD_LOCATION_TEST_PG_URL"),
                     "Ephemeral Postgres required for integration")
class RealPostgresLocationIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with _db() as conn:
            cur = conn.cursor()
            # Test DB is private/temporary in GitHub Actions. Fail closed if it
            # is not our dedicated CI DB, never drop production tables.
            cur.execute("SELECT current_database() AS name")
            assert cur.fetchone()["name"] == "verifyd_location_ci", "Refusing non-CI database"
            for table in (
                "verifyd_circle_location_native_tokens",
                "verifyd_circle_latest_locations",
                "verifyd_circle_location_grants",
                "verifyd_circle_location_shares",
                "verifyd_trust_circle_members", "verifyd_voice_sessions",
                "verifyd_identities",
            ):
                cur.execute(f'DROP TABLE IF EXISTS "{table}" CASCADE')
            cur.execute("""
                CREATE TABLE verifyd_identities (
                    id TEXT PRIMARY KEY, handle TEXT NOT NULL,
                    display_name TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'active'
                )
            """)
            cur.execute("""
                CREATE TABLE verifyd_trust_circle_members (
                    owner_identity_id TEXT NOT NULL,
                    member_identity_id TEXT NOT NULL,
                    PRIMARY KEY(owner_identity_id, member_identity_id)
                )
            """)
            cur.execute("""
                CREATE TABLE verifyd_voice_sessions (
                    session_id TEXT PRIMARY KEY,
                    identity_id TEXT NOT NULL,
                    revoked_at TEXT
                )
            """)
            for ident in ("owner", "friend", "stranger"):
                cur.execute("INSERT INTO verifyd_identities(id,handle,display_name) VALUES (%s,%s,%s)",
                            (ident,ident,ident.capitalize()))
            cur.execute("INSERT INTO verifyd_trust_circle_members VALUES ('owner','friend')")
            cur.execute("INSERT INTO verifyd_voice_sessions VALUES ('sess_owner','owner',NULL)")
        cls.location = load_location(skip_schema=True)
        cls.location.get_db = _db
        cls.location._identity_by_id = lambda identity: {
            "id": identity, "handle": identity, "display_name": identity.title(), "status": "active"
        } if identity in ("owner","friend","stranger") else None
        cls.location._identity_by_handle = lambda handle: cls.location._identity_by_id(handle)
        cls.location.trust_circle_contains = lambda owner, member: owner=="owner" and member=="friend"
        cls.location.identities_blocked = lambda owner, viewer: False
        cls.location._verify_token = lambda raw: {"identity_id":"owner","session_id":"sess_owner"}
        cls.location._schema()
        assert cls.location._SCHEMA_READY

    def setUp(self):
        self.location._identity = lambda _: {"id":"owner", "status":"active"}
        with _db() as conn:
            cur = conn.cursor()
            for table in (
                "verifyd_circle_location_native_tokens",
                "verifyd_circle_latest_locations",
                "verifyd_circle_location_grants",
                "verifyd_circle_location_shares",
            ):
                cur.execute(f'DELETE FROM "{table}"')

    def _share(self):
        share = types.SimpleNamespace(
            enabled=True, scope="selected", precision="approximate",
            selected_handles=["friend"], duration_minutes=60,
        )
        result = self.location.set_sharing(share,Response(),authorization="Bearer owner")
        self.assertTrue(result["ok"])

    def test_schema_and_selected_recipient_visibility(self):
        self._share()
        loc = types.SimpleNamespace(latitude=40.71283, longitude=-73.99428, accuracy_m=6)
        self.location.update_position(loc,Response(),authorization="Bearer owner")
        self.location._identity = lambda _: {"id":"friend","status":"active"}
        visible = self.location.visible_locations(Response(),authorization="Bearer friend")
        self.assertEqual(len(visible["locations"]),1)
        self.assertEqual(visible["locations"][0]["latitude"],40.71)
        self.assertEqual(visible["locations"][0]["longitude"],-73.99)
        self.assertIsNone(visible["locations"][0]["accuracy_m"])
        self.location._identity = lambda _: {"id":"stranger","status":"active"}
        stranger = self.location.visible_locations(Response(),authorization="Bearer stranger")
        self.assertEqual(stranger["locations"],[])

    def test_stop_sharing_clears_native_credentials_and_last_position(self):
        self._share()
        self.location._identity = lambda _: {"id":"owner","status":"active"}
        self.location.issue_native_upload_token(Response(),authorization="Bearer owner")
        point = types.SimpleNamespace(latitude=40.1, longitude=-73.9,accuracy=8,time=time.time()*1000)
        self.location.upload_native_position(point,Response(),authorization="Bearer "+self._current_native_token())
        self.location.stop_sharing(Response(),authorization="Bearer owner")
        with _db() as conn:
            cur = conn.cursor()
            for table in ("verifyd_circle_location_native_tokens","verifyd_circle_latest_locations","verifyd_circle_location_grants"):
                cur.execute(f'SELECT count(*) AS n FROM "{table}"')
                self.assertEqual(cur.fetchone()["n"],0)

    def _current_native_token(self):
        # Native upload tokens are hashed, never retrievable from DB. Explicitly
        # issue a new one in the test and use its one-time response instead.
        return self.location.issue_native_upload_token(Response(),authorization="Bearer owner")["upload_token"]

    def test_cascade_delete_identity_removes_location_rows(self):
        self._share()
        self.location.update_position(
            types.SimpleNamespace(latitude=40.1,longitude=-73.9,accuracy_m=10),
            Response(),authorization="Bearer owner",
        )
        self.location.issue_native_upload_token(Response(),authorization="Bearer owner")
        with _db() as conn:
            cur=conn.cursor()
            cur.execute("DELETE FROM verifyd_identities WHERE id='owner'")
            for table in (
                "verifyd_circle_location_shares",
                "verifyd_circle_location_grants",
                "verifyd_circle_latest_locations",
                "verifyd_circle_location_native_tokens",
            ):
                cur.execute(f'SELECT count(*) AS n FROM "{table}"')
                self.assertEqual(cur.fetchone()["n"],0)
