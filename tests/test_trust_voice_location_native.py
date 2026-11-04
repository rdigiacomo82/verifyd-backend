"""Isolated native-location security tests (no production DB; no network)."""
import os
import time
import types
import unittest
from contextlib import contextmanager
from unittest.mock import patch

from test_trust_voice_location import (
    load_location, Cursor, Connection, HTTPException, Response,
)


class NativeCursor(Cursor):
    def __init__(self, share=None, token_owner=None, session_active=True):
        super().__init__(share=share)
        self.token_owner = token_owner
        self.session_active = session_active

    def fetchone(self):
        if "SELECT owner_identity_id FROM verifyd_circle_location_native_tokens" in self.last_sql:
            return {"owner_identity_id": self.token_owner} if self.token_owner else None
        if "SELECT 1 FROM verifyd_circle_location_native_tokens t" in self.last_sql:
            return {"allowed": True} if self.session_active else None
        if "SELECT enabled,expires_at FROM verifyd_circle_location_shares" in self.last_sql:
            return self.share
        return super().fetchone()


class NativeLocationSecurityTests(unittest.TestCase):
    def setUp(self):
        self.location = load_location()

    def install_cursor(self, cursor):
        @contextmanager
        def db():
            yield Connection(cursor)
        self.location.get_db = db

    def test_native_upload_is_disabled_until_feature_flag_enabled(self):
        point = types.SimpleNamespace(latitude=40.1, longitude=-73.1, accuracy=10.0, time=time.time()*1000)
        with patch.dict(os.environ, {"VERIFYD_TRUST_VOICE_LOCATION_ENABLED": "0"}):
            with self.assertRaises(HTTPException) as e:
                self.location.upload_native_position(point, Response(), authorization="Bearer loc_bad")
            self.assertEqual(e.exception.status_code, 404)

    def test_rejects_primary_session_token_for_native_upload(self):
        point = types.SimpleNamespace(latitude=40.1, longitude=-73.1, accuracy=None, time=time.time()*1000)
        with patch.dict(os.environ, {"VERIFYD_TRUST_VOICE_LOCATION_ENABLED": "1"}):
            with self.assertRaises(HTTPException) as e:
                self.location.upload_native_position(point, Response(), authorization="Bearer valid")
            self.assertEqual(e.exception.status_code, 401)

    def test_rejects_expired_native_point_before_touching_db(self):
        point = types.SimpleNamespace(latitude=40.1, longitude=-73.1, accuracy=12, time=(time.time()-700)*1000)
        with patch.dict(os.environ, {"VERIFYD_TRUST_VOICE_LOCATION_ENABLED": "1"}):
            with self.assertRaises(HTTPException) as e:
                self.location.upload_native_position(point, Response(), authorization="Bearer loc_test")
            self.assertEqual(e.exception.status_code, 422)

    def test_cannot_issue_native_upload_token_from_legacy_session(self):
        with patch.dict(os.environ, {"VERIFYD_TRUST_VOICE_LOCATION_ENABLED": "1"}):
            with self.assertRaises(HTTPException) as e:
                self.location.issue_native_upload_token(Response(), authorization="Bearer valid")
            self.assertEqual(e.exception.status_code, 403)

    def test_stop_sharing_removes_native_token_and_coordinate(self):
        cursor = NativeCursor()
        self.install_cursor(cursor)
        with patch.dict(os.environ, {"VERIFYD_TRUST_VOICE_LOCATION_ENABLED": "1"}):
            self.location.stop_sharing(Response(), authorization="Bearer valid")
        statements = " ".join(sql for sql, _ in cursor.statements)
        self.assertIn("DELETE FROM verifyd_circle_location_native_tokens", statements)
        self.assertIn("DELETE FROM verifyd_circle_latest_locations", statements)

    def test_denies_native_upload_after_revoked_session(self):
        cursor = NativeCursor(
            share={"enabled": True, "precision_mode": "precise", "expires_at": None},
            token_owner="owner",
            session_active=False,
        )
        self.install_cursor(cursor)
        point = types.SimpleNamespace(latitude=40.1, longitude=-73.1, accuracy=20, time=time.time()*1000)
        with patch.dict(os.environ, {"VERIFYD_TRUST_VOICE_LOCATION_ENABLED": "1"}):
            with self.assertRaises(HTTPException) as e:
                self.location.upload_native_position(point, Response(), authorization="Bearer loc_native")
            self.assertEqual(e.exception.status_code, 401)
        self.assertFalse(any("INSERT INTO verifyd_circle_latest_locations" in sql for sql, _ in cursor.statements))

    def test_null_native_timestamp_is_supported_without_location_history(self):
        cursor = NativeCursor(
            share={"enabled": True, "precision_mode": "approximate", "expires_at": None},
            token_owner="owner",
            session_active=True,
        )
        self.install_cursor(cursor)
        point = types.SimpleNamespace(latitude=40.11231, longitude=-73.82451, accuracy=None, time=None)
        with patch.dict(os.environ, {"VERIFYD_TRUST_VOICE_LOCATION_ENABLED": "1"}):
            result = self.location.upload_native_position(point, Response(), authorization="Bearer loc_native")
        self.assertTrue(result["ok"])
        writes = [(sql, args) for sql, args in cursor.statements if "INSERT INTO verifyd_circle_latest_locations" in sql]
        self.assertEqual(len(writes), 1)
        self.assertEqual(writes[0][1][1:4], (40.11, -73.82, None))

    def test_approximate_native_upload_redacts_coordinates(self):
        cursor = NativeCursor(
            share={"enabled": True, "precision_mode": "approximate", "expires_at": None},
            token_owner="owner",
            session_active=True,
        )
        self.install_cursor(cursor)
        point = types.SimpleNamespace(latitude=40.71283, longitude=-73.99428, accuracy=11, time=time.time()*1000)
        with patch.dict(os.environ, {"VERIFYD_TRUST_VOICE_LOCATION_ENABLED": "1"}):
            result = self.location.upload_native_position(point, Response(), authorization="Bearer loc_native")
        self.assertTrue(result["ok"])
        writes = [(sql, args) for sql, args in cursor.statements if "INSERT INTO verifyd_circle_latest_locations" in sql]
        self.assertEqual(len(writes), 1)
        self.assertEqual(writes[0][1][1:4], (40.71, -73.99, None))


if __name__ == "__main__":
    unittest.main()
