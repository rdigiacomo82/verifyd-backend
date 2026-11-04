"""Dependency-free tests for the disabled-by-default Circle location API.

Run:
    python -m unittest discover -s tests -p test_trust_voice_location.py -v

This exercises endpoint security logic against fakes, not the production database.
"""
from __future__ import annotations

import importlib.util
import os
import pathlib
import sys
import types
import unittest
from contextlib import contextmanager
from unittest.mock import patch


BACKEND = pathlib.Path(__file__).resolve().parents[1] / "trust_voice_location.py"


class HTTPException(Exception):
    def __init__(self, status_code, detail=None):
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"{status_code}: {detail}")


class Response:
    def __init__(self):
        self.headers = {}


class Router:
    def __init__(self, **kwargs):
        self.options = kwargs

    def get(self, path):
        return lambda fn: fn

    def post(self, path):
        return lambda fn: fn

    def put(self, path):
        return lambda fn: fn

    def delete(self, path):
        return lambda fn: fn


class BaseModel:
    def __init__(self, **kwargs):
        for name in getattr(type(self), "__annotations__", {}):
            setattr(self, name, kwargs.get(name, getattr(type(self), name, None)))


def Field(default=None, **kwargs):
    return kwargs.get("default_factory", lambda: default)()


class Cursor:
    def __init__(self, share=None, positions=None):
        self.share = share
        self.positions = positions
        self.statements = []
        self.last_sql = ""

    def execute(self, sql, params=None):
        self.last_sql = " ".join(sql.split())
        self.statements.append((self.last_sql, params))

    def fetchone(self):
        if "SELECT enabled,precision_mode,expires_at" in self.last_sql:
            return self.share
        if "SELECT recorded_at FROM" in self.last_sql:
            return self.positions
        return None

    def fetchall(self):
        return []


class Connection:
    def __init__(self, cursor):
        self._cursor = cursor

    def cursor(self):
        return self._cursor


def load_location(skip_schema=False):
    fake_fastapi = types.ModuleType("fastapi")
    fake_fastapi.APIRouter = Router
    fake_fastapi.Header = lambda default=None: default
    fake_fastapi.HTTPException = HTTPException
    fake_fastapi.Response = Response
    fake_pydantic = types.ModuleType("pydantic")
    fake_pydantic.BaseModel = BaseModel
    fake_pydantic.Field = Field
    database = types.ModuleType("database")
    database.get_db = lambda: None
    blocking = types.ModuleType("trust_voice_blocking")
    blocking.identities_blocked = lambda owner, viewer: False
    permissions = types.ModuleType("trust_voice_permissions")
    permissions.trust_circle_contains = lambda owner, viewer: True
    signaling = types.ModuleType("trust_voice_signaling")
    signaling._verify_token = lambda token: {"identity_id": "owner"} if token == "valid" else {}
    signaling._identity_by_id = lambda i: {"id": i, "status": "active", "handle": i} if i else None
    signaling._identity_by_handle = lambda h: {"id": h, "status": "active"} if h else None
    stub_modules = {
        "fastapi": fake_fastapi,
        "pydantic": fake_pydantic,
        "database": database,
        "trust_voice_blocking": blocking,
        "trust_voice_permissions": permissions,
        "trust_voice_signaling": signaling,
    }
    with patch.dict(sys.modules, stub_modules):
        spec = importlib.util.spec_from_file_location("location_under_test", BACKEND)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    if not skip_schema:
        module._schema = lambda: None
    return module


class LocationSecurityTest(unittest.TestCase):
    def setUp(self):
        self.module = load_location()

    def test_public_status_is_off_without_schema_when_flag_disabled(self):
        calls = []
        self.module._schema = lambda: calls.append("schema")
        with patch.dict(os.environ, {"VERIFYD_TRUST_VOICE_LOCATION_ENABLED": "0"}):
            response = Response()
            result = self.module.location_status(response)
        self.assertEqual(result, {"enabled": False, "ready": False})
        self.assertEqual(calls, [])
        self.assertEqual(response.headers.get("Cache-Control"), "private, no-store")

    def test_enabled_status_creates_isolated_schema_before_user_session(self):
        calls = []
        self.module._schema = lambda: calls.append("schema")
        with patch.dict(os.environ, {"VERIFYD_TRUST_VOICE_LOCATION_ENABLED": "1"}):
            result = self.module.location_status(Response())
        self.assertEqual(result, {"enabled": True, "ready": True})
        self.assertEqual(calls, ["schema"])

    def test_feature_disabled_by_default_without_access_or_database(self):
        with patch.dict(os.environ, {"VERIFYD_TRUST_VOICE_LOCATION_ENABLED": "0"}):
            for fn, args in [
                (self.module.get_sharing, (Response(),)),
                (self.module.stop_sharing, (Response(),)),
                (self.module.visible_locations, (Response(),)),
            ]:
                with self.assertRaises(HTTPException) as error:
                    fn(*args, authorization="Bearer valid")
                self.assertEqual(error.exception.status_code, 404)

    def test_requires_session_even_when_enabled(self):
        with patch.dict(os.environ, {"VERIFYD_TRUST_VOICE_LOCATION_ENABLED": "1"}):
            for token in (None, "", "Token valid", "Bearer ", "Bearer invalid"):
                with self.assertRaises(HTTPException) as error:
                    self.module.get_sharing(Response(), authorization=token)
                self.assertEqual(error.exception.status_code, 401)

    def test_selected_contact_cannot_be_outside_trust_circle(self):
        self.module.trust_circle_contains = lambda owner, member: False
        body = types.SimpleNamespace(
            enabled=True, scope="selected", precision="approximate",
            selected_handles=["outsider"], duration_minutes=15,
        )
        with patch.dict(os.environ, {"VERIFYD_TRUST_VOICE_LOCATION_ENABLED": "1"}):
            with self.assertRaises(HTTPException) as error:
                self.module.set_sharing(body, Response(), authorization="Bearer valid")
            self.assertEqual(error.exception.status_code, 403)

    def test_blocked_contact_cannot_be_granted_access(self):
        self.module.identities_blocked = lambda owner, member: True
        body = types.SimpleNamespace(
            enabled=True, scope="selected", precision="precise",
            selected_handles=["blocked"], duration_minutes=60,
        )
        with patch.dict(os.environ, {"VERIFYD_TRUST_VOICE_LOCATION_ENABLED": "1"}):
            with self.assertRaises(HTTPException) as error:
                self.module.set_sharing(body, Response(), authorization="Bearer valid")
            self.assertEqual(error.exception.status_code, 403)

    def test_revoking_sharing_purges_grants_and_location(self):
        cursor = Cursor()
        @contextmanager
        def get_db():
            yield Connection(cursor)
        self.module.get_db = get_db
        with patch.dict(os.environ, {"VERIFYD_TRUST_VOICE_LOCATION_ENABLED": "1"}):
            result = self.module.stop_sharing(Response(), authorization="Bearer valid")
        self.assertFalse(result["enabled"])
        queries = " ".join(statement for statement, _ in cursor.statements)
        self.assertIn("SET enabled=FALSE", queries)
        self.assertIn("DELETE FROM verifyd_circle_location_grants", queries)
        self.assertIn("DELETE FROM verifyd_circle_latest_locations", queries)

    def test_position_denied_if_sharing_is_off(self):
        cursor = Cursor(share=None)
        @contextmanager
        def get_db():
            yield Connection(cursor)
        self.module.get_db = get_db
        body = types.SimpleNamespace(latitude=40.0, longitude=-73.0, accuracy_m=10.0)
        with patch.dict(os.environ, {"VERIFYD_TRUST_VOICE_LOCATION_ENABLED": "1"}):
            with self.assertRaises(HTTPException) as error:
                self.module.update_position(body, Response(), authorization="Bearer valid")
            self.assertEqual(error.exception.status_code, 403)
        self.assertFalse(any("INSERT INTO verifyd_circle_latest_locations" in s for s, _ in cursor.statements))

    def test_approximate_location_rounded_before_storage(self):
        cursor = Cursor(share={"enabled": True, "precision_mode": "approximate", "expires_at": None})
        @contextmanager
        def get_db():
            yield Connection(cursor)
        self.module.get_db = get_db
        body = types.SimpleNamespace(latitude=40.71283, longitude=-73.99428, accuracy_m=12.0)
        with patch.dict(os.environ, {"VERIFYD_TRUST_VOICE_LOCATION_ENABLED": "1"}):
            result = self.module.update_position(body, Response(), authorization="Bearer valid")
        self.assertTrue(result["ok"])
        insert = [(s, p) for s, p in cursor.statements if "INSERT INTO verifyd_circle_latest_locations" in s]
        self.assertEqual(len(insert), 1)
        self.assertEqual(insert[0][1][1:4], (40.71, -73.99, None))

    def test_account_deletion_cascades_all_location_rows(self):
        source = BACKEND.read_text(encoding="utf-8")
        # Three owner-keyed tables and both columns of the grants table.
        self.assertGreaterEqual(
            source.count("REFERENCES verifyd_identities(id) ON DELETE CASCADE"), 5
        )
        self.assertIn("i.status='active'", source)

    def test_visible_results_require_membership_and_expiry_filters(self):
        source = BACKEND.read_text(encoding="utf-8")
        self.assertIn("m.member_identity_id=%s", source)
        self.assertIn("s.expires_at IS NULL OR s.expires_at>NOW()", source)
        self.assertIn("g.viewer_identity_id=%s", source)
        self.assertIn("identities_blocked(owner_id, viewer_id)", source)
        self.assertIn("_purge_expired(cur)", source)


if __name__ == "__main__":
    unittest.main()
