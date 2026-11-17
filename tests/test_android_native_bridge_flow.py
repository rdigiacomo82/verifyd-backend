"""Mocked redemption checks; no live DB or API touched. Concurrency still requires PostgreSQL tests."""
import base64
import hashlib
import secrets
import sys
import time
import types
import unittest
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from unittest.mock import patch
from uuid import UUID, uuid4

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from fastapi import HTTPException, Response

_db = types.ModuleType("database")
_db.get_db = lambda: None
_spec = spec_from_file_location("verifyd_bridge_flow_under_test", Path(__file__).resolve().parents[1] / "trust_voice_android_bridge.py")
with patch.dict(sys.modules, {"database": _db}):
    _bridge = module_from_spec(_spec)
    _spec.loader.exec_module(_bridge)


def b64(value):
    return base64.urlsafe_b64encode(value).decode().rstrip("=")


class FakeCursor:
    rowcount = 0
    def __init__(self, row):
        self.row = row
    def execute(self, sql, params):
        if "UPDATE verifyd_android_native_bridge_intents SET redeemed_at" in sql:
            self.rowcount = int(self.row["redeemed_at"] is None)
            if self.rowcount:
                self.row["redeemed_at"] = params[0]
    def fetchone(self):
        return dict(self.row)


class FakeConn:
    def __init__(self, cursor):
        self._cursor = cursor
    def __enter__(self):
        return self
    def __exit__(self, *args):
        return False
    def cursor(self):
        return self._cursor


class BridgeRedemptionFlow(unittest.TestCase):
    def setUp(self):
        key = ec.generate_private_key(ec.SECP256R1())
        raw = key.public_key().public_bytes(serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo)
        public_key = b64(raw)
        key_hash = _bridge.parse_public_key(public_key)[1]
        installation = str(uuid4())
        intent = "nbi_" + secrets.token_urlsafe(24)
        challenge = secrets.token_urlsafe(32)
        signature = key.sign(_bridge.message_to_sign(intent, challenge, installation, _bridge.PACKAGE, key_hash), ec.ECDSA(hashes.SHA256()))
        # Pydantic parsing is tested separately; this exercises transaction guards.
        self.body = types.SimpleNamespace(installation_id=UUID(installation), package_name=_bridge.PACKAGE,
                     public_key_spki_b64=public_key, opted_in=True, intent_id=intent, challenge=challenge,
                     signature_b64=b64(signature), fcm_token="a" * 100)
        self.row = {"identity_id": "test-identity", "session_id": "test-session",
                    "installation_id": installation, "package_name": _bridge.PACKAGE,
                    "public_key_hash": key_hash,
                    "challenge_hash": hashlib.sha256(challenge.encode()).hexdigest(),
                    "redeemed_at": None, "revoked_at": None, "expires_at": time.time() + 45}
        self.cursor = FakeCursor(self.row)
        self.registrations = []
        self.registry_ready = False
        calls = types.ModuleType("trust_voice_android_calls")
        calls.AndroidDeviceRegistration = lambda **kwargs: types.SimpleNamespace(**kwargs)
        def record_device(*args):
            assert self.registry_ready, "device registry must exist before redemption writes"
            self.registrations.append(args)
        calls._write_device = record_device
        calls.ensure_schema = lambda: setattr(self, "registry_ready", True)
        calls.allowed_packages = lambda: {_bridge.PACKAGE}
        calls.pilot_allowed = lambda identity: True
        signaling = types.ModuleType("trust_voice_signaling")
        signaling._identity_by_id = lambda identity_id: {"id": identity_id}
        patches = [
            patch.object(_bridge, "_require_enabled", lambda: None),
            patch.object(_bridge, "_schema", lambda: None),
            patch.object(_bridge, "get_db", lambda: FakeConn(self.cursor)),
            patch.dict(sys.modules, {"trust_voice_android_calls": calls, "trust_voice_signaling": signaling}),
        ]
        for item in patches:
            item.start()
        self.addCleanup(lambda: [item.stop() for item in reversed(patches)])

    def redeem(self):
        return _bridge.redeem_intent(self.body, Response())

    def assert_denied(self):
        with self.assertRaises(HTTPException) as caught:
            self.redeem()
        self.assertEqual(caught.exception.status_code, 401)
        self.assertFalse(self.registrations)

    def test_success_then_replay_denied(self):
        self.assertEqual(self.redeem()["schema"], "verifyd.native-bridge.android.v1")
        self.assertEqual(len(self.registrations), 1)
        with self.assertRaises(HTTPException) as caught:
            self.redeem()
        self.assertEqual(caught.exception.status_code, 401)
        self.assertEqual(len(self.registrations), 1)

    def test_expired_denied(self):
        self.row["expires_at"] = time.time() - 1
        self.assert_denied()

    def test_revoked_session_denied(self):
        self.row["revoked_at"] = "2026-11-16T12:00:00Z"
        self.assert_denied()

    def test_device_key_mismatch_denied(self):
        self.row["public_key_hash"] = "0" * 64
        self.assert_denied()

    def test_challenge_mismatch_denied(self):
        self.row["challenge_hash"] = "0" * 64
        self.assert_denied()


if __name__ == "__main__":
    unittest.main()
