"""Startup diagnostics must not expose secrets, send pushes or change delivery."""
from contextlib import asynccontextmanager
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec
from fastapi import FastAPI
from fastapi.testclient import TestClient

import trust_voice_native_calls as native


class NativeReadiness(unittest.TestCase):
    def test_missing_settings_do_not_fail_startup_or_expose_exception(self):
        secret = "DO-NOT-LOG-THIS-PRIVATE-PATH"
        with patch.dict(os.environ, {"VERIFYD_APNS_PRIVATE_KEY_FILE": secret}, clear=True), \
             patch.object(native, "_provider_token", side_effect=ValueError(secret)), \
             patch.object(native, "_token_cipher", side_effect=ValueError(secret)):
            report = native.readiness_report()
        self.assertEqual(5, len(report))
        self.assertTrue(all(type(value) is bool for value in report.values()))
        self.assertFalse(any(report.values()))
        self.assertNotIn(secret, json.dumps(report))

    def test_valid_local_key_checked_without_push_db_or_enabling_delivery(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "private.p8"
            pem = ec.generate_private_key(ec.SECP256R1()).private_bytes(
                serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                serialization.NoEncryption())
            path.write_bytes(pem)
            encryption = Fernet.generate_key().decode()
            env = {"VERIFYD_APNS_TEAM_ID": "3623QELM75", "VERIFYD_APNS_KEY_ID": "ABCDEFGHIJ",
                   "VERIFYD_APNS_PRIVATE_KEY_FILE": str(path),
                   "VERIFYD_NATIVE_CALL_TOKEN_ENCRYPTION_KEY": encryption,
                   "VERIFYD_NATIVE_CALLS_PILOT_HANDLES": "test", "VERIFYD_NATIVE_CALLS_ENABLED": "0"}
            with patch.dict(os.environ, env, clear=True), patch.object(native, "_jwt_cache", None), \
                 patch.object(native.httpx, "AsyncClient", side_effect=AssertionError("No network")) as network, \
                 patch.object(native, "get_db", side_effect=AssertionError("No database")) as database:
                report = native.readiness_report()
                self.assertEqual("0", os.environ["VERIFYD_NATIVE_CALLS_ENABLED"])
                network.assert_not_called()
                database.assert_not_called()
            self.assertFalse(report.pop("native_delivery_enabled"))
            self.assertTrue(all(report.values()))
            for secret in [str(path), pem.decode(), encryption, "ABCDEFGHIJ"]:
                self.assertNotIn(secret, json.dumps(report))

    def test_router_diagnostic_runs_with_existing_parent_lifespan(self):
        events = []
        @asynccontextmanager
        async def parent(_app):
            events.append("start")
            yield
            events.append("stop")
        app = FastAPI(lifespan=parent)
        app.include_router(native.router)
        with patch.object(native, "readiness_report", return_value={"native_delivery_enabled": False}) as report:
            with self.assertLogs("uvicorn.error", level="INFO") as logs:
                with TestClient(app):
                    self.assertEqual(["start"], events)
            report.assert_called_once_with()
        self.assertEqual(["start", "stop"], events)
        self.assertIn('VERIFYD_NATIVE_CALL_READINESS {"native_delivery_enabled": false}', logs.output[0])


if __name__ == "__main__":
    unittest.main()
