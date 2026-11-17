"""Tests for the non-delivering Render FCM credential preflight.

These tests use synthetic metadata only, never real keys or Firebase access.
"""
import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[1] / "scripts" / "check_circle_fcm_sender.py"
spec = importlib.util.spec_from_file_location("verifyd_circle_fcm_preflight_under_test", SOURCE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class FcmPreflight(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "test.json"
        self.data = {
            "type": "service_account",
            "project_id": "verifyd-detector-pilot",
            "private_key": "synthetic_test_key_not_a_real_secret",
            "client_email": "synthetic-test@example.invalid",
        }
        self.path.write_text(json.dumps(self.data), encoding="utf-8")

    def test_request_is_validation_only_topic_without_device_token(self):
        payload = module.request_body()
        self.assertIs(payload["validate_only"], True)
        self.assertEqual(payload["message"]["topic"], "verifyd-circle-preflight")
        self.assertNotIn("token", payload["message"])
        self.assertNotIn("notification", payload["message"])

    def test_metadata_accepts_project_match(self):
        self.assertIsNone(module.check_metadata(str(self.path), "verifyd-detector-pilot"))

    def test_missing_file_denied(self):
        self.assertEqual(
            module.check_metadata(str(self.path) + ".missing", "verifyd-detector-pilot"),
            "SECRET_FILE_MISSING",
        )

    def test_mismatched_project_denied(self):
        self.assertEqual(
            module.check_metadata(str(self.path), "another-project"),
            "PROJECT_ID_MISMATCH",
        )

    def test_secret_contents_not_exposed_in_error_reason(self):
        self.path.write_text("{private_key: fake}", encoding="utf-8")
        self.assertEqual(
            module.check_metadata(str(self.path), "verifyd-detector-pilot"),
            "SECRET_FILE_UNREADABLE",
        )

    def test_existing_valid_encryption_key_is_accepted(self):
        from cryptography.fernet import Fernet
        with patch.dict(os.environ, {
            "VERIFYD_NATIVE_CALL_TOKEN_ENCRYPTION_KEY": Fernet.generate_key().decode()
        }, clear=True):
            self.assertTrue(module.token_encryption_ready())

    def test_missing_or_invalid_encryption_key_is_blocked(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(module.token_encryption_ready())
        with patch.dict(os.environ, {
            "VERIFYD_NATIVE_CALL_TOKEN_ENCRYPTION_KEY": "invalid"
        }, clear=True):
            self.assertFalse(module.token_encryption_ready())

    def test_default_run_fails_closed_without_env_or_network(self):
        emitted = []
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(module.run(emit=emitted.append), 1)
        self.assertEqual(emitted, [
            "FCM_PREFLIGHT RESULT=FAIL REASON=PROJECT_ID_UNSET_OR_INVALID"
        ])

    def test_only_sanitized_status_passed_to_emitter(self):
        emitted = []
        with patch.dict(os.environ, {
            "GOOGLE_APPLICATION_CREDENTIALS": str(self.path),
            "VERIFYD_ANDROID_FIREBASE_PROJECT_ID": "other-project"
        }, clear=True):
            self.assertEqual(module.run(emit=emitted.append), 1)
        self.assertTrue(all(line.startswith("FCM_PREFLIGHT ") for line in emitted))
        self.assertNotIn("synthetic_test_key_not_a_real_secret", " ".join(emitted))


if __name__ == "__main__":
    unittest.main()
