"""Isolated tests for Android FCM configuration readiness; no network or credentials."""
import ast
import os
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cryptography.fernet import Fernet

SOURCE = Path(__file__).resolve().parents[1] / "trust_voice_android_calls.py"
node = next(
    n for n in ast.parse(SOURCE.read_text(encoding="utf-8")).body
    if isinstance(n, ast.FunctionDef) and n.name == "configured"
)
namespace = {"os": os, "re": re, "Fernet": Fernet}
exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), "exec"), namespace)
configured = namespace["configured"]


class AndroidCallConfigurationReadiness(unittest.TestCase):
    def setUp(self):
        self.credentials = tempfile.TemporaryDirectory()
        self.addCleanup(self.credentials.cleanup)
        self.path = Path(self.credentials.name) / "fcm-pilot.json"
        self.path.write_text("{}", encoding="utf-8")
        self.env = patch.dict(os.environ, {
            "VERIFYD_ANDROID_FIREBASE_PROJECT_ID": "verifyd-detector-pilot",
            "GOOGLE_APPLICATION_CREDENTIALS": str(self.path),
            "VERIFYD_NATIVE_CALL_TOKEN_ENCRYPTION_KEY": Fernet.generate_key().decode(),
        }, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_present_valid_prerequisites_are_ready_for_pilot_enrollment(self):
        # This validates only local prerequisites, NOT live Firebase IAM/send rights.
        self.assertTrue(configured())

    def test_missing_credential_file_makes_enrollment_unavailable(self):
        self.path.unlink()
        self.assertFalse(configured())

    def test_invalid_fernet_key_makes_enrollment_unavailable(self):
        os.environ["VERIFYD_NATIVE_CALL_TOKEN_ENCRYPTION_KEY"] = "invalid"
        self.assertFalse(configured())

    def test_invalid_firebase_project_identifier_makes_enrollment_unavailable(self):
        os.environ["VERIFYD_ANDROID_FIREBASE_PROJECT_ID"] = "bad/name"
        self.assertFalse(configured())


if __name__ == "__main__":
    unittest.main()
