"""No production services: test that RQ readiness checks expose capabilities only."""
import ast
import importlib.util
import pathlib
import shutil
import unittest
from unittest.mock import Mock, patch

PATH = pathlib.Path(__file__).resolve().parent.parent / "verifyd_worker.py"
TREE = ast.parse(PATH.read_text())


class VoicemailWorkerReadinessTests(unittest.TestCase):
    def test_reports_only_booleans_and_never_exposes_secrets(self):
        fn = next(n for n in TREE.body if isinstance(n, ast.FunctionDef) and n.name == "_log_voicemail_readiness")
        logger = Mock()
        namespace = {"os": __import__("os"), "log": logger}
        exec(compile(ast.Module(body=[fn], type_ignores=[]), "<worker-readiness>", "exec"), namespace)
        fake_key = "DONT_PRINT_EXAMPLE_SECRET"
        with patch.dict("os.environ", {
            "OPENAI_API_KEY": fake_key,
            "VERIFYD_VOICEMAIL_INTELLIGENCE_ENABLED": "1",
            "VERIFYD_TRUST_VOICE_ANALYSIS_QUEUE": "verifyd"
        }), patch.object(shutil, "which", return_value="/usr/bin/ffmpeg"), patch.object(
            importlib.util, "find_spec", return_value=Mock()
        ):
            namespace["_log_voicemail_readiness"](["verifyd"])
        args = logger.info.call_args.args
        self.assertEqual(args[0], "Voicemail Intelligence worker readiness (no user data): %s")
        checks = args[1]
        self.assertTrue(checks["transcription_key_present"])
        self.assertTrue(checks["analysis_queue_consumed"])
        self.assertTrue(all(isinstance(v, bool) for v in checks.values()))
        self.assertNotIn(fake_key, str(checks))

    def test_mismatched_queue_is_not_ready(self):
        fn = next(n for n in TREE.body if isinstance(n, ast.FunctionDef) and n.name == "_log_voicemail_readiness")
        logger = Mock()
        namespace = {"os": __import__("os"), "log": logger}
        exec(compile(ast.Module(body=[fn], type_ignores=[]), "<worker-readiness>", "exec"), namespace)
        with patch.dict("os.environ", {"VERIFYD_TRUST_VOICE_ANALYSIS_QUEUE": "verifyd"}, clear=True), patch.object(
            shutil, "which", return_value=None
        ), patch.object(importlib.util, "find_spec", return_value=None):
            namespace["_log_voicemail_readiness"](["verifyd-documents"])
        checks = logger.info.call_args.args[1]
        self.assertFalse(checks["analysis_queue_consumed"])
        self.assertFalse(checks["transcription_key_present"])
        self.assertFalse(checks["ffmpeg_available"])


if __name__ == "__main__":
    unittest.main()
