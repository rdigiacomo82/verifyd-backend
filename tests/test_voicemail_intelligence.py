"""Zero-network tests for voicemail triage and recipient authorization."""
import ast
import pathlib
import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Optional

ROOT = pathlib.Path(__file__).resolve().parent.parent
from trust_voice_voicemail_risk import review_voicemail_content


class RiskTests(unittest.TestCase):
    def test_scams_are_red_with_explanations(self):
        result = review_voicemail_content(
            "Your account is locked. Send a gift card and your verification code today.",
            {"available": False},
        )
        self.assertEqual(result["level"], "red")
        self.assertTrue(result["explanations"])

    def test_urgency_without_strong_evidence_needs_review(self):
        self.assertEqual(review_voicemail_content(
            "Urgent response needed. Call back immediately.", {"available": False},
        )["level"], "blue")

    def test_unknown_audio_or_empty_transcript_never_green(self):
        self.assertEqual(review_voicemail_content("", {"available": False})["level"], "blue")
        self.assertEqual(review_voicemail_content("Hello", {"available": False})["level"], "blue")

    def test_ai_voice_score_is_uncertain_not_fraud_proof(self):
        result = review_voicemail_content(
            "Hello, I hope you are doing well today.",
            {"available": True, "has_audio": True, "audio_ai_score": 88,
             "confidence": "medium", "evidence": ["Unusual spectral characteristics"]},
        )
        self.assertEqual(result["level"], "blue")
        self.assertEqual(result["synthetic_voice"]["assessment"], "possible")
        self.assertIn("probabilistic", result["synthetic_voice"]["disclaimer"])

    def test_low_confidence_ai_score_does_not_claim_ai_voice(self):
        result = review_voicemail_content(
            "Hello, your package has arrived at the front desk.",
            {"available": True, "has_audio": True, "audio_ai_score": 95, "confidence": "low"},
        )
        self.assertEqual(result["level"], "green")
        self.assertNotEqual(result["synthetic_voice"]["assessment"], "possible")


class AccessControlTests(unittest.TestCase):
    def setUp(self):
        source = (ROOT / "trust_voice_voicemail_intelligence.py").read_text()
        parsed = ast.parse(source)
        node = next(n for n in parsed.body if isinstance(n, ast.FunctionDef) and n.name == "_incoming_voicemail")
        namespace = {}
        class HTTPException(Exception):
            def __init__(self, status_code, detail=None):
                self.status_code = status_code
                self.detail = detail

        @contextmanager
        def db():
            class Cur:
                def execute(self, query, params):
                    self.params = params
                def fetchone(self):
                    return {"deleted_at": None, "sender_identity_id": "sender"}
            yield SimpleNamespace(cursor=lambda: Cur())

        namespace.update({
            "HTTPException": HTTPException,
            "Optional": Optional,
            "_require_attachment_member": lambda aid, owner: {
                "attachment_id": aid, "conversation_id": "conversation",
                "message_id": "message", "filename": "VeriFYD_Voicemail_123.webm",
                "uploader_identity_id": "sender", "media_category": "audio",
            },
            "get_db": db,
            "PREFIX": "VeriFYD_Voicemail_",
        })
        exec(compile(ast.Module(body=[node], type_ignores=[]), "<voicemail-access>", "exec"), namespace)
        self.get = namespace["_incoming_voicemail"]
        self.HTTPException = HTTPException

    def test_sender_cannot_request_own_sent_voicemail_analysis(self):
        with self.assertRaises(self.HTTPException) as caught:
            self.get("attachment", "sender")
        self.assertEqual(caught.exception.status_code, 403)

    def test_receiver_is_authorized_only_after_conversation_membership(self):
        self.assertEqual(self.get("attachment", "receiver")["message_id"], "message")

    def test_pilot_has_two_bounded_recovery_retries_after_db_tls_outage(self):
        code = (ROOT / "trust_voice_voicemail_intelligence.py").read_text()
        self.assertIn("MAX_REVIEW_ATTEMPTS = 5", code)
        self.assertIn('if int(existing["attempt_count"] or 0) >= MAX_REVIEW_ATTEMPTS:', code)

    def test_background_worker_rechecks_recipient_and_pilot(self):
        code = (ROOT / "trust_voice_voicemail_intelligence.py").read_text()
        self.assertIn('recipient = _identity_by_id(str(owner_id))', code)
        self.assertIn('if not recipient or not _in_pilot(recipient):', code)
        self.assertIn('str(msg["sender_identity_id"]) != str(row.get("uploader_identity_id"))', code)

    def test_account_and_attachment_deletion_cleanup_is_cascade(self):
        code = (ROOT / "trust_voice_voicemail_intelligence.py").read_text()
        self.assertIn("REFERENCES verifyd_voice_attachments(attachment_id) ON DELETE CASCADE", code)
        self.assertIn("REFERENCES verifyd_identities(id) ON DELETE CASCADE", code)
        self.assertIn("AND deleted_at IS NULL", code)
        self.assertIn("review_summaries", code)
        self.assertIn('final_status = "completed" if transcript_status == "available" else "partial"', code)
        self.assertIn("consent_to_external_transcription", code)

    def test_router_is_feature_gated_and_not_public_business_api(self):
        code = (ROOT / "trust_voice_voicemail_intelligence.py").read_text()
        main = (ROOT / "main.py").read_text()
        self.assertIn("VERIFYD_VOICEMAIL_INTELLIGENCE_ENABLED", code)
        self.assertIn("consent_to_external_transcription", code)
        self.assertIn("from trust_voice_voicemail_intelligence import router", main)
        self.assertIn("app.include_router(trust_voice_voicemail_intelligence_router)", main)
        self.assertNotIn('@router.post("/business/', code)


class PilotAuthorizationTests(unittest.TestCase):
    def setUp(self):
        import os
        import ast
        source = (ROOT / "trust_voice_voicemail_intelligence.py").read_text()
        parsed = ast.parse(source)
        nodes = [n for n in parsed.body if isinstance(n, ast.FunctionDef) and
                 n.name in {"_pilot_handles", "_in_pilot", "_require_pilot", "voicemail_intelligence_health"}]
        namespace = {"os": os, "Optional": Optional, "Header": lambda default=None: default}
        class HTTPException(Exception):
            def __init__(self, status_code, detail=None):
                self.status_code, self.detail = status_code, detail
        namespace["HTTPException"] = HTTPException
        namespace["_enabled"] = lambda: True
        namespace["_owner"] = lambda auth: {"handle_lower": "roberttest22"} if auth == "test_valid" else {"handle_lower": "outsider"}
        # Compile the original function bodies without FastAPI decorators.
        for node in nodes:
            node.decorator_list = []
        code = ast.Module(body=[ast.ImportFrom(module="__future__",names=[ast.alias(name="annotations")],level=0), *nodes], type_ignores=[])
        exec(compile(ast.fix_missing_locations(code), "<pilot-rules>", "exec"), namespace)
        self.ns = namespace
        self.HTTPException = HTTPException

    def test_health_only_enables_allowlisted_identity(self):
        from unittest.mock import patch
        with patch.dict("os.environ", {"VERIFYD_VOICEMAIL_INTELLIGENCE_PILOT_HANDLES": "@RobertTest22,@DatabyDesign"}):
            self.assertTrue(self.ns["voicemail_intelligence_health"](authorization="test_valid")["enabled"])
            self.assertFalse(self.ns["voicemail_intelligence_health"](authorization="somebody_else")["enabled"])
            self.assertFalse(self.ns["voicemail_intelligence_health"](authorization=None)["enabled"])

    def test_server_enforces_pilot_not_only_frontend(self):
        from unittest.mock import patch
        with patch.dict("os.environ", {"VERIFYD_VOICEMAIL_INTELLIGENCE_PILOT_HANDLES": "roberttest22,databydesign"}):
            self.ns["_require_pilot"]({"handle_lower":"databydesign"})
            with self.assertRaises(self.HTTPException) as err:
                self.ns["_require_pilot"]({"handle_lower":"outsider"})
            self.assertEqual(err.exception.status_code, 404)

    def test_missing_allowlist_fail_closed(self):
        from unittest.mock import patch
        with patch.dict("os.environ", {"VERIFYD_VOICEMAIL_INTELLIGENCE_PILOT_HANDLES": ""}):
            self.assertFalse(self.ns["_in_pilot"]({"handle_lower": "roberttest22"}))
            self.assertFalse(self.ns["voicemail_intelligence_health"](authorization="test_valid")["enabled"])

    def test_all_five_private_operations_check_pilot(self):
        source = (ROOT / "trust_voice_voicemail_intelligence.py").read_text()
        self.assertGreaterEqual(source.count("    _require_pilot(owner)\n"), 7)
        self.assertIn("VERIFYD_VOICEMAIL_INTELLIGENCE_PILOT_HANDLES", source)


if __name__ == "__main__":
    unittest.main()
