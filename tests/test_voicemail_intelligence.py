"""Zero-network tests for voicemail triage and recipient authorization."""
import ast
import pathlib
import unittest
from contextlib import contextmanager
from types import SimpleNamespace

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

    def test_router_is_feature_gated_and_not_public_business_api(self):
        code = (ROOT / "trust_voice_voicemail_intelligence.py").read_text()
        main = (ROOT / "main.py").read_text()
        self.assertIn("VERIFYD_VOICEMAIL_INTELLIGENCE_ENABLED", code)
        self.assertIn("consent_to_external_transcription", code)
        self.assertIn("from trust_voice_voicemail_intelligence import router", main)
        self.assertIn("app.include_router(trust_voice_voicemail_intelligence_router)", main)
        self.assertNotIn('@router.post("/business/', code)


if __name__ == "__main__":
    unittest.main()
