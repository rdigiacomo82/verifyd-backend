"""Safe staged Resend receiving of registered @vfvid.com handles."""
import os
import pathlib
import unittest
from unittest.mock import patch

from trust_voice_circle_email_resend_apex import (
    resend_apex_enabled, resend_apex_recipient_handle,
)

ROOT = pathlib.Path(__file__).resolve().parents[1]


def gates(apex="1", public="1", extra=None):
    return patch.dict(os.environ, {
        "VERIFYD_CIRCLE_EMAIL_RESEND_APEX_ENABLED": apex,
        "VERIFYD_CIRCLE_EMAIL_PUBLIC_HANDLE_ENABLED": public,
        "VERIFYD_CIRCLE_EMAIL_RESERVED_LOCALPARTS": extra or "",
    })


class ResendApexTests(unittest.TestCase):
    def test_independent_apex_gate_defaults_off(self):
        with gates(apex="0"):
            self.assertFalse(resend_apex_enabled())
            self.assertIsNone(resend_apex_recipient_handle(["roberttest22@vfvid.com"]))

    def test_public_gate_also_required(self):
        with gates(public="0"):
            self.assertFalse(resend_apex_enabled())
            self.assertIsNone(resend_apex_recipient_handle(["roberttest22@vfvid.com"]))

    def test_exact_single_registered_shape(self):
        with gates():
            self.assertEqual(resend_apex_recipient_handle(["RobertTest22@vfvid.com"]), "roberttest22")
            self.assertEqual(
                resend_apex_recipient_handle(["Robert <roberttest22@vfvid.com>"]),
                "roberttest22",
            )

    def test_never_claim_operational_mailboxes(self):
        with gates():
            for name in ("support", "admin", "contact", "security", "billing", "mail", "postmaster"):
                with self.subTest(name=name):
                    self.assertIsNone(resend_apex_recipient_handle([name + "@vfvid.com"]))

    def test_custom_reserved_names(self):
        with gates(extra="roberttest22, marketing"):
            self.assertIsNone(resend_apex_recipient_handle(["roberttest22@vfvid.com"]))
            self.assertIsNone(resend_apex_recipient_handle(["marketing@vfvid.com"]))
            self.assertEqual(resend_apex_recipient_handle(["annslp@vfvid.com"]), "annslp")

    def test_reject_wrong_domain_aliases_and_multiple_recipients(self):
        with gates():
            for recipients in (
                [], ["roberttest22@reply.vfvid.com"], ["roberttest22@gmail.com"],
                ["roberttest22@vfvid.com", "someone@vfvid.com"],
                ["roberttest22@vfvid.com, support@vfvid.com"],
                ["support@vfvid.com, roberttest22@vfvid.com"],
                ["@vfvid.com"], ["abc+token@vfvid.com"],
                ["a@vfvid.com"], ["john@example.com", "roberttest22@vfvid.com"],
                "roberttest22@vfvid.com", [None], [42],
            ):
                with self.subTest(recipients=recipients):
                    self.assertIsNone(resend_apex_recipient_handle(recipients))

    def test_existing_reply_routing_remains_separate(self):
        inbound = (ROOT / "trust_voice_circle_email_inbound.py").read_text()
        self.assertIn("public_recipient_from_to(received.get('to') or [], domain)", inbound)
        self.assertIn("apex_handle=resend_apex_recipient_handle(received.get('to') or [])", inbound)
        self.assertIn("return store_public_received(received,provider_id,handle=apex_handle)", inbound)
        self.assertIn("candidates=_candidate_routes(received.get('to') or [], domain)", inbound)
        self.assertIn("verify_resend_webhook(raw, request.headers)", inbound)
        self.assertIn("_fetch_received_email,provider_id", inbound)


if __name__ == "__main__":
    unittest.main()
