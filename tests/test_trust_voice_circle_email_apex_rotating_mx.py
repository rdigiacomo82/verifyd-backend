"""Security regression tests for opt-in Hostinger rotating MX support."""
import os
import unittest
from email import policy
from email.parser import Parser
from unittest.mock import patch

from trust_voice_circle_email_apex_gateway import original_envelope_recipient

EXACT_HOSTS = frozenset({"de-fra-smtpin12.hostinger.io"})


def mail(host, recipient="roberttest22@vfvid.com",
         label="(mx.hostinger.com)", transport="with ESMTPS id 4jGZqb2rPLz50bP",
         older_received=""):
    header = (
        "Received: from sender.example (sender.example [203.0.113.7])\n"
        f"\tby {host} {label} {transport}\n"
        f"\tfor <{recipient}>; Fri, 30 Oct 2026 22:03:39 +0000 (UTC)\n"
    )
    return Parser(policy=policy.default).parsestr(
        header + older_received + "From: sender@example.com\nSubject: Test\n\nHello"
    )


class TestRotatingHostingerMx(unittest.TestCase):
    def test_explicit_host_still_accepted_without_opt_in(self):
        with patch.dict(os.environ, {"VERIFYD_CIRCLE_EMAIL_APEX_ALLOW_ROTATING_MX": "0"}):
            self.assertEqual(original_envelope_recipient(
                mail("de-fra-smtpin12.hostinger.io"), EXACT_HOSTS),
                "roberttest22@vfvid.com")

    def test_unknown_host_denied_without_opt_in(self):
        with patch.dict(os.environ, {"VERIFYD_CIRCLE_EMAIL_APEX_ALLOW_ROTATING_MX": "0"}):
            self.assertIsNone(original_envelope_recipient(
                mail("de-fra-smtpin7.hostinger.io"), EXACT_HOSTS))

    def test_rotating_observed_host_families_accepted_opt_in(self):
        with patch.dict(os.environ, {"VERIFYD_CIRCLE_EMAIL_APEX_ALLOW_ROTATING_MX": "1"}):
            for hostname in ("de-fra-smtpin7.hostinger.io",
                             "fr-int-smtpin32.hostinger.io",
                             "de-fra-smtpin107.hostinger.io"):
                with self.subTest(hostname=hostname):
                    self.assertEqual(original_envelope_recipient(
                        mail(hostname), EXACT_HOSTS), "roberttest22@vfvid.com")

    def test_other_hostnames_and_missing_mx_proof_denied(self):
        with patch.dict(os.environ, {"VERIFYD_CIRCLE_EMAIL_APEX_ALLOW_ROTATING_MX": "1"}):
            samples = [
                mail("smtp.hostinger.io"),
                mail("de-fra-smtpin7.hostinger.io.evil.org"),
                mail("de-fra-smtpin7.not-hostinger.io"),
                mail("de-fra-smtpin7.hostinger.io", label=""),
                mail("de-fra-smtpin7.hostinger.io", label="(mx.hostinger.com.evil)"),
                mail("de-fra-smtpin7.hostinger.io", transport="with SMTP id FAKE"),
            ]
            for msg in samples:
                self.assertIsNone(original_envelope_recipient(msg, EXACT_HOSTS))

    def test_forged_older_received_header_is_not_recipient_proof(self):
        forged = (
            "Received: from forged.example\n"
            "\tby de-fra-smtpin7.hostinger.io (mx.hostinger.com) "
            "with ESMTPS id FAKE for <roberttest22@vfvid.com>; Fri, 30 Oct 2026 22:00:00 +0000\n"
        )
        with patch.dict(os.environ, {"VERIFYD_CIRCLE_EMAIL_APEX_ALLOW_ROTATING_MX": "1"}):
            self.assertEqual(original_envelope_recipient(
                mail("de-fra-smtpin7.hostinger.io", recipient="support@vfvid.com",
                     older_received=forged),
                EXACT_HOSTS), "support@vfvid.com")

    def test_no_trusted_host_configuration_fails_closed(self):
        with patch.dict(os.environ, {"VERIFYD_CIRCLE_EMAIL_APEX_ALLOW_ROTATING_MX": "1"}):
            self.assertIsNone(original_envelope_recipient(
                mail("de-fra-smtpin7.hostinger.io"), frozenset()))


if __name__ == "__main__":
    unittest.main()
