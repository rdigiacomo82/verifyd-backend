"""Optional fast ingress and safe MIME capture security regressions."""
import os
import unittest
from email.message import EmailMessage
from unittest.mock import patch

import trust_voice_circle_email_apex_gateway as gateway
import trust_voice_circle_email_idle as idle
from trust_voice_circle_email_transport import extract_hostinger_mime_files


def sample_attachment(filename="note.txt", content=b"Safe plain text"):
    mail = EmailMessage()
    mail["From"] = "sender@gmail.com"
    mail["To"] = "roberttest22@vfvid.com"
    mail["Subject"] = "New attachment"
    mail["Received"] = ("from google.example by fr-int-smtpin22.hostinger.io "
                        "(mx.hostinger.com) with ESMTPS "
                        "for <roberttest22@vfvid.com>;")
    mail.set_content("Test")
    mail.add_attachment(content, maintype="text", subtype="plain",
                        filename=filename)
    return mail


class HostingerAttachmentTests(unittest.TestCase):
    TRUSTED = frozenset({"fr-int-smtpin22.hostinger.io"})

    def test_attachment_flag_off_never_adds_mime_object(self):
        with patch.dict(os.environ,
                        {"VERIFYD_CIRCLE_EMAIL_APEX_ATTACHMENT_CAPTURE_ENABLED": "false"}):
            owner, received = gateway.parse_catchall(
                sample_attachment().as_bytes(), self.TRUSTED)
        self.assertEqual(owner, "roberttest22")
        self.assertNotIn("_hostinger_verified_mime", received)
        self.assertTrue(received["attachments"])

    def test_opt_in_uses_exact_verified_envelope(self):
        with patch.dict(os.environ,
                        {"VERIFYD_CIRCLE_EMAIL_APEX_ATTACHMENT_CAPTURE_ENABLED": "true"}):
            owner, received = gateway.parse_catchall(
                sample_attachment().as_bytes(), self.TRUSTED)
        self.assertEqual(owner, "roberttest22")
        mime = received["_hostinger_verified_mime"]
        files, omitted = extract_hostinger_mime_files(mime)
        self.assertEqual(omitted, 0)
        self.assertEqual(len(files), 1)
        self.assertEqual(files[0]["filename"], "note.txt")
        self.assertEqual(files[0]["bytes"], b"Safe plain text")
        self.assertEqual(files[0]["content_type"], "text/plain")

    def test_spoofed_hostinger_header_cannot_enable_capture(self):
        raw = sample_attachment()
        raw.replace_header(
            "Received", "from x by untrusted.example "
                        "for <roberttest22@vfvid.com>;")
        with patch.dict(os.environ,
                        {"VERIFYD_CIRCLE_EMAIL_APEX_ATTACHMENT_CAPTURE_ENABLED": "true"}):
            self.assertEqual(gateway.parse_catchall(
                raw.as_bytes(), self.TRUSTED), (None, None))

    def test_executable_and_large_files_are_withheld(self):
        mail = sample_attachment(filename="malware.exe", content=b"unsafe")
        files, omitted = extract_hostinger_mime_files(mail)
        self.assertEqual(files, [])
        self.assertEqual(omitted, 1)
        oversized = sample_attachment(
            filename="large.txt", content=b"A" * (8 * 1024 * 1024 + 1))
        files, omitted = extract_hostinger_mime_files(oversized)
        self.assertEqual(files, [])
        self.assertEqual(omitted, 1)

    def test_more_than_three_files_are_not_exposed(self):
        mail = sample_attachment()
        for i in range(4):
            mail.add_attachment(b"X", maintype="text", subtype="plain",
                                filename=f"extra{i}.txt")
        files, omitted = extract_hostinger_mime_files(mail)
        self.assertEqual(len(files), 3)
        self.assertEqual(omitted, 2)


class ListenerTests(unittest.TestCase):
    def test_disabled_refuses_to_run_without_imap(self):
        with patch.dict(os.environ, {"VERIFYD_CIRCLE_EMAIL_IDLE_ENABLED": "false"}):
            with patch.object(idle.imaplib, "IMAP4_SSL") as factory:
                self.assertEqual(idle.main(), 2)
                factory.assert_not_called()

    def test_gateway_disabled_refuses_to_run(self):
        with patch.dict(os.environ, {"VERIFYD_CIRCLE_EMAIL_IDLE_ENABLED": "true"}):
            with patch.object(idle, "configured", return_value=False):
                self.assertEqual(idle.main(), 2)

    def test_idle_uses_read_only_mailbox(self):
        class Idler:
            def __enter__(self):
                return iter([("EXISTS", [b"4"])])
            def __exit__(self, *_):
                return False

        class FakeConn:
            capabilities = (b"IDLE",)
            def __init__(self):
                self.selected = []
                self.logged_out = False
            def login(self, *_):
                return "OK", []
            def select(self, name, readonly=False):
                self.selected.append((name, readonly))
                return "OK", []
            def idle(self, duration=None):
                return Idler()
            def logout(self):
                self.logged_out = True

        fake = FakeConn()
        count = []
        def done():
            count.append(True)
            if len(count) == 2:
                idle._STOP.set()
        idle._STOP.clear()
        try:
            with patch.object(idle.imaplib, "IMAP4_SSL", return_value=fake):
                with patch.object(idle, "_ingest", side_effect=done):
                    with patch.dict(os.environ, {
                        "VERIFYD_CIRCLE_EMAIL_HOSTINGER_IMAP_USER": "circleinbound@vfvid.com",
                        "VERIFYD_CIRCLE_EMAIL_HOSTINGER_IMAP_PASSWORD": "test-only"
                    }):
                        idle._watch_connection()
            self.assertEqual(count, [True, True])
            self.assertEqual(fake.selected, [("INBOX", True)])
            self.assertTrue(fake.logged_out)
        finally:
            idle._STOP.clear()


if __name__ == "__main__":
    unittest.main()
