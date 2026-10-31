"""Public mail delivery must survive broken/unavailable private attachment storage."""
from __future__ import annotations

import os
import unittest
from unittest.mock import patch

import trust_voice_circle_email_public as public
import trust_voice_circle_email_transport as transport
import trust_voice_circle_email_apex_gateway as gateway


class FakeCursor:
    def __init__(self):
        self.last_rows = []
        self.statements = []

    def execute(self, statement, args=None):
        self.statements.append((statement, args))
        if "SELECT id, handle_lower FROM verifyd_identities" in statement:
            self.last_rows = [{"id": "owner123", "handle_lower": "roberttest22"}]
        elif "SELECT 1 FROM verifyd_circle_email_received_events" in statement:
            self.last_rows = []
        else:
            self.last_rows = []

    def fetchone(self):
        return self.last_rows[0] if self.last_rows else None


class FakeDB:
    def __init__(self):
        self.c = FakeCursor()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def cursor(self):
        return self.c


class AttachmentFallbackTest(unittest.TestCase):
    def test_r2_upload_failure_preserves_email_and_provider_checkpoint(self):
        db = FakeDB()
        received = {
            "from": "test@example.com",
            "to": ["roberttest22@vfvid.com"],
            "subject": "Attachment that must not lose its email",
            "text": "Please see my PDF",
            "html": "",
            "attachments": [{"withheld": True}],
            "_hostinger_verified_mime": object(),
        }
        env = {
            "VERIFYD_CIRCLE_EMAIL_APEX_ATTACHMENT_CAPTURE_ENABLED": "true",
            "VERIFYD_CIRCLE_EMAIL_PUBLIC_HANDLE_ENABLED": "true",
        }
        with patch.dict(os.environ, env):
            with patch.object(public, "get_db", return_value=db):
                with patch.object(public, "_public_spam_limits", return_value=True):
                    with patch("trust_voice_attachments._storage_configured", return_value=True):
                        with patch.object(transport, "extract_hostinger_mime_files",
                                          return_value=([{"filename": "document.pdf",
                                                          "bytes": b"%PDF-1.7",
                                                          "content_type": "application/pdf"}], 0)):
                            with patch.object(transport, "persist_inbound_attachments",
                                              side_effect=OSError("mock r2 failure")):
                                owner, mail_id = public.store_public_received(
                                    received,
                                    "hostinger-testvalidproviderid-1234567",
                                    handle="roberttest22")
        self.assertEqual(owner, "owner123")
        self.assertTrue(mail_id.startswith("tvcmail_"))
        sql = "\n".join(item[0] for item in db.c.statements)
        self.assertIn("SAVEPOINT circle_email_external_file_write", sql)
        self.assertIn("ROLLBACK TO SAVEPOINT circle_email_external_file_write", sql)
        self.assertIn("RELEASE SAVEPOINT circle_email_external_file_write", sql)
        self.assertIn("UPDATE verifyd_circle_email_messages SET body=", sql)
        self.assertIn("INSERT INTO verifyd_circle_email_received_events", sql)
        body_updates = [args for statement, args in db.c.statements
                        if "UPDATE verifyd_circle_email_messages SET body=" in statement]
        self.assertIn("files withheld", body_updates[0][0])


class PrivateStorageProbeTest(unittest.TestCase):
    def test_probe_disabled_makes_no_storage_calls(self):
        with patch.dict(os.environ, {"VERIFYD_CIRCLE_EMAIL_APEX_R2_PROBE_ENABLED": "false"}):
            with patch("trust_voice_attachments._get_attachment_client") as mock:
                gateway._probe_private_attachment_storage()
                mock.assert_not_called()

    def test_probe_missing_credentials_makes_no_storage_calls(self):
        with patch.dict(os.environ, {"VERIFYD_CIRCLE_EMAIL_APEX_R2_PROBE_ENABLED": "true"}):
            with patch("trust_voice_attachments._storage_configured", return_value=False):
                with patch("trust_voice_attachments._get_attachment_client") as mock:
                    gateway._probe_private_attachment_storage()
                    mock.assert_not_called()

    def test_probe_reads_back_and_deletes_private_object(self):
        class Body:
            def read(self, _):
                return b"verifyd-circle-mail-storage-probe-v1"
            def close(self):
                pass
        class Client:
            def __init__(self):
                self.put = []
                self.removed = []
            def put_object(self, **kw):
                self.put.append(kw)
            def get_object(self, **kw):
                return {"Body": Body()}
            def delete_object(self, **kw):
                self.removed.append(kw)
        client = Client()
        with patch.dict(os.environ, {"VERIFYD_CIRCLE_EMAIL_APEX_R2_PROBE_ENABLED": "true"}):
            with patch("trust_voice_attachments._storage_configured", return_value=True):
                with patch("trust_voice_attachments._get_attachment_client", return_value=client):
                    gateway._probe_private_attachment_storage()
        self.assertEqual(len(client.put), 1)
        self.assertEqual(len(client.removed), 1)
        self.assertEqual(client.put[0]["Key"], client.removed[0]["Key"])
        self.assertNotIn("password", str(client.put))


if __name__ == "__main__":
    unittest.main()
