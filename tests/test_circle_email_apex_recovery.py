"""Security regression: replay Hostinger quarantines without changing recipient ownership."""
import hashlib
import sys
import types
import unittest
from email.parser import BytesParser
from email import policy
from unittest.mock import patch

import trust_voice_circle_email_apex_gateway as gw


class FakeIMAP:
    def __init__(self, row):
        self.row = row
        self.calls = []

    def uid(self, *args):
        self.calls.append(args)
        if args[0] != "search":
            raise AssertionError("replay must not modify Hostinger messages")
        return "OK", [self.row]


class FakeCursor:
    def __init__(self, eligible):
        self.eligible = eligible
        self.updates = []
        self.rows = []

    def execute(self, sql, args):
        if sql.lstrip().startswith("SELECT provider_id"):
            self.rows = [{"provider_id": p} for p in args[0] if p in self.eligible]
        elif sql.lstrip().startswith("UPDATE verifyd_circle_email_apex_quarantine"):
            self.updates.append(args)
        else:
            raise AssertionError("Unexpected SQL")

    def fetchall(self):
        return self.rows


class FakeDB:
    def __init__(self, cursor):
        self.the_cursor = cursor

    def cursor(self):
        return self.the_cursor

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


def message(host):
    return (
        "Received: from mail-pl1-x630.google.com "
        "by " + host + " (mx.hostinger.com) with ESMTPS "
        "for <roberttest22@vfvid.com>; Fri, 30 Oct 2026 21:03:33 +0000 (UTC)\n"
        "From: gmail-sender@example.com\n"
        "To: another-member@vfvid.com\n"
        "Subject: Security test\n"
        "Content-Type: text/plain; charset=utf-8\n\n"
        "Hello\n"
    ).encode("ascii")


class ReplayTests(unittest.TestCase):
    def test_only_bounded_existing_checkpointed_uids(self):
        imap = FakeIMAP(b"95 100 98 105 100")
        self.assertEqual(gw._recent_processed_uids(imap, 100, 3), [100, 98])
        self.assertEqual(imap.calls, [("search", None, "UID", "98:100")])
        self.assertEqual(gw._recent_processed_uids(FakeIMAP(b""), 0), [])

    def test_malformed_uid_search_fails_closed(self):
        for response in (b"-1", b"123x", b"0", b"4294967296"):
            with self.subTest(response=response):
                with self.assertRaises(RuntimeError):
                    gw._recent_processed_uids(FakeIMAP(response), 200)

    def test_new_hostinger_topmost_hop_is_authoritative_not_to_header(self):
        trusted = frozenset({"fr-int-smtpin22.hostinger.io"})
        raw = message("fr-int-smtpin22.hostinger.io")
        handle, mail = gw.parse_catchall(raw, trusted)
        self.assertEqual(handle, "roberttest22")
        self.assertEqual(mail["to"], ["roberttest22@vfvid.com"])
        self.assertEqual(gw.parse_catchall(raw, frozenset()), (None, None))
        spoof = message("untrusted.fake.example")
        self.assertEqual(gw.parse_catchall(spoof, trusted), (None, None))

    def test_quarantine_replay_stores_only_after_verifying_trusted_hop(self):
        key, valid, uid = "mailbox-hash", "1234", 77
        provider_id = "hostinger-" + hashlib.sha256(
            f"{key}:{valid}:{uid}".encode("ascii")).hexdigest()
        cur = FakeCursor({provider_id})
        saved = []

        def save(mail, provider, *, handle):
            saved.append((provider, handle, mail["to"]))
            return "account-1", "saved-mail-1"

        with patch.dict(sys.modules, {
            "database": types.SimpleNamespace(get_db=lambda: FakeDB(cur)),
            "trust_voice_circle_email_public": types.SimpleNamespace(
                store_public_received=save),
        }):
            with patch.object(gw, "_fetch_size", return_value=1000):
                with patch.object(gw, "_fetch_raw",
                                  return_value=message("fr-int-smtpin22.hostinger.io")):
                    recovered, checked = gw._replay_recent_quarantine(
                        FakeIMAP(b"77"), key, valid, uid,
                        frozenset({"fr-int-smtpin22.hostinger.io"}))
        self.assertEqual((recovered, checked), (1, 1))
        self.assertEqual(saved, [(provider_id, "roberttest22",
                                  ["roberttest22@vfvid.com"])])
        self.assertEqual(len(cur.updates), 1)
        self.assertEqual(cur.updates[0][1:], (True, provider_id))

    def test_untrusted_mail_never_imported_even_if_quarantined(self):
        key, valid, uid = "mailbox-hash", "1234", 77
        provider_id = "hostinger-" + hashlib.sha256(
            f"{key}:{valid}:{uid}".encode("ascii")).hexdigest()
        cur = FakeCursor({provider_id})
        calls = []

        def save(*args, **kwargs):
            calls.append((args, kwargs))
            raise AssertionError("Untrusted recipient must never route")

        with patch.dict(sys.modules, {
            "database": types.SimpleNamespace(get_db=lambda: FakeDB(cur)),
            "trust_voice_circle_email_public": types.SimpleNamespace(
                store_public_received=save),
        }):
            with patch.object(gw, "_fetch_size", return_value=1000):
                with patch.object(gw, "_fetch_raw",
                                  return_value=message("fake.example")):
                    recovered, checked = gw._replay_recent_quarantine(
                        FakeIMAP(b"77"), key, valid, uid,
                        frozenset({"fr-int-smtpin22.hostinger.io"}))
        self.assertEqual((recovered, checked), (0, 1))
        self.assertFalse(calls)
        self.assertEqual(cur.updates[0][1:], (False, provider_id))


class ReplayWatermarkTests(unittest.TestCase):
    """No IMAP rescan when both mailbox UID and exact trust list are unchanged."""

    class Cursor:
        def __init__(self, row):
            self.row = row
            self.commands = []

        def execute(self, sql, args):
            self.commands.append((sql, args))

        def fetchone(self):
            return self.row

    def test_replay_idle_skip_and_new_mail_or_hostlist_recheck(self):
        hosts = frozenset({"de-fra-smtpin6.hostinger.io"})
        digest = gw._replay_policy_hash(hosts)
        cur = self.Cursor({"replay_checked_uid": 841,
                           "replay_policy_hash": digest})
        with patch.dict(sys.modules, {
            "database": types.SimpleNamespace(get_db=lambda: FakeDB(cur))
        }):
            self.assertFalse(gw._replay_needed("mailbox", "valid", 841, hosts))
            self.assertTrue(gw._replay_needed("mailbox", "valid", 842, hosts))
            self.assertTrue(gw._replay_needed(
                "mailbox", "valid", 841,
                hosts | {"fr-int-smtpin22.hostinger.io"}))
        self.assertEqual(len(cur.commands), 3)

    def test_replay_marks_completed_state_only_on_success(self):
        hosts = frozenset({"de-fra-smtpin6.hostinger.io"})
        cur = self.Cursor(None)
        with patch.dict(sys.modules, {
            "database": types.SimpleNamespace(get_db=lambda: FakeDB(cur))
        }):
            gw._mark_replay_checked("mailbox", "valid", 841, hosts)
        self.assertEqual(len(cur.commands), 1)
        _, args = cur.commands[0]
        self.assertEqual(args, (841, gw._replay_policy_hash(hosts),
                                "mailbox", "valid"))

    def test_ingest_keeps_draining_full_replay_batches(self):
        from pathlib import Path
        source = (Path(__file__).resolve().parents[1] /
                  "trust_voice_circle_email_apex_gateway.py").read_text()
        self.assertIn("if _replay_needed(mailbox_key, valid, last, trusted_hosts):",
                      source)
        self.assertIn("if rechecked < _REPLAY_BATCH:", source)
        self.assertIn("_mark_replay_checked(mailbox_key, valid, last, trusted_hosts)",
                      source)
        self.assertIn("_replay_recent_quarantine(", source)


if __name__ == "__main__":
    unittest.main()
