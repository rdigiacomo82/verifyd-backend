"""Regression: normal email threads must not silently exhaust sender quota."""
from __future__ import annotations

import ast
import hashlib
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import trust_voice_circle_email_apex_gateway as gateway

ROOT = Path(__file__).resolve().parents[1]


class Cursor:
    def __init__(self, *, existing=(), complete=False):
        self.existing = set(existing)
        self.complete = complete
        self.rows = []
        self.writes = []

    def execute(self, query, args):
        if "SELECT skipped_repair_version" in query:
            self.rows = ([{"skipped_repair_version": "sender-limit-30-v1"}]
                         if self.complete else [{"skipped_repair_version": None}])
        elif "SELECT provider_email_id AS pid" in query:
            self.rows = [{"pid": x} for x in args[0] if x in self.existing]
        elif "UPDATE verifyd_circle_email_apex_cursor" in query:
            self.writes.append(tuple(args))
            self.rows = []
        else:
            raise AssertionError("Unexpected recovery SQL " + query[:70])

    def fetchone(self):
        return self.rows[0] if self.rows else None

    def fetchall(self):
        return self.rows


class DB:
    def __init__(self, cur):
        self.cur = cur

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def cursor(self):
        return self.cur


class IMAP:
    def __init__(self):
        self.calls = []

    def uid(self, *args):
        self.calls.append(args)
        if args[0] != "search":
            raise AssertionError("Recovery must not modify Hostinger mailbox")
        return "OK", [b"9 10"]


def raw_test_mail():
    return (
        "Received: from google.example by fr-int-smtpin22.hostinger.io "
        "(mx.hostinger.com) with ESMTPS for <roberttest22@vfvid.com>;\n"
        "From: rdigiacomo82@gmail.com\n"
        "To: roberttest22@vfvid.com\n"
        "Subject: Once and for all\n"
        "Content-Type: text/plain; charset=UTF-8\n\n"
    ).encode("ascii")


class RecoveryTests(unittest.TestCase):
    def test_recovers_only_previously_skipped_proven_member(self):
        mailbox, valid = "mailboxhash", "someuidvalidity"
        previous_id = "hostinger-" + hashlib.sha256(
            f"{mailbox}:{valid}:9".encode("ascii")).hexdigest()
        last_id = "hostinger-" + hashlib.sha256(
            f"{mailbox}:{valid}:10".encode("ascii")).hexdigest()
        cur = Cursor(existing={previous_id})
        imap = IMAP()
        saved = []

        def store(mail, provider_id, *, handle):
            saved.append((mail["subject"], provider_id, handle))
            return "verified-owner", "stored-email"

        with patch.dict(sys.modules, {
            "database": types.SimpleNamespace(get_db=lambda: DB(cur)),
            "trust_voice_circle_email_public": types.SimpleNamespace(
                store_public_received=store),
        }), patch.object(gateway, "_fetch_size", return_value=500), patch.object(
                gateway, "_fetch_raw", return_value=raw_test_mail()):
            result = gateway._recover_recent_untracked(
                imap, mailbox, valid, 10,
                frozenset({"fr-int-smtpin22.hostinger.io"}))

        self.assertEqual(result, (1, 1))
        self.assertEqual(saved, [("Once and for all", last_id, "roberttest22")])
        self.assertEqual(cur.writes,
                         [("sender-limit-30-v1", mailbox, valid)])
        self.assertEqual(imap.calls, [("search", None, "UID", "1:10")])

    def test_repeat_run_skips_all_mailbox_io(self):
        cur = Cursor(complete=True)
        imap = IMAP()
        with patch.dict(sys.modules, {
            "database": types.SimpleNamespace(get_db=lambda: DB(cur)),
            "trust_voice_circle_email_public": types.SimpleNamespace(
                store_public_received=lambda *a, **kw: (None, None)),
        }):
            result = gateway._recover_recent_untracked(
                imap, "hash", "uidvalidity", 10,
                frozenset({"fr-int-smtpin22.hostinger.io"}))
        self.assertEqual(result, (0, 0))
        self.assertEqual(imap.calls, [])

    def test_untrusted_mail_will_not_be_imported(self):
        cur = Cursor()
        imap = IMAP()
        saved = []
        def store(*a, **kw):
            saved.append(True)
            return "a", "b"

        with patch.dict(sys.modules, {
            "database": types.SimpleNamespace(get_db=lambda: DB(cur)),
            "trust_voice_circle_email_public": types.SimpleNamespace(
                store_public_received=store),
        }), patch.object(gateway, "_fetch_size", return_value=500), patch.object(
                gateway, "_fetch_raw", return_value=raw_test_mail()):
            result = gateway._recover_recent_untracked(
                imap, "hash", "uidvalidity", 10,
                frozenset({"unknown-host.example"}))
        self.assertEqual(result, (0, 2))
        self.assertEqual(saved, [])

    def test_normal_thread_limit_and_owner_flood_ceiling(self):
        source = (ROOT / "trust_voice_circle_email_public.py").read_text()
        tree = ast.parse(source)
        constants = {target.id: node.value.value
                     for node in tree.body if isinstance(node, ast.Assign)
                     for target in node.targets if isinstance(target, ast.Name)
                     if isinstance(node.value, ast.Constant)}
        self.assertEqual(constants["_MAX_OWNER_DAY"], 60)
        self.assertEqual(constants["_MAX_SENDER_DAY"], 30)
        fn = next(x for x in tree.body if isinstance(x, ast.FunctionDef)
                  and x.name == "_public_spam_limits")
        ns = {k: constants[k] for k in ("_MAX_OWNER_DAY", "_MAX_SENDER_DAY")}
        exec(compile(ast.Module(body=[fn], type_ignores=[]),
                     "trust_voice_circle_email_public.py", "exec"), ns)

        class RateCursor:
            def __init__(self, counts):
                self.counts = iter(counts)
                self.calls = []
            def execute(self, sql, args):
                self.calls.append((sql, args))
            def fetchone(self):
                return {"n": next(self.counts)}
        self.assertTrue(ns["_public_spam_limits"](
            RateCursor([15, 8]), "owner", "sender@example.com"))
        self.assertTrue(ns["_public_spam_limits"](
            RateCursor([59, 29]), "owner", "sender@example.com"))
        self.assertFalse(ns["_public_spam_limits"](
            RateCursor([59, 30]), "owner", "sender@example.com"))
        self.assertFalse(ns["_public_spam_limits"](
            RateCursor([60]), "owner", "sender@example.com"))


if __name__ == "__main__":
    unittest.main()
