"""No-network tests for opt-in public VeriFYD Circle handle mailboxes."""
from __future__ import annotations

import ast
import os
import pathlib
import re
import uuid
from email.utils import parseaddr, getaddresses
from unittest import TestCase, main
from unittest.mock import patch

ROOT = pathlib.Path(__file__).resolve().parents[1]
PUBLIC = ROOT / "trust_voice_circle_email_public.py"
INBOUND = ROOT / "trust_voice_circle_email_inbound.py"
MAIL = ROOT / "trust_voice_circle_email.py"
DOMAIN = "reply.vfvid.com"


def helpers():
    tree = ast.parse(PUBLIC.read_text(encoding="utf-8"))
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    imports = [node for node in tree.body
               if isinstance(node, ast.ImportFrom) and node.module in {
                   "__future__",
               }]
    env = {
        "os": os, "re": re, "uuid": uuid, "parseaddr": parseaddr, "getaddresses": getaddresses,
        "_HANDLE": re.compile(r"^[a-z][a-z0-9_]{2,23}$"),
        "_MAX_OWNER_DAY": 60, "_MAX_SENDER_DAY": 8,
    }
    exec(compile(ast.Module(body=imports + nodes, type_ignores=[]), str(PUBLIC), "exec"), env)
    return env


class PublicHandleTests(TestCase):
    def config(self, handles="roberttest22"):
        return patch.dict(os.environ, {
            "VERIFYD_CIRCLE_EMAIL_PUBLIC_HANDLE_ENABLED": "true",
            "VERIFYD_CIRCLE_EMAIL_PUBLIC_HANDLE_ALLOWLIST": handles
        }, clear=True)

    def test_disabled_by_default(self):
        h = helpers()
        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(h["public_address_for_handle"]("roberttest22", DOMAIN))
            self.assertIsNone(h["public_recipient_from_to"](
                ["roberttest22@reply.vfvid.com"], DOMAIN))

    def test_allowlisted_alias_and_mixed_case(self):
        h = helpers()
        with self.config():
            self.assertEqual(
                h["public_address_for_handle"]("@RobertTest22", DOMAIN),
                "roberttest22@reply.vfvid.com"
            )
            self.assertEqual(
                h["public_recipient_from_to"](["RobertTest22@reply.vfvid.com"], DOMAIN),
                "roberttest22"
            )
            self.assertIsNone(h["public_address_for_handle"]("otherperson", DOMAIN))

    def test_domain_and_multiple_recipient_are_not_auto_routed(self):
        h = helpers()
        with self.config():
            for addrs in (
                ["roberttest22@vfvid.com"],
                ["roberttest22@otherdomain.com"],
                ["roberttest22+aaa@reply.vfvid.com"],
                ["roberttest22@reply.vfvid.com", "other@reply.vfvid.com"],
                ["roberttest22@reply.vfvid.com, other@reply.vfvid.com"]
            ):
                with self.subTest(addrs=addrs):
                    self.assertIsNone(h["public_recipient_from_to"](addrs, DOMAIN))

    def test_no_wildcard_or_unapproved_handle(self):
        h = helpers()
        with self.config("*, roberttest22, admin, invalid.dot, support"):
            self.assertEqual(h["_allowed_handles"](), frozenset({"roberttest22"}))
        with self.config("*"):
            self.assertIsNone(h["public_address_for_handle"]("roberttest22", DOMAIN))

    def test_sender_and_owner_daily_limits(self):
        h = helpers()
        class Cursor:
            def __init__(self, values):
                self.values = iter(values)
                self.params = []
            def execute(self, sql, params):
                self.params.append(params)
            def fetchone(self):
                return {"n": next(self.values)}
        c = Cursor([59,7])
        self.assertTrue(h["_public_spam_limits"](c, "owner1", "friend@gmail.com"))
        self.assertEqual(c.params[1], ("owner1", "external:friend@gmail.com"))
        c = Cursor([60])
        self.assertFalse(h["_public_spam_limits"](c,"owner1","friend@gmail.com"))
        c = Cursor([10,8])
        self.assertFalse(h["_public_spam_limits"](c,"owner1","friend@gmail.com"))

    def test_sending_short_alias_does_not_remove_legacy_routing(self):
        mail = MAIL.read_text(encoding="utf-8")
        inbound = INBOUND.read_text(encoding="utf-8")
        self.assertIn("reply_address=create_reply_route(db,", mail)
        self.assertIn("reply_to=public_reply or reply_address", mail)
        self.assertIn("_HEX_TOKEN.fullmatch(address)", inbound)
        self.assertIn("_HANDLE_TOKEN.fullmatch(address)", inbound)
        self.assertIn("public_recipient_from_to(received.get('to')", inbound)

    def test_new_inbound_is_idempotent_and_keeps_attachments_withheld(self):
        src = PUBLIC.read_text(encoding="utf-8")
        self.assertIn("WHERE provider_email_id=%s", src)
        self.assertIn("SELECT pg_advisory_xact_lock(hashtext(%s))", src)
        self.assertIn("attachments withheld", src.lower())
        self.assertIn("email_verified=TRUE AND status='active'", src)
        self.assertIn("Unsolicited public-handle email; sender unverified.", src)
        self.assertIn("received_external", src)


if __name__ == "__main__":
    main()
