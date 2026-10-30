"""Safety tests for Hostinger all-member catch-all ingress, no live accounts."""
import ast
import email
import os
import pathlib
import re
import unittest
from email import policy
from email.parser import BytesParser

ROOT = pathlib.Path(__file__).resolve().parents[1]
FILE = ROOT / "trust_voice_circle_email_apex_gateway.py"


def load():
    tree = ast.parse(FILE.read_text(encoding="utf-8"))
    names = ("_trusted_hosts", "original_envelope_recipient", "parse_catchall")
    functions=[node for node in tree.body
        if isinstance(node,ast.FunctionDef) and node.name in names]
    assert len(functions)==3
    mod = __import__("types").SimpleNamespace()
    ns = {
        "os": os, "re": re,"email":email, "policy":policy,
        "BytesParser":BytesParser,
        "parseaddr":__import__("email.utils",fromlist=["parseaddr"]).parseaddr,
        "_APEX":"vfvid.com",
        "_MAX_RAW_BYTES":18*1024*1024,
        "_LOCALPART":re.compile(r"^[a-z][a-z0-9_]{2,23}$"),
        "_TRUSTED_BY":re.compile(r"\bby\s+([a-z0-9][a-z0-9.-]+)",re.I),
        "_FOR_ADDRESS":re.compile(
            r"\bfor\s+<?([a-z][a-z0-9_]{2,23}@vfvid\.com)>?(?=[\s;,]|$)", re.I)
    }
    for a,b in list(ns.items()):
        if isinstance(b,type(re.compile(""))):
            # File is literal untrusted source under test; compile exactly as declared.
            pass
    exec(compile(ast.Module(body=functions,type_ignores=[]),str(FILE),"exec"),ns)
    return ns


def mail(*, top, to="attacker@vfvid.com", lower=None, body="Test email"):
    source = f"Received: {top}\n"
    if lower: source+=f"Received: {lower}\n"
    source+=f"From: sender@gmail.com\nTo: {to}\nSubject: Test\n"
    source+="MIME-Version: 1.0\nContent-Type: text/plain; charset=utf-8\n\n"+body
    return source.encode("utf-8")


class ApexGatewayTests(unittest.TestCase):
    def setUp(self):
        self.h=load()
        self.hosts=frozenset({"mx.test.hostinger.com"})

    def test_trusted_outermost_received_controls_routing(self):
        raw=mail(top="from gmail.com by mx.test.hostinger.com with ESMTP for <annslp@vfvid.com>; Thu, 29 Oct 2026 15:00:00 +0000",
            to="roberttest22@vfvid.com")
        handle,data=self.h["parse_catchall"](raw,self.hosts)
        self.assertEqual(handle,"annslp")
        self.assertEqual(data["to"],["annslp@vfvid.com"])
        self.assertEqual(data["text"],"Test email")

    def test_spoofed_to_header_cannot_select_recipient(self):
        raw=mail(top="from gmail.com by unrelated.mailhost.com with ESMTP for <attacker@vfvid.com>;",
            to="roberttest22@vfvid.com")
        self.assertEqual(self.h["parse_catchall"](raw,self.hosts),(None,None))

    def test_spoofed_older_received_hop_not_accepted(self):
        raw=mail(top="from unknown.example by other.mta.net for <attacker@vfvid.com>;",
            lower="from gmail.com by mx.test.hostinger.com for <annslp@vfvid.com>;",
            to="annslp@vfvid.com")
        self.assertEqual(self.h["parse_catchall"](raw,self.hosts),(None,None))

    def test_user_controlled_original_to_header_not_accepted(self):
        raw=mail(top="from gmail.com by mx.test.hostinger.com with ESMTP;")
        raw=raw.replace(b"To: ",b"X-Original-To: annslp@vfvid.com\nDelivered-To: annslp@vfvid.com\nTo: ",1)
        self.assertEqual(self.h["parse_catchall"](raw,self.hosts),(None,None))

    def test_trusted_hop_without_for_is_quarantined(self):
        raw=mail(top="from gmail.com by mx.test.hostinger.com with ESMTP;")
        self.assertEqual(self.h["parse_catchall"](raw,self.hosts),(None,None))

    def test_untrusted_server_cannot_be_suffix_lookalike(self):
        raw=mail(top="from gmail.com by mx.test.hostinger.com.attacker.net for <annslp@vfvid.com>;")
        self.assertEqual(self.h["parse_catchall"](raw,self.hosts),(None,None))

    def test_to_multiple_or_malformed_recipients_quarantined(self):
        top="from gmail.com by mx.test.hostinger.com for <annslp@vfvid.com> and for <roberttest22@vfvid.com>;"
        self.assertEqual(self.h["parse_catchall"](mail(top=top),self.hosts),(None,None))
        top="from gmail.com by mx.test.hostinger.com for <support@vfvid.com>;"
        handle,record=self.h["parse_catchall"](mail(top=top),self.hosts)
        self.assertEqual(handle,"support")
        # Database identity/operational-role gate is in public importer, not parser.
        src=(ROOT/"trust_voice_circle_email_public.py").read_text(encoding="utf-8")
        self.assertIn("public_address_for_handle(handle, reply_domain())",src)

    def test_body_size_limit_and_attachment_withheld(self):
        top="from gmail.com by mx.test.hostinger.com for <annslp@vfvid.com>;"
        self.assertEqual(self.h["parse_catchall"](b"",self.hosts),(None,None))
        self.assertEqual(self.h["parse_catchall"](b"x"*(18*1024*1024+1),self.hosts),(None,None))
        raw=mail(top=top)
        self.assertEqual(self.h["parse_catchall"](raw,self.hosts)[1]["attachments"],[])

    def test_cli_requires_global_gate_and_does_not_modify_workers(self):
        src=FILE.read_text(encoding="utf-8")
        self.assertIn('VERIFYD_CIRCLE_EMAIL_APEX_GATEWAY_ENABLED',src)
        self.assertIn('public_handles_enabled()',src)
        self.assertIn('imaplib.IMAP4_SSL(_IMAP_HOST, 993',src)
        self.assertIn('pg_try_advisory_xact_lock',src)
        self.assertIn('UIDVALIDITY',src)
        self.assertIn('BODY.PEEK[]',src)
        self.assertIn('provider_id = "hostinger-"',src)
        self.assertIn('store_public_received(received,provider_id,handle=handle)',src)
        self.assertNotIn('os.system(',src)
        self.assertNotIn('eval(',src)


if __name__=="__main__": unittest.main()
