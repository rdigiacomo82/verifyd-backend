"""Legacy-compatible secure human-readable email reply alias tests.

No live database, SMTP, email API keys or network required.
"""
from __future__ import annotations
import ast
import hashlib
import os
import pathlib
import re
import secrets
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
INBOUND = ROOT / "trust_voice_circle_email_inbound.py"
OUTBOUND = ROOT / "trust_voice_circle_email.py"


class FakeHTTPException(Exception):
    def __init__(self, status_code, detail=None):
        self.status_code = status_code
        self.detail = detail


class FakeCursor:
    def __init__(self):
        self.calls = []
    def execute(self, sql, values):
        self.calls.append((sql,values))


class FakeDb:
    def __init__(self):
        self.cur = FakeCursor()
    def cursor(self):
        return self.cur


def isolated():
    tree = ast.parse(INBOUND.read_text(encoding="utf-8"))
    nodes = [node for node in tree.body if
             isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
             and node.name in {"_candidate_routes", "create_reply_route"}]
    assert len(nodes)==2
    for node in nodes:
        node.decorator_list=[]
        node.returns=None
        for a in (*node.args.args, *node.args.kwonlyargs):
            a.annotation=None
    source=INBOUND.read_text(encoding="utf-8")
    assert "_HANDLE_TOKEN" in source and "_HEX_TOKEN" in source
    env=dict(
        re=re,os=os,hashlib=hashlib,secrets=secrets,
        _HEX_TOKEN=re.compile(r"^reply\\+([a-f0-9]{40})@([a-z0-9.-]+)$",re.I),
        _HANDLE_TOKEN=re.compile(r"^([a-z][a-z0-9_]{2,23})\\+([a-f0-9]{32})@([a-z0-9.-]+)$",re.I),
        reply_domain=lambda:"reply.vfvid.com",inbound_configured=lambda:True,
        HTTPException=FakeHTTPException,parseaddr=__import__("email.utils",fromlist=["parseaddr"]).parseaddr,
    )
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(INBOUND),"exec"),env)
    return env


class AliasRegressions(unittest.TestCase):
    def test_new_alias_includes_lowercase_handle_and_128_bit_token(self):
        env=isolated();db=FakeDb()
        addr=env["create_reply_route"](db, owner_id="owner",
            sent_mail_id="mail",thread_id="thread",external_recipient="Friend@Example.com",
            stamp="now",sender_handle="@RobertTest22")
        self.assertRegex(addr,r"^roberttest22\\+[0-9a-f]{32}@reply\\.vfvid\\.com$")
        self.assertEqual(env["_candidate_routes"]([addr],"reply.vfvid.com"),
                         [addr.split("+")[1].split("@")[0]])
        self.assertEqual(db.cur.calls[0][1][4],"friend@example.com")
        self.assertEqual(db.cur.calls[0][1][0],hashlib.sha256(
            addr.split("+")[1].split("@")[0].encode()).hexdigest())
        self.assertLessEqual(len(addr.split("@")[0]),64)

    def test_old_40_character_reply_addresses_still_work(self):
        env=isolated()
        token="a"*40
        addr=f"reply+{token}@reply.vfvid.com"
        self.assertEqual(env["_candidate_routes"]([addr],"reply.vfvid.com"),[token])

    def test_wrong_domain_unknown_name_malformed_token_rejected(self):
        env=isolated()
        invalid=[
            "roberttest22+"+"f"*32+"@external.com",
            "roberttest22@reply.vfvid.com",
            "roberttest22+"+"f"*40+"@reply.vfvid.com",
            "reply+"+"f"*32+"@reply.vfvid.com",
            "roberttest22+"+"g"*32+"@reply.vfvid.com",
        ]
        self.assertEqual(env["_candidate_routes"](invalid,"reply.vfvid.com"),[])

    def test_invalid_handle_not_silently_mapped(self):
        env=isolated()
        for bad in ("", "a", "contains.dot", "123bad", "a"*25):
            with self.subTest(bad=bad):
                with self.assertRaises(FakeHTTPException):
                    env["create_reply_route"](FakeDb(), owner_id="owner",
                        sent_mail_id="mail",thread_id="thread",
                        external_recipient="x@example.com",stamp="now",
                        sender_handle=bad)

    def test_backend_passes_authenticated_sender_handle(self):
        source=OUTBOUND.read_text(encoding="utf-8")
        self.assertIn("sender_handle=str(actor.get('handle_lower')",source)
        self.assertIn("create_reply_route(db,owner_id=uid",source)


if __name__ == "__main__":
    unittest.main()
