"""Hostinger IMAP UIDVALIDITY regression and privacy-safe phase diagnostics."""
from __future__ import annotations
import ast
from pathlib import Path
import unittest

SRC=Path(__file__).resolve().parents[1]/"trust_voice_circle_email_apex_gateway.py"


def extract(name):
    source=ast.parse(SRC.read_text(encoding="utf-8"))
    funcs=[x for x in source.body if isinstance(x,ast.FunctionDef) and x.name==name]
    assert len(funcs)==1
    ctx={}
    exec(compile(ast.Module(body=funcs,type_ignores=[]),str(SRC),"exec"),ctx)
    return ctx[name]


class IMAP:
    def __init__(self,data,status="UIDVALIDITY"):
        self.data=data
        self.status=status
    def response(self,name):
        assert name=="UIDVALIDITY"
        return self.status,self.data


class HostingerDiagnosticTests(unittest.TestCase):
    def test_uidvalidity_bytes(self):
        self.assertEqual(extract("_uidvalidity")(IMAP([b"1771286388"])),"1771286388")

    def test_uidvalidity_string(self):
        self.assertEqual(extract("_uidvalidity")(IMAP(["1771286388"])),"1771286388")

    def test_uidvalidity_malformed_is_rejected(self):
        fn=extract("_uidvalidity")
        for data in (None,[],[None],[b""],[b"12X34"],["12X34"],
                     [b"177",b"128"],[b"\xff"],[{"uid":123}],["12\u0661"]):
            with self.subTest(data=data),self.assertRaises((RuntimeError,UnicodeDecodeError)):
                fn(IMAP(data))
        with self.assertRaises(RuntimeError):
            fn(IMAP([b"1771286388"],"NO"))

    def test_failed_stage_logs_have_no_sensitive_data(self):
        source=SRC.read_text(encoding="utf8")
        expected=("schema","advisory_lock","imap_connect","imap_login",
                  "imap_select","uid_validity","cursor","uid_search","fetch_size",
                  "fetch_raw","parse_catchall","store_received","checkpoint")
        for phase in expected:
            self.assertIn('_stage("'+phase+'")',source)
        self.assertIn("Apex mail batch failed stage=%s error=%s",source)
        self.assertNotIn("log.exception(",source)
        self.assertNotIn("log.error(str(exc))",source)
        self.assertNotIn("x.isdecimal()",source)


if __name__=="__main__":
    unittest.main()
