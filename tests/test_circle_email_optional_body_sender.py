"""Offline tests for flexible Circle email and authenticated sender display."""
from __future__ import annotations

import ast
import os
import pathlib
import re
import unittest
from email.utils import parseaddr
from unittest.mock import patch

FILE = pathlib.Path(__file__).resolve().parents[1] / "trust_voice_circle_email.py"


class FakeHTTPException(Exception):
    def __init__(self, status, detail=None):
        self.status = status
        self.detail = detail


def load_helpers():
    source=FILE.read_text(encoding="utf-8")
    nodes=[
        node for node in ast.parse(source).body
        if isinstance(node,ast.FunctionDef) and node.name in (
            "fail","_has_mail_content","_personal_from_header",
        )
    ]
    assert len(nodes)==3
    namespace=dict(
        re=re,os=os,parseaddr=parseaddr,HTTPException=FakeHTTPException,
        is_valid_email=lambda v: bool(re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+",v)),
    )
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(FILE),"exec"),namespace)
    return namespace


class FlexibleMailTests(unittest.TestCase):
    def test_subject_only_accepted(self):
        h=load_helpers()
        self.assertTrue(h["_has_mail_content"]("Meeting details","",[]))
    def test_attachment_only_accepted(self):
        h=load_helpers()
        self.assertTrue(h["_has_mail_content"]("","",["tvcfile_a"]))
    def test_body_only_accepted(self):
        h=load_helpers()
        self.assertTrue(h["_has_mail_content"]("","Hello",[]))
    def test_fully_blank_rejected(self):
        h=load_helpers()
        self.assertFalse(h["_has_mail_content"]("  ","   ",[]))
    def test_sender_name_is_verified_handle(self):
        h=load_helpers()
        with patch.dict(os.environ,{},clear=True):
            self.assertEqual(
                h["_personal_from_header"]({"handle":"RobertTest22"},True),
                '"@RobertTest22" <mail@vfvid.com>')
    def test_sender_from_config_keeps_verified_address_not_generic_label(self):
        h=load_helpers()
        with patch.dict(os.environ,{"VERIFYD_CIRCLE_EMAIL_FROM":"VeriFYD Circle <mail@vfvid.com>"},clear=True):
            self.assertEqual(
                h["_personal_from_header"]({"handle":"AnnSLP"},True),
                '"@AnnSLP" <mail@vfvid.com>')
    def test_invalid_handle_cannot_be_sender(self):
        h=load_helpers()
        with patch.dict(os.environ,{},clear=True):
            with self.assertRaises(FakeHTTPException):
                h["_personal_from_header"]({"handle":"Bad\\r\\nBcc:victim"},True)
    def test_model_and_wire_contract(self):
        src=FILE.read_text(encoding="utf-8")
        self.assertIn("body: str=Field(default='',max_length=12000)",src)
        self.assertIn("if not _has_mail_content(supplied_subject,body,payload.file_ids):",src)
        self.assertIn("from_address=_personal_from_header(actor,inbound_reply_active)",src)
        self.assertNotIn("From VeriFYD Circle user @",src)
        self.assertIn("if attachment_payload:",src)
        self.assertIn("send_params['attachments']=attachment_payload",src)


if __name__=="__main__":
    unittest.main()
