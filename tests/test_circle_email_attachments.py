"""Offline contract tests for VeriFYD Circle mail attachment transport."""
from __future__ import annotations

import ast
import base64
import hashlib
import io
import os
from pathlib import Path
import re
import tempfile
import unittest
import urllib.parse
import urllib.request
import uuid

ROOT = Path(__file__).resolve().parents[1]
TRANS = ROOT / "trust_voice_circle_email_transport.py"
EMAIL = ROOT / "trust_voice_circle_email.py"
INBOUND = ROOT / "trust_voice_circle_email_inbound.py"


class FakeHTTP(Exception):
    def __init__(self, status_code, detail=None):
        self.status_code = status_code
        self.detail = detail


class FakeLog:
    def warning(self, *_args, **_kwargs):
        pass


class FakeBody:
    def __init__(self, data):
        self.stream = io.BytesIO(data)

    def read(self, n=-1):
        return self.stream.read(n)

    def close(self):
        self.stream.close()


class FakeR2:
    def __init__(self):
        self.storage = {}
        self.deleted = []

    def get_object(self, Bucket, Key):
        return {"Body": FakeBody(self.storage[Key])}

    def put_object(self, Bucket, Key, Body, **kwargs):
        self.storage[Key] = Body

    def delete_object(self, Bucket, Key):
        self.deleted.append(Key)
        self.storage.pop(Key, None)


def load_transport(client=None):
    tree = ast.parse(TRANS.read_text(encoding="utf-8"))
    funcs = [node for node in tree.body
             if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
    for node in funcs:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            node.decorator_list = []
            node.returns = None
            for a in (*node.args.args, *node.args.kwonlyargs):
                a.annotation = None
        if isinstance(node, ast.ClassDef):
            node.decorator_list = []
    # No third-party imports, secrets, network, storage, or database in tests.
    env = dict(base64=base64, hashlib=hashlib, os=os, re=re, tempfile=tempfile,
               urllib=urllib, uuid=uuid, json=__import__("json"),
               HTTPException=FakeHTTP, log=FakeLog(), MAX_FILE_BYTES=8*1024*1024,
               MAX_TOTAL_BYTES=16*1024*1024, MAX_FILES=3,
               ALLOWED_EXTENSIONS=frozenset({".txt", ".pdf", ".png"}),
               _RESEND_ID=re.compile(r"^[A-Za-z0-9-]{15,90}$"),
               _CDN_HOST="inbound-cdn.resend.com",
               TRUST_VOICE_BUCKET="test", _get_attachment_client=lambda: client,
               validate_attachment_content=lambda *a, **k: {"content_type": "text/plain"},
               AttachmentContentError=ValueError)
    exec(compile(ast.Module(body=funcs, type_ignores=[]), str(TRANS), "exec"), env)
    return env


class AttachmentTests(unittest.TestCase):
    def test_outbound_base64_and_integrity(self):
        client = FakeR2()
        client.storage["a"] = b"hello"
        env = load_transport(client)
        rows = [dict(byte_size=5,storage_key="a",sha256=hashlib.sha256(b"hello").hexdigest(),
                     filename="hello.txt",content_type="text/plain")]
        payload = env["prepare_outbound_attachments"](rows)
        self.assertEqual(len(payload),1)
        self.assertEqual(payload[0]["filename"], "hello.txt")
        self.assertEqual(base64.b64decode(payload[0]["content"]),b"hello")
        client.storage["a"] = b"other"
        with self.assertRaises(FakeHTTP):
            env["prepare_outbound_attachments"](rows)

    def test_outbound_limits_three_files_and_sixteen_mb(self):
        env = load_transport(FakeR2())
        rows = [dict(byte_size=6*1024*1024) for _ in range(3)]
        with self.assertRaises(FakeHTTP) as ctx:
            env["prepare_outbound_attachments"](rows)
        self.assertEqual(ctx.exception.status_code,413)
        with self.assertRaises(FakeHTTP):
            env["prepare_outbound_attachments"]([dict(byte_size=1)]*4)

    def test_strict_cdn_url_and_path(self):
        env = load_transport()
        email_id = "aaf8f78e-e2b4-4903-8423-76de7bffed2b"
        att_id = "2a0c9ce0-3112-4728-976e-47ddcd16a318"
        good = ("https://inbound-cdn.resend.com/" + email_id +
                "/attachments/" + att_id + "?signature=abc")
        env["_verify_cdn_url"](good,email_id,att_id)
        for evil in (
            good.replace("inbound-cdn.resend.com","evil.example"),
            good.replace("https:","http:"),
            good.replace(att_id,"other-id"),
            good.replace("inbound-cdn.resend.com","inbound-cdn.resend.com.evil.test"),
            good.replace("inbound-cdn.resend.com","inbound-cdn.resend.com:444"),
            good.replace("https://","https://example@"),
            good.split("?")[0],
        ):
            with self.subTest(evil=evil), self.assertRaises(ValueError):
                env["_verify_cdn_url"](evil,email_id,att_id)

    def test_inbound_oversized_and_unsupported_files_omitted(self):
        env = load_transport()
        eid = "aaf8f78e-e2b4-4903-8423-76de7bffed2b"
        env["_received_list"] = lambda provider_id: [
            {"filename":"bad.exe","size":10,"id":"2a0c9ce0-3112-4728-976e-47ddcd16a318"},
            {"filename":"large.txt","size":999999999,"id":"3a0c9ce0-3112-4728-976e-47ddcd16a319"},
        ]
        received, omitted = env["fetch_inbound_attachment_files"](eid, [{"id":"2a0c9ce0-3112-4728-976e-47ddcd16a318"}])
        self.assertEqual(received, [])
        self.assertEqual(omitted, 2)

    def test_inbound_persistence_marks_external_not_scanned(self):
        env_client = FakeR2()
        env = load_transport(env_client)
        params = []
        class Cursor:
            def execute(self, sql, args):
                params.append((sql,args))
        class Db:
            def cursor(self):
                return Cursor()
        keys = env["persist_inbound_attachments"](Db(),
            files=[dict(filename="image.png",bytes=b"hello",content_type="image/png")],
            owner_id="alice",mail_id="message",stamp="now")
        self.assertEqual(len(keys),1)
        self.assertIn(keys[0],env_client.storage)
        self.assertIn("'external','not_scanned'",params[0][0])

    def test_external_download_requires_acknowledgement(self):
        source = EMAIL.read_text(encoding="utf-8")
        self.assertIn("attachment_confirmation_required", source)
        self.assertIn("acknowledge_unscanned:bool=Query(default=False)", source)
        self.assertIn("Content-Disposition", source)
        self.assertIn("X-Content-Type-Options", source)

    def test_inbound_attach_persistence_runs_before_event_idempotency(self):
        source = INBOUND.read_text(encoding="utf-8")
        self.assertLess(source.index("persist_inbound_attachments(db,"),
                        source.index("INSERT INTO verifyd_circle_email_received_events"))
        self.assertIn("fetch_inbound_attachment_files(provider_id, advertised)",source)

    def test_external_send_no_longer_rejects_all_files(self):
        source = EMAIL.read_text(encoding="utf-8")
        self.assertNotIn("external_files_not_ready",source)
        self.assertIn("prepare_outbound_attachments(rows)",source)
        self.assertIn("send_params['attachments']=attachment_payload",source)
        self.assertIn("MAX_EMAIL_TOTAL_BYTES",source)


if __name__ == "__main__":
    unittest.main()
