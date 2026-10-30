"""Isolated Resend receiving regression tests; no network or database access."""
from __future__ import annotations

import ast
import io
import json
import os
import pathlib
import re
import types
import unittest
import urllib.error
import urllib.request
from unittest.mock import patch

ROOT = pathlib.Path(__file__).resolve().parents[1]
SOURCE = ROOT / "trust_voice_circle_email_inbound.py"
PROVIDER_ID = "aaf8f78e-e2b4-4903-8423-76de7bffed2b"


class FakeHTTPException(Exception):
    def __init__(self, status_code, detail=None):
        self.status_code = status_code
        self.detail = detail


class SafeLog:
    def __init__(self):
        self.warnings = []

    def warning(self, message, *args, **kwargs):
        self.warnings.append((message, args))


def load_helpers():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    names = {
        "inbound_enabled", "reply_domain", "_inbound_secret",
        "_receiving_api_key", "inbound_configured", "_fetch_received_email",
    }
    nodes = [node for node in tree.body
             if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
             and node.name in names]
    assert len(nodes) == len(names), "expected inbound helpers not found"
    log = SafeLog()
    env = dict(
        os=os, re=re, urllib=urllib, json=json, log=log,
        HTTPException=FakeHTTPException,
        _PROVIDER_ID=re.compile(r"^[a-zA-Z0-9-]{15,90}$"),
    )
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), "exec"), env)
    return env, log


class ReceivingTests(unittest.TestCase):
    def test_dedicated_key_preferred_without_changing_send_key(self):
        h, _ = load_helpers()
        with patch.dict(os.environ, {
            "RESEND_API_KEY": "sending-only-test-key",
            "VERIFYD_CIRCLE_EMAIL_RECEIVING_API_KEY": "read-test-key",
            "VERIFYD_CIRCLE_EMAIL_INBOUND_WEBHOOK_SECRET": "whsec_dummy",
            "VERIFYD_CIRCLE_EMAIL_INBOUND_ENABLED": "true",
        }, clear=True):
            self.assertEqual(h["_receiving_api_key"](), "read-test-key")
            self.assertEqual(os.environ["RESEND_API_KEY"], "sending-only-test-key")
            self.assertTrue(h["inbound_configured"]())

    def test_fallback_read_key_keeps_existing_config_valid(self):
        h, _ = load_helpers()
        with patch.dict(os.environ, {
            "RESEND_API_KEY": "shared-full-access-key",
            "VERIFYD_CIRCLE_EMAIL_INBOUND_WEBHOOK_SECRET": "whsec_dummy",
            "VERIFYD_CIRCLE_EMAIL_INBOUND_ENABLED": "true",
        }, clear=True):
            self.assertEqual(h["_receiving_api_key"](), "shared-full-access-key")
            self.assertTrue(h["inbound_configured"]())

    def test_missing_keys_report_not_configured(self):
        h, _ = load_helpers()
        with patch.dict(os.environ, {
            "VERIFYD_CIRCLE_EMAIL_INBOUND_WEBHOOK_SECRET": "whsec_dummy",
            "VERIFYD_CIRCLE_EMAIL_INBOUND_ENABLED": "true",
        }, clear=True):
            self.assertFalse(h["inbound_configured"]())

    def test_fetch_uses_dedicated_key_and_explicit_user_agent(self):
        h, _ = load_helpers()
        payload = {"id": PROVIDER_ID, "to": ["reply+token@reply.vfvid.com"],
                   "from": "someone@example.com", "text": "hello"}
        requests = []
        def fake_open(req, timeout):
            requests.append((req, timeout))
            return io.BytesIO(json.dumps(payload).encode("utf-8"))
        with patch.dict(os.environ, {
            "RESEND_API_KEY": "send-key",
            "VERIFYD_CIRCLE_EMAIL_RECEIVING_API_KEY": "read-key",
        }, clear=True), patch("urllib.request.urlopen", side_effect=fake_open):
            response = h["_fetch_received_email"](PROVIDER_ID)
        self.assertEqual(response["id"], PROVIDER_ID)
        self.assertEqual(len(requests), 1)
        req, timeout = requests[0]
        self.assertEqual(req.get_method(), "GET")
        self.assertEqual(timeout, 12)
        self.assertTrue(req.full_url.endswith("/emails/receiving/" + PROVIDER_ID))
        self.assertEqual(req.get_header("Authorization"), "Bearer read-key")
        self.assertIn("VeriFYD-Circle", req.get_header("User-agent"))

    def test_http_401_is_retryable_and_logs_only_numeric_status(self):
        h, log = load_helpers()
        error = urllib.error.HTTPError(
            "https://api.resend.com/emails/receiving/id", 401,
            "Unauthorized", {}, None,
        )
        with patch.dict(os.environ, {"RESEND_API_KEY": "private-test-key"}, clear=True), patch(
            "urllib.request.urlopen", side_effect=error
        ), self.assertRaises(FakeHTTPException) as raised:
            h["_fetch_received_email"](PROVIDER_ID)
        self.assertEqual(raised.exception.status_code, 503)
        self.assertEqual(raised.exception.detail["error"], "inbound_provider_retry")
        self.assertEqual(log.warnings[-1][1], (401,))
        self.assertNotIn("private-test-key", str(log.warnings))

    def test_wrong_provider_id_is_rejected_before_network(self):
        h, _ = load_helpers()
        with self.assertRaises(FakeHTTPException) as raised:
            h["_fetch_received_email"]("invalid/")
        self.assertEqual(raised.exception.status_code, 400)

    def test_response_id_mismatch_is_rejected(self):
        h, _ = load_helpers()
        with patch.dict(os.environ, {"RESEND_API_KEY": "test-key"}, clear=True), patch(
            "urllib.request.urlopen",
            return_value=io.BytesIO(b'{"id":"another-id"}'),
        ), self.assertRaises(FakeHTTPException) as raised:
            h["_fetch_received_email"](PROVIDER_ID)
        self.assertEqual(raised.exception.status_code, 503)


if __name__ == "__main__":
    unittest.main()
