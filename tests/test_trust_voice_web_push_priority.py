"""Check that Web Push urgency is raised only for live incoming calls."""
import ast
import hashlib
import hmac
import json
import re
import threading
import uuid
from collections import OrderedDict
import unittest
from pathlib import Path
from unittest.mock import Mock


class WebPushPriorityTests(unittest.TestCase):
    def test_call_urgency_preserves_ttl_and_other_notification_defaults(self):
        source = Path(__file__).resolve().parents[1] / "trust_voice_push.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        node = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == "notify_identity_push")
        module = ast.parse("from __future__ import annotations")
        module.body.append(node)
        push = Mock()
        log = Mock()
        env = {
            "json": json, "re": re, "log": log, "webpush": push,
            "_enabled": lambda: True, "_configured": lambda: True,
            "_active_subscriptions": lambda _identity: [
                {"subscription_id": "test-1", "endpoint": "https://example.com/1"},
                {"subscription_id": "test-2", "endpoint": "https://example.com/2"},
            ],
            "preferred_locale_for_identity": lambda _identity: "en-US",
            "localize_notification": lambda _locale, _kind, title, body: (title, body),
            "_now_iso": lambda: "2026-11-10T00:00:00Z",
            "_vapid_private_key": lambda: "test-only",
            "_vapid_subject": lambda: "mailto:test@example.com",
            "_mark_subscription_success": Mock(),
            "_issue_call_receipt": lambda _kind, _call_id: "test-receipt",
        }
        exec(compile(module, str(source), "exec"), env)
        call_id = "tvcall_" + "a" * 32
        for kind, ttl, headers in (
            ("call_incoming", 30, {"Urgency": "high"}),
            ("call_missed", 3600, None),
            ("push_test", 120, None),
        ):
            with self.subTest(kind=kind):
                push.reset_mock()
                log.reset_mock()
                result = env["notify_identity_push"](
                    "callee", notification_type=kind, title="test", body="test",
                    data={"call_id": call_id}, ttl=ttl,
                )
                self.assertEqual(result["sent"], 2)
                self.assertEqual(push.call_count, 2)
                for sent in push.call_args_list:
                    self.assertEqual(sent.kwargs["headers"], headers)
                    self.assertEqual(sent.kwargs["ttl"], ttl)
                    payload = json.loads(sent.kwargs["data"])
                    self.assertEqual(payload.get("call_receipt"), "test-receipt" if kind.startswith("call_") else None)
                if kind.startswith("call_"):
                    self.assertEqual(log.info.call_args.args[1:], (
                        call_id, kind, "high" if headers else "normal", 2, 0,
                    ))
                else:
                    log.info.assert_not_called()

    def test_worker_receipts_are_signed_expiring_and_deduplicated(self):
        source = Path(__file__).resolve().parents[1] / "trust_voice_push.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        start = next(i for i, node in enumerate(tree.body) if isinstance(node, ast.Assign)
                     and isinstance(node.targets[0], ast.Name)
                     and node.targets[0].id == "_CALL_RECEIPT_PATTERN")
        end = next(i for i, node in enumerate(tree.body) if isinstance(node, ast.AsyncFunctionDef)
                   and node.name == "receive_call_receipt")
        module = ast.parse("from __future__ import annotations")
        module.body.extend(tree.body[start:end])
        clock = Mock()
        clock.time.return_value = 1794330000
        log = Mock()
        env = {
            "re": re, "threading": threading, "OrderedDict": OrderedDict,
            "time": clock, "uuid": uuid, "hashlib": hashlib, "hmac": hmac,
            "_secret": lambda: b"test-only-signing-key", "log": log,
        }
        exec(compile(module, str(source), "exec"), env)
        call_id = "tvcall_" + "a" * 32
        issue, record = env["_issue_call_receipt"], env["_record_call_receipt"]
        receipt = issue("call_incoming", call_id)
        record(receipt.replace("call_incoming", "call_missed"), "display_accepted")
        record(receipt, "unknown")
        log.info.assert_not_called()
        record(receipt, "fallback_accepted")
        record(receipt, "display_accepted")
        log.info.assert_called_once()
        self.assertEqual(log.info.call_args.args[1:], (call_id, "call_incoming", "fallback_accepted"))
        unused_receipt = issue("call_missed", call_id)
        clock.time.return_value += 901
        record(unused_receipt, "display_accepted")
        self.assertEqual(log.info.call_count, 1)
