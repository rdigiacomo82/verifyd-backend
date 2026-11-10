"""Check that Web Push urgency is raised only for live incoming calls."""
import ast
import json
import re
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
                if kind.startswith("call_"):
                    self.assertEqual(log.info.call_args.args[1:], (
                        call_id, kind, "high" if headers else "normal", 2, 0,
                    ))
                else:
                    log.info.assert_not_called()
