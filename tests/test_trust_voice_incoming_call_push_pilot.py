"""Fail-closed tests for staged incoming-call browser push and ringing replay."""
from __future__ import annotations

import ast
import asyncio
import logging
import os
from pathlib import Path
import sys
import time
import types
import unittest
from unittest.mock import patch

SRC = Path(__file__).resolve().parents[1] / "trust_voice_signaling.py"
FN_NAMES = {
    "_incoming_call_push_pilot_enabled",
    "_incoming_call_push_subscription_ready",
    "_send_incoming_call_pilot_push",
    "_replay_pilot_ringing_calls",
}


def harness():
    tree = ast.parse(SRC.read_text(encoding="utf8"))
    functions = [node for node in tree.body if isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef)) and node.name in FN_NAMES]
    assert {n.name for n in functions} == FN_NAMES
    mod = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)] + functions,
        type_ignores=[],
    )
    ast.fix_missing_locations(mod)
    sent = []

    async def socket_send(socket, payload):
        sent.append((socket, payload))

    identity_by_id = {
        "caller-a": {"id": "caller-a", "handle": "caller"},
        "callee-b": {"id": "callee-b", "handle": "pilot"},
    }
    env = dict(
        asyncio=asyncio,
        os=os,
        time=time,
        log=logging.getLogger("test"),
        INVITE_TTL_SECONDS=30,
        _state_lock=asyncio.Lock(),
        _active_calls={},
        _identity_by_id=lambda id: identity_by_id.get(id),
        call_permission_decision=lambda **kwargs: (True, "allowed"),
        _public_identity=lambda i: dict(i),
        _send=socket_send,
    )
    exec(compile(mod, str(SRC), "exec"), env)
    return env, sent


class IncomingPushPilotTests(unittest.IsolatedAsyncioTestCase):
    async def test_feature_off_by_default_even_if_allowlist_present(self):
        env, _ = harness()
        with patch.dict(os.environ, {
            "VERIFYD_INCOMING_CALL_PUSH_PILOT_ENABLED": "0",
            "VERIFYD_INCOMING_CALL_PUSH_PILOT_HANDLES": "pilot",
        }):
            self.assertFalse(env["_incoming_call_push_pilot_enabled"]({"handle": "pilot"}))

    async def test_empty_allowlist_fails_closed(self):
        env, _ = harness()
        with patch.dict(os.environ, {
            "VERIFYD_INCOMING_CALL_PUSH_PILOT_ENABLED": "1",
            "VERIFYD_INCOMING_CALL_PUSH_PILOT_HANDLES": "",
        }):
            self.assertFalse(env["_incoming_call_push_pilot_enabled"]({"handle": "pilot"}))

    async def test_only_exact_explicit_pilot_handle_works(self):
        env, _ = harness()
        with patch.dict(os.environ, {
            "VERIFYD_INCOMING_CALL_PUSH_PILOT_ENABLED": "on",
            "VERIFYD_INCOMING_CALL_PUSH_PILOT_HANDLES": " @Pilot , @other ",
        }):
            self.assertTrue(env["_incoming_call_push_pilot_enabled"]({"handle_lower": "pilot"}))
            self.assertFalse(env["_incoming_call_push_pilot_enabled"]({"handle": "pilot_extra"}))

    async def test_subscription_readiness_requires_enabled_config_and_live_sub(self):
        env, _ = harness()
        mod = types.ModuleType("trust_voice_push")
        mod._active_subscriptions = lambda identity: [{"endpoint": "push"}] if identity == "callee-b" else []
        mod._enabled = lambda: True
        mod._configured = lambda: True
        with patch.dict(sys.modules, {"trust_voice_push": mod}):
            self.assertTrue(env["_incoming_call_push_subscription_ready"]("callee-b"))
            self.assertFalse(env["_incoming_call_push_subscription_ready"]("unknown"))
            mod._configured = lambda: False
            self.assertFalse(env["_incoming_call_push_subscription_ready"]("callee-b"))
            mod._configured = lambda: True
            mod._enabled = lambda: False
            self.assertFalse(env["_incoming_call_push_subscription_ready"]("callee-b"))

    async def test_pilot_push_has_expiring_invitation_and_no_sensitive_contents(self):
        env, _ = harness()
        recorded = []

        def fake_push(identity_id, **kwargs):
            recorded.append((identity_id, kwargs))
            return {"sent": 1, "attempted": 1}

        mod = types.ModuleType("trust_voice_push")
        mod.notify_identity_push = fake_push
        with patch.dict(sys.modules, {"trust_voice_push": mod}):
            ok = await env["_send_incoming_call_pilot_push"]("callee-b", "tvcall_test1", "audio")
        self.assertTrue(ok)
        identity, payload = recorded[0]
        self.assertEqual(identity, "callee-b")
        self.assertEqual(payload["notification_type"], "call_incoming")
        self.assertEqual(payload["ttl"], 30)
        self.assertEqual(payload["data"]["call_id"], "tvcall_test1")
        self.assertNotIn("token", str(payload).lower())
        self.assertNotIn("latitude", str(payload).lower())

    async def test_replay_only_nonexpired_ringing_pilot_calls(self):
        env, sent = harness()
        now = time.time()
        env["_active_calls"] = {
            "ringing": dict(call_id="tvcall_live", caller_id="caller-a", callee_id="callee-b",
                state="ringing", call_type="video", created_at=now-2, pilot_incoming_push=True),
            "expired": dict(call_id="tvcall_expired", caller_id="caller-a", callee_id="callee-b",
                state="ringing", call_type="audio", created_at=now-40, pilot_incoming_push=True),
            "nonpilot": dict(call_id="tvcall_other", caller_id="caller-a", callee_id="callee-b",
                state="ringing", call_type="audio", created_at=now-2, pilot_incoming_push=False),
            "answered": dict(call_id="tvcall_answered", caller_id="caller-a", callee_id="callee-b",
                state="answered", call_type="audio", created_at=now-2, pilot_incoming_push=True),
        }
        mod = types.ModuleType("trust_voice_blocking")
        mod.identities_blocked = lambda a, b: False
        with patch.dict(sys.modules, {"trust_voice_blocking": mod}):
            await env["_replay_pilot_ringing_calls"]("callee-b", object())
        self.assertEqual(len(sent), 1)
        self.assertEqual(sent[0][1]["type"], "incoming_call")
        self.assertEqual(sent[0][1]["call_id"], "tvcall_live")
        self.assertEqual(sent[0][1]["call_type"], "video")
        self.assertLessEqual(sent[0][1]["expires_in_seconds"], 28)

    async def test_blocked_sender_is_not_replayed(self):
        env, sent = harness()
        env["_active_calls"]["r"] = dict(
            call_id="r", caller_id="caller-a", callee_id="callee-b",
            state="ringing", call_type="audio", created_at=time.time(), pilot_incoming_push=True,
        )
        mod = types.ModuleType("trust_voice_blocking")
        mod.identities_blocked = lambda a, b: True
        with patch.dict(sys.modules, {"trust_voice_blocking": mod}):
            await env["_replay_pilot_ringing_calls"]("callee-b", object())
        self.assertEqual(sent, [])

    async def test_legacy_foreground_and_timeout_flow_remains(self):
        text = SRC.read_text(encoding="utf8")
        self.assertIn("if not recipient_online and not pilot_push_reachable:", text)
        self.assertIn("if delivered == 0 and not push_sent", text)
        self.assertIn("_create_call_history(", text)
        self.assertIn("_schedule_call_expiry(call_id)", text)
        self.assertIn('if _incoming_call_push_pilot_enabled(identity):', text)
        self.assertIn("if msg_type == \"call_invite\":", text)


    def load_unregister(self, env, terminal, cancelled):
        from collections import defaultdict
        tree = ast.parse(SRC.read_text(encoding="utf8"))
        fn = [n for n in tree.body if isinstance(n, ast.AsyncFunctionDef) and n.name == "_unregister"]
        assert len(fn) == 1
        mod = ast.Module(
            body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)] + fn,
            type_ignores=[],
        )
        ast.fix_missing_locations(mod)
        env["_connections"] = defaultdict(set)
        env["_connection_meta"] = {}
        env["_cancel_call_expiry"] = lambda call_id: cancelled.append(call_id)
        env["_record_call_terminal"] = lambda call_id, status, reason: terminal.append((call_id, status, reason))
        env["_send_to_identity"] = env["_send"]
        exec(compile(mod, str(SRC), "exec"), env)
        return env["_unregister"]

    async def test_pilot_callee_socket_loss_keeps_pending_ringing_invitation(self):
        env, sent = harness()
        terminal, cancelled = [], []
        unregister = self.load_unregister(env, terminal, cancelled)
        ws = object()
        env["_connections"]["callee-b"].add(ws)
        env["_active_calls"]["tvcall_live"] = dict(
            call_id="tvcall_live", caller_id="caller-a", callee_id="callee-b",
            state="ringing", created_at=time.time(), pilot_incoming_push=True,
        )
        await unregister("callee-b", ws)
        self.assertIn("tvcall_live", env["_active_calls"])
        self.assertEqual(terminal, [])
        self.assertEqual(cancelled, [])
        self.assertEqual(sent, [])

    async def test_legacy_callee_disconnect_still_marks_missed_immediately(self):
        env, sent = harness()
        terminal, cancelled = [], []
        unregister = self.load_unregister(env, terminal, cancelled)
        ws = object()
        env["_connections"]["callee-b"].add(ws)
        env["_active_calls"]["tvcall_legacy"] = dict(
            call_id="tvcall_legacy", caller_id="caller-a", callee_id="callee-b",
            state="ringing", created_at=time.time(), pilot_incoming_push=False,
        )
        await unregister("callee-b", ws)
        self.assertNotIn("tvcall_legacy", env["_active_calls"])
        self.assertEqual(terminal, [("tvcall_legacy", "missed", "callee_disconnected")])
        self.assertEqual(cancelled, ["tvcall_legacy"])
        self.assertEqual(len(sent), 2)

    async def test_pilot_caller_disconnect_cancels_invitation_immediately(self):
        env, sent = harness()
        terminal, cancelled = [], []
        unregister = self.load_unregister(env, terminal, cancelled)
        ws = object()
        env["_connections"]["caller-a"].add(ws)
        env["_active_calls"]["tvcall_pilot"] = dict(
            call_id="tvcall_pilot", caller_id="caller-a", callee_id="callee-b",
            state="ringing", created_at=time.time(), pilot_incoming_push=True,
        )
        await unregister("caller-a", ws)
        self.assertNotIn("tvcall_pilot", env["_active_calls"])
        self.assertEqual(terminal, [("tvcall_pilot", "canceled", "caller_disconnected")])
        self.assertEqual(cancelled, ["tvcall_pilot"])
        self.assertEqual(len(sent), 1)
        self.assertEqual(sent[0][1]["type"], "call_ended")



if __name__ == "__main__":
    unittest.main()
