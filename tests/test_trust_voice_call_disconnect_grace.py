"""Isolated regression tests for OFF-by-default answered-call WebSocket grace.

These tests exercise the real functions extracted from trust_voice_signaling.py
without importing production FastAPI, queues or database services.
"""
from __future__ import annotations

import ast
import asyncio
from collections import defaultdict
import os
from pathlib import Path
import time
import unittest
from unittest.mock import patch

SIGNAL = Path(__file__).resolve().parents[1] / "trust_voice_signaling.py"
NAMES = {
    "_call_reconnect_grace_seconds",
    "_call_reconnect_grace_allowed",
    "_cancel_call_disconnect_grace",
    "_expire_call_after_disconnect_grace",
    "_schedule_call_disconnect_grace",
    "_register",
    "_unregister",
}


def load_functions():
    tree = ast.parse(SIGNAL.read_text(encoding="utf-8"))
    selected = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in NAMES]
    assert len(selected) == len(NAMES), "A reconnect-grace function was renamed or lost"
    mod = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)] + selected,
        type_ignores=[],
    )
    ast.fix_missing_locations(mod)
    terminations, sent = [], []

    async def send(identity, payload):
        sent.append((identity, payload))
        return 1

    def record(call_id, status, reason):
        terminations.append((call_id, status, reason))
        return "2026-11-07T00:00:00Z"

    env = {
        "__name__": "isolated_grace",
        "asyncio": asyncio,
        "os": os,
        "time": time,
        "_state_lock": asyncio.Lock(),
        "_connections": defaultdict(set),
        "_connection_meta": {},
        "_active_calls": {},
        "_call_disconnect_grace_tasks": {},
        "_send_to_identity": send,
        "_record_call_terminal": record,
        "_cancel_call_expiry": lambda _id: None,
    }
    exec(compile(mod, str(SIGNAL), "exec"), env)
    return env, terminations, sent


class CallDisconnectGraceTests(unittest.IsolatedAsyncioTestCase):
    def make_answered(self, env):
        ws = object()
        env["_connections"]["a"].add(ws)
        env["_active_calls"]["test-call"] = {
            "call_id": "test-call", "caller_id": "a", "callee_id": "b", "state": "answered",
        }
        return ws

    async def test_default_off_disconnect_ends_call(self):
        with patch.dict(os.environ, {"VERIFYD_CALL_RECONNECT_GRACE_ENABLED": "0",
                                     "VERIFYD_CALL_RECONNECT_GRACE_PILOT_IDS": "a"}):
            env, terminal, sent = load_functions()
            ws = self.make_answered(env)
            await env["_unregister"]("a", ws)
            self.assertNotIn("test-call", env["_active_calls"])
            self.assertEqual(terminal, [("test-call", "ended", "peer_disconnected")])
            self.assertEqual(sent[0][0], "b")
            self.assertEqual(env["_call_disconnect_grace_tasks"], {})

    async def test_disabled_without_exact_pilot_identity(self):
        with patch.dict(os.environ, {"VERIFYD_CALL_RECONNECT_GRACE_ENABLED": "1",
                                     "VERIFYD_CALL_RECONNECT_GRACE_PILOT_IDS": "some-other-id"}):
            env, terminal, sent = load_functions()
            ws = self.make_answered(env)
            await env["_unregister"]("a", ws)
            self.assertNotIn("test-call", env["_active_calls"])
            self.assertEqual(len(terminal), 1)
            self.assertEqual(env["_call_disconnect_grace_tasks"], {})

    async def test_authenticated_reconnect_cancels_grace(self):
        with patch.dict(os.environ, {"VERIFYD_CALL_RECONNECT_GRACE_ENABLED": "1",
                                     "VERIFYD_CALL_RECONNECT_GRACE_PILOT_IDS": "a"}):
            env, terminal, sent = load_functions()
            env["_call_reconnect_grace_seconds"] = lambda: 0.01
            ws = self.make_answered(env)
            await env["_unregister"]("a", ws)
            self.assertIn("test-call", env["_active_calls"])
            self.assertIn(("test-call", "a"), env["_call_disconnect_grace_tasks"])
            await env["_register"]("a", object(), {"session_id": "verified-session"})
            await asyncio.sleep(0.04)
            self.assertIn("test-call", env["_active_calls"])
            self.assertEqual(terminal, [])
            self.assertEqual(sent, [])

    async def test_no_reconnect_ends_call_once(self):
        with patch.dict(os.environ, {"VERIFYD_CALL_RECONNECT_GRACE_ENABLED": "1",
                                     "VERIFYD_CALL_RECONNECT_GRACE_PILOT_IDS": "a"}):
            env, terminal, sent = load_functions()
            env["_call_reconnect_grace_seconds"] = lambda: 0.01
            ws = self.make_answered(env)
            await env["_unregister"]("a", ws)
            await asyncio.sleep(0.05)
            self.assertNotIn("test-call", env["_active_calls"])
            self.assertEqual(terminal, [("test-call", "ended", "peer_disconnected")])
            self.assertEqual(len(sent), 1)
            self.assertEqual(sent[0][1]["call_id"], "test-call")
            self.assertEqual(env["_call_disconnect_grace_tasks"], {})

    async def test_ringing_calls_still_terminate_immediately(self):
        with patch.dict(os.environ, {"VERIFYD_CALL_RECONNECT_GRACE_ENABLED": "1",
                                     "VERIFYD_CALL_RECONNECT_GRACE_PILOT_IDS": "a"}):
            env, terminal, sent = load_functions()
            ws = self.make_answered(env)
            env["_active_calls"]["test-call"]["state"] = "ringing"
            await env["_unregister"]("a", ws)
            self.assertNotIn("test-call", env["_active_calls"])
            self.assertEqual(terminal, [("test-call", "canceled", "caller_disconnected")])
            self.assertEqual(env["_call_disconnect_grace_tasks"], {})

    async def test_other_live_socket_no_grace_needed(self):
        with patch.dict(os.environ, {"VERIFYD_CALL_RECONNECT_GRACE_ENABLED": "1",
                                     "VERIFYD_CALL_RECONNECT_GRACE_PILOT_IDS": "a"}):
            env, terminal, sent = load_functions()
            ws = self.make_answered(env)
            env["_connections"]["a"].add(object())
            await env["_unregister"]("a", ws)
            self.assertIn("test-call", env["_active_calls"])
            self.assertEqual(env["_call_disconnect_grace_tasks"], {})
            self.assertEqual(terminal, [])

    async def test_config_clamped_and_invalid_input_safe(self):
        env, _, _ = load_functions()
        with patch.dict(os.environ, {"VERIFYD_CALL_RECONNECT_GRACE_SECONDS": "999"}):
            self.assertEqual(env["_call_reconnect_grace_seconds"](), 30)
        with patch.dict(os.environ, {"VERIFYD_CALL_RECONNECT_GRACE_SECONDS": "-4"}):
            self.assertEqual(env["_call_reconnect_grace_seconds"](), 5)
        with patch.dict(os.environ, {"VERIFYD_CALL_RECONNECT_GRACE_SECONDS": "garbage"}):
            self.assertEqual(env["_call_reconnect_grace_seconds"](), 20)


if __name__ == "__main__":
    unittest.main()
