"""Protocol and concurrency regression checks for Trust Voice WebSocket auth.

No credentials, network calls, database mutations or active sockets required.
"""
import ast
import unittest
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "trust_voice_signaling.py").read_text(encoding="utf-8")
TREE = ast.parse(SOURCE)
NAME = "_websocket_auth_from_scope"
HELPER = next(n for n in TREE.body if isinstance(n, ast.FunctionDef) and n.name == NAME)
ENV = {
    "WebSocket": object,
    "Optional": Optional,
    "_WS_AUTH_PROTOCOL": "verifyd-auth-v1",
    "_WS_TOKEN_PREFIX": "verifyd-session.",
}
exec(compile(ast.Module(body=[HELPER], type_ignores=[]), str(ROOT / "trust_voice_signaling.py"), "exec"), ENV)
extract = ENV[NAME]


class FakeWebSocket:
    def __init__(self, protocols):
        self.scope = {"subprotocols": protocols}


class WebSocketAuthRegressionTests(unittest.TestCase):
    def test_existing_query_token_is_still_supported(self):
        self.assertEqual(extract(FakeWebSocket([]), "legacy.test"), ("legacy.test", None))

    def test_header_token_negotiates_fixed_protocol(self):
        self.assertEqual(
            extract(FakeWebSocket(["verifyd-auth-v1", "verifyd-session.private.test"]), ""),
            ("private.test", "verifyd-auth-v1"),
        )

    def test_header_token_overrides_legacy_query_when_both_present(self):
        self.assertEqual(
            extract(FakeWebSocket(["verifyd-auth-v1", "verifyd-session.new.test"]), "legacy.test"),
            ("new.test", "verifyd-auth-v1"),
        )

    def test_malformed_protocol_cannot_downgrade_to_query(self):
        for protocols in (
            ["verifyd-auth-v1"],
            ["verifyd-session.private.test"],
            ["verifyd-auth-v1", "verifyd-session."],
            ["verifyd-auth-v1", "verifyd-session.first", "verifyd-session.second"],
        ):
            with self.subTest(protocols=[p.split(".")[0] for p in protocols]):
                self.assertEqual(extract(FakeWebSocket(protocols), "legacy.test"), ("", None))

    def test_unrelated_subprotocol_does_not_break_legacy_auth(self):
        self.assertEqual(extract(FakeWebSocket(["unrelated"]), "legacy.test"), ("legacy.test", None))

    def test_signal_checks_are_offloaded_from_event_loop(self):
        func = next(n for n in TREE.body if isinstance(n, ast.AsyncFunctionDef) and n.name == "signaling_ws")
        calls = []
        for node in ast.walk(func):
            if isinstance(node, ast.Await) and isinstance(node.value, ast.Call):
                c = node.value
                if isinstance(c.func, ast.Name) and c.func.id == "run_in_threadpool":
                    if c.args and isinstance(c.args[0], ast.Name):
                        calls.append(c.args[0].id)
        self.assertGreaterEqual(calls.count("_verify_token"), 2)
        self.assertIn("_identity_by_id", calls)

    def test_fixed_protocol_is_echoed_instead_of_token(self):
        self.assertIn("await websocket.accept(subprotocol=negotiated_protocol)", SOURCE)
        self.assertNotIn("websocket.accept(subprotocol=token)", SOURCE)


if __name__ == "__main__":
    unittest.main()
