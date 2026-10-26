"""Regression gates for Trust Voice security, without a live database.
The production integration suite still must test real PostgreSQL and WebSocket.
"""
import ast
import pathlib
import types
import unittest
from contextlib import contextmanager

ROOT = pathlib.Path(__file__).resolve().parents[1]

def extract(file, names, env):
    tree = ast.parse((ROOT / file).read_text(encoding="utf-8"))
    nodes = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
             and node.name in names]
    for node in nodes:
        node.decorator_list = []
        for arg in node.args.args:
            arg.annotation = None
        node.returns = None
    code = compile(ast.Module(body=nodes, type_ignores=[]), file, "exec")
    exec(code, env)
    return env

class FakeHTTPException(Exception):
    def __init__(self, status_code, detail=None):
        self.status_code = status_code
        self.detail = detail

class Cursor:
    def __init__(self, rows=(), count=1):
        self.rows = list(rows)
        self.rowcount = count
        self.calls = []
    def execute(self, sql, values=()):
        self.calls.append((sql, tuple(values)))
    def fetchone(self):
        return self.rows.pop(0) if self.rows else None
    def fetchall(self):
        return list(self.rows)

class Db:
    def __init__(self, cursor):
        self.c = cursor
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass
    def cursor(self):
        return self.c

class SecurityRegressions(unittest.TestCase):
    def test_revocation_rejects_other_owner_and_current_session(self):
        c=Cursor(count=0)
        env=dict(Optional=object, Header=lambda **kw: None, HTTPException=FakeHTTPException,
                 RevokeDevice=object, get_db=lambda: Db(c), ensure_persistent_session_schema=lambda:None,
                 _now_iso=lambda:"now", _session_context=lambda auth:("alice","current"))
        ns=extract("trust_voice_devices.py", ["revoke_device"], env)
        body=types.SimpleNamespace(session_id="other")
        with self.assertRaises(FakeHTTPException) as ctx:
            ns["revoke_device"](body, "Bearer token")
        self.assertEqual(ctx.exception.status_code,404)
        self.assertEqual(c.calls[-1][1],("now","other","alice"))
        with self.assertRaises(FakeHTTPException) as ctx:
            ns["revoke_device"](types.SimpleNamespace(session_id="current"), "Bearer token")
        self.assertEqual(ctx.exception.status_code,409)

    def test_block_is_symmetric_and_identity_scoped(self):
        c=Cursor(rows=[{"exists":1}])
        env=dict(get_db=lambda:Db(c),ensure_block_schema=lambda:None,blocking_enabled=lambda:True)
        ns=extract("trust_voice_blocking.py", ["identities_blocked"], env)
        self.assertTrue(ns["identities_blocked"]("alice","bob"))
        self.assertEqual(c.calls[-1][1], ("alice","bob","bob","alice"))
        self.assertFalse(ns["identities_blocked"]("alice","alice"))
        self.assertFalse(ns["identities_blocked"]("","bob"))

    def test_direct_messages_check_blocks_before_privacy(self):
        source=(ROOT/"trust_voice_messages.py").read_text()
        fn=source[source.index("def _message_permission_decision("):source.index("\ndef _require_message_allowed(",source.index("def _message_permission_decision("))]
        self.assertLess(fn.index("identities_blocked("),fn.index("_settings_for_identity("))
        self.assertIn('return False, "blocked_user"',fn)

    def test_calls_check_blocks_before_feature_flag(self):
        source=(ROOT/"trust_voice_permissions.py").read_text()
        fn=source[source.index("def call_permission_decision("):]
        self.assertLess(fn.index("identities_blocked("),fn.index("if not trust_circle_enabled():"))

    def test_revoked_ws_session_closes_on_401(self):
        source=(ROOT/"trust_voice_signaling.py").read_text()
        self.assertIn("except HTTPException as exc:",source)
        self.assertIn("if exc.status_code == 401:",source)
        self.assertIn("await websocket.close(code=4401)",source)

if __name__=="__main__":
    unittest.main()
