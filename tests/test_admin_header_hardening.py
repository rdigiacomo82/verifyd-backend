"""Isolated admin-header migration regressions. No real secrets or HTTP server."""
import ast
import hmac
import unittest
from contextvars import ContextVar
from pathlib import Path

SOURCE=(Path(__file__).resolve().parents[1]/"main.py").read_text("utf-8")
TREE=ast.parse(SOURCE)
FUNC=next(n for n in TREE.body if isinstance(n,ast.FunctionDef) and n.name=="_is_admin")
ENV={"_ADMIN_KEY":"synthetic-test-admin-key","hmac":hmac,
     "_ADMIN_HEADER_CONTEXT":ContextVar("test_header",default=""),
     "_ADMIN_LEGACY_QUERY_ALLOWED":True}
exec(compile(ast.Module(body=[FUNC],type_ignores=[]),"main.py","exec"),ENV)
check=ENV["_is_admin"]

class AdminHeaderMigration(unittest.TestCase):
    def setUp(self):
        self.token=ENV["_ADMIN_HEADER_CONTEXT"].set("")
        ENV["_ADMIN_LEGACY_QUERY_ALLOWED"]=True

    def tearDown(self):
        ENV["_ADMIN_HEADER_CONTEXT"].reset(self.token)

    def test_existing_query_compatibility_during_migration(self):
        self.assertTrue(check("synthetic-test-admin-key"))
        self.assertFalse(check("bad-key"))

    def test_header_auth_without_key_in_url(self):
        ENV["_ADMIN_HEADER_CONTEXT"].set("synthetic-test-admin-key")
        self.assertTrue(check(""))
        self.assertFalse(check("different-value"))

    def test_disable_legacy_query_without_locking_out_headers(self):
        ENV["_ADMIN_LEGACY_QUERY_ALLOWED"]=False
        self.assertFalse(check("synthetic-test-admin-key"))
        ENV["_ADMIN_HEADER_CONTEXT"].set("synthetic-test-admin-key")
        self.assertTrue(check(""))
        self.assertFalse(check("synthetic-test-admin-key"))

    def test_middleware_wires_header_to_context(self):
        self.assertIn('request.headers.get("x-verifyd-admin-key", "")',SOURCE)
        self.assertIn("_ADMIN_HEADER_CONTEXT.reset(token)",SOURCE)

if __name__=="__main__":
    unittest.main()
