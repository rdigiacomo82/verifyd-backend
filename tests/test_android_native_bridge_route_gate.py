"""Contract checks for the gated native Android pilot router.

No imports of main.py or production services; this is source/AST inspection only.
"""
import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
FEATURE = "VERIFYD_ANDROID_NATIVE_BRIDGE_ENABLED"


class AndroidBridgeRouterGateContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.main_source = (ROOT / "main.py").read_text(encoding="utf-8")
        cls.tree = ast.parse(cls.main_source, filename="main.py")

    def test_main_syntax_valid(self):
        compile(self.tree, "main.py", "exec")

    def test_native_route_is_not_unconditionally_mounted(self):
        top_level_mounts = [
            node for node in self.tree.body
            if isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Attribute)
            and node.value.func.attr == "include_router"
            and node.value.args
            and isinstance(node.value.args[0], ast.Name)
            and node.value.args[0].id == "trust_voice_android_bridge_router"
        ]
        self.assertEqual(top_level_mounts, [])

    def test_router_mounted_only_in_flag_guard(self):
        guarded = []
        for node in self.tree.body:
            if not isinstance(node, ast.If) or FEATURE not in ast.unparse(node.test):
                continue
            text = ast.unparse(node)
            if "trust_voice_android_bridge_router" in text:
                guarded.append(text)
        self.assertEqual(len(guarded), 1)
        guard = guarded[0]
        self.assertIn('os.environ.get(\'VERIFYD_ANDROID_NATIVE_BRIDGE_ENABLED\', \'0\')', guard)
        self.assertIn("app.include_router(trust_voice_android_bridge_router)", guard)
        self.assertIn("from trust_voice_android_bridge import router as trust_voice_android_bridge_router", guard)
        self.assertNotIn("app.include_router(trust_voice_android_bridge_router)", ast.unparse(self.tree).split(guard)[0])

    def test_existing_android_calls_route_remains(self):
        self.assertIn("app.include_router(trust_voice_android_calls_router)", self.main_source)


if __name__ == "__main__":
    unittest.main()
