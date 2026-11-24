"""Authorization contract tests for the TRAINING-ONLY private Safety conversation.

These tests deliberately compile the real helper from the source AST, avoiding
production DB calls and web push. Full database/browser tests remain required.
"""
import ast
import pathlib
import unittest


SOURCE = pathlib.Path(__file__).resolve().parents[1] / "trust_voice_school_incident_chat.py"


def source_tree():
    return ast.parse(SOURCE.read_text(encoding="utf-8"), filename=str(SOURCE))


def functions():
    return {node.name: node for node in source_tree().body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}


class FakeHTTPException(Exception):
    def __init__(self, status_code, detail):
        super().__init__(str(detail))
        self.status_code, self.detail = status_code, detail


class SafetyFamilyTextContractTests(unittest.TestCase):
    def helper(self, *, role="guardian", allowed=True):
        calls = []
        report = {"school_id": "test_school", "student_id": "student_1"}
        def authorize(cur, rid, who):
            calls.append(("authorize", who))
            if not allowed:
                raise FakeHTTPException(403, {"error": "revoked_guardian"})
            return report
        def membership(cur, school, who, roles=None):
            calls.append(("membership", who, roles))
            if roles and role not in roles:
                raise FakeHTTPException(403, {"error": "wrong_role"})
            return role
        node = functions()["_family_chat_authorize"]
        ns = {"_authorize": authorize, "_membership": membership, "HTTPException": FakeHTTPException}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), "exec"), ns)
        return ns["_family_chat_authorize"], calls

    def test_school_staff_cannot_read_family_conversation(self):
        for role in ("principal", "responder", "operator"):
            with self.subTest(role=role):
                helper, _ = self.helper(role=role)
                with self.assertRaises(FakeHTTPException) as error:
                    helper(None, "report", "school_staff")
                self.assertEqual(error.exception.status_code, 403)

    def test_student_can_use_own_family_conversation(self):
        helper, calls = self.helper(role="student")
        self.assertEqual(helper(None, "report", "student_1")["student_id"], "student_1")
        self.assertIn(("authorize", "student_1"), calls)

    def test_verified_guardian_can_use_conversation(self):
        helper, calls = self.helper(role="guardian")
        self.assertEqual(helper(None, "report", "guardian_1")["school_id"], "test_school")
        self.assertIn(("authorize", "guardian_1"), calls)

    def test_revoked_guardian_does_not_bypass_report_access(self):
        helper, _ = self.helper(role="guardian", allowed=False)
        with self.assertRaises(FakeHTTPException):
            helper(None, "report", "guardian_1")

    def test_both_message_routes_require_private_authorization(self):
        for route in ("conversation", "send_reply"):
            node = functions()[route]
            calls = [n.func.id for n in ast.walk(node) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]
            self.assertIn("_family_chat_authorize", calls, route)

    def test_reply_dispatch_rechecks_private_authorization(self):
        node = functions()["_dispatch"]
        calls = [n.func.id for n in ast.walk(node) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]
        self.assertIn("_family_chat_authorize", calls)
        self.assertIn("notify_identity_push", calls)

    def test_school_initial_alert_queues_separately_from_family_text(self):
        node = functions()["queue_report"]
        calls = [n.func.id for n in ast.walk(node) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]
        self.assertIn("_authorize", calls)


if __name__ == "__main__":
    unittest.main()
