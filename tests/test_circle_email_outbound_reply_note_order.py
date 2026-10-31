"""Regression: user-visible reply instruction precedes outgoing external text."""
import ast
from pathlib import Path
import unittest


SOURCE = Path(__file__).resolve().parents[1] / "trust_voice_circle_email.py"


def render_outbound_text(body, reply_note):
    """Evaluate only the pure 'text' value from the outbound send_params dict."""
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    assignments = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "send_params"
                for target in node.targets)
    ]
    if len(assignments) != 1:
        raise AssertionError("Expected exactly one outbound Resend send_params")
    mapping = assignments[0].value
    if not isinstance(mapping, ast.Dict):
        raise AssertionError("send_params must remain a dictionary")
    values = {
        key.value: value for key, value in zip(mapping.keys, mapping.values)
        if isinstance(key, ast.Constant) and isinstance(key.value, str)
    }
    if "text" not in values:
        raise AssertionError("Missing outbound email text")
    return eval(compile(ast.Expression(values["text"]), str(SOURCE), "eval"),
                {"__builtins__": {}}, {"body": body, "reply_note": reply_note})


class TestReplyInstructionAtTop(unittest.TestCase):
    NOTE = "Reply to this message to reach the sender in VeriFYD Circle."

    def test_regular_mail(self):
        self.assertEqual(
            render_outbound_text("Hello from Circle.", self.NOTE),
            self.NOTE + "\n\nHello from Circle."
        )

    def test_multiline_body_preserved(self):
        body = "First line\n\nSecond line with emojis 👋"
        self.assertEqual(render_outbound_text(body, self.NOTE), self.NOTE + "\n\n" + body)

    def test_empty_body_only_note(self):
        self.assertEqual(render_outbound_text("", self.NOTE), self.NOTE)

    def test_previous_fallback_message_is_also_first(self):
        fallback = "Replies go to the sender's verified email address until VeriFYD inbound routing is enabled."
        self.assertEqual(render_outbound_text("Hello", fallback), fallback + "\n\nHello")


if __name__ == "__main__":
    unittest.main()
