"""Ensure Hostinger gateway logs actual saved deliveries only."""
from pathlib import Path
import ast
import unittest

ROOT=Path(__file__).resolve().parents[1]
GATEWAY=ROOT/"trust_voice_circle_email_apex_gateway.py"


class ActualImportCounts(unittest.TestCase):
    def test_logged_imports_require_saved_member_message(self):
        text=GATEWAY.read_text(encoding="utf-8")
        block=text[text.index("def ingest_batch("):text.index("def main()")]
        self.assertIn("delivered_owner, delivered_id = store_public_received(",block)
        self.assertIn("if delivered_owner and delivered_id:",block)
        self.assertIn("handled += 1",block)
        self.assertIn("skipped += 1",block)
        self.assertIn('"processed": handled',block)
        self.assertIn('"skipped": skipped',block)
        self.assertLess(block.index("store_public_received(received,provider_id,handle=handle)"),
                        block.index("if delivered_owner and delivered_id:"))
        self.assertLess(block.index("if delivered_owner and delivered_id:"),
                        block.index("handled += 1"))

    def test_log_has_non_sensitive_counters(self):
        s=GATEWAY.read_text(encoding="utf-8")
        self.assertIn("status=%s imported=%d quarantined=%d skipped=%d",s)
        self.assertIn("result.get(\"skipped\",0)",s)
        self.assertNotIn("log.error(str(exc))",s)
        # Successful counting still requires recipient checks.
        self.assertIn("parse_catchall(raw,_trusted_hosts())",s)
        self.assertIn("original_envelope_recipient(",s)

if __name__=="__main__":
    unittest.main()
