"""Standard-library source contract for audio-only group sessions (CI smoke test)."""
import ast
from pathlib import Path
import unittest


SOURCE = (Path(__file__).resolve().parents[1] / "trust_voice_group_video.py").read_text()


class GroupAudioSourceContractTests(unittest.TestCase):
    def test_module_parses(self):
        ast.parse(SOURCE)

    def test_session_metadata_persisted_with_video_migration(self):
        self.assertIn("ADD COLUMN IF NOT EXISTS call_mode TEXT NOT NULL DEFAULT 'video'", SOURCE)
        self.assertIn("call_mode: Literal['audio', 'video'] = Query('video')", SOURCE)
        self.assertIn("ended_by_identity_id, ended_reason, call_mode", SOURCE)

    def test_group_session_mode_is_public_to_invitees(self):
        self.assertIn("'call_mode': 'audio' if session.get('call_mode') == 'audio' else 'video'", SOURCE)
        self.assertIn("'type':'group_video_started'", SOURCE)
        self.assertIn("'session':public_session", SOURCE)

    def test_all_legacy_status_and_join_handlers_remain(self):
        for method in ("video_status", "video_start", "video_join", "video_end"):
            self.assertIn("async def " + method + "(", SOURCE)


if __name__ == "__main__":
    unittest.main()
