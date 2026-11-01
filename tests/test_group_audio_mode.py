"""Group Call remains audio-only across start, status, join and invitations."""
import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi import BackgroundTasks
import trust_voice_group_video as group


class GroupCallModeTests(unittest.TestCase):
    def test_existing_video_is_default_and_audio_mode_is_public(self):
        with patch.object(group, "_identity_by_id", return_value=None):
            ordinary = group._public_session({"conversation_id": "group1", "status": "active"})
            audio = group._public_session({"conversation_id": "group1", "status": "active", "call_mode": "audio"})
        self.assertEqual(ordinary["call_mode"], "video")
        self.assertEqual(audio["call_mode"], "audio")

    def test_schema_alter_is_additive_and_existing_sessions_default_video(self):
        queries = []
        class Cursor:
            def execute(self, statement, args=None):
                queries.append(statement)
        class Db:
            def __enter__(self): return self
            def __exit__(self, *_): return False
            def cursor(self): return Cursor()
        original = group._schema_ready
        try:
            group._schema_ready = False
            with patch.object(group, "ensure_group_schema"):
                with patch.object(group, "get_db", return_value=Db()):
                    group.ensure_schema()
        finally:
            group._schema_ready = original
        self.assertTrue(any("ADD COLUMN IF NOT EXISTS call_mode TEXT NOT NULL DEFAULT 'video'" in q for q in queries))

    def test_new_group_audio_session_is_persisted_and_notified(self):
        executed = []
        class Cursor:
            def execute(self, query, args=None): executed.append((query, args))
        class Db:
            def __enter__(self): return self
            def __exit__(self, *_): return False
            def cursor(self): return Cursor()
        created = {
            "session_id": "one",
            "conversation_id": "group1",
            "room_name": "room1",
            "status": "active",
            "call_mode": "audio",
        }
        sessions = [None, created]
        pending = BackgroundTasks()

        with patch.object(group, "_require_available"):
            with patch.object(group, "_identity_from_bearer", return_value={"id": "user1"}):
                with patch.object(group, "_require_group_member", return_value={"owner_identity_id": "user1"}):
                    with patch.object(group, "_active_session", side_effect=lambda gid: sessions.pop(0)):
                        with patch.object(group, "_create_room", new_callable=AsyncMock):
                            with patch.object(group, "_room_name", return_value="room1"):
                                with patch.object(group, "get_db", return_value=Db()):
                                    with patch.object(group, "_public_session", return_value={"session_id": "one", "call_mode": "audio"}):
                                        with patch.object(group, "_connection", return_value={"participant_token": "test"}):
                                            result = asyncio.run(group.video_start("group1", pending, authorization="Bearer token", call_mode="audio"))
        self.assertTrue(result["created"])
        self.assertEqual(result["session"]["call_mode"], "audio")
        inserts = [(q, a) for q, a in executed if "INSERT INTO verifyd_voice_group_video_sessions" in q]
        self.assertEqual(len(inserts), 1)
        self.assertEqual(inserts[0][1][-1], "audio")
        self.assertEqual(len(pending.tasks), 1)
        self.assertEqual(pending.tasks[0].kwargs.get("payload", {}).get("session", {}).get("call_mode"), "audio")
