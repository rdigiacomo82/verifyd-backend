"""Pure/mock unit checks for optional voicemail greetings (no DB/R2/calls).
Run: python -m unittest tests/test_circle16_voicemail_greetings.py
"""
from unittest import TestCase
from unittest.mock import patch
from fastapi import HTTPException
from trust_voice_voicemail_greetings import (
    _public, _caller_and_target, _stream_audio, update_my_greeting, GreetingUpdate,
)


class GreetingTests(TestCase):
    def test_public_metadata_never_leaks_r2_key(self):
        row = {"mode": "audio", "greeting_text": "", "storage_key": "private/secret",
               "content_type": "audio/webm", "updated_at": "2026-11-17"}
        result = _public(row)
        self.assertEqual(result["mode"], "audio")
        self.assertTrue(result["has_audio"])
        self.assertNotIn("storage_key", result)
        self.assertNotIn("content_type", result)

    def test_default_does_not_require_recording(self):
        result = _public({"mode": "default"})
        self.assertIn("leave a message", result["text"])

    def test_audio_activation_requires_prior_owner_upload(self):
        with patch("trust_voice_voicemail_greetings._owner",
                   return_value={"id": "owner"}), \
             patch("trust_voice_voicemail_greetings._read",
                   return_value={"mode": "default", "storage_key": None}):
            with self.assertRaises(HTTPException) as error:
                update_my_greeting(GreetingUpdate(mode="audio"), "Bearer test")
            self.assertEqual(error.exception.status_code, 409)

    def test_unauthorized_caller_cannot_fetch_greeting(self):
        with patch("trust_voice_voicemail_greetings._require_enabled"), \
             patch("trust_voice_voicemail_greetings._identity_from_http_authorization",
                   return_value={"id": "caller"}), \
             patch("trust_voice_voicemail_greetings._normalize_handle", return_value="target"), \
             patch("trust_voice_voicemail_greetings._identity_by_handle",
                   return_value={"id": "recipient"}), \
             patch("trust_voice_voicemail_greetings._require_message_allowed",
                   side_effect=HTTPException(status_code=403, detail="not_allowed")):
            with self.assertRaises(HTTPException) as error:
                _caller_and_target("target", "Bearer test")
            self.assertEqual(error.exception.status_code, 403)

    def test_audio_stream_requires_active_recording(self):
        with self.assertRaises(HTTPException) as error:
            _stream_audio({"mode": "text", "greeting_text": "Vacation"})
        self.assertEqual(error.exception.status_code, 404)
