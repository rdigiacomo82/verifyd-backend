"""Web Push regression checks: a provider-expired endpoint cannot be revived.

No external provider, Firebase credential, database or notification is touched.
Run with: python -m unittest tests/test_circle16_push_expired_guard.py
"""
from contextlib import contextmanager
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock, patch

from fastapi import HTTPException

import trust_voice_push as push


class PushExpiryRegistrationTests(TestCase):
    def _register(self, existing_row):
        cursor = Mock()
        cursor.fetchone.return_value = existing_row

        @contextmanager
        def fake_get_db():
            yield SimpleNamespace(cursor=lambda: cursor)

        payload = SimpleNamespace(
            endpoint="https://fcm.googleapis.com/fcm/send/valid-test-endpoint",
            keys=SimpleNamespace(p256dh="test-p256dh", auth="test-auth"),
        )
        request = SimpleNamespace(headers={"user-agent": "Test/1.0"})
        patches = [
            patch.object(push, "_enabled", return_value=True),
            patch.object(push, "_configured", return_value=True),
            patch.object(push, "_auth_context", return_value=(
                {"id": "test-identity"}, {"session_id": "test-session"}
            )),
            patch.object(push, "ensure_push_schema", return_value=None),
            patch.object(push, "_validate_endpoint", side_effect=lambda x: x),
            patch.object(push, "get_db", side_effect=fake_get_db),
        ]
        for entry in patches:
            entry.start()
        try:
            if existing_row is None:
                with self.assertRaises(HTTPException) as context:
                    push.register_push_subscription(payload, request, "Bearer test")
                self.assertEqual(context.exception.status_code, 409)
                self.assertEqual(
                    context.exception.detail["error"], "push_subscription_expired"
                )
                result = None
            else:
                result = push.register_push_subscription(
                    payload, request, "Bearer test"
                )
            self.assertTrue(cursor.execute.called)
            sql = cursor.execute.call_args.args[0]
            self.assertIn("ON CONFLICT (endpoint)", sql)
            self.assertIn(
                "IS DISTINCT FROM 'expired_or_gone'", sql
            )
            return result
        finally:
            for entry in reversed(patches):
                entry.stop()

    def test_new_endpoint_can_register(self):
        result = self._register({"subscription_id": "tvpush_new"})
        self.assertEqual(result["subscription_id"], "tvpush_new")
        self.assertTrue(result["subscribed"])

    def test_expired_endpoint_requires_rotation(self):
        self._register(None)
