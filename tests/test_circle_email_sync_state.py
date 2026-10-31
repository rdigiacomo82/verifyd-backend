"""Circle email sync-state: auth, privacy, no-body polling, stable revision."""
from __future__ import annotations

import json
import unittest
from unittest.mock import patch

import trust_voice_circle_email as mail


class Cursor:
    def __init__(self, item):
        self.item = item
        self.queries = []

    def execute(self, sql, parameters):
        self.queries.append((sql, parameters))

    def fetchone(self):
        return self.item


class Database:
    def __init__(self, item):
        self.cur = Cursor(item)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def cursor(self):
        return self.cur


class CircleEmailSyncTests(unittest.TestCase):
    def setUp(self):
        self.row = {
            "total_count": 6,
            "inbox_count": 3,
            "spam_count": 0,
            "trash_count": 2,
            "sent_count": 1,
            "archive_count": 0,
            "custom_count": 0,
            "unread": 2,
            "latest_created_at": "2026-10-31T02:00:00+00:00",
        }
        self.db = Database(self.row)

    def call(self):
        with patch.object(mail, "feature"), patch.object(mail, "schema"):
            with patch.object(mail, "_identity_from_bearer", return_value={"id": "user-private-id"}):
                with patch.object(mail, "get_db", return_value=self.db):
                    return mail.sync_state(authorization="Bearer session-token")

    def test_returns_small_non_caching_response_without_sender_data(self):
        response = self.call()
        data = json.loads(response.body)
        self.assertEqual(data["unread"], 2)
        self.assertEqual(data["inbox_count"], 3)
        self.assertEqual(len(data["revision"]), 24)
        self.assertNotIn("user-private-id", response.body.decode("utf-8"))
        self.assertNotIn("message", data)
        self.assertEqual(response.headers["cache-control"], "private, no-store")
        self.assertEqual(self.db.cur.queries[0][1], ("user-private-id",))

    def test_same_state_reuses_revision(self):
        first = json.loads(self.call().body)["revision"]
        second = json.loads(self.call().body)["revision"]
        self.assertEqual(first, second)

    def test_new_message_and_moved_folder_change_revision(self):
        first = json.loads(self.call().body)["revision"]
        self.row["latest_created_at"] = "2026-10-31T02:01:00+00:00"
        second = json.loads(self.call().body)["revision"]
        self.assertNotEqual(first, second)
        self.row["inbox_count"] = 2
        self.row["trash_count"] = 3
        third = json.loads(self.call().body)["revision"]
        self.assertNotEqual(second, third)

    def test_bad_auth_does_not_access_mail(self):
        with patch.object(mail, "feature"), patch.object(mail, "_identity_from_bearer", side_effect=ValueError("unauthenticated")):
            with patch.object(mail, "schema") as schema:
                with self.assertRaises(ValueError):
                    mail.sync_state(authorization="nope")
                schema.assert_not_called()


if __name__ == "__main__":
    unittest.main()
