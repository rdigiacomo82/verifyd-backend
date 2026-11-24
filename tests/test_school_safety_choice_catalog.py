"""Contract tests for editable school-specific TEST Safety Button choices.

These tests never call a live school database, email, push, SMS, or 911.
"""
import inspect
import unittest
from unittest.mock import patch

from fastapi import HTTPException
from pydantic import ValidationError

import trust_voice_school_choice_catalog as catalog
import trust_voice_school_safety_training as safety


class Cursor:
    def __init__(self, row=None):
        self.row = row
        self.sql = []

    def execute(self, query, values=None):
        self.sql.append((str(query), values))

    def fetchone(self):
        return self.row


class DB:
    def __init__(self, cursor):
        self.cursor_value = cursor

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def cursor(self):
        return self.cursor_value


class SchoolSafetyChoiceContractTests(unittest.TestCase):
    def valid(self):
        return catalog.SafetyChoiceUpdate(
            expected_revision=0,
            choices=[catalog.SafetyChoice(**item) for item in catalog.default_choices()])

    def test_twelve_defaults_and_free_text_other(self):
        choices = catalog.default_choices()
        self.assertEqual(len(choices), 12)
        self.assertEqual(choices[-1]["id"], "other")
        self.assertEqual(choices[-1]["category"], "other")
        self.assertFalse(choices[-1]["hint"])
        self.assertIn("weapon", [choice["id"] for choice in choices])

    def test_guardrail_other_is_always_available_and_free_text(self):
        good = self.valid()
        self.assertEqual(good.choices[-1].id, "other")
        with self.assertRaises(ValidationError):
            catalog.SafetyChoiceUpdate(expected_revision=0, choices=good.choices[:-1])
        with self.assertRaises(ValidationError):
            catalog.SafetyChoiceUpdate(expected_revision=0, choices=good.choices + [good.choices[0]])
        with self.assertRaises(ValidationError):
            catalog.SafetyChoice(id="other", label="Type concern", category="fire", hint="")
        with self.assertRaises(ValidationError):
            catalog.SafetyChoice(id="other", label="Type concern", category="other", hint="Prefill somebody's report")

    def test_catalog_disallows_invalid_category_and_untrimmed_label(self):
        with self.assertRaises(ValidationError):
            catalog.SafetyChoice(id="bad", label="Unsafe", category="call_911", hint="")
        with self.assertRaises(ValidationError):
            catalog.SafetyChoice(id="bad", label=" Leading ", category="other", hint="")
        with self.assertRaises(ValidationError):
            catalog.SafetyChoiceUpdate(expected_revision=0, choices=[
                catalog.SafetyChoice(id=f"custom_{i}", label="Concern", category="other")
                for i in range(21)])

    def test_unknown_stale_choice_denied_server_side(self):
        cur = Cursor()
        with patch.object(catalog, "_read_catalog", return_value={
            "revision": 4, "choices": catalog.default_choices()
        }):
            with self.assertRaises(HTTPException) as ctx:
                catalog.resolve_report_choice(cur, "test-school", "custom_deleted")
        self.assertEqual(ctx.exception.status_code, 409)

    def test_student_or_unapproved_backup_cannot_save_choice_updates(self):
        cursor = Cursor()
        with patch.object(catalog, "_actor", return_value={"id": "student-1"}):
            with patch.object(catalog, "_catalog_schema"):
                with patch.object(catalog, "get_db", return_value=DB(cursor)):
                    with patch.object(catalog, "_school_admin", side_effect=HTTPException(403, detail="not_admin")):
                        with self.assertRaises(HTTPException) as ctx:
                            catalog.update_school_safety_choices("school", self.valid())
        self.assertEqual(ctx.exception.status_code, 403)
        self.assertFalse(any("INSERT INTO verifyd_safety_choice_catalog" in q for q, _ in cursor.sql))

    def test_optimistic_revision_blocks_overwrite(self):
        cursor = Cursor({"revision": 5})
        with patch.object(catalog, "_actor", return_value={"id": "principal-1"}):
            with patch.object(catalog, "_catalog_schema"):
                with patch.object(catalog, "get_db", return_value=DB(cursor)):
                    with patch.object(catalog, "_school_admin", return_value="principal"):
                        with self.assertRaises(HTTPException) as ctx:
                            catalog.update_school_safety_choices("school", self.valid())
        self.assertEqual(ctx.exception.status_code, 409)

    def test_report_uses_authoritative_choice_not_client_label(self):
        route_src = inspect.getsource(safety.create_training_report)
        self.assertIn("_create_training_report_for_student", route_src)
        src = inspect.getsource(safety._create_training_report_for_student)
        self.assertIn("resolve_report_choice(cur, payload.school_id, payload.choice_id)", src)
        self.assertIn('selected_choice["category"] != payload.category', src)
        self.assertIn('selected_choice["id"] == "other"', src)
        self.assertIn('payload.description.strip()', src)
        self.assertIn("initial_description", src)
        # No new 911 or distress endpoint was introduced.
        paths = [route.path for route in catalog.router.routes]
        self.assertIn("/trust-voice/school-safety/choices/me", paths)
        self.assertIn("/trust-voice/school-safety/schools/{school_id}/choices", paths)
        self.assertNotIn("/trust-voice/safety/distress", paths)


if __name__ == "__main__":
    unittest.main()
