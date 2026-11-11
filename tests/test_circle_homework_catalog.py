"""Unit tests for the Circle Homework pilot (no live PostgreSQL used)."""
import importlib.util
import json
import os
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
db = types.ModuleType("database")
db.get_db = lambda: None
auth = types.ModuleType("trust_voice_messages")
auth._identity_from_bearer = lambda token: {"id": "test-identity", "status": "active"}
with patch.dict(sys.modules, {"database": db, "trust_voice_messages": auth}):
    spec = importlib.util.spec_from_file_location("homework_test_module", ROOT / "trust_voice_homework.py")
    homework = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(homework)


class HomeworkTests(unittest.TestCase):
    def test_grade_bands_and_subjects(self):
        self.assertEqual(len(homework.ASSIGNMENTS), 16)
        self.assertEqual({a["band"] for a in homework.ASSIGNMENTS.values()}, {"K-2", "3-5", "6-8", "9-12"})
        for band in {"K-2", "3-5", "6-8", "9-12"}:
            self.assertEqual(
                {a["subject"] for a in homework.ASSIGNMENTS.values() if a["band"] == band},
                {"Mathematics", "Reading & Writing", "Science", "Social Studies"},
            )

    def test_public_catalog_redacts_answer_keys(self):
        for assignment in homework.ASSIGNMENTS.values():
            safe = homework._public_assignment(assignment)
            self.assertEqual(len(safe["questions"]), 3)
            for q in safe["questions"]:
                self.assertNotIn("answer", q)

    def test_two_of_three_questions_pass(self):
        item = next(iter(homework.ASSIGNMENTS.values()))
        answers = {q["id"]: q["answer"] for q in item["questions"]}
        self.assertTrue(homework.grade_answers(item, answers)["passed"])
        q = item["questions"][0]
        answers[q["id"]] = next(choice["id"] for choice in q["choices"] if choice["id"] != q["answer"])
        self.assertTrue(homework.grade_answers(item, answers)["passed"])
        for q in item["questions"][1:]:
            answers[q["id"]] = next(choice["id"] for choice in q["choices"] if choice["id"] != q["answer"])
        self.assertFalse(homework.grade_answers(item, answers)["passed"])

    def test_missing_answers_rejected(self):
        item = next(iter(homework.ASSIGNMENTS.values()))
        with self.assertRaises(homework.HTTPException):
            homework.grade_answers(item, {})

    def test_invalid_choice_rejected(self):
        item = next(iter(homework.ASSIGNMENTS.values()))
        with self.assertRaises(homework.HTTPException):
            homework._validate_partial(item, {"q1": "scripted-score"})

    def test_fixed_point_budget(self):
        self.assertTrue(all(x["points"] == 60 for x in homework.ASSIGNMENTS.values()))

    def test_disabled_without_flag(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(homework.enabled())


if __name__ == "__main__":
    unittest.main()
