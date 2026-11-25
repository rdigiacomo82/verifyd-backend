"""Regression tests for school staff invitation prefix initialization (TEST only)."""
import inspect
import unittest
from unittest.mock import patch

import trust_voice_school_staff_invites as staff


class FakeNamingCursor:
    def __init__(self, *, existing=None, school_name="Test School", taken=()):
        self.existing = existing
        self.school_name = school_name
        self.taken = set(taken)
        self.last = None
        self.queries = []
        self.inserted = None

    def execute(self, query, params):
        self.queries.append((query, params))
        if "SELECT school_code FROM verifyd_safety_managed_roster_settings" in query:
            self.last = {"school_code": self.existing} if self.existing else None
        elif "SELECT school_name FROM verifyd_safety_training_schools" in query:
            self.last = {"school_name": self.school_name}
        elif "SELECT 1 FROM verifyd_safety_managed_roster_settings" in query:
            self.last = {"exists": True} if params[0] in self.taken else None
        elif "INSERT INTO verifyd_safety_managed_roster_settings" in query:
            self.inserted = params
            self.last = None
        else:
            self.last = None

    def fetchone(self):
        return self.last


class SchoolStaffNamingTests(unittest.TestCase):
    def test_preserves_manually_configured_prefix(self):
        cur = FakeNamingCursor(existing="EXISTING")
        with patch.object(staff, "_audit") as audit:
            self.assertEqual(staff._ensure_school_naming(cur, "school1", "principal1"), ("EXISTING", False))
            audit.assert_not_called()
        self.assertIsNone(cur.inserted)

    def test_first_invite_initializes_school_prefix_without_identity_or_consent(self):
        cur = FakeNamingCursor()
        with patch.object(staff, "_audit") as audit:
            code, created = staff._ensure_school_naming(cur, "school1", "principal1")
            self.assertEqual((code, created), ("TS", True))
            audit.assert_called_once()
        self.assertEqual(cur.inserted, ("school1", "TS", "principal1"))
        self.assertTrue(any("pg_advisory_xact_lock" in query for query, _ in cur.queries))
        self.assertNotIn("verifyd_identities", inspect.getsource(staff._ensure_school_naming))
        self.assertNotIn("verifyd_safety_training_guardians", inspect.getsource(staff._ensure_school_naming))

    def test_collision_generates_unique_fallback_code(self):
        cur = FakeNamingCursor(taken={"TS"})
        with patch.object(staff, "_audit"):
            code, created = staff._ensure_school_naming(cur, "school1", "principal1")
        self.assertTrue(created)
        self.assertTrue(code.startswith("TS"))
        self.assertNotEqual(code, "TS")
        self.assertLessEqual(len(code), 10)

    def test_expired_staff_invite_cleared_without_activating_account(self):
        src = inspect.getsource(staff.create_staff_invite)
        self.assertIn("_ensure_school_naming", src)
        self.assertIn("expires_at<=NOW()", src)
        self.assertIn("school_prefix_auto_initialized", src)
        self.assertIn('"new_account_active":False', src)
        self.assertNotIn("INSERT INTO verifyd_identities", src)


if __name__ == "__main__":
    unittest.main()
