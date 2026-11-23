"""Safety batch roster: identity isolation and no silent elevation."""
import inspect
import unittest
from unittest.mock import patch
from fastapi import HTTPException
from pydantic import ValidationError
import trust_voice_school_roster_batch as batch


class SchoolRosterBatchContract(unittest.TestCase):
    def test_batch_limit(self):
        with self.assertRaises(ValidationError):
            batch.BatchInput(handles=["test01"]*51,role="student")

    def test_duplicate_and_invalid_handles_are_rejected(self):
        with self.assertRaises(HTTPException):
            batch._normalized(["Student01","student01"])
        with self.assertRaises(HTTPException):
            batch._normalized(["student01","student 02"])

    def test_preview_and_commit_authorize_existing_school_admin(self):
        for function in (batch.preview_batch,batch.commit_batch,batch.list_roster_references):
            source=inspect.getsource(function)
            self.assertIn("_actor(authorization)",source)
            self.assertIn("_school_admin(cur,school_id",source)

    def test_batch_does_not_grant_protected_roles(self):
        self.assertEqual(set(batch.BatchInput.model_fields["role"].annotation.__args__),{"student","guardian"})
        source=inspect.getsource(batch.commit_batch)
        self.assertIn("cannot_change_own_school_role",source)
        self.assertIn("conflicting_existing_role_review_each_account",source)
        self.assertNotIn("DELETE FROM verifyd_identities",source)

    def test_preview_never_returns_email_or_dob(self):
        source=inspect.getsource(batch._resolve)
        self.assertNotIn("email_lower",source)
        self.assertNotIn("birth_date",source)
        self.assertNotIn("student_number",source)

    def test_batch_cannot_be_called_without_session(self):
        with patch.object(batch, "_actor", side_effect=HTTPException(401, "session_required")):
            with self.assertRaises(HTTPException) as e:
                batch.preview_batch("sschool_test", batch.BatchInput(handles=["student01"]), None)
            self.assertEqual(e.exception.status_code, 401)


if __name__ == "__main__":
    unittest.main()
