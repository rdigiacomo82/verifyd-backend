"""School-managed enrollment safety and authority boundaries (no database)."""
import inspect
import os
import unittest
from datetime import date, timedelta
from unittest.mock import MagicMock, patch

from fastapi import HTTPException
from pydantic import ValidationError
import trust_voice_school_managed_roster as roster


class SchoolManagedRosterContracts(unittest.TestCase):
    def test_feature_requires_explicit_enabled_school(self):
        with patch.dict(os.environ, {"VERIFYD_SAFETY_MANAGED_ROSTER_ENABLED": "0",
                                     "VERIFYD_SAFETY_MANAGED_ROSTER_SCHOOL_IDS": "school-1"}):
            with self.assertRaises(HTTPException) as error:
                roster._require_enrollment_enabled("school-1")
            self.assertEqual(error.exception.status_code, 404)
        with patch.dict(os.environ, {"VERIFYD_SAFETY_MANAGED_ROSTER_ENABLED": "1",
                                     "VERIFYD_SAFETY_MANAGED_ROSTER_SCHOOL_IDS": "school-1"}):
            roster._require_enrollment_enabled("school-1")
            with self.assertRaises(HTTPException):
                roster._require_enrollment_enabled("different-school")

    def test_handle_format_respects_24_character_global_limit(self):
        cursor = MagicMock()
        cursor.fetchone.return_value = None
        named = roster._new_handle(cursor,"WMHS","name","John","Smith",set())
        self.assertEqual(named,"WMHS_JohnSmith")
        code = roster._new_handle(cursor,"WMHS","code","John","Smith",set())
        self.assertRegex(code,r"^WMHS_S[0-9]{6}$")
        self.assertLessEqual(len(code),24)
        abbreviated = roster._new_handle(cursor,"WMHS","name","Alexandria",
                                          "VeryLongLastNameOfStudent",set())
        self.assertLessEqual(len(abbreviated),24)

    def test_duplicate_name_gets_numeric_suffix(self):
        cursor = MagicMock()
        cursor.fetchone.return_value = None
        used = set()
        self.assertEqual(roster._new_handle(cursor,"WMHS","name","John","Smith",used),
                         "WMHS_JohnSmith")
        self.assertEqual(roster._new_handle(cursor,"WMHS","name","John","Smith",used),
                         "WMHS_JohnSmith2")

    def test_missing_paper_signature_evidence_is_rejected(self):
        with self.assertRaises(HTTPException) as exc:
            roster._check_evidence(roster.SchoolAuthorizationItem(
                local_ref="A123",method="signed_paper",
                evidence_ref="folder-A-34",individually_verified=True))
        self.assertEqual(exc.exception.status_code,422)
        self.assertIn("paper_requires_guardian_name",str(exc.exception.detail))

    def test_future_paper_signature_is_rejected(self):
        with self.assertRaises(HTTPException) as exc:
            roster._check_evidence(roster.SchoolAuthorizationItem(
                local_ref="A123",method="signed_paper",
                guardian_name="Parent Smith",
                signed_date=date.today()+timedelta(days=2),
                evidence_ref="paper-A123",individually_verified=True))
        self.assertEqual(exc.exception.status_code,422)

    def test_school_authorization_requires_documented_authority(self):
        with self.assertRaises(HTTPException) as exc:
            roster._check_evidence(roster.SchoolAuthorizationItem(
                local_ref="A123",method="school_authority",
                evidence_ref="policy-2026",individually_verified=True))
        self.assertEqual(exc.exception.status_code,422)
        roster._check_evidence(roster.SchoolAuthorizationItem(
            local_ref="A123",method="school_authority",
            evidence_ref="policy-2026",authority_reference="signed-contract-v1",
            individually_verified=True))

    def test_batch_cannot_skip_per_student_verification(self):
        with self.assertRaises(ValidationError):
            roster.SchoolAuthorizationItem(
                local_ref="A123",method="signed_paper",evidence_ref="paper-A123",
                guardian_name="John Smith",signed_date=date.today(),
                individually_verified=False)
        with self.assertRaises(ValidationError):
            roster.SchoolAuthorizationBatch(approvals=[],principal_confirms_evidence_on_file=True,
                                            school_use_only=True)
        with self.assertRaises(ValidationError):
            roster.SchoolAuthorizationBatch(approvals=[roster.SchoolAuthorizationItem(
                local_ref="A123",method="school_authority",
                evidence_ref="document12345",authority_reference="signed-board-policy",
                individually_verified=True)],
                principal_confirms_evidence_on_file=False,school_use_only=True)

    def test_duplicates_rejected_before_database_or_approval(self):
        item = roster.SchoolAuthorizationItem(local_ref="A123",
            method="signed_paper",evidence_ref="paper-A123",
            guardian_name="Parent Smith",signed_date=date.today(),
            individually_verified=True)
        payload = roster.SchoolAuthorizationBatch(
            approvals=[item,item],principal_confirms_evidence_on_file=True,school_use_only=True)
        with patch.object(roster, "_require_enrollment_enabled", return_value=None), \
             patch.object(roster, "_actor",return_value={"id":"principal-1"}), \
             patch.object(roster,"_schema",side_effect=AssertionError("db must not be reached")):
            with self.assertRaises(HTTPException) as exc:
                roster.approve_managed_school_enrollment("school-1",payload)
        self.assertEqual(exc.exception.status_code,422)

    def test_school_admin_gate_and_no_global_signin_or_family_authorization(self):
        for fn in [roster.save_managed_roster_settings,roster.preview_managed_roster,
                   roster.commit_managed_roster,roster.list_managed_roster,
                   roster.approve_managed_school_enrollment]:
            self.assertIn("_school_admin(cur, school_id, str(actor[\"id\"]))",
                          inspect.getsource(fn))
        src = inspect.getsource(roster.approve_managed_school_enrollment)
        self.assertIn("'supervised_child'",src)
        self.assertIn("'nobody'",src)
        self.assertIn("guardian_private_message_access_granted",src)
        self.assertIn("emergency_dispatch_enabled",src)
        self.assertNotIn("verifyd_safety_training_guardians",src)

    def test_existing_consent_and_student_reporting_gates_are_unchanged(self):
        import trust_voice_school_safety_training as training
        source = inspect.getsource(training._create_training_report_for_student)
        self.assertIn("require_training_signoffs", source)
        self.assertIn("_verified_training_contact", source)
        self.assertIn("_school_response_recipients", source)


if __name__=="__main__":
    unittest.main()
