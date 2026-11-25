"""TEST School Safety family callbacks: optional notes, pending/history and safe bulk review."""
import inspect
import unittest
from fastapi import HTTPException
from pydantic import ValidationError
import trust_voice_school_family_interest as family

class CallbackReviewContracts(unittest.TestCase):
    def test_optional_single_and_bulk_notes(self):
        self.assertEqual(family.FamilyInterestDecision(status="reviewed").review_note,"")
        self.assertEqual(family.FamilyInterestDecision(status="closed").review_note,"")
        bulk=family.FamilyInterestBulkDecision(
            request_ids=["ssfinterest_"+"a"*32],
            school_staff_confirms_independent_review=True)
        self.assertEqual(bulk.review_note,"")
        self.assertTrue(bulk.require_electronic_acknowledgment)
        with self.assertRaises(ValidationError):
            family.FamilyInterestBulkDecision(
                request_ids=["ssfinterest_"+"a"*32],
                school_staff_confirms_independent_review=False)
        with self.assertRaises(ValidationError):
            family.FamilyInterestBulkDecision(
                request_ids=["ssfinterest_"+"a"*32]*51,
                school_staff_confirms_independent_review=True)

    def test_pending_refresh_and_completed_history_are_separate(self):
        src=inspect.getsource(family.list_school_family_interests)
        self.assertIn('view:Literal["pending","history","all"]="pending"',src)
        self.assertIn("awaiting_school_verification",src)
        self.assertIn("('reviewed','closed')",src)
        self.assertIn("_school_admin",src)
        self.assertIn("_matched_electronic_ack",src)

    def test_bulk_requires_authenticated_staff_and_independent_review(self):
        src=inspect.getsource(family.bulk_review_family_interests)
        self.assertIn("_school_admin(cur,school_id,str(actor[\"id\"]))",src)
        self.assertIn("family_callback_selection_changed_refresh",src)
        self.assertIn("family_callback_missing_verified_electronic_acknowledgment",src)
        self.assertIn("FOR UPDATE",src)
        self.assertIn("_audit",src)
        self.assertNotIn("INSERT INTO verifyd_safety_training_guardians",src)
        self.assertNotIn("INSERT INTO verifyd_identities",src)
        self.assertNotIn("UPDATE verifyd_safety_managed_students",src)

    def test_public_child_name_is_not_verification(self):
        src=inspect.getsource(family._matched_electronic_ack)
        self.assertIn("a.verified_at IS NOT NULL",src)
        self.assertIn("verifyd_safety_electronic_guardian_ack",src)
        self.assertIn("verifyd_safety_managed_students",src)
        self.assertIn("lower(a.guardian_email)=%s",src)
        self.assertIn("len(acknowledged)==len(children)",src)
        self.assertIn('row["contact_method"]!="email"',src)

    def test_existing_individual_review_still_distinct_from_approval(self):
        src=inspect.getsource(family.review_family_interest)
        self.assertIn("status='awaiting_school_verification'",src)
        self.assertIn("guardian_verified",src)
        self.assertIn("family_access_granted",src)
        self.assertIn("student_record_modified",src)

if __name__=="__main__":
    unittest.main()
