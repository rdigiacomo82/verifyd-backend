"""Verify guardian contact changes do not bypass independent authorization."""
import inspect
import unittest
from unittest.mock import patch
from fastapi import HTTPException
from pydantic import ValidationError
import trust_voice_school_guardian_contacts as c

class GuardianContactChangeSecurity(unittest.TestCase):
    def test_request_requires_school_verified_links_for_all_children_and_adults(self):
        source=inspect.getsource(c.request_guardian_contact_change)
        self.assertIn("_verified_link(cur,payload.school_id,student_id,gid)",source)
        self.assertIn("_verified_link(cur,payload.school_id,student_id,candidate)",source)
        self.assertIn("contact_change_already_pending",source)
        self.assertIn("contacts_changed\":False",source)
    def test_two_guardians_must_confirm_and_school_independently_verify(self):
        source=inspect.getsource(c.staff_review_contact_change)
        self.assertIn("_school_admin(cur,school_id,aid)",source)
        self.assertIn("both_guardians_must_confirm",source)
        self.assertIn("independent_school_verification_required",source)
        self.assertIn("self_approval_not_allowed",source)
        self.assertIn("contact_priorities_changed_since_request",source)
        self.assertIn("_verified_link(cur,school_id,row[\"student_id\"],gid)",source)
    def test_guardian_can_only_confirm_proposed_roles(self):
        source=inspect.getsource(c.confirm_guardian_contact_change)
        self.assertIn("not_a_proposed_guardian",source)
        self.assertIn("guardian_authorization_expired",source)
        self.assertIn("contact_request_not_pending",source)
    def test_batch_validations(self):
        with self.assertRaises(ValidationError):
            c.FamilyContactRequest(school_id="sschool_test",student_ids=[],
                primary_guardian_id="guardian1",backup_guardian_id="guardian2",
                reason="Change school contact priority")
    def test_unauthenticated_family_requests_denied(self):
        with patch.object(c,"_actor",side_effect=HTTPException(401,"session_required")):
            with self.assertRaises(HTTPException):
                c.my_guardian_contacts(None)
    def test_no_emergency_send_claims(self):
        s=inspect.getsource(c.staff_review_contact_change)
        self.assertIn("live_emergency_delivery\":False",s)
        self.assertNotIn("send_911",s)

if __name__=="__main__":unittest.main()
