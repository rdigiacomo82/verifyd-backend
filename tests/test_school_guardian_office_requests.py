"""Guardian school-office requests are NOT an unrestricted distress call or 911."""
import inspect
import os
import unittest
from unittest.mock import patch
from fastapi import HTTPException
from pydantic import ValidationError
import trust_voice_school_guardian_office_requests as office
import trust_voice_school_family_interest as registration


class GuardianOfficePilotSafetyContracts(unittest.TestCase):
    def test_disabled_by_default_and_school_allowlist_required(self):
        with patch.dict(os.environ,{},clear=True):
            with self.assertRaises(HTTPException) as cm:
                office._gate("sschool_any")
            self.assertEqual(cm.exception.status_code,404)
        with patch.dict(os.environ,{
            "VERIFYD_SAFETY_GUARDIAN_OFFICE_REQUEST_ENABLED":"1",
            "VERIFYD_SAFETY_GUARDIAN_OFFICE_REQUEST_SCHOOL_IDS":"sschool_allowed"
        }):
            office._gate("sschool_allowed")
            with self.assertRaises(HTTPException):
                office._gate("sschool_other")

    def test_guardian_request_requires_confirmed_policy_and_real_guardian_link(self):
        with self.assertRaises(ValidationError):
            office.GuardianOfficeRequest(school_id="sschool_test",student_id="childtest123",
                purpose="urgent_family_contact",
                understands_school_mediated_not_emergency=False,
                agrees_not_to_abuse_school_requests=True)
        src=inspect.getsource(office.create_office_request)
        for assertion in ("_verified_training_contact","_school_response_authorized",
                          "guardian_office_daily_limit","guardian_office_request_already_open",
                          "pending_school","student_not_notified"):
            self.assertIn(assertion,src)
        self.assertNotIn("_send(",src)
        self.assertNotIn("911",src)

    def test_school_action_required_before_any_child_notice(self):
        school=inspect.getsource(office.staff_review_office_request)
        self.assertIn("_school_response_authorized",school)
        self.assertIn("staff_confirms_school_follow_up",inspect.getsource(office.StaffDecision))
        student=inspect.getsource(office.student_office_notices)
        self.assertIn("q.status='approved_to_notify'",student)
        self.assertIn("m.role='student'",student)
        self.assertNotIn("guardian_email",student)
        self.assertNotIn("guardian_id",student)
        self.assertNotIn("push",student)

    def test_response_status_is_scoped_per_account(self):
        parent=inspect.getsource(office.my_office_requests)
        staff=inspect.getsource(office.staff_office_queue)
        self.assertIn("WHERE guardian_id=%s",parent)
        self.assertIn("_school_response_authorized",staff)
        self.assertIn("WHERE q.school_id=%s",staff)
        self.assertNotIn("private_family_messages",staff)

    def test_routes_registered_under_single_school_prefix(self):
        paths={r.path for r in registration.router.routes}
        self.assertIn("/trust-voice/school-safety/family/office-requests",paths)
        self.assertIn("/trust-voice/school-safety/student/office-notices",paths)
        self.assertIn("/trust-voice/school-safety/schools/{school_id}/guardian-office-requests",paths)


if __name__ == "__main__":
    unittest.main()
