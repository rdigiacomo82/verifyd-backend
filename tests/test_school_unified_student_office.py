"""TEST-only unified Safety: supervised children must receive school-approved
office checks in the same Safety application, not by parent-triggered push.
"""
import inspect
import unittest
from trust_voice_school_guardian_office_requests import (
    _guardian_children,supervised_student_office_notices,
    supervised_student_office_seen, router
)

class UnifiedSchoolDeviceOfficeNoticeContracts(unittest.TestCase):
    def test_guardian_can_select_supervised_school_child_after_approval(self):
        sql=inspect.getsource(_guardian_children)
        self.assertIn("'supervised_child'",sql)
        self.assertIn("verifyd_safety_training_guardians",sql)
        self.assertIn("_verified_training_contact",sql)
        self.assertNotIn("guardian_email",sql)

    def test_student_device_only_sees_staff_approved_neutral_office_notice(self):
        src=inspect.getsource(supervised_student_office_notices)
        self.assertIn("from trust_voice_school_student_device import _session",src)
        self.assertIn('ctx["school_id"]',src)
        self.assertIn('ctx["student_id"]',src)
        self.assertIn("status='approved_to_notify'",src)
        self.assertIn("INTERVAL '1 day'",src)
        self.assertIn("student_device_delivery_guaranteed",src)
        self.assertNotIn("guardian_id",src)
        self.assertNotIn("guardian_email",src)
        self.assertNotIn("school_note",src)
        self.assertNotIn("self.registration",src)

    def test_seen_receipt_not_guaranteed_student_arrival(self):
        src=inspect.getsource(supervised_student_office_seen)
        self.assertIn("from trust_voice_school_student_device import _session",src)
        self.assertIn('ctx["student_id"]',src)
        self.assertIn('ctx["school_id"]',src)
        self.assertIn("student_viewed_at=COALESCE",src)
        self.assertIn("status='approved_to_notify'",src)
        self.assertIn("school_follow_up_required",src)
        self.assertNotIn("_actor(",src)

    def test_student_device_endpoints_exist_as_pilot_routes(self):
        paths={route.path for route in router.routes}
        self.assertIn("/student-devices/office-notices",paths)
        self.assertIn("/student-devices/office-notices/{request_id}/seen",paths)

if __name__ == "__main__":
    unittest.main()
