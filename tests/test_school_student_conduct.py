"""VeriFYD TEST student acceptable-use contract: student assent and school review."""
import inspect
import os
import unittest
from unittest.mock import patch
from fastapi import HTTPException
from pydantic import ValidationError

import trust_voice_school_student_conduct as conduct
import trust_voice_school_safety_consents as consent
import trust_voice_school_student_device as device


class StudentConductPolicyTests(unittest.TestCase):
    def test_default_off_and_school_only_allowlist(self):
        with patch.dict(os.environ,{},clear=True):
            self.assertFalse(conduct._enabled("sschool_test"))
        with patch.dict(os.environ,{
            "VERIFYD_SAFETY_STUDENT_CONDUCT_ENABLED":"1",
            "VERIFYD_SAFETY_STUDENT_CONDUCT_SCHOOL_IDS":"sschool_allowed",
        },clear=True):
            self.assertTrue(conduct._enabled("sschool_allowed"))
            self.assertFalse(conduct._enabled("sschool_other"))

    def test_parent_and_principal_policy_versions_are_unchanged(self):
        self.assertEqual(consent.POLICY_VERSION,"2026-11-22.training.v1")
        policy=consent.training_signoff_policy()
        self.assertEqual(policy["student_conduct"]["version"],conduct.CONDUCT_POLICY_VERSION)
        self.assertTrue(policy["student_conduct"]["student_signs_separately"])
        self.assertGreaterEqual(len(policy["student_conduct"]["statements"]),6)
        self.assertTrue(any("good-faith" in s.lower() for s in conduct.CONDUCT_RULES))
        self.assertTrue(any("school may review" in s.lower() for s in conduct.CONDUCT_RULES))

    def test_student_must_accept_versioned_both_rules(self):
        source=inspect.getsource(consent.accept_signoff)
        self.assertIn('payload.kind == "student"',source)
        self.assertIn('payload.understands_acceptable_use',source)
        self.assertIn('payload.understands_school_review_consequences',source)
        self.assertIn('record_conduct',source)
        self.assertIn('_scope(cur, school_id, actor_id',source)
        payload=consent.SignoffInput(kind="student",policy_version=consent.POLICY_VERSION,
                                    accepted=True)
        self.assertFalse(payload.understands_acceptable_use)

    def test_supervised_device_requires_student_not_guardian_acceptance(self):
        student=inspect.getsource(device.device_student_acknowledgment)
        self.assertIn("conduct_policy_version",student)
        self.assertIn("understands_acceptable_use",student)
        self.assertIn("understands_school_review_consequences",student)
        self.assertIn("record_conduct",student)
        self.assertIn("_session(x_safety_device_session)",student)
        status=inspect.getsource(device.student_device_me)
        self.assertIn("student_conduct",status)
        self.assertIn('not conduct["restricted"]',status)

    def test_staff_may_not_restrict_without_recorded_school_review(self):
        with self.assertRaises(ValidationError):
            conduct.StaffConductDecision(decision="restrict_test_privileges",reason="bad",
                school_confirms_alternative_safety_reporting=True,
                school_confirms_fair_review=True)
        with self.assertRaises(ValidationError):
            conduct.StaffConductDecision(decision="restrict_test_privileges",
                reason="School tested a documented violation.",
                school_confirms_alternative_safety_reporting=False,
                school_confirms_fair_review=True)
        source=inspect.getsource(conduct.admin_student_conduct_decision)
        self.assertIn("_school_admin",source)
        self.assertIn('review_reason=%s',source)
        self.assertIn("restricted_at=NOW()",source)
        self.assertIn("school_alternative_reporting_required",source)
        self.assertNotIn("UPDATE verifyd_identities",source)

    def test_training_report_requires_separate_student_conduct(self):
        source=inspect.getsource(consent.require_training_signoffs)
        self.assertIn("_require_accepted",source)
        check=inspect.getsource(conduct._require_accepted)
        self.assertIn("school_student_practice_access_restricted",check)
        self.assertIn("student_conduct_acknowledgment_required",check)
        self.assertNotIn("911",check)

    def test_role_routing_is_school_scoped(self):
        # The full app resolves included routes; FastAPI's internal test router
        # descriptor is not a standalone APIRoute in this repository's test stack.
        source=inspect.getsource(conduct)
        self.assertIn('router=APIRouter(tags=["TEST Student Acceptable Use"])',source)
        self.assertIn('@router.get("/schools/{school_id}/student-conduct/status")',source)
        self.assertIn('@router.post("/schools/{school_id}/student-conduct/{student_id}/decision")',source)
        self.assertTrue(hasattr(consent, "_student_conduct_router"))
        self.assertIn("router.include_router(_student_conduct_router)",inspect.getsource(consent))


if __name__=="__main__":
    unittest.main()
