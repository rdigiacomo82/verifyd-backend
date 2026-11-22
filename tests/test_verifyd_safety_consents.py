"""School Safety TRAINING only: role, signoff, version, and report gating tests.

No real students, no network, no database writes, no incident delivery.
"""
import os
import unittest
from unittest.mock import Mock, patch
from fastapi import HTTPException
from pydantic import ValidationError

import trust_voice_school_safety_consents as consent


class SchoolTrainingSignoffsTests(unittest.TestCase):
    def test_policy_is_specific_version_and_disclaims_emergency_service(self):
        policy = consent.training_signoff_policy()
        self.assertEqual(policy["policy_version"], consent.POLICY_VERSION)
        self.assertFalse(policy["live_emergency_delivery"])
        self.assertFalse(policy["location_tracking"])
        self.assertEqual(set(policy["policies"]), {"school", "guardian", "student"})

    def test_acceptance_must_be_explicit_and_version_current(self):
        with self.assertRaises(ValidationError):
            consent.SignoffInput(kind="student", accepted=False, policy_version=consent.POLICY_VERSION)
        with self.assertRaises(ValidationError):
            consent.SignoffInput(kind="student", accepted=True, policy_version="old-version")

    def test_principal_not_self_assigned_by_checking_school_box(self):
        with patch.object(consent, "_membership", return_value="student"):
            with self.assertRaises(HTTPException) as raised:
                consent._scope(Mock(), "school", "student", "school", None)
        self.assertEqual(raised.exception.status_code, 403)

    def test_student_cannot_acknowledge_someone_else(self):
        with patch.object(consent, "_membership", return_value="student"):
            with self.assertRaises(HTTPException) as raised:
                consent._scope(Mock(), "school", "student1", "student", "student2")
        self.assertEqual(raised.exception.status_code, 403)

    def test_guardian_must_have_specific_operator_verified_link(self):
        cur = Mock()
        cur.fetchone.return_value = None
        with patch.object(consent, "_membership", return_value="guardian"):
            with self.assertRaises(HTTPException) as raised:
                consent._scope(cur, "school", "parent", "guardian", "child")
        self.assertEqual(raised.exception.status_code, 403)

    def test_valid_guardian_requires_link_and_student_enrollment(self):
        cur = Mock()
        cur.fetchone.return_value = {"exists": True}
        with patch.object(consent, "_membership", side_effect=["guardian", "student"]):
            self.assertEqual(consent._scope(cur, "school", "parent", "guardian", "child"), "child")
        self.assertIn("verifyd_safety_training_guardians", cur.execute.call_args.args[0])

    def test_report_requires_all_three_independent_roles(self):
        cur = Mock()
        with patch.object(consent, "_school_accepted", return_value=False), \
             patch.object(consent, "_student_accepted", return_value=True), \
             patch.object(consent, "_guardian_count", return_value=1):
            with self.assertRaises(HTTPException) as raised:
                consent.require_training_signoffs(cur, "school", "student")
        self.assertEqual(raised.exception.detail["error"], "school_training_signoff_required")
        with patch.object(consent, "_school_accepted", return_value=True), \
             patch.object(consent, "_student_accepted", return_value=False), \
             patch.object(consent, "_guardian_count", return_value=1):
            with self.assertRaises(HTTPException) as raised:
                consent.require_training_signoffs(cur, "school", "student")
        self.assertEqual(raised.exception.detail["error"], "student_training_acknowledgment_required")
        with patch.object(consent, "_school_accepted", return_value=True), \
             patch.object(consent, "_student_accepted", return_value=True), \
             patch.object(consent, "_guardian_count", return_value=0):
            with self.assertRaises(HTTPException) as raised:
                consent.require_training_signoffs(cur, "school", "student")
        self.assertEqual(raised.exception.detail["error"], "verified_guardian_training_acknowledgment_required")
        with patch.object(consent, "_school_accepted", return_value=True), \
             patch.object(consent, "_student_accepted", return_value=True), \
             patch.object(consent, "_guardian_count", return_value=1):
            self.assertIsNone(consent.require_training_signoffs(cur, "school", "student"))

    def test_recording_uses_actor_identity_not_request_author_claim(self):
        # School role, student subject, and linked guardian all derive from
        # authenticated actor and operator-provisioned membership.
        self.assertFalse("actor_id" in consent.SignoffInput.model_fields)
        self.assertFalse("school_id" in consent.SignoffInput.model_fields)

    def test_feature_off_fails_closed(self):
        with patch.dict(os.environ, {"VERIFYD_SAFETY_TRAINING_ENABLED": "0"}):
            with self.assertRaises(HTTPException) as raised:
                consent._actor("Bearer sample")
            self.assertEqual(raised.exception.status_code, 503)


if __name__ == "__main__":
    unittest.main()
