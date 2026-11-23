"""Security contract tests for isolated school family enrollment pilot."""
import inspect
import unittest
from unittest.mock import patch
from fastapi import HTTPException
from pydantic import ValidationError
import trust_voice_school_family_portal as family


class FamilyPortalSecurityContract(unittest.TestCase):
    def test_request_requires_valid_relationship(self):
        with self.assertRaises(ValidationError):
            family.FamilyRequest(school_id="sschool_testonly", student_handle="TestStudent01",
                                 relationship="self")
        valid = family.FamilyRequest(school_id="sschool_testonly", student_handle="TestStudent01",
                                    relationship="parent")
        self.assertEqual(valid.relationship, "parent")

    def test_approval_defaults_unverified(self):
        review = family.FamilyReview(decision="approve", note="Verified offline school record")
        self.assertFalse(review.verified_relationship)

    def test_every_route_rechecks_authenticated_identity(self):
        endpoints = (family.available_family_schools, family.request_family_link,
                     family.my_family_requests, family.staff_family_requests,
                     family.review_family_request, family.family_portal,
                     family.staff_school_updates, family.publish_school_update)
        for fn in endpoints:
            self.assertIn("_actor(authorization)", inspect.getsource(fn), fn.__name__)

    def test_no_guardian_self_approval_or_automatic_access(self):
        source = inspect.getsource(family.review_family_request)
        self.assertIn("_school_admin(cur,school_id,actor_id)", source)
        self.assertIn('if not payload.verified_relationship:', source)
        self.assertIn('cannot_approve_own_family_link', source)
        self.assertIn('{"student"}',source)
        self.assertIn('existing["role"]!="guardian"', source)

    def test_portal_requires_active_guardian_role_and_policy_acknowledgment(self):
        source = inspect.getsource(family.family_portal)
        self.assertIn("m.role='guardian'", source)
        self.assertIn("x.kind='guardian'", source)
        self.assertIn("x.revoked_at IS NULL", source)
        self.assertIn("x.policy_version=%s", source)
        self.assertNotIn("emergency_notification", source)

    def test_school_update_cannot_be_used_as_dispatch(self):
        source = inspect.getsource(family.publish_school_update)
        self.assertIn("_school_admin", source)
        self.assertIn("use_official_emergency_channels", source)
        self.assertIn('"in_app_only":True', source)

    def test_no_unauthenticated_school_directory(self):
        with patch.object(family, "_actor", side_effect=HTTPException(401, "session_required")):
            with self.assertRaises(HTTPException) as problem:
                family.available_family_schools(authorization=None)
            self.assertEqual(problem.exception.status_code, 401)


if __name__ == "__main__":
    unittest.main()
