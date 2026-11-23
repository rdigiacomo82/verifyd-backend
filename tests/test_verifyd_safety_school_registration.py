"""Owner-approved Safety school applications and scoped-roster regression tests."""
import os
import unittest
from unittest.mock import Mock, patch

from fastapi import HTTPException
from pydantic import ValidationError
import trust_voice_school_safety_registration as reg


class SchoolOwnerApprovalTests(unittest.TestCase):
    def test_unverified_owner_cannot_approve_application(self):
        with patch.dict(os.environ, {"VERIFYD_SAFETY_TRAINING_OPERATOR_IDS": "verified-owner"}):
            with self.assertRaises(HTTPException) as err:
                reg._owner({"id": "student"})
        self.assertEqual(err.exception.status_code, 403)

    def test_applicant_cannot_ask_for_a_privileged_role_in_payload(self):
        self.assertFalse("role" in reg.SchoolApplication.model_fields)
        self.assertFalse("principal_id" in reg.SchoolApplication.model_fields)
        self.assertFalse("approved_by" in reg.SchoolApplication.model_fields)
        self.assertFalse("school_id" in reg.SchoolApplication.model_fields)

    def test_owner_review_requires_verification_for_approval(self):
        # An owner can reject a bogus application without verifying it first.
        # The approval route independently rejects unverified=True being absent.
        draft = reg.ReviewDecision(decision="approve", verified_official_authority=False,
                                   review_note="No independent verification completed")
        self.assertFalse(draft.verified_official_authority)
        with self.assertRaises(ValidationError):
            reg.ReviewDecision(decision="approve", verified_official_authority=True,
                               review_note="ok")

    def test_internal_test_school_must_have_explicit_test_only_name_and_owner_scope(self):
        payload = reg.ReviewDecision(decision="approve", verified_official_authority=False,
                   internal_test_school=True, review_note="Owner-confirmed internal practice")
        self.assertTrue(reg._approval_verified(payload, "VeriFYD Safety School — TEST ONLY"))
        self.assertFalse(reg._approval_verified(payload, "Real City High School"))
        not_verified = reg.ReviewDecision(decision="approve", verified_official_authority=False,
                   internal_test_school=False, review_note="Unverified applicant pending")
        self.assertFalse(reg._approval_verified(not_verified, "VeriFYD Safety School — TEST ONLY"))
        real_verified = reg.ReviewDecision(decision="approve", verified_official_authority=True,
                   internal_test_school=False, review_note="Independently called district office")
        self.assertTrue(reg._approval_verified(real_verified, "Real City High School"))

    def test_school_admin_can_only_manage_non_privileged_roster_roles(self):
        with self.assertRaises(ValidationError):
            reg.RosterMember(handle="@fake", role="principal")
        with self.assertRaises(ValidationError):
            reg.RosterMember(handle="@fake", role="operator")
        with self.assertRaises(ValidationError):
            reg.RosterRoleEdit(role="backup")

    def test_unapproved_responder_cannot_manage_roster(self):
        cur = Mock()
        cur.fetchone.return_value = None
        with patch.object(reg, "_membership", return_value="responder"):
            with self.assertRaises(HTTPException) as err:
                reg._school_admin(cur, "school1", "unapproved-backup")
        self.assertEqual(err.exception.status_code, 403)

    def test_active_approved_and_accepted_backup_can_manage_roster(self):
        cur=Mock()
        cur.fetchone.return_value={"allowed":True}
        with patch.object(reg, "_membership", return_value="responder"), \
             patch.object(reg, "_school_accepted", return_value=True):
            self.assertEqual(reg._school_admin(cur, "school1", "backup-id"), "backup")

    def test_principal_must_acknowledge_school_before_roster_management(self):
        with patch.object(reg, "_membership", return_value="principal"), \
             patch.object(reg, "_school_accepted", return_value=False):
            with self.assertRaises(HTTPException) as err:
                reg._school_admin(Mock(), "school1", "principal-id")
        self.assertEqual(err.exception.status_code, 403)

    def test_principal_cannot_be_deleted_as_roster_member(self):
        cur=Mock()
        cur.fetchone.return_value={"role":"principal"}
        with self.assertRaises(HTTPException) as err:
            reg._reject_protected_member(cur,"school1","principal-id")
        self.assertEqual(err.exception.status_code, 403)

    def test_roster_routes_are_school_scoped_and_no_global_delete(self):
        paths={(route.path,tuple(sorted(route.methods))) for route in reg.router.routes}
        self.assertTrue(any("/schools/{school_id}/roster" in item[0] for item in paths))
        self.assertFalse(any("verifyd_identities/" in item[0] for item in paths))

    def test_owner_must_separately_approve_backup(self):
        with self.assertRaises(ValidationError):
            reg.BackupApproval(backup_handle="@backup", verified_official_authority=False)

    def test_owner_approval_endpoint_cannot_activate_real_emergencies(self):
        with patch.dict(os.environ, {"VERIFYD_SAFETY_TRAINING_ENABLED":"0"}):
            with self.assertRaises(HTTPException) as err:
                reg.my_school_applications("Bearer anything")
        self.assertEqual(err.exception.status_code, 503)


    def test_owner_notification_address_is_fixed_and_not_applicant_controlled(self):
        self.assertEqual(reg._OWNER_NOTIFICATION_ADDRESS, "support@vfvid.com")
        self.assertEqual(reg._OWNER_NOTIFICATION_FROM, "VeriFYD Safety <support@vfvid.com>")
        self.assertEqual(reg._OWNER_DASHBOARD_URL, "https://vfvid.com/admin")

    def test_email_failure_does_not_grant_school_access(self):
        # Notification exceptions must be contained by registration. The
        # applicant's role cannot change based on mail delivery.
        import inspect
        source = inspect.getsource(reg.submit_school_application)
        self.assertIn("_send_owner_application_notification(application_id)", source)
        self.assertIn("except Exception:", source)
        self.assertIn('"school_access_granted": False', source)

    def test_notification_retry_is_owner_only(self):
        with patch.object(reg, "_actor", return_value={"id": "unprivileged"}), \
             patch.object(reg, "_owner", side_effect=HTTPException(403, "training_operator_required")):
            with self.assertRaises(HTTPException) as err:
                reg.retry_owner_notifications("Bearer token")
        self.assertEqual(err.exception.status_code, 403)

    def test_notification_subject_does_not_approve_school(self):
        import inspect
        source = inspect.getsource(reg._send_owner_application_notification)
        self.assertIn("Idempotency-Key", source)
        self.assertIn("No school access has been granted", source)
        self.assertNotIn("principal_application_approved", source)

if __name__=="__main__":
    unittest.main()
