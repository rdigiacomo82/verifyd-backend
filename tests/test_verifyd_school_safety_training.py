"""Safe, non-network unit checks for school safety TRAINING ONLY backend."""
import os
import unittest
from unittest.mock import patch, MagicMock
from fastapi import HTTPException

import trust_voice_school_safety_training as safety


class SchoolSafetyTrainingTests(unittest.TestCase):
    def test_default_off(self):
        with patch.dict(os.environ, {"VERIFYD_SAFETY_TRAINING_ENABLED": "0"}):
            self.assertFalse(safety._enabled())
            with self.assertRaises(HTTPException) as error:
                safety._require()
            self.assertEqual(error.exception.status_code, 503)

    def test_training_health_never_claims_emergency_delivery(self):
        with patch.dict(os.environ, {"VERIFYD_SAFETY_TRAINING_ENABLED": "1"}):
            body = safety.health()
        self.assertTrue(body["training_enabled"])
        self.assertFalse(body["live_emergency_delivery"])
        self.assertFalse(body["emergency_services_connected"])
        self.assertFalse(body["location_sharing"])

    def test_training_receipt_does_not_claim_staff_or_parent_notification(self):
        result = safety._report_receipt(
            {"report_id": "report-test", "incident_id": "incident-test"}, 12
        )
        self.assertTrue(result["stored_on_server"])
        self.assertEqual(result["incident_reports"], 12)
        self.assertEqual(result["delivery_status"], "TRAINING_RECORD_ONLY")
        self.assertFalse(result["staff_notified"])
        self.assertFalse(result["parents_notified"])
        self.assertFalse(result["emergency_services_notified"])

    def test_unauthorized_operator_cannot_provision_school(self):
        with patch.dict(os.environ, {"VERIFYD_SAFETY_TRAINING_OPERATOR_IDS": ""}):
            with self.assertRaises(HTTPException) as error:
                safety._operator({"id": "not-authorized"})
            self.assertEqual(error.exception.status_code, 403)

    def test_test_route_removal_is_explicitly_scoped_to_verified_admin(self):
        from trust_voice_school_safety_registration import _school_admin
        conn = MagicMock()
        cursor = conn.__enter__.return_value.cursor.return_value
        cursor.fetchone.return_value = {"primary_id": "guardian-1", "backup_id": "principal-1"}
        with patch.object(safety, "_actor", return_value={"id": "principal-1"}), \
             patch.object(safety, "_schema"), \
             patch.object(safety, "get_db", return_value=conn), \
             patch("trust_voice_school_safety_registration._school_admin") as admin, \\
             patch.object(safety, "_membership") as membership, \\
             patch.object(safety, "_audit") as audit:
            result = safety.disable_verified_school_contact_routing(
                "test-school", "student-1", authorization="test-session"
            )
        self.assertTrue(result["removed"])
        self.assertTrue(result["test_reporting_disabled_until_new_route"])
        self.assertFalse(result["accounts_deleted"])
        self.assertFalse(result["prior_reports_deleted"])
        self.assertFalse(result["notifications_sent"])
        admin.assert_called_once_with(cursor, "test-school", "principal-1")
        membership.assert_called_once_with(cursor, "test-school", "student-1", {"student"})
        sql, args = cursor.execute.call_args.args
        self.assertIn("DELETE FROM verifyd_safety_training_contacts", sql)
        self.assertIn("RETURNING primary_id, backup_id", sql)
        self.assertEqual(args, ("test-school", "student-1"))
        self.assertIn("verified_training_contact_route_disabled", str(audit.call_args))

    def test_nonadmin_cannot_clear_school_contact_route(self):
        conn = MagicMock()
        cursor = conn.__enter__.return_value.cursor.return_value
        with patch.object(safety, "_actor", return_value={"id": "guardian-1"}), \
             patch.object(safety, "_schema"), \
             patch.object(safety, "get_db", return_value=conn), \
             patch("trust_voice_school_safety_registration._school_admin", side_effect=HTTPException(403, detail="school_roster_admin_required")):
            with self.assertRaises(HTTPException) as exc:
                safety.disable_verified_school_contact_routing(
                    "test-school", "student-1", authorization="guardian-session"
                )
        self.assertEqual(exc.exception.status_code, 403)
        cursor.execute.assert_not_called()

    def test_test_route_removal_is_idempotent_without_claiming_deletion(self):
        conn = MagicMock()
        cursor = conn.__enter__.return_value.cursor.return_value
        cursor.fetchone.return_value = None
        with patch.object(safety, "_actor", return_value={"id": "principal-1"}), \
             patch.object(safety, "_schema"), \
             patch.object(safety, "get_db", return_value=conn), \
             patch("trust_voice_school_safety_registration._school_admin"), \
             patch.object(safety, "_membership"), \
             patch.object(safety, "_audit") as audit:
            result = safety.disable_verified_school_contact_routing(
                "test-school", "student-1", authorization="test-session"
            )
        self.assertFalse(result["removed"])
        audit.assert_not_called()

    def test_same_contact_is_rejected_before_database_use(self):
        with patch.object(safety, "_actor", return_value={"id": "principal-1"}):
            with patch.object(safety, "_schema", side_effect=AssertionError("DB should not be touched")):
                with self.assertRaises(HTTPException) as error:
                    safety.save_verified_school_contact_routing(
                        "training-school", "student-1",
                        safety.SchoolVerifiedContactRoute(
                            primary_id="guardian-1", backup_id="guardian-1",
                            independently_verified=True))
                self.assertEqual(error.exception.status_code, 422)

    def test_student_contact_self_assignment_is_denied(self):
        with patch.object(safety, "_actor", return_value={"id": "student-1"}):
            with self.assertRaises(HTTPException) as error:
                safety.set_training_contacts("training-school", safety.ContactInput(
                    primary_id="guardian-1", backup_id="admin-1"))
            self.assertEqual(error.exception.status_code, 403)

    def test_create_requires_authenticated_session_even_if_training_enabled(self):
        with patch.dict(os.environ, {"VERIFYD_SAFETY_TRAINING_ENABLED": "0"}):
            with self.assertRaises(HTTPException) as error:
                safety.create_training_report(safety.ReportInput(
                    school_id="test-school", category="fire",
                    idempotency_key="0123456789abcdef"), "Bearer bogus")
            self.assertEqual(error.exception.status_code, 503)

    def test_update_and_resolve_models_require_valid_status(self):
        from pydantic import ValidationError
        with self.assertRaises(ValidationError):
            safety.UpdateInput(status="official_all_clear", message="fake")
        with self.assertRaises(ValidationError):
            safety.ResolveInput(reason="ok")

    def test_training_end_confirmation_required(self):
        from pydantic import ValidationError
        with self.assertRaises(ValidationError):
            safety.EndInput(confirmation=False)

    def test_staff_initial_report_api_never_exposes_student_followup_status(self):
        import inspect
        incidents = inspect.getsource(safety.training_school_incidents)
        reports = inspect.getsource(safety.incident_training_reports)
        # Principals and approved responders see the original description/category,
        # not the child's later 'need_help' status, update history or closure.
        self.assertIn("COUNT(r.report_id) AS report_count", incidents)
        self.assertNotIn("r.student_status", incidents)
        self.assertNotIn("r.student_status", reports)
        self.assertNotIn("r.closed_at", reports)
        self.assertIn("r.description", reports)

    def test_live_calling_functions_are_not_registered_in_training_router(self):
        paths = {r.path for r in safety.router.routes}
        self.assertIn("/trust-voice/school-safety/reports", paths)
        self.assertNotIn("/trust-voice/safety/distress", paths)
        self.assertNotIn("/trust-voice/push/subscriptions", paths)


if __name__ == "__main__":
    unittest.main()
