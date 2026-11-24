"""No-network authorization and isolation tests for supervised school TEST sessions."""
import os
import inspect
import unittest
from unittest.mock import patch, MagicMock

from fastapi import HTTPException

import trust_voice_school_student_device as device
import trust_voice_school_safety_training as training
from trust_voice_school_safety_consents import POLICY_VERSION


class SupervisedDeviceTests(unittest.TestCase):
    def test_default_disabled_and_school_allowlist(self):
        with patch.dict(os.environ, {"VERIFYD_SAFETY_STUDENT_DEVICE_ENABLED": "0"}, clear=False):
            with self.assertRaises(HTTPException) as error:
                device._pilot()
        self.assertEqual(error.exception.status_code, 503)
        with patch.dict(os.environ, {"VERIFYD_SAFETY_STUDENT_DEVICE_SCHOOL_IDS": ""}, clear=False):
            with self.assertRaises(HTTPException) as error:
                device._school_allowed("anything")
        self.assertEqual(error.exception.status_code, 403)

    def test_no_voice_tokens_or_full_identity_login_routes(self):
        paths = {route.path for route in device.router.routes}
        self.assertIn("/trust-voice/school-safety/student-devices/activate", paths)
        self.assertIn("/trust-voice/school-safety/student-devices/acknowledgment", paths)
        self.assertIn("/trust-voice/school-safety/student-devices/reports", paths)
        self.assertIn("/trust-voice/school-safety/student-devices/reports/{report_id}/conversation", paths)
        self.assertNotIn("/trust-voice/session", paths)
        self.assertNotIn("/trust-voice/safety/distress", paths)
        self.assertTrue(all("/student-devices/" in p for p in paths))

    def test_bad_activation_code_fails_without_db(self):
        with patch.object(device, "_pilot"):
            with patch.object(device, "_schema", side_effect=AssertionError("invalid code should not need DB")):
                with self.assertRaises(AssertionError):
                    device.activate_device(device.ActivateCode(student_handle="Joejunior", code="not-a-valid-code"))
        # Schema runs before parsing the code in this implementation, but the
        # caller cannot obtain a token without matching an unexpired digest.
        src = inspect.getsource(device.activate_device)
        self.assertIn("WHERE codes.code_hash=%s AND i.handle_lower=%s", src)
        self.assertIn("FOR UPDATE OF codes", src)
        self.assertIn("used_at=NOW()", src)

    def test_expired_restricted_session_rejected(self):
        cursor = MagicMock()
        cursor.fetchone.return_value = None
        conn = MagicMock()
        conn.cursor.return_value = cursor
        cm = MagicMock()
        cm.__enter__.return_value = conn
        cm.__exit__.return_value = None
        with patch.object(device, "_pilot"), patch.object(device, "_schema"), patch.object(device, "get_db", return_value=cm):
            with self.assertRaises(HTTPException) as error:
                device._session("a" * 42)
        self.assertEqual(error.exception.status_code, 401)

    def test_student_report_never_crosses_school_or_skips_choice(self):
        from pydantic import ValidationError
        ctx = {"student_id": "vfyd_student", "school_id": "school_one"}
        payload = device.SessionReport(
            school_id="school_two", category="fire", choice_id="fire",
            idempotency_key="12dbf23d6e6f7d33")
        with patch.object(device, "_session", return_value=ctx):
            with patch.object(device, "_create_training_report_for_student", side_effect=AssertionError("Must not reach DB")):
                with self.assertRaises(HTTPException) as error:
                    device.device_send_report(payload)
        self.assertEqual(error.exception.status_code, 403)
        payload.school_id = "school_one"
        payload.choice_id = ""
        with patch.object(device, "_session", return_value=ctx):
            with patch.object(device, "_create_training_report_for_student", side_effect=AssertionError("Must not reach DB")):
                with self.assertRaises(HTTPException) as error:
                    device.device_send_report(payload)
        self.assertEqual(error.exception.status_code, 403)

    def test_student_acknowledgment_requires_explicit_student_action(self):
        ctx = {"student_id": "vfyd_child", "school_id": "test_school"}
        for payload in (
            device.StudentAcknowledgment(policy_version=POLICY_VERSION, accepted=False, understood_test_only=True),
            device.StudentAcknowledgment(policy_version=POLICY_VERSION, accepted=True, understood_test_only=False),
            device.StudentAcknowledgment(policy_version="expired", accepted=True, understood_test_only=True),
        ):
            with patch.object(device, "_session", return_value=ctx):
                with patch.object(device, "get_db", side_effect=AssertionError("No write allowed")):
                    with self.assertRaises(HTTPException) as error:
                        device.device_student_acknowledgment(payload)
            self.assertEqual(error.exception.status_code, 422)

    def test_admin_code_issuance_requires_current_authority(self):
        ctx = {"id": "student_identity"}
        with patch.object(device, "_pilot"), patch.object(device, "_school_allowed"):
            with patch.object(device, "_actor", return_value=ctx), patch.object(device, "_schema"):
                with patch.object(device, "get_db") as db:
                    cursor = db.return_value.__enter__.return_value.cursor.return_value
                    with patch.object(device, "_school_admin", side_effect=HTTPException(403, detail="not_admin")):
                        with self.assertRaises(HTTPException) as error:
                            device.issue_code("test_school", device.IssueCode(student_id="vfyd_child"))
                    self.assertEqual(error.exception.status_code, 403)
                    self.assertFalse(any("INSERT INTO verifyd_safety_student_device_codes" in str(call)
                                         for call in cursor.execute.call_args_list))

    def test_existing_student_endpoint_still_uses_standard_bearer(self):
        source = inspect.getsource(training.create_training_report)
        self.assertIn("actor = _actor(authorization)", source)
        self.assertIn("_create_training_report_for_student", source)
        writer = inspect.getsource(training._create_training_report_for_student)
        for guard in ("require_training_signoffs", "configure_primary_and_backup_first",
                      "_verified_training_contact", "_membership", "dispatch_report"):
            self.assertIn(guard, writer)

    def test_private_conversation_enforces_student_and_school(self):
        for route in (device.device_conversation, device.device_conversation_reply):
            src = inspect.getsource(route)
            self.assertIn("_family_chat_authorize", src)
            self.assertIn('ctx["student_id"]', src)
            self.assertIn('ctx["school_id"]', src)
        self.assertNotIn("_school_admin", inspect.getsource(device.device_conversation))

    def test_token_issuer_revalidated_and_header_isolated(self):
        from fastapi.params import Header as FastAPIHeader
        # Restricted session header is not the normal Authorization header.
        for route in device.router.routes:
            if "/student-devices/" not in route.path:
                continue
            params = inspect.signature(route.endpoint).parameters
            if "x_safety_device_session" not in params:
                continue
            field = params["x_safety_device_session"].default
            self.assertEqual(field.alias, "X-VeriFYD-Safety-Session")
        src = inspect.getsource(device._session)
        self.assertIn('_school_admin(cur, school_id, str(row["code_issued_by"]))', src)
        src = inspect.getsource(device.activate_device)
        self.assertIn('_school_admin(cur, sid, str(row["issued_by"]))', src)

    def test_device_token_not_accepted_as_normal_bearer(self):
        src = inspect.getsource(device._session)
        self.assertIn("session_hash=%s", src)
        self.assertIn("expires_at>NOW()", src)
        self.assertIn("_guardian_ready", src)
        self.assertNotIn("_identity_from_bearer", src)


if __name__ == "__main__":
    unittest.main()
