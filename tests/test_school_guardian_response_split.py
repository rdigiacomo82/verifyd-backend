"""Contract tests: guardians and school staff must never share a recipient role."""
import inspect
import unittest
from unittest.mock import MagicMock, patch
from fastapi import HTTPException

import trust_voice_school_safety_training as school
import trust_voice_school_incident_chat as chat
import trust_voice_school_student_device as device


class GuardianResponseSplit(unittest.TestCase):
    def test_backup_guardian_is_optional_but_staff_are_never_guardians(self):
        item = school.SchoolVerifiedContactRoute(primary_id="guardian-one", independently_verified=True)
        self.assertIsNone(item.backup_id)
        src = inspect.getsource(school._verified_training_contact)
        self.assertIn('_membership(cur, school_id, adult_id, {"guardian"})', src)
        self.assertIn("verified_guardian_training_acknowledgment_required", src)
        self.assertNotIn("_school_admin", src)

    def test_guardian_update_rejects_same_identity_before_db(self):
        with patch.object(school, "_actor", return_value={"id": "principal-1"}):
            with patch.object(school, "_schema", side_effect=AssertionError("DB must not be called")):
                with self.assertRaises(HTTPException) as exc:
                    school.save_verified_school_contact_routing(
                        "test-school","child-1",school.SchoolVerifiedContactRoute(
                            primary_id="guardian-one", backup_id="guardian-one",
                            independently_verified=True))
        self.assertEqual(exc.exception.status_code, 422)

    def test_staff_cannot_be_guardian_even_with_roster_access(self):
        cursor = MagicMock()
        with patch.object(school, "_membership", side_effect=HTTPException(403, detail="school_membership_required")) as check:
            with self.assertRaises(HTTPException) as exc:
                school._verified_training_contact(cursor, "test-school","student-1","principal-1")
        self.assertEqual(exc.exception.status_code,403)
        check.assert_called_once_with(cursor,"test-school","principal-1",{"guardian"})

    def test_unselected_school_responder_cannot_read_incident(self):
        cur = MagicMock()
        cur.fetchone.side_effect = [{"status":"active"}, None]
        with patch.object(school, "_membership", return_value="responder"):
            with self.assertRaises(HTTPException) as exc:
                school._school_response_authorized(cur,"test-school","responder-1")
        self.assertEqual(exc.exception.status_code,403)

    def test_staff_authority_scoped_to_initial_incident(self):
        authorize = inspect.getsource(chat._authorize)
        self.assertIn("_school_response_authorized(cur,report[",authorize)
        family = inspect.getsource(chat._family_chat_authorize)
        self.assertIn("private_family_conversation_only",family)
        response = inspect.getsource(school._school_response_recipients)
        self.assertIn("role='principal'",response)
        self.assertIn("verifyd_safety_school_response_team",response)
        self.assertNotIn("verifyd_safety_training_guardians",response)

    def test_training_report_routes_to_guardian_and_school_staff_separately(self):
        code = inspect.getsource(school._create_training_report_for_student)
        self.assertIn("_school_response_recipients",code)
        self.assertIn("if contacts[\"backup_id\"]:",code)
        self.assertIn("recipients.add",code)
        self.assertIn("_verified_training_contact",code)
        self.assertIn("require_training_signoffs",code)

    def test_device_ready_with_one_verified_guardian(self):
        code = inspect.getsource(device.student_device_me)
        self.assertIn("not contacts[\"backup_id\"]",code)
        self.assertIn("recipients_ready = True",code)
        self.assertIn("_verified_training_contact",code)

    def test_school_response_team_is_admin_only_and_principal_fixed(self):
        code = inspect.getsource(school.update_verified_school_response_team)
        self.assertIn("_school_admin(cur, school_id, str(actor[\"id\"]))",code)
        self.assertIn("school_response_staff_must_be_verified_responder",code)
        self.assertIn("principal_always_included",code)
        self.assertNotIn("trust_voice_session_token",code)

    def test_response_team_rejects_duplicate_ids(self):
        with patch.object(school, "_actor",return_value={"id":"principal-1"}), patch.object(school,"_schema"), patch("trust_voice_school_safety_consents._consent_schema"):
            with self.assertRaises(HTTPException) as exc:
                school.update_verified_school_response_team("test-school",
                    school.SchoolResponseTeamSelection(responder_ids=["responder-1","responder-1"],independently_verified=True))
        self.assertEqual(exc.exception.status_code,422)

    def test_schema_migrates_principal_backup_without_erasing_guardian(self):
        code = inspect.getsource(school._schema)
        self.assertIn("ALTER COLUMN backup_id DROP NOT NULL",code)
        self.assertIn("SET backup_id=NULL",code)
        self.assertIn("m.role IN ('principal','responder')",code)
        self.assertIn("school_staff_removed_from_guardian_backup",code)
        self.assertNotIn("SET primary_id=NULL",code)

    def test_removed_staff_cannot_inherit_stale_response_team_designation(self):
        from trust_voice_school_safety_registration import (
            change_school_roster_member, deactivate_school_roster_member,
        )
        for fn in (change_school_roster_member, deactivate_school_roster_member):
            source = inspect.getsource(fn)
            self.assertIn("DELETE FROM verifyd_safety_school_response_team", source)
            self.assertIn("responder_id=%s", source)
            self.assertIn("school_id=%s", source)

    def test_school_incident_reads_require_designation(self):
        for action in (school.training_school_incidents,school.incident_training_reports,school.resolve_training_incident):
            self.assertIn("_school_response_authorized",inspect.getsource(action))


if __name__ == "__main__":
    unittest.main()
