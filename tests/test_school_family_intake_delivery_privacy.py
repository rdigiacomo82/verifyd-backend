"""School Safety: regress 500 family-interest bug, duplicate mail and consent gates."""
import inspect
import unittest

from fastapi import HTTPException
from pydantic import ValidationError
import trust_voice_school_family_interest as intake
import trust_voice_school_electronic_ack as electronic
from trust_voice_school_data_notice import DATA_NOTICE_VERSION


class FamilyIntakeDeliveryPrivacyContracts(unittest.TestCase):
    def test_public_interest_db_advisory_lock_uses_psycopg_tuple(self):
        source=inspect.getsource(intake.submit_family_contact_request)
        # psycopg2 previously raised TypeError on this incorrectly packed parameter.
        self.assertIn('("school_family_interest:"+school_id,)',source)
        self.assertNotIn('("school_family_interest:"+school_id))',source)
        self.assertIn("student_information_disclosed",source)
        self.assertIn("family_access_granted",source)

    def test_versioned_intake_privacy_confirmation_does_not_approve_student(self):
        record=intake.FamilyInterest(
            public_code="SC_"+"A"*24,guardian_name="Jordan Sample",
            children=["Avery Sample"],relationship="parent",
            contact_method="email",contact_value="guardian@example.org",
            permission_for_school_follow_up=True,
            data_notice_version=DATA_NOTICE_VERSION,
            acknowledge_registration_data_use=True,
            acknowledge_app_use_instructions=True)
        self.assertTrue(record.acknowledge_app_use_instructions)
        self.assertIn("data_notice_acknowledged_at",inspect.getsource(intake._schema))
        self.assertIn("app_use_notice_acknowledged_at",inspect.getsource(intake._schema))
        self.assertNotIn("INSERT INTO verifyd_safety_training_guardians",
                         inspect.getsource(intake.submit_family_contact_request))
        with self.assertRaises(HTTPException) as error:
            intake.submit_family_contact_request(record.model_copy(update={"data_notice_version":"stale"}))
        self.assertEqual(error.exception.status_code,409)

    def test_electronic_acknowledgment_enforces_both_separate_notices(self):
        with self.assertRaises(ValidationError):
            electronic.ElectronicAccept(
                claim_session="C"*40,children=["TEST-AVERY-001"],
                relationship_confirmed=True,acknowledge_school_safety_policy=True,
                policy_version="2026-11-22.training.v1",
                data_notice_version=DATA_NOTICE_VERSION,
                acknowledge_school_data_use_notice=False,
                acknowledge_app_use_instructions=True)
        source=inspect.getsource(electronic.acknowledge_schoolwide)
        self.assertIn("school_data_notice_version_changed_reload_registration",source)
        self.assertIn("data_notice_version",source)
        self.assertIn("app_use_acknowledged_at",source)
        self.assertIn("family_access_granted",source)
        self.assertNotIn("INSERT INTO verifyd_safety_training_guardians",source)
        self.assertNotIn("INSERT INTO verifyd_identities",source)

    def test_public_notice_endpoint_is_content_only(self):
        result=electronic.family_data_use_notice()
        self.assertEqual(result["data_notice_version"],DATA_NOTICE_VERSION)
        self.assertTrue(result["data_use_statements"])
        self.assertTrue(result["app_use_statements"])
        self.assertFalse(result["live_emergency_dispatch"])

    def test_campaign_history_is_admin_only_and_prevents_duplicates(self):
        history=inspect.getsource(electronic.schoolwide_invitation_history)
        self.assertIn("_school_admin(cur,school_id,str(actor[\"id\"]))",history)
        self.assertIn("_require_electronic_enabled(school_id)",history)
        self.assertIn("accepted_recipient_emails",history)
        self.assertIn("recent_delivery_attempts",history)
        reserve=inspect.getsource(electronic._reserve_mail)
        self.assertIn("accepted_by_provider IS TRUE",reserve)
        self.assertIn("recipient_hash=%s",reserve)
        send=inspect.getsource(electronic.send_schoolwide_campaign)
        self.assertIn("campaign_id",send)
        self.assertIn("student_refs=refs_for_email",send)
        self.assertNotIn("s.first_name",send)
        self.assertIn("skipped_due_to_quota_or_recent_send",send)


if __name__=="__main__":
    unittest.main()
