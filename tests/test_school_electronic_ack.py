"""School Safety: generic two-channel registration and electronic claim contract."""
import inspect
import unittest
from pydantic import ValidationError
import trust_voice_school_electronic_ack as e
import trust_voice_school_managed_roster as roster


class ElectronicGuardianConsentContracts(unittest.TestCase):
    def test_public_name_never_matches_a_student(self):
        match=inspect.getsource(e._matches)
        request=inspect.getsource(e.request_schoolwide_otp)
        self.assertIn("family_contact_drafts",match)
        self.assertIn("guardian_email=%s",match)
        self.assertIn("enrollment_status='staged'",match)
        self.assertNotIn("lower(first_name)",match)
        self.assertIn("student_information_disclosed",request)
        self.assertIn("_reserve_mail",request)
        self.assertNotIn("student_name",request)

    def test_disclosure_only_after_otp(self):
        verify=inspect.getsource(e.verify_schoolwide_otp)
        self.assertIn("verify_otp(email,payload.email_code)",verify)
        self.assertLess(verify.index("verify_otp"),verify.index('"children":'))
        self.assertIn("claim_session",verify)
        self.assertIn("school_registration_link_rotated",verify)

    def test_acknowledgment_does_not_activate_a_family(self):
        ack=inspect.getsource(e.acknowledge_schoolwide)
        self.assertIn("policy_version",ack)
        self.assertIn("claim_session",ack)
        self.assertIn("pending_school_review",ack)
        self.assertIn("family_access_granted",ack)
        self.assertNotIn("INSERT INTO verifyd_identities",ack)
        self.assertNotIn("INSERT INTO verifyd_safety_training_guardians",ack)
        with self.assertRaises(ValidationError):
            e.ElectronicAccept(claim_session="x"*42,children=["TEST-001"],
                policy_version="v1",relationship_confirmed=False,
                acknowledge_school_safety_policy=True)
        with self.assertRaises(ValidationError):
            e.Campaign(local_refs=["TEST-001"],principal_confirms_send=False,
                       school_confirmed_guardian_recipient_assignments=True)

    def test_atomic_principal_approval_requires_real_claim(self):
        body=inspect.getsource(roster.approve_managed_school_enrollment)
        self.assertIn("verified_electronic_guardian_acknowledgment_required",body)
        self.assertIn("administrator_cannot_approve_own_guardian_acknowledgment",body)
        self.assertIn("_new_guardian_identity",body)
        self.assertIn("guardian",body)
        self.assertIn("school_staff_cannot_claim_guardian_role",body)
        self.assertIn("ON CONFLICT DO NOTHING",body)
        self.assertIn("school_authority",inspect.getsource(roster.SchoolAuthorizationItem))
        self.assertIn("signed_paper",inspect.getsource(roster.SchoolAuthorizationItem))
        self.assertIn("electronic",inspect.getsource(roster.SchoolAuthorizationItem))
        self.assertIn("EACK-",inspect.getsource(roster._check_evidence))

    def test_automated_blast_has_verified_recipients_and_quota(self):
        send=inspect.getsource(e.send_schoolwide_campaign)
        self.assertIn("_school_admin",send)
        self.assertIn("family_contact_drafts",send)
        self.assertIn("school_confirmed_guardian_recipient_assignments",inspect.getsource(e.Campaign))
        self.assertIn("_reserve_mail",send)
        self.assertNotIn("s.first_name",send)
        self.assertTrue(any(r.path.endswith("/send-schoolwide") for r in e.router.routes))
        self.assertEqual(e._limit("NOT_SET_TEST_LIMIT",40,250),40)

    def test_electronic_school_feature_is_off_without_explicit_allowlist(self):
        import os
        from fastapi import HTTPException
        from unittest.mock import patch
        with patch.object(e,"_invites_enabled",return_value=None):
            with patch.dict(os.environ,{},clear=True):
                with self.assertRaises(HTTPException) as ctx:
                    e._require_electronic_enabled("test-school")
                self.assertEqual(ctx.exception.status_code,404)
            with patch.dict(os.environ,{
                "VERIFYD_SAFETY_ELECTRONIC_ACK_ENABLED":"1",
                "VERIFYD_SAFETY_ELECTRONIC_ACK_SCHOOL_IDS":"other-school"
            },clear=True):
                with self.assertRaises(HTTPException):
                    e._require_electronic_enabled("test-school")
            with patch.dict(os.environ,{
                "VERIFYD_SAFETY_ELECTRONIC_ACK_ENABLED":"1",
                "VERIFYD_SAFETY_ELECTRONIC_ACK_SCHOOL_IDS":"test-school"
            },clear=True):
                self.assertIsNone(e._require_electronic_enabled("test-school"))

    def test_paper_and_authority_still_required(self):
        from fastapi import HTTPException
        for kwargs in [
            dict(method="signed_paper",guardian_name="Paper Guardian",signed_date=None),
            dict(method="school_authority",authority_reference=None),
            dict(method="electronic",evidence_ref="FAKE-REFERENCE",authority_reference=None),
        ]:
            item=roster.SchoolAuthorizationItem(
               local_ref="TEST-001",evidence_ref="GENUINE-EXAMPLE-REF",
               individually_verified=True,**{k:v for k,v in kwargs.items() if k!="evidence_ref"})
            with self.assertRaises(HTTPException):
                roster._check_evidence(item)


if __name__=="__main__":
    unittest.main()
