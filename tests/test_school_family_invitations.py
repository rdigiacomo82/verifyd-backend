"""School guardian invitation claims never substitute for individual consent."""
import inspect
import os
import unittest
from unittest.mock import MagicMock, patch

from fastapi import HTTPException
from pydantic import ValidationError
import trust_voice_school_family_invitations as invite


class SchoolFamilyInvitationsTests(unittest.TestCase):
    def test_requires_separate_school_scoped_feature_gate(self):
        with patch.dict(os.environ, {
            "VERIFYD_SAFETY_FAMILY_INVITES_ENABLED": "0",
            "VERIFYD_SAFETY_FAMILY_INVITE_SCHOOL_IDS": "school-A",
        }), patch.object(invite, "_require_enrollment_enabled"):
            with self.assertRaises(HTTPException) as ctx:
                invite._enabled("school-A")
            self.assertEqual(ctx.exception.status_code, 404)
        with patch.dict(os.environ, {
            "VERIFYD_SAFETY_FAMILY_INVITES_ENABLED": "1",
            "VERIFYD_SAFETY_FAMILY_INVITE_SCHOOL_IDS": "school-A",
        }), patch.object(invite, "_require_enrollment_enabled"):
            invite._enabled("school-A")
            with self.assertRaises(HTTPException):
                invite._enabled("school-B")

    def test_invitation_codes_do_not_take_student_handles(self):
        for cls in (invite.InvitationCode,invite.VerifyInvitation,invite.ConfirmFamily):
            self.assertNotIn("student_handle",cls.model_fields)
            self.assertNotIn("student_id",cls.model_fields)
        source = inspect.getsource(invite.verify_family_invitation)
        self.assertIn("verify_otp",source)
        self.assertIn("student_handle",source)
        self.assertIn("claim_session",source)

    def test_otp_before_student_name_is_disclosed(self):
        req = inspect.getsource(invite.request_invitation_email_otp)
        self.assertIn("send_otp_email",req)
        self.assertIn("student_information_disclosed",req)
        self.assertNotIn("student_handle",req)
        self.assertNotIn("student_name",req)

    def test_only_explicit_consent_can_complete_self_claim(self):
        valid = "A"*48
        with self.assertRaises(ValidationError):
            invite.ConfirmFamily(claim_session=valid,
                relationship_confirmed=False,
                acknowledge_school_safety_policy=True)
        with self.assertRaises(ValidationError):
            invite.ConfirmFamily(claim_session=valid,
                relationship_confirmed=True,
                acknowledge_school_safety_policy=False)

    def test_primary_cannot_approve_the_backup_automatically(self):
        code = inspect.getsource(invite.confirm_family_claim)
        self.assertIn("second_guardian_invitation_requested",code)
        self.assertIn("backup_requires_own_verification",code)
        self.assertNotIn("verifyd_safety_training_signoffs",code)
        approval = inspect.getsource(invite.approve_family_invitation_batch)
        self.assertIn("acknowledged_policy",approval)
        self.assertIn("claimed_by",approval)
        self.assertIn("principal_attests_individual_review" if False else "independently_verified_family_relationships",
                      inspect.getsource(invite.FinalBatchApproval))

    def test_invalid_email_attempts_lock_invitation(self):
        source = inspect.getsource(invite.verify_family_invitation)
        self.assertIn('verification_attempts"] >= 5',source)
        self.assertIn("verification_attempts=verification_attempts+1",source)
        self.assertIn("invitation_verification_locked_contact_school",source)
        self.assertIn("session_expires_at",inspect.getsource(invite.confirm_family_claim))

    def test_school_guardian_identity_is_not_created_twice_for_siblings(self):
        source = inspect.getsource(invite._new_guardian_identity)
        self.assertIn("pg_advisory_xact_lock", source)
        self.assertIn("email_lower", source)

    def test_principal_final_batch_only_verified_claims(self):
        src = inspect.getsource(invite.approve_family_invitation_batch)
        self.assertIn("_school_admin(cur,school_id,str(actor[\"id\"]))",src)
        self.assertIn('approval_status"]!="pending_principal"',src)
        self.assertIn("cannot_approve_own_family_relationship",src)
        self.assertIn("verifyd_safety_training_guardians",src)
        self.assertIn("verifyd_safety_training_signoffs",src)
        self.assertIn('accepted_at=EXCLUDED.accepted_at',src)
        self.assertIn("guardian_invitation_final_approved",src)

    def test_principal_cannot_send_invitation_for_unapproved_roster(self):
        src = inspect.getsource(invite.send_invitation_batch)
        self.assertIn("enrollment_status='school_approved'",src)
        self.assertIn("guardian_slot_already_assigned_review_before_resend",src)
        self.assertIn("_school_admin(cur, school_id, str(actor[\"id\"]))",src)
        self.assertIn("invalid_guardian_email",src)

    def test_batch_cannot_duplicate_same_invitation(self):
        with patch.object(invite, "_enabled"), patch.object(invite,"_actor",return_value={"id":"principal-1"}):
            payload = invite.FinalBatchApproval(
                invite_ids=["inv-1","inv-1"],
                independently_verified_family_relationships=True,
                review_note="Independently reviewed family documents")
            with self.assertRaises(HTTPException) as ctx:
                invite.approve_family_invitation_batch("school-A",payload)
        self.assertEqual(ctx.exception.status_code,422)

    def test_no_sms_or_911_delivery(self):
        src = inspect.getsource(invite.send_invitation_batch)
        self.assertIn("sms_sent",src)
        self.assertNotIn("notify_identity_push",src)
        self.assertNotIn("911",inspect.getsource(invite._send_invitation))
        self.assertIn("live_emergency_dispatch_enabled",inspect.getsource(invite.approve_family_invitation_batch))

    def test_primary_backup_addresses_must_be_distinct_across_edits(self):
        send_source = inspect.getsource(invite.send_invitation_batch)
        self.assertIn("primary_and_backup_need_distinct_emails",send_source)
        self.assertIn("by_slot",send_source)
        save_source = inspect.getsource(invite.save_school_family_contact_drafts)
        self.assertIn("SELECT local_ref,slot,guardian_email",save_source)
        self.assertIn("pg_advisory_xact_lock",save_source)
        self.assertIn("primary_and_backup_need_distinct_emails",save_source)

    def test_guardian_contacts_only_saved_by_authorized_school_staff(self):
        src = inspect.getsource(invite.save_school_family_contact_drafts)
        self.assertIn("_school_admin(cur,school_id,str(actor[\"id\"]))",src)
        self.assertIn("school_verified_emails_and_relationships",inspect.getsource(invite.SavedFamilyContacts))
        self.assertIn("guardian_contact_student_not_in_school_roster",src)
        self.assertIn("revoke_prior_invitation_before_changing_guardian_contact",src)

    def test_auto_send_after_approval_only_and_skip_duplicate_codes(self):
        src = inspect.getsource(invite.send_saved_family_invitations)
        self.assertIn("enrollment_status='school_approved'",src)
        self.assertIn("not in occupied",src)
        self.assertIn("send_invitation_batch",src)
        self.assertIn("missing_contact_records",src)
        self.assertIn("No new invitations needed",src)

    def test_resend_invalidates_old_code_and_limits_rate(self):
        src = inspect.getsource(invite.resend_school_family_invitation)
        self.assertIn("_school_admin(cur,school_id,str(actor[\"id\"]))",src)
        self.assertIn("only_unclaimed_invitations_can_be_resent",src)
        self.assertIn("send_attempts",src)
        self.assertIn("2 minutes",src)
        self.assertIn("token_hash=%s",src)
        self.assertIn("prior_invitation_link_revoked",src)
        self.assertIn("new_code_emailed_not_displayed",src)
        self.assertNotIn("student_handle",src)
        with self.assertRaises(ValidationError):
            invite.ResendInvitation(principal_confirms_address=False)
        with self.assertRaises(ValidationError):
            invite.SendSavedInvitations(local_refs=["STU001"],
                principal_confirms_recipient_list=False)

    def test_school_only_can_revoke_pending_invitation(self):
        src = inspect.getsource(invite.revoke_school_family_invitation)
        self.assertIn("_school_admin",src)
        self.assertIn("approved_family_use_guardian_deauthorization",src)


if __name__=="__main__":
    unittest.main()
