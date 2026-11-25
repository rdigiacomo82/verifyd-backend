"""Non-DB contracts: Principal delegation and secure school entry links."""
import inspect
import unittest
from pydantic import ValidationError
import trust_voice_school_multi_admin as admins
import trust_voice_school_family_link as family_link
import trust_voice_school_managed_roster as managed


class SchoolPrincipalDelegationAndFamilyLinkTests(unittest.TestCase):
    def test_principal_may_authorize_only_verified_staff_with_acceptance(self):
        source=inspect.getsource(admins.principal_authorize_school_administrator)
        self.assertIn('_school_admin(cur,school_id,principal_id)!="principal"',source)
        self.assertIn('email_verified',source)
        self.assertIn('candidate_needs_active_verified_identity',source)
        self.assertIn('existing_student_or_guardian_role_cannot_be_promoted',source)
        self.assertIn('school_admin_limit_reached',source)
        self.assertIn('accepted_at=NULL',source)
        self.assertIn('"administrator_access_granted":False',source)
        self.assertNotIn('_owner(',source)
        with self.assertRaises(ValidationError):
            admins.PrincipalDelegate(candidate_handle="testhandle",
                principal_verified_staff_role=False,reason="A verified staff member")

    def test_principal_only_revoke_keeps_user_and_school_roster(self):
        source=inspect.getsource(admins.principal_revoke_school_administrator)
        self.assertIn('_school_admin(cur,school_id,principal_id)!="principal"',source)
        self.assertIn('SET revoked_at=NOW(),accepted_at=NULL',source)
        self.assertIn('"global_account_deleted":False',source)
        self.assertNotIn('DELETE FROM',source)

    def test_public_school_link_cannot_grant_access_or_reveal_students(self):
        source=inspect.getsource(family_link.request_personal_invitation_via_school_link)
        self.assertIn('generic=',source)
        self.assertIn('student_information_disclosed',source)
        self.assertIn('family_access_granted',source)
        self.assertIn('guardian_email=%s',source)
        self.assertIn("enrollment_status='school_approved'",source)
        self.assertIn('_invites_enabled',source)
        self.assertIn('recipient_hash',source)
        self.assertIn('>=2',source)
        self.assertIn('>=300',source)
        self.assertIn('_send_invitation',source)
        self.assertNotIn('"student_name"',source)
        self.assertNotIn('"student_handle"',source)
        self.assertNotIn('"invitation_code"',source)
        self.assertNotIn('"guardian_email":',inspect.getsource(family_link.get_or_create_school_registration_link).split("return {")[-1])

    def test_link_rotation_is_principal_only(self):
        source=inspect.getsource(family_link.rotate_school_registration_link)
        self.assertIn('school_principal_required',source)
        self.assertIn('principal_confirms_new_link',source)
        self.assertIn('prior_public_link_invalidated',source)

    def test_roster_preview_warns_but_does_not_block_identical_names(self):
        source=inspect.getsource(managed.preview_managed_roster)
        self.assertIn('possible_same_name_warning',source)
        self.assertIn('distinct students may keep distinct assigned handles',source)
        self.assertIn('_new_handle',source)
        self.assertNotIn('duplicate_name_in_upload',source)


if __name__=="__main__":
    unittest.main()
