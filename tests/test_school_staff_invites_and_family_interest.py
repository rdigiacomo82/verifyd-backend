"""Safety boundaries for email-verified Principal-created staff and family intake."""
import inspect
import unittest
from pydantic import ValidationError
import trust_voice_school_staff_invites as staff
import trust_voice_school_family_interest as family
import trust_voice_school_managed_roster as students
import trust_voice_school_roster_documents as doc


class SchoolStaffAndFamilySafetyContracts(unittest.TestCase):
    def test_staff_identity_is_not_activated_until_email_otp_and_acceptance(self):
        issue=inspect.getsource(staff.create_staff_invite)
        confirm=inspect.getsource(staff.staff_accept_invitation)
        self.assertIn('_school_admin(cur,school_id,str(actor["id"]))!="principal"',issue)
        self.assertIn("proposed_handle",issue)
        self.assertIn("school_admin_limit_reached",issue)
        self.assertIn("principal_verified_employment_and_authority",inspect.getsource(staff.StaffInvite))
        self.assertIn('"new_account_active":False',issue)
        self.assertNotIn("INSERT INTO verifyd_identities",issue)
        self.assertIn("verify_otp(email,payload.email_code)",confirm)
        self.assertIn('accept_school_administrator_responsibility',inspect.getsource(staff.StaffAccept))
        self.assertIn("INSERT INTO verifyd_identities",confirm)
        self.assertIn("accepted_at=NOW()",confirm)
        self.assertIn("cannot_promote_guardian_or_student",confirm)
        with self.assertRaises(ValidationError):
            staff.StaffInvite(staff_name="Demo Teacher",
                 official_email="teacher@example.org",
                 position="Dean",principal_verified_employment_and_authority=False)

    def test_staff_invites_are_restricted_and_expire(self):
        code=inspect.getsource(staff._active_invitation)
        self.assertIn("revoked_at",code)
        self.assertIn("expires_at",code)
        self.assertIn("accepted_at",code)
        self.assertIn("_hash(code.strip())",code)
        routes=[r.path for r in staff.router.routes]
        for suffix in ("/staff-invitations/request-code","/staff-invitations/accept"):
            self.assertTrue(any(x.endswith(suffix) for x in routes))

    def test_public_parent_names_are_only_review_requests(self):
        submit=inspect.getsource(family.submit_family_contact_request)
        self.assertIn("student_information_disclosed",submit)
        self.assertIn("family_access_granted",submit)
        self.assertIn("verified_guardian",submit)
        self.assertIn("contact_fingerprint",submit)
        self.assertNotIn("verifyd_safety_training_guardians",submit)
        self.assertNotIn("verifyd_identities",submit)
        self.assertNotIn("student_identity_id",submit)
        self.assertIn(">=2",submit)
        self.assertIn(">=250",submit)
        list_src=inspect.getsource(family.list_school_family_interests)
        self.assertIn("_school_admin",list_src)
        self.assertIn("requires_separate_school_verification",list_src)
        with self.assertRaises(ValidationError):
            family.FamilyInterest(public_code="a"*25,
                guardian_name="Example Parent",children=["Child One"],
                contact_method="phone",contact_value="555-111-9999",
                permission_for_school_follow_up=False)
        record=family.FamilyInterest(public_code="a"*25,
            guardian_name="Example Parent",
            children=["Child One","Child Two"],
            contact_method="phone",contact_value="555-111-9999",
            permission_for_school_follow_up=True)
        self.assertEqual(len(record.children),2)

    def test_contact_import_stays_school_only_and_unverified(self):
        self.assertIn("student_email",doc.COLUMNS)
        self.assertIn("primary_guardian_phone",doc.COLUMNS)
        self.assertIn("backup_guardian_phone",doc.COLUMNS)
        school_source=inspect.getsource(students.update_school_student_contact_fields)
        self.assertIn("_school_admin",school_source)
        self.assertIn("verified_guardian_relationship",school_source)
        self.assertIn("email_or_sms_sent",school_source)
        self.assertIn("student_phone",inspect.getsource(students.SchoolStudentContacts))
        self.assertIn("SchoolStudentContacts.model_fields",
                      inspect.getsource(students.commit_managed_roster))
        with self.assertRaises(ValidationError):
            students.SchoolStudentContacts(primary_guardian_email="not an email")
        with self.assertRaises(ValidationError):
            students.SchoolStudentContacts(backup_guardian_phone="123")


if __name__=="__main__":
    unittest.main()
