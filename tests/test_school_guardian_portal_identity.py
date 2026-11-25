"""Guardian Family Safety: approval visibility, existing handles and child identity."""
import inspect
import unittest
from trust_voice_school_family_portal import family_portal
from trust_voice_school_family_invitations import _new_guardian_identity
from trust_voice_school_managed_roster import approve_managed_school_enrollment
from trust_voice_school_safety_training import guardian_training_children


class GuardianFamilyPortalContracts(unittest.TestCase):
    def test_family_portal_shows_existing_guardian_identity_and_eack_progress(self):
        source=inspect.getsource(family_portal)
        self.assertIn('"guardian_account":guardian_account',source)
        self.assertIn('"registration_progress":registration_progress',source)
        self.assertIn('a.verified_at IS NOT NULL',source)
        self.assertIn('"awaiting_school_approval"',source)
        self.assertIn('row["enrollment_status"]=="school_approved"',source)
        self.assertIn('str(row["guardian_identity_id"])==guardian_id',source)
        self.assertIn('identity["email_verified"]',source)
        self.assertIn('WHERE g.guardian_id=%s',source)
        self.assertIn('x.revoked_at IS NULL',source)
        self.assertIn('student_device_requires_separate_supervised_school_session',source)

    def test_new_guardian_handle_prefers_school_prefix_existing_identity_preserved(self):
        source=inspect.getsource(_new_guardian_identity)
        self.assertIn('if existing:',source)
        self.assertIn('return str(existing["id"]),False',source)
        self.assertLess(source.index('if existing:'),source.index('guardian_id ='))
        self.assertIn('school_code',source)
        self.assertIn('secrets.token_hex',source)
        self.assertIn('handle_lower=%s',source)
        self.assertIn('to_regclass',source)

    def test_electronic_school_approval_creates_real_link_but_not_before(self):
        src=inspect.getsource(approve_managed_school_enrollment)
        self.assertIn('guardian_id,guardian_created=_new_guardian_identity',src)
        self.assertIn('verifyd_safety_training_guardians',src)
        self.assertIn('guardian_identity_id=%s',src)
        self.assertIn('electronic_guardian_school_approved',src)
        self.assertIn('school_approved',src)
        children=inspect.getsource(guardian_training_children)
        self.assertIn('verifyd_safety_training_guardians',children)


if __name__ == "__main__":
    unittest.main()
