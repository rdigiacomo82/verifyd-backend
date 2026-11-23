"""Unified new-parent registration stays email-verified and child-safe."""
import inspect
import unittest
from unittest.mock import patch
from fastapi import HTTPException
from pydantic import ValidationError
import trust_voice_school_unified_signup as signup

class UnifiedFamilyRegistration(unittest.TestCase):
    def test_one_to_five_children(self):
        fields=dict(school_id="sschool_test",email="example@example.com",
            email_code="123456",parent_handle="ExampleParent",
            parent_display_name="Example Parent",parental_authority_acknowledged=True)
        with self.assertRaises(ValidationError):
            signup.UnifiedFamilyApplication(**fields,children=[])
        with self.assertRaises(ValidationError):
            signup.UnifiedFamilyApplication(**fields,
                children=[{"handle":f"ExampleChild{i}","display_label":"Example"} for i in range(6)])

    def test_new_parent_requires_email_code_and_authority_ack(self):
        payload=signup.UnifiedFamilyApplication(
            school_id="sschool_test",email="example@example.com",
            email_code="123456",parent_handle="ExampleParent",
            parent_display_name="Example Parent",
            children=[{"handle":"ExampleChild","display_label":"Child"}])
        with patch.object(signup,"_voice_enabled"), patch.object(signup,"_school_enabled",return_value=True):
            with self.assertRaises(HTTPException) as error:
                signup.register_family(payload)
            self.assertEqual(error.exception.status_code,422)

    def test_does_not_activate_child_accounts_or_emergency(self):
        source=inspect.getsource(signup.register_family)
        self.assertIn("verify_otp(email,payload.email_code)",source)
        self.assertIn("parental_authority_acknowledged",source)
        self.assertIn("verifyd_safety_family_intake",source)
        self.assertIn("duplicate_family_handles",source)
        self.assertIn("_availability(cur,lower)",source)
        self.assertIn("pg_advisory_xact_lock",source)
        self.assertIn('"children_activated":False',source)
        self.assertIn('"emergency_delivery_active":False',source)
        # Only a single INSERT into the global identity table, for the adult.
        self.assertEqual(source.count("INSERT INTO verifyd_identities"),1)

    def test_school_directory_is_approved_only(self):
        src=inspect.getsource(signup.open_schools)
        self.assertIn("a.status='approved'",src)
        self.assertNotIn("official_email",src)

if __name__=="__main__":unittest.main()
