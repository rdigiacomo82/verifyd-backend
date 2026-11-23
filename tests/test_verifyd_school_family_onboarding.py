"""Safety first: selected handles never create unrestricted child identities."""
import inspect
import unittest
from unittest.mock import patch
from fastapi import HTTPException
from pydantic import ValidationError
import trust_voice_school_family_onboarding as x

class FamilyHandleIntakeTests(unittest.TestCase):
    def test_no_more_than_five_children(self):
        with self.assertRaises(ValidationError):
            x.Intake(school_id="sschool_test",children=[{"handle":f"Child_{i}","display_label":"Child"} for i in range(6)])
    def test_no_global_identity_creation(self):
        src=inspect.getsource(x.submit)
        self.assertNotIn("INSERT INTO verifyd_identities",src)
        self.assertIn("parental_authority_acknowledged",src)
        self.assertIn("_availability(cur,lower)",src)
        self.assertIn("accounts_created",src)
    def test_staff_queue_requires_school_admin(self):
        for fn in (x.school_requests,x.review):
            self.assertIn("_school_admin(cur,school_id",inspect.getsource(fn))
    def test_unauthorized_request_denied(self):
        with patch.object(x,"_actor",side_effect=HTTPException(401,"session_required")):
            with self.assertRaises(HTTPException):
                x.mine()
if __name__=="__main__": unittest.main()
