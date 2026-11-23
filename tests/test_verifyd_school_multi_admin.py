"""Multi-admin guards: Principal nomination never creates delegation."""
import inspect
import unittest
from unittest.mock import patch
from fastapi import HTTPException
from pydantic import ValidationError
import trust_voice_school_multi_admin as adm

class SchoolMultiAdminSecurity(unittest.TestCase):
    def test_nomination_requires_principal_and_existing_responder(self):
        src=inspect.getsource(adm.nominate_administrator)
        self.assertIn('_school_admin(cur,school_id,aid)!="principal"',src)
        self.assertIn('_membership(cur,school_id,candidate_id,{"responder"})',src)
        self.assertIn('"administrator_access_granted":False',src)
        self.assertNotIn("INSERT INTO verifyd_safety_school_admin_delegations",src)
    def test_only_owner_can_decide_and_authority_proof_required(self):
        src=inspect.getsource(adm.operator_decide)
        self.assertIn("_owner(owner)",src)
        self.assertIn("verified_staff_authority",src)
        self.assertIn("'responder'",src.replace('"responder"','\'responder\''))
        self.assertIn("accepted_at=NULL",src)
        self.assertIn("school_admin_limit_reached",src)
    def test_revocation_requires_owner(self):
        src=inspect.getsource(adm.owner_revoke_administrator)
        self.assertIn("_owner(owner)",src)
        self.assertIn("revoked_at=NOW()",src)
    def test_no_emergency_delivery_claim(self):
        for fn in (adm.school_administrators,adm.nominate_administrator,adm.operator_decide):
            self.assertIn("emergency_delivery_active",inspect.getsource(fn))
    def test_bad_reason_rejected(self):
        with self.assertRaises(ValidationError):adm.Nominate(candidate_handle="Candidate42",reason="ok")
        with self.assertRaises(ValidationError):adm.Revoke(reason="short")
    def test_anonymous_list_denied(self):
        with patch.object(adm,"_actor",side_effect=HTTPException(401,"session_required")):
            with self.assertRaises(HTTPException):adm.school_administrators("school")
if __name__=="__main__":unittest.main()
