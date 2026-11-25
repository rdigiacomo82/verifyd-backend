"""TEST School: approved-only guardian routing, principal override and private school directory."""
import inspect
import os
import unittest
from unittest.mock import patch
from pydantic import ValidationError
import trust_voice_school_auto_routing as routing
import trust_voice_school_directory as directory
from trust_voice_school_safety_training import get_verified_school_contact_routing,save_verified_school_contact_routing
from trust_voice_school_family_portal import family_portal
from trust_voice_school_student_device import student_device_me


class FakeCursor:
    def __init__(self,student=True,saved=None,eligible=None,slots=None):
        self.student=student
        self.saved=saved
        self.eligible=eligible if eligible is not None else ["g1"]
        self.slots=slots if slots is not None else [(1,"g1")]
        self.last=""
        self.writes=[]
    def execute(self,sql,args=()):
        self.last=sql
        if sql.lstrip().upper().startswith(("INSERT","UPDATE","DELETE")):
            self.writes.append((sql,args))
    def fetchone(self):
        sql=self.last
        if "SELECT s.local_ref,s.enrollment_status" in sql:
            return {"local_ref":"TEST-STUDENT-1","enrollment_status":"school_approved"} if self.student else None
        if "SELECT primary_id,backup_id,route_source" in sql:
            return self.saved
        return None
    def fetchall(self):
        if "SELECT g.guardian_id" in self.last:
            return [{"guardian_id":x} for x in self.eligible]
        if "SELECT a.slot,a.guardian_identity_id" in self.last:
            return [{"slot":k,"guardian_id":v} for k,v in self.slots]
        return []


class RoutePolicyContracts(unittest.TestCase):
    def setUp(self):
        self.gate=patch.object(routing,"_enabled",return_value=True)
        self.check=patch.object(routing,"_verified_training_contact",return_value="guardian")
        self.audit=patch.object(routing,"_audit")
        self.gate.start();self.check.start();self.audit.start()
    def tearDown(self):
        self.audit.stop();self.check.stop();self.gate.stop()

    def test_off_by_default_and_test_school_allowlist(self):
        self.gate.stop()
        try:
            with patch.dict(os.environ,{},clear=True):
                self.assertFalse(routing._enabled("sschool_test_real"))
            with patch.dict(os.environ,{
                "VERIFYD_SAFETY_AUTO_ROUTING_ENABLED":"1",
                "VERIFYD_SAFETY_AUTO_ROUTING_SCHOOL_IDS":"sschool_allowed",
            },clear=True):
                self.assertTrue(routing._enabled("sschool_allowed"))
                self.assertFalse(routing._enabled("sschool_other"))
        finally:
            self.gate.start()

    def test_approved_single_guardian_auto_primary(self):
        cur=FakeCursor()
        result=routing.sync_approved_guardian_route(cur,"school","student","principal")
        self.assertTrue(result["updated"])
        self.assertTrue(result["primary_assigned"])
        self.assertFalse(result["backup_assigned"])
        self.assertEqual(len(cur.writes),1)
        self.assertIn("'approved_auto'",cur.writes[0][0])

    def test_independently_approved_backup_slot_auto(self):
        cur=FakeCursor(eligible=["g1","g2"],slots=[(1,"g1"),(2,"g2")])
        result=routing.sync_approved_guardian_route(cur,"school","student","principal")
        self.assertTrue(result["backup_assigned"])
        self.assertEqual(cur.writes[0][1][2],"g1")
        self.assertEqual(cur.writes[0][1][3],"g2")

    def test_no_unapproved_student(self):
        cur=FakeCursor(student=False)
        result=routing.sync_approved_guardian_route(cur,"school","student","principal")
        self.assertFalse(result["updated"])
        self.assertEqual(cur.writes,[])

    def test_manual_override_is_never_replaced(self):
        cur=FakeCursor(saved={"route_source":"manual","primary_id":"g1","backup_id":None})
        result=routing.sync_approved_guardian_route(cur,"school","student","principal")
        self.assertEqual(result["reason"],"principal_manual_override")
        self.assertEqual(cur.writes,[])

    def test_multiple_guardians_no_primary_slot_requires_school_choice(self):
        cur=FakeCursor(eligible=["g1","g2"],slots=[])
        result=routing.sync_approved_guardian_route(cur,"school","student","principal")
        self.assertEqual(result["reason"],"ambiguous_primary_requires_school_review")
        self.assertEqual(cur.writes,[])

    def test_guardian_revocation_removes_auto_only(self):
        cur=FakeCursor(saved={"route_source":"approved_auto","primary_id":"g1","backup_id":None},eligible=[],slots=[])
        result=routing.sync_approved_guardian_route(cur,"school","student","principal")
        self.assertTrue(result["updated"])
        self.assertIn("DELETE FROM",cur.writes[0][0])

    def test_directory_never_uses_principal_private_email(self):
        src=inspect.getsource(directory._details)
        self.assertIn("principal_name",src)
        self.assertIn("principal_handle",src)
        self.assertIn("approved_backup_administrators",src)
        self.assertNotIn("official_email",src)
        self.assertNotIn("email_lower",src)
        self.assertNotIn("recipient_email",src)
        self.assertFalse(directory.DirectoryUpdate(main_phone="",office_phone="").main_phone)
        with self.assertRaises(ValidationError):
            directory.DirectoryUpdate(main_phone="123",office_phone="")
        with self.assertRaises(ValidationError):
            directory.DirectoryUpdate(public_instructions="Email principal@private.example")
        self.assertIn("approved_guardian_family_link_required",inspect.getsource(directory._require_directory_read))
        self.assertIn("_school_admin",inspect.getsource(directory.update_school_public_directory))

    def test_dashboard_contract_preserves_manual_review(self):
        source=inspect.getsource(get_verified_school_contact_routing)
        self.assertIn("sync_approved_guardian_route",source)
        self.assertIn("_auto_enabled",source)
        manual=inspect.getsource(save_verified_school_contact_routing)
        self.assertIn("route_source='manual'",manual)
        self.assertIn("_school_admin",manual)
        self.assertIn("_verified_training_contact",manual)
        family=inspect.getsource(family_portal)
        self.assertIn("school_directories",family)
        self.assertIn('"safety_route"',family)
        child=inspect.getsource(student_device_me)
        self.assertIn('"school_directory":directory',child)
        self.assertIn('"guardian_handles":family_handles',child)


if __name__=="__main__":
    unittest.main()
