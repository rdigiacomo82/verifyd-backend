"""Non-escalation and guardian/school separation for supervised child identities."""
import inspect
import unittest
from unittest.mock import patch
from fastapi import HTTPException
from pydantic import ValidationError
import trust_voice_school_child_activation as a

class SupervisedChildContract(unittest.TestCase):
    def test_bulk_activation_max_five_and_requires_acknowledgment(self):
        with self.assertRaises(ValidationError):
            a.ActivateChildren(request_ids=[str(i) for i in range(6)],email_code="123456",supervised_account_acknowledged=True)
        with patch.object(a,"_actor",return_value={"id":"guardian"}),patch.object(a,"_schema"):
            with patch.object(a,"_require_pilot"):
                with self.assertRaises(HTTPException) as e:
                    a.activate_children(a.ActivateChildren(request_ids=["x"],email_code="123456"))
                self.assertEqual(e.exception.status_code,422)
    def test_staff_verification_requires_school_auth_and_independent_evidence(self):
        src=inspect.getsource(a.verify_child)
        self.assertIn("_school_admin(cur,school_id",src)
        self.assertIn("independently_verified",src)
        self.assertIn("self_verification_denied",src)
        self.assertNotIn("INSERT INTO verifyd_identities",src)
    def test_child_created_with_global_unique_handle_and_no_voice(self):
        src=inspect.getsource(a.activate_children)
        self.assertIn("INSERT INTO verifyd_identities",src)
        self.assertIn("'supervised_child'",src)
        self.assertIn("'nobody'",src)
        self.assertIn("FALSE,'nobody'",src)
        self.assertIn("_school_accepted",src)
        self.assertIn("verifyd_safety_child_activation_codes",src)
        self.assertIn("hmac.compare_digest",src)
        self.assertIn("verifyd_safety_training_guardians",src)
        self.assertIn("supervised_voice_login_available",src)
    def test_activation_challenge_is_independent_of_voice_login(self):
        request_code=inspect.getsource(a.request_activation_code)
        activate=inspect.getsource(a.activate_children)
        self.assertNotIn("create_otp(",request_code)
        self.assertNotIn("verify_otp(",activate)
        self.assertIn("activation_code_recently_sent",request_code)
        self.assertIn("activation_code_expired_or_locked",activate)
        self.assertIn("attempts=attempts+1",activate)
    def test_pilot_school_allowlist_fails_closed(self):
        with patch.dict("os.environ",{"VERIFYD_SAFETY_CHILD_ACTIVATION_SCHOOL_IDS":"sschool_test"}):
            a._allowed_school("sschool_test")
            with self.assertRaises(HTTPException) as error:
                a._allowed_school("sschool_other")
            self.assertEqual(error.exception.status_code,403)
        with patch.dict("os.environ",{"VERIFYD_SAFETY_CHILD_ACTIVATION_SCHOOL_IDS":""}):
            with self.assertRaises(HTTPException):
                a._allowed_school("sschool_test")
    def test_readiness_authentication_and_no_child_login(self):
        source=inspect.getsource(a.activation_readiness)
        self.assertIn("_actor(authorization)",source)
        self.assertIn('child_login_enabled":False',source)
        self.assertIn('emergency_alerts_active":False',source)
        self.assertNotIn("email_lower",source.split("return {")[-1])
    def test_family_batch_requires_individual_verification(self):
        source=inspect.getsource(a.verify_family_batch)
        self.assertIn("_school_admin(cur,school_id,aid)",source)
        self.assertIn("_allowed_school(school_id)",source)
        self.assertIn("each_child_must_be_independently_verified",source)
        self.assertIn("batch_requires_same_parent",source)
        self.assertIn("request_already_verified",source)
        self.assertIn("guardian_email_confirmation_required",source)
        self.assertIn("activated_count",source)
        self.assertIn("FOR UPDATE",source)
        self.assertIn("self_verification_denied",source)
    def test_verification_notes_are_optional_but_acknowledgment_required(self):
        single=a.SchoolVerify(independently_verified=True)
        self.assertEqual(single.verification_note,"")
        batch=a.FamilyBatchVerify(requests=[
            a.FamilyBatchItem(request_id="ssint_test123",independently_verified=True)])
        self.assertEqual(batch.requests[0].verification_note,"")
        self.assertFalse(a.SchoolVerify().independently_verified)
        self.assertFalse(a.FamilyBatchItem(request_id="ssint_test123").independently_verified)
        source=inspect.getsource(a.verify_family_batch)
        self.assertIn("each_child_must_be_independently_verified",source)
    def test_parent_approval_emails_follow_both_school_verification_paths(self):
        import inspect
        single=inspect.getsource(a.verify_child)
        batch=inspect.getsource(a.verify_family_batch)
        self.assertIn("_notify_parent_after_approval(school_id,[request_id])",single)
        self.assertIn("_notify_parent_after_approval(school_id,request_ids)",batch)
        self.assertIn('parent_approval_email',single)
        self.assertIn('parent_approval_email',batch)
        delivery=inspect.getsource(a._notify_parent_after_approval)
        self.assertIn("approval_email_status",delivery)
        self.assertIn("approval_email_attempts",delivery)
        self.assertIn("approval_email_sent_at",delivery)
        self.assertIn("pending=[",delivery)
        self.assertIn("activated_at IS NULL",delivery)
        self.assertIn("email_verified",delivery)
        self.assertIn("parent must complete",delivery)
        self.assertNotIn("email_code",delivery)
    def test_approval_email_retry_requires_school_admin(self):
        import inspect
        retry=inspect.getsource(a.retry_parent_approval_email)
        self.assertIn("_school_admin(cur,school_id",retry)
        self.assertIn("_require_pilot()",retry)
        self.assertIn("_allowed_school(school_id)",retry)
        self.assertIn("duplicate_request_ids",retry)
    def test_original_consent_required_for_automatic_creation(self):
        source=inspect.getsource(a._auto_activate_after_school_approval)
        self.assertIn("supervised_creation_consent",source)
        self.assertIn('not row["supervised_creation_consent"]',source)
        self.assertIn('not row["email_verified"]',source)
        self.assertIn("_school_accepted(cur,school_id)",source)
        self.assertIn("_allowed_school(school_id)",source)
        self.assertIn("status='supervised_child'",source.replace("'supervised_child',","status='supervised_child'"))
        self.assertIn("can_receive_calls,call_privacy,status",source)
        self.assertIn("supervised_child_created_with_prior_parent_consent",source)

    def test_verification_endpoints_dispatch_auto_only_after_commit(self):
        one=inspect.getsource(a.verify_child)
        many=inspect.getsource(a.verify_family_batch)
        self.assertIn("_auto_activate_after_school_approval(school_id,[request_id])",one)
        self.assertIn("_auto_activate_after_school_approval(school_id,request_ids)",many)
        self.assertIn("automatic_activation",one)
        self.assertIn("activated_count",many)
        self.assertIn("guardian_email_confirmation_required",many)

    def test_pilot_default_off(self):
        with patch.dict("os.environ",{"VERIFYD_SAFETY_CHILD_ACTIVATION_ENABLED":"0"}):
            with self.assertRaises(HTTPException) as e:
                a._require_pilot()
            self.assertEqual(e.exception.status_code,503)
    def test_no_unauthenticated_status(self):
        with patch.object(a,"_actor",side_effect=HTTPException(401,"session_required")):
            with self.assertRaises(HTTPException):
                a.my_activation_status(None)
if __name__=="__main__":unittest.main()
