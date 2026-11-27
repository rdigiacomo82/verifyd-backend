"""App Store 5.1.1(v): authenticated deletion request regression tests."""
import inspect
import unittest
from datetime import datetime,timezone,timedelta
from unittest.mock import MagicMock, patch
from fastapi import HTTPException

import trust_voice_account_deletion as deletion

def db(cur):
    cm=MagicMock()
    cm.__enter__.return_value.cursor.return_value=cur
    cm.__exit__.return_value=False
    return cm

class DeleteRequestTests(unittest.TestCase):
    def test_confirmation_required(self):
        from pydantic import ValidationError
        with self.assertRaises(ValidationError):
            deletion.DeleteRequest.model_validate({})
        with self.assertRaises(ValidationError):
            deletion.DeleteRequest.model_validate({"confirm":"yes"})
        self.assertEqual(deletion.DeleteRequest(confirm="DELETE").confirm,"DELETE")

    def test_request_uses_authenticated_bearer_not_client_identity(self):
        src=inspect.getsource(deletion.create_deletion_request)
        self.assertIn("actor = _actor(authorization)",src)
        self.assertIn('identity_id = str(actor["id"])',src)
        self.assertNotIn("payload.identity_id",src)
        self.assertIn("pg_advisory_xact_lock",src)

    def test_existing_request_reused_without_emailing_again(self):
        now=datetime.now(timezone.utc)
        cur=MagicMock()
        cur.fetchone.return_value={"request_id":"tvdel_existing","status":"pending",
                  "created_at":now,"target_completion_at":now+timedelta(days=30)}
        with patch.object(deletion,"_actor",return_value={"id":"id_1","status":"active"}),patch.object(
            deletion,"_schema"),patch.object(deletion,"get_db",return_value=db(cur)),patch.object(
            deletion,"_notify_operator") as mail:
            response=deletion.create_deletion_request(deletion.DeleteRequest(confirm="DELETE"),"Bearer valid")
        self.assertEqual(response["status"],"pending")
        self.assertFalse(response["deletion_complete"])
        mail.assert_not_called()

    def test_operator_queue_off_by_default(self):
        with patch.object(deletion,"_actor",return_value={"id":"id_1","status":"active"}),patch.dict(
            deletion.os.environ,{"VERIFYD_ACCOUNT_DELETION_OPERATOR_IDS":""}):
            with self.assertRaises(HTTPException) as err:
                deletion.list_account_deletion_queue("Bearer valid")
        self.assertEqual(err.exception.status_code,403)

    def test_completion_requires_operator_attestations(self):
        from pydantic import ValidationError
        with self.assertRaises(ValidationError):
            deletion.ConfirmErasure(request_id="tvdel_"+"a"*32,note="Just deleting the account")
        with self.assertRaises(ValidationError):
            deletion.ConfirmErasure(
                request_id="tvdel_"+"a"*32,
                data_erasure_completed=True,
                dependent_records_reviewed=True,
                legally_required_retention_reviewed=True,
                customer_completion_confirmation_sent=False,
                note="Account and dependents deleted and verified")

    def test_never_misrepresents_pending_request_as_completed(self):
        src=inspect.getsource(deletion.create_deletion_request)
        self.assertNotIn("DELETE FROM verifyd_identities",src)
        row={"request_id":"tvdel_1","status":"pending",
             "created_at":datetime.now(timezone.utc),
             "target_completion_at":datetime.now(timezone.utc)+timedelta(days=30)}
        result=deletion._request_public(row)
        self.assertFalse(result["deletion_complete"])
        self.assertIn("request has been received",result["message"])

    def test_status_endpoint_requires_login(self):
        src=inspect.getsource(deletion.account_deletion_status)
        self.assertIn("_actor(authorization)",src)
        self.assertIn('str(actor["id"])',src)
        self.assertNotIn("email_lower",src)

if __name__=="__main__":
    unittest.main()
