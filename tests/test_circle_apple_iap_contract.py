"""Contract tests for fail-closed Apple In-App Purchase integration.

No Apple accounts, certificates, purchases or production database involved.
"""
from __future__ import annotations

import os
import sys
import types
import unittest
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

# Isolate this router from unrelated app/ML/DB dependencies.
db = types.ModuleType("database")
db.get_db = MagicMock()
messages = types.ModuleType("trust_voice_messages")
messages._identity_from_bearer = lambda token: {"id":"reviewer", "status":"active"}
sys.modules.setdefault("database",db)
sys.modules.setdefault("trust_voice_messages",messages)

import trust_voice_circle_apple_iap as iap
from fastapi import HTTPException

def signed_transaction(**overrides):
    data = {
      "bundleId": "com.verifyd.circle",
      "productId": "com.verifyd.circle.monthly",
      "originalTransactionId": "orig-1",
      "transactionId": "tx-1",
      "expiresDate": int((datetime.now(timezone.utc)+timedelta(days=1)).timestamp()*1000),
      "revocationDate": None,
      "inAppOwnershipType": "PURCHASED",
    }
    data.update(overrides)
    return SimpleNamespace(**data)

class AppleIAPTests(unittest.TestCase):
    def setUp(self):
        iap._verifiers.clear()

    def test_feature_flags_default_closed(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(iap._configured())
            self.assertFalse(iap._sandbox_allowed("reviewer"))
            with self.assertRaises(HTTPException) as failure:
                iap._require_ready()
            self.assertEqual(failure.exception.status_code,503)

    def test_product_id_never_guessed(self):
        self.assertEqual(iap.PRODUCT, "com.verifyd.circle.monthly")
        self.assertEqual(iap.BUNDLE, "com.verifyd.circle")
        self.assertEqual(iap.APPLE_APP_ID,6826285833)

    def test_valid_signed_transaction(self):
        verifier=MagicMock()
        verifier.verify_and_decode_signed_transaction.return_value=signed_transaction()
        with patch.object(iap,"_verifier",return_value=verifier):
            result=iap._verify_signed_transaction("signed-jws")
        self.assertEqual(result["transaction_id"],"tx-1")
        self.assertEqual(result["environment"],"Production")
        self.assertFalse(result["revoked"])

    def test_untrusted_or_wrong_bundle_fails(self):
        verifier=MagicMock()
        verifier.verify_and_decode_signed_transaction.side_effect=Exception("Bad signature")
        with patch.object(iap,"_verifier",return_value=verifier):
            with self.assertRaises(HTTPException) as err:
                iap._verify_signed_transaction("bogus")
        self.assertEqual(err.exception.status_code,400)
        verifier.verify_and_decode_signed_transaction.side_effect=None
        verifier.verify_and_decode_signed_transaction.return_value=signed_transaction(bundleId="com.other")
        with patch.object(iap,"_verifier",return_value=verifier):
            with self.assertRaises(HTTPException):
                iap._verify_signed_transaction("signed-jws")

    def test_expired_and_revoked_never_granted(self):
        for change in [{"revocationDate":1000},{"expiresDate":1000}]:
            mock=MagicMock()
            mock.verify_and_decode_signed_transaction.return_value=signed_transaction(**change)
            with patch.object(iap,"_user",return_value={"id":"reviewer","status":"active"}),\
                 patch.object(iap,"_require_ready"),\
                 patch.object(iap,"_verifier",return_value=mock):
                with self.assertRaises(HTTPException) as err:
                    iap.verify(iap.TransactionClaim(transaction_jws="x"*40), "Bearer ok")
                self.assertEqual(err.exception.status_code,403)

    def test_sandbox_denied_without_allowlisted_account(self):
        self.assertFalse(iap._sandbox_allowed("reviewer"))
        with patch.dict(os.environ,{"VERIFYD_CIRCLE_APPLE_IAP_SANDBOX_ACCOUNT_IDS":"reviewer,sample"}):
            self.assertTrue(iap._sandbox_allowed("reviewer"))
            self.assertFalse(iap._sandbox_allowed("unknown"))

    def test_nonmatching_original_purchase_owner_denied(self):
        # Genuine JWS is NOT sufficient: same original Apple purchase must
        # never be attached to a second VeriFYD identity.
        tx={"environment":"Production","original":"orig-1","transaction_id":"tx-1",
            "expires":datetime.now(timezone.utc)+timedelta(days=1),"revoked":False,"ownership":"PURCHASED"}
        cur=MagicMock()
        cur.fetchone.side_effect=[None, {"circle_identity_id":"someone-else"}]
        conn=MagicMock()
        conn.cursor.return_value=cur
        cm=MagicMock()
        cm.__enter__.return_value=conn
        with patch.object(iap,"_user",return_value={"id":"reviewer","status":"active"}),\
             patch.object(iap,"_require_ready"),\
             patch.object(iap,"_verify_signed_transaction",return_value=tx),\
             patch.object(iap,"_schema"),\
             patch.object(iap,"get_db",return_value=cm):
            with self.assertRaises(HTTPException) as err:
                iap.verify(iap.TransactionClaim(transaction_jws="x"*40), "Bearer ok")
        self.assertEqual(err.exception.status_code,409)

    def test_refund_ledger_blocks_replay_before_original_claim(self):
        tx={"environment":"Production","original":"orig-1","transaction_id":"tx-refunded",
            "expires":datetime.now(timezone.utc)+timedelta(days=10),
            "revoked":False,"ownership":"PURCHASED"}
        cur=MagicMock()
        cur.fetchone.return_value={"transaction_id":"tx-refunded"}
        conn=MagicMock(); conn.cursor.return_value=cur
        cm=MagicMock(); cm.__enter__.return_value=conn
        with patch.object(iap,"_user",return_value={"id":"reviewer","status":"active"}), \
             patch.object(iap,"_require_ready"), \
             patch.object(iap,"_verify_signed_transaction",return_value=tx), \
             patch.object(iap,"_schema"), \
             patch.object(iap,"get_db",return_value=cm):
            with self.assertRaises(HTTPException) as err:
                iap.verify(iap.TransactionClaim(transaction_jws="x"*40), "Bearer ok")
        self.assertEqual(err.exception.status_code,403)
        self.assertIn("revoked",str(err.exception.detail))
        sql=" ".join(str(c.args[0]) for c in cur.execute.call_args_list)
        self.assertIn("circle_apple_subscription_revocations",sql)
        self.assertNotIn("INSERT INTO circle_apple_subscription_owners",sql)

    def test_signed_refund_records_revocation_before_any_account_ownership(self):
        signed_event=SimpleNamespace(
            notificationUUID="uuid-refund",
            notificationType="NotificationTypeV2.REFUND",
            data=SimpleNamespace(signedTransactionInfo="signed-jws",signedRenewalInfo=None))
        fake=MagicMock()
        fake.verify_and_decode_notification.return_value=signed_event
        tx={"environment":"Production","original":"orig-1","transaction_id":"tx-1",
            "expires":datetime.now(timezone.utc)+timedelta(days=1),
            "revoked":False,"ownership":"PURCHASED"}
        cur=MagicMock()
        # New UUID insertion returns row, no existing Apple owner.
        cur.fetchone.side_effect=[{"notification_uuid":"uuid-refund"},None]
        conn=MagicMock();conn.cursor.return_value=cur
        cm=MagicMock();cm.__enter__.return_value=conn
        with patch.object(iap,"_require_ready"), \
             patch.object(iap,"_schema"), \
             patch.object(iap,"_verifier",return_value=fake), \
             patch.object(iap,"_verify_signed_transaction",return_value=tx), \
             patch.object(iap,"get_db",return_value=cm):
            result=iap.signed_notification(iap.NotificationRequest(signedPayload="x"*40))
        self.assertEqual(result,{"accepted":True,"duplicate":False})
        sql=" ".join(str(c.args[0]) for c in cur.execute.call_args_list)
        self.assertIn("INSERT INTO circle_apple_subscription_revocations",sql)
        self.assertNotIn("INSERT INTO circle_apple_subscription_transactions",sql)

    def test_client_flag_cannot_grant_entitlement(self):
        from pydantic import ValidationError
        with self.assertRaises(ValidationError):
            iap.TransactionClaim.model_validate({"paid":True})
        with self.assertRaises(ValidationError):
            iap.TransactionClaim.model_validate({"transaction_jws":"fake"})

    def test_transaction_record_never_removes_previous_revocation(self):
        import inspect
        source=inspect.getsource(iap._record)
        self.assertIn("revoked OR EXCLUDED.revoked",source)

if __name__=="__main__":
    unittest.main()
