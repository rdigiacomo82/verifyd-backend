"""TEST-only supervised student email sign-in regressions; no network/DB required."""
import inspect
import os
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

from fastapi import HTTPException

import trust_voice_school_student_email as email_auth
import trust_voice_school_student_device as device


def _db(cursor):
    conn = MagicMock()
    conn.cursor.return_value = cursor
    manager = MagicMock()
    manager.__enter__.return_value = conn
    manager.__exit__.return_value = None
    return manager


class SchoolEmailAuthTests(unittest.TestCase):
    def test_disabled_by_default_and_separate_school_allowlist(self):
        with patch.object(email_auth, "_pilot"), patch.dict(
            os.environ, {"VERIFYD_SAFETY_STUDENT_EMAIL_ENABLED": "0"},
        ):
            with self.assertRaises(HTTPException) as err:
                email_auth._enabled()
            self.assertEqual(err.exception.status_code, 503)
        with patch.object(email_auth, "_school_allowed"), patch.dict(
            os.environ, {"VERIFYD_SAFETY_STUDENT_EMAIL_SCHOOL_IDS": ""},
        ):
            with self.assertRaises(HTTPException) as err:
                email_auth._allow_email_school("school_demo")
            self.assertEqual(err.exception.status_code, 403)

    def test_email_validation_and_isolated_routes(self):
        self.assertEqual(email_auth.EmailRequest(email="  SCHOOL@example.edu  ").email,
                         "school@example.edu")
        with self.assertRaises(ValueError):
            email_auth.EmailRequest(email="not an email")
        paths = [route.path for route in email_auth.router.routes]
        self.assertIn("/trust-voice/school-safety/student-devices/email/request",paths)
        self.assertIn("/trust-voice/school-safety/student-devices/email/verify",paths)
        self.assertNotIn("/trust-voice/session",paths)
        self.assertNotIn("/trust-voice/safety/distress",paths)

    def test_unknown_student_request_never_sends_and_is_generic(self):
        cursor=MagicMock()
        with patch.object(email_auth,"_enabled"),patch.object(email_auth,"_schema"),patch.object(
            email_auth,"_candidate",return_value=None
        ),patch.object(email_auth,"get_db",return_value=_db(cursor)),patch.object(
            email_auth,"_send_code"
        ) as send:
            result=email_auth.request_email_code(email_auth.EmailRequest(email="other@example.edu"))
        self.assertTrue(result["no_account_information_disclosed"])
        self.assertNotIn("student_id",result)
        send.assert_not_called()

    def test_duplicate_roster_addresses_never_qualify(self):
        cursor=MagicMock()
        cursor.fetchall.return_value=[{"school_id":"one","student_identity_id":"one"},
                                       {"school_id":"two","student_identity_id":"two"}]
        with patch.object(email_auth,"_allow_email_school",side_effect=AssertionError("ambiguous")):
            self.assertIsNone(email_auth._candidate(cursor,"shared@example.edu"))

    def test_email_cannot_be_a_registered_adult_account(self):
        cursor=MagicMock()
        cursor.fetchall.return_value=[{"school_id":"school","student_identity_id":"child"}]
        cursor.fetchone.return_value={"exists":1}
        with patch.object(email_auth,"_allow_email_school"),patch.object(
            email_auth,"_supervised"),patch.object(email_auth,"_guardian_ready"):
            self.assertIsNone(email_auth._candidate(cursor,"adult@example.edu"))

    def test_otp_is_not_stored_as_plaintext_and_uses_attempt_limit(self):
        src=inspect.getsource(email_auth.request_email_code)
        self.assertIn("_hash_code(salt,code_str)",src)
        self.assertIn("_MAX_DAILY",src)
        self.assertIn("_COOLDOWN_SECONDS",src)
        self.assertIn("FOR UPDATE",src)
        src=inspect.getsource(email_auth.verify_email_code)
        self.assertIn("hmac.compare_digest",src)
        self.assertIn('challenge["attempts"] < 5',src)
        self.assertIn("used_at=NOW()",src)

    def test_bad_verification_code_increments_attempt_and_never_mints_session(self):
        cursor=MagicMock()
        cursor.fetchone.return_value={
            "school_id":"school","student_id":"child","salt":"abcd","code_hash":"notmatching",
            "attempts":0,"used_at":None,
            "expires_at":datetime.now(timezone.utc)+timedelta(minutes=5)
        }
        with patch.object(email_auth,"_enabled"),patch.object(email_auth,"_schema"),patch.object(
            email_auth,"get_db",return_value=_db(cursor)
        ):
            with self.assertRaises(HTTPException) as err:
                email_auth.verify_email_code(email_auth.EmailVerify(email="child@example.edu",code="12345678"))
        self.assertEqual(err.exception.status_code,403)
        self.assertTrue(any("attempts=attempts+1" in str(c) for c in cursor.execute.call_args_list))
        self.assertFalse(any("INSERT INTO verifyd_safety_student_device_sessions" in str(c)
                             for c in cursor.execute.call_args_list))

    def test_valid_code_mints_only_school_restricted_token(self):
        cursor=MagicMock()
        now=datetime.now(timezone.utc)
        salt="012345abcdef"
        challenge={
            "school_id":"school","student_id":"child","salt":salt,
            "code_hash":email_auth._hash_code(salt,"12345678"),"attempts":0,
            "used_at":None,"expires_at":now+timedelta(minutes=5)
        }
        cursor.fetchone.side_effect=[
            challenge,{"identity_id":"principal"},{"expires_at":now+timedelta(minutes=45)}
        ]
        with patch.object(email_auth,"_enabled"),patch.object(email_auth,"_schema"),patch.object(
            email_auth,"_candidate",return_value={"school_id":"school","student_id":"child"}
        ),patch.object(email_auth,"get_db",return_value=_db(cursor)),patch.object(
            email_auth,"_school_admin"),patch.object(email_auth,"_audit"):
            result=email_auth.verify_email_code(email_auth.EmailVerify(
                email="child@example.edu",code="12345678"))
        self.assertTrue(result["restricted_safety_only"])
        self.assertFalse(result["normal_voice_login_enabled"])
        self.assertFalse(result["emergency_dispatch"])
        self.assertTrue(result["email_verified_for_safety"])
        self.assertGreater(len(result["session_token"]),32)
        self.assertTrue(any("'student_email'" in str(c) for c in cursor.execute.call_args_list))

    def test_changed_student_email_invalidates_existing_session(self):
        cursor=MagicMock()
        cursor.fetchone.side_effect=[
            {"school_id":"school","student_id":"child","code_issued_by":"principal",
             "expires_at":datetime.now(timezone.utc)+timedelta(minutes=10),
             "auth_method":"student_email","bound_email":"old@example.edu"},
            None,
        ]
        with patch.object(device,"_pilot"),patch.object(device,"_schema"),patch.object(
            device,"_school_allowed"),patch.object(device,"_school_admin"),patch.object(
            device,"get_db",return_value=_db(cursor)):
            with self.assertRaises(HTTPException) as err:
                device._session("x"*64)
        self.assertEqual(err.exception.status_code,401)
        self.assertEqual(err.exception.detail["error"],"restricted_student_email_session_revoked")

    def test_revoke_is_principal_or_approved_admin_only(self):
        cursor=MagicMock()
        with patch.object(email_auth,"_enabled"),patch.object(email_auth,"_allow_email_school"),patch.object(
            email_auth,"_actor",return_value={"id":"not_an_admin"}),patch.object(
            email_auth,"_schema"),patch.object(email_auth,"get_db",return_value=_db(cursor)),patch.object(
            email_auth,"_school_admin",side_effect=HTTPException(403,detail="not_admin")
        ):
            with self.assertRaises(HTTPException) as err:
                email_auth.revoke_email_access("school",email_auth.RevokeEmail(student_id="child123"))
        self.assertEqual(err.exception.status_code,403)
        self.assertFalse(any("UPDATE verifyd_safety_student_email_bindings" in str(c)
                             for c in cursor.execute.call_args_list))


if __name__ == "__main__":
    unittest.main()
