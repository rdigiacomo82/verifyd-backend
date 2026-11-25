"""Circle 16 TEST school-response channel isolation (no DB or network)."""
import inspect
import unittest
from unittest.mock import MagicMock, patch

from fastapi import HTTPException
import trust_voice_school_staff_responses as staff
import trust_voice_school_student_device as device
import trust_voice_school_incident_chat as chat


def db(cursor):
    cm=MagicMock()
    cm.__enter__.return_value.cursor.return_value=cursor
    cm.__exit__.return_value=False
    return cm


class SchoolResponseContractTests(unittest.TestCase):
    def test_guardian_cannot_enter_school_thread(self):
        cursor=MagicMock()
        report={"school_id":"school_1","student_id":"student_1"}
        with patch.object(staff,"_authorize",return_value=report),patch.object(
             staff,"_membership",return_value="guardian"):
            with self.assertRaises(HTTPException) as err:
                staff._school_chat_authorize(cursor,"report_1","guardian_1")
        self.assertEqual(err.exception.status_code,403)
        self.assertIn("school_response_student_or_approved_staff_only",str(err.exception.detail))

    def test_reporting_student_allowed_but_other_student_denied_by_authorize(self):
        cursor=MagicMock()
        report={"school_id":"school_1","student_id":"student_1"}
        with patch.object(staff,"_authorize",return_value=report),patch.object(
             staff,"_membership",return_value="student") as membership:
            self.assertIs(staff._school_chat_authorize(cursor,"report_1","student_1"),report)
            membership.assert_called_with(cursor,"school_1","student_1",{"student"})
        with patch.object(staff,"_authorize",side_effect=HTTPException(403,detail="not_incident_recipient")):
            with self.assertRaises(HTTPException):
                staff._school_chat_authorize(cursor,"report_1","unrelated_student")

    def test_approved_principal_and_responder_only(self):
        cursor=MagicMock()
        report={"school_id":"school_1","student_id":"student_1"}
        for role in ("principal","responder"):
            with patch.object(staff,"_authorize",return_value=report),patch.object(
                 staff,"_membership",return_value=role):
                self.assertIs(staff._school_chat_authorize(cursor,"report_1","staff_1"),report)

    def test_cross_school_session_never_reads_staff_messages(self):
        cursor=MagicMock()
        report={"school_id":"school_1","student_id":"student_1"}
        with patch.object(staff,"_schema"),patch.object(staff,"get_db",return_value=db(cursor)),patch.object(
             staff,"_school_chat_authorize",return_value=report):
            with self.assertRaises(HTTPException) as err:
                staff._read_school_response("report_1","student_1",school_id="school_2")
        self.assertEqual(err.exception.status_code,403)
        self.assertFalse(any("FROM verifyd_safety_incident_school_responses" in str(c)
                             for c in cursor.execute.call_args_list))

    def test_closed_incident_rejects_school_response_write(self):
        cursor=MagicMock()
        report={"school_id":"school_1","student_id":"student_1","closed_at":None,"incident_status":"closed"}
        with patch.object(staff,"_schema"),patch.object(staff,"get_db",return_value=db(cursor)),patch.object(
             staff,"_school_chat_authorize",return_value=report):
            with self.assertRaises(HTTPException) as err:
                staff._send_school_response("report_1","principal_1",chat.Reply(client_id="idempotent001",body="I can help"))
        self.assertEqual(err.exception.status_code,409)
        self.assertFalse(any("INSERT INTO verifyd_safety_incident_school_responses" in str(c)
                             for c in cursor.execute.call_args_list))

    def test_guards_before_any_message_read_or_write(self):
        src=inspect.getsource(staff._read_school_response)
        self.assertIn("_school_chat_authorize",src)
        self.assertIn("verifyd_safety_incident_school_responses",src)
        src=inspect.getsource(staff._send_school_response)
        self.assertIn("_school_chat_authorize",src)
        self.assertIn("ON CONFLICT(report_id,sender_id,client_id) DO NOTHING",src)
        self.assertIn("_allowed",src)
        self.assertNotIn("verifyd_safety_incident_messages",src)

    def test_school_thread_never_opens_private_family_table(self):
        for fn in (staff._read_school_response,staff._send_school_response):
            src=inspect.getsource(fn)
            self.assertNotIn("verifyd_safety_incident_messages",src)
            self.assertNotIn("_family_chat_authorize",src)
        private=inspect.getsource(chat._family_chat_authorize)
        self.assertIn('!= "guardian"',private)

    def test_push_rechecks_school_thread_authorization_and_not_message_body(self):
        src=inspect.getsource(chat._dispatch)
        self.assertIn('startswith("schoolreply:")',src)
        self.assertIn("_school_chat_authorize",src)
        self.assertIn("A school TEST response is ready",src)
        self.assertNotIn("payload.body",src)

    def test_restricted_student_routes_use_session_and_belong_to_same_school(self):
        for fn in (device.device_school_response,device.device_school_response_reply):
            src=inspect.getsource(fn)
            self.assertIn("_session(x_safety_device_session)",src)
            self.assertIn('school_id=ctx["school_id"]',src)
        for route in device.router.routes:
            if "student-devices/reports/{report_id}/school-response" in route.path:
                field = inspect.signature(route.endpoint).parameters["x_safety_device_session"].default
                self.assertEqual(field.alias, "X-VeriFYD-Safety-Session")

    def test_staff_routes_use_signed_in_identity(self):
        for fn in (staff.read_school_response,staff.send_school_response):
            self.assertIn("_actor(authorization)",inspect.getsource(fn))


if __name__=="__main__":
    unittest.main()
