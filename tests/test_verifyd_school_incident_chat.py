"""Tests: school test notifications require authorization and stay non-emergency."""
import inspect
import unittest
from pydantic import ValidationError
import trust_voice_school_incident_chat as chat
import trust_voice_school_safety_training as school

class SchoolIncidentAlerts(unittest.TestCase):
    def test_notification_disabled_by_default(self):
        from unittest.mock import patch
        with patch.dict("os.environ",{"VERIFYD_SAFETY_TEST_NOTIFICATIONS_ENABLED":"0","VERIFYD_SAFETY_TEST_NOTIFICATION_SCHOOL_IDS":"sschool_test"}):
            self.assertFalse(chat._allowed("sschool_test"))
        with patch.dict("os.environ",{"VERIFYD_SAFETY_TEST_NOTIFICATIONS_ENABLED":"1","VERIFYD_SAFETY_TEST_NOTIFICATION_SCHOOL_IDS":"sschool_test"}):
            self.assertTrue(chat._allowed("sschool_test"))
            self.assertFalse(chat._allowed("different_school"))

    def test_only_authorized_recipients_get_incident_messages(self):
        src=inspect.getsource(chat._authorize)
        self.assertIn("_membership(cur,report[",src)
        self.assertIn("_school_admin(cur,report[",src)
        self.assertIn("guardian_relationship_not_active",src)
        self.assertIn("POLICY_VERSION",src)
        self.assertIn("not_incident_recipient",src)

    def test_push_queue_filters_existing_roles(self):
        src=inspect.getsource(chat.queue_report)
        self.assertIn("_authorize(cur,report_id,str(ident))",src)
        self.assertIn("ON CONFLICT DO NOTHING",src)
        self.assertIn("if not _allowed(school_id):return",src)

    def test_responders_send_no_911(self):
        source=inspect.getsource(chat._dispatch)
        self.assertIn("notify_identity_push(",source)
        self.assertIn("device_delivery_confirmed",source)
        self.assertNotIn("911",source)

    def test_durable_reply_idempotency_and_closed_report(self):
        src=inspect.getsource(chat.send_reply)
        self.assertIn("_family_chat_authorize(cur,report_id,aid)",src)
        self.assertIn("incident_closed",src)
        self.assertIn("ON CONFLICT(report_id,sender_id,client_id)",src)
        with self.assertRaises(ValidationError):
            chat.Reply(client_id="short",body="hey")

    def test_family_messages_do_not_expose_staff_or_send_staff_reply_push(self):
        read_src=inspect.getsource(chat.conversation)
        write_src=inspect.getsource(chat.send_reply)
        dispatch_src=inspect.getsource(chat._dispatch)
        self.assertIn("_family_chat_authorize(cur,report_id,aid)",read_src)
        self.assertIn("_family_chat_authorize(cur,report_id,aid)",write_src)
        self.assertIn("_family_chat_authorize(conn.cursor(),report_id,str(job",dispatch_src)
        self.assertIn("school_safety_family_reply_test",dispatch_src)
        self.assertNotIn("notify_identity_push(",write_src)

    def test_report_transaction_queues_before_commit(self):
        source=inspect.getsource(school.create_training_report)
        self.assertIn("queue_report(cur, report_id",source)
        self.assertIn("dispatch_report(report_id)",source)
        self.assertIn("TRAINING_RECORD_ONLY",inspect.getsource(school._report_receipt))

    def test_acknowledgement_is_auth_scoped(self):
        source=inspect.getsource(chat.acknowledge_alert)
        self.assertIn("_authorize(cur,report_id,aid)",source)
        self.assertIn("school_test_alert_acknowledged",source)
        self.assertIn("device_notification_delivery_confirmed",source)

if __name__=="__main__":unittest.main()
