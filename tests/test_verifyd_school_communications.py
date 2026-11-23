"""School communications isolation invariants (training only)."""
import inspect
import unittest
from unittest.mock import patch
from fastapi import HTTPException
from pydantic import ValidationError
import trust_voice_school_communications as c

class SchoolCommsContracts(unittest.TestCase):
    def test_audience_enum_and_limits(self):
        with self.assertRaises(ValidationError):
            c.PostInput(title="Hello",body="Long enough",audience_type="everyone")
        with self.assertRaises(ValidationError):
            c.AttachmentInput(filename="oversized.pdf",mime_type="application/pdf",base64_data="A"*1400001)

    def test_publish_stays_non_emergency(self):
        s=inspect.getsource(c.create_post)
        self.assertIn("_school_admin(cur,school_id",s)
        self.assertIn("_eligible(cur,school_id)",s)
        self.assertIn("push_sent",s)
        self.assertIn("_BLOCKED",s)
        self.assertNotIn("send_push",s)

    def test_every_staff_operation_checks_school_auth(self):
        for fn in (c.create_post,c.staff_posts,c.set_student_group,c.attach,c.staff_download):
            self.assertIn("_school_admin(cur,school_id",inspect.getsource(fn))

    def test_guardian_access_filters_current_authority(self):
        for fn in (c.guardian_posts,c.mark_read,c.attachment_list,c.download):
            self.assertIn("_eligible(cur,",inspect.getsource(fn))

    def test_unauthenticated_request_fails_closed(self):
        with patch.object(c,"_actor",side_effect=HTTPException(401,"session_required")):
            with self.assertRaises(HTTPException) as e:
                c.guardian_posts(None)
            self.assertEqual(e.exception.status_code,401)

if __name__=="__main__":
    unittest.main()
