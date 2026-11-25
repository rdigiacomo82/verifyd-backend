"""Circle 16 School Safety privacy and account header contracts (no live database)."""
import inspect
import unittest
from unittest.mock import MagicMock,patch

import trust_voice_school_directory as directory
import trust_voice_school_safety_training as training


class Circle16SchoolSafetyPrivacyTests(unittest.TestCase):
    def test_shared_directory_never_selects_or_returns_school_address(self):
        src=inspect.getsource(directory._details)
        self.assertNotIn('SELECT district_name,school_address',src)
        self.assertNotIn('"school_address":',src)
        # Principal setup application still retains its private address separately.
        self.assertIn('"school_name"',src)

    def test_directory_keeps_approved_public_school_contacts(self):
        src=inspect.getsource(directory._details)
        for name in ('school_name','district_name','school_main_phone',
                     'school_office_phone','principal_handle'):
            self.assertIn(name,src)

    def test_enrollment_identity_handle_from_authenticated_server_identity(self):
        src=inspect.getsource(training.enrollment)
        self.assertIn("current_user_handle",src)
        self.assertIn("WHERE id=%s AND status='active'",src)
        self.assertIn("(identity_id,)",src)
        self.assertIn("_actor(authorization)",src)

    def test_no_school_address_in_shared_directory_response_mock(self):
        cur=MagicMock()
        cur.fetchone.side_effect=[
            {"school_name":"Test School"},
            {"district_name":"Test District","school_website":"","principal_name":"Test Principal"},
            None,
            {"handle":"Principal"},
        ]
        cur.fetchall.return_value=[]
        value=directory._details(cur,"test_school")
        self.assertNotIn("school_address",value)
        self.assertEqual(value["school_name"],"Test School")
        self.assertEqual(value["principal_handle"],"Principal")


if __name__=="__main__":
    unittest.main()
