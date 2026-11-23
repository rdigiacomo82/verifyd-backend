"""School demo geolocation must never dispatch or expose cross-school coordinates."""
import inspect
import unittest
from unittest.mock import patch
from fastapi import HTTPException
from pydantic import ValidationError
import trust_voice_school_demo_location as loc

class DemoLocationContracts(unittest.TestCase):
    def test_opt_in_default_false_and_coord_validation(self):
        x=loc.DemoLocation(latitude=40.7,longitude=-73.9,accuracy_m=30,captured_at="2026-11-23T16:00:00+00:00")
        self.assertFalse(x.explicit_opt_in)
        with self.assertRaises(ValidationError):
            loc.DemoLocation(latitude=140,longitude=-73.9,accuracy_m=30,captured_at="2026-11-23T16:00:00+00:00")
    def test_student_self_report_and_finite_fresh_point(self):
        s=inspect.getsource(loc.share_demo_location)
        for x in ("explicit_opt_in","math.isfinite","_membership(","student_id","location_timestamp_out_of_range","_enabled_for(","_purge(cur)","real_notifications_sent"):
            self.assertIn(x,s)
        self.assertNotIn("send_sms",s)
    def test_staff_location_view_checks_school_admin(self):
        s=inspect.getsource(loc.incident_demo_locations)
        self.assertIn("_school_admin(cur,incident[\"school_id\"]",s)
        self.assertIn("r.incident_id=%s",s)
        self.assertIn("expires_at",inspect.getsource(loc._purge))
    def test_default_disabled(self):
        with patch.dict("os.environ",{"VERIFYD_SAFETY_DEMO_LOCATION_ENABLED":"0","VERIFYD_SAFETY_DEMO_LOCATION_SCHOOL_IDS":"sschool_example"}):
            with self.assertRaises(HTTPException) as x:loc._enabled_for("sschool_example")
            self.assertEqual(x.exception.status_code,403)
    def test_unauthenticated_fails(self):
        with patch.object(loc,"_actor",side_effect=HTTPException(401,"session_required")):
            with self.assertRaises(HTTPException):loc.incident_demo_locations("incident")
if __name__=="__main__":unittest.main()
