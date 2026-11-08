"""Validation tests for standalone native-call protocol: no network or secrets."""
import unittest

from native_call_contract import SCHEMA, build_invite, fcm_data, apns_voip_payload, validate_data

CALL_ID = "tvcall_" + "a" * 32


class NativeCallContractTests(unittest.TestCase):
    def setUp(self):
        self.invite = build_invite(CALL_ID, "audio", 1000, now=1005)

    def test_30_second_expiry_does_not_reset(self):
        self.assertEqual(1030, self.invite.expires_at)

    def test_apns_aps_voip(self):
        p = apns_voip_payload(self.invite)
        self.assertEqual(p["aps"], {"content-available": 1})
        self.assertEqual(p["schema"], SCHEMA)

    def test_fcm_data_strings(self):
        self.assertTrue(all(isinstance(x, str) for x in fcm_data(self.invite).values()))

    def test_fcm_no_secrets_or_message_text(self):
        self.assertEqual(
            set(fcm_data(self.invite)),
            {"schema", "kind", "call_id", "call_uuid", "call_type", "expires_at"},
        )

    def test_reject_bad_call_id(self):
        with self.assertRaises(ValueError):
            build_invite("invalid-id", "audio", 1000, now=1005)

    def test_reject_expired(self):
        with self.assertRaisesRegex(ValueError, "expired"):
            build_invite(CALL_ID, "audio", 1000, now=1030)

    def test_reject_extra_schema(self):
        data = fcm_data(self.invite)
        data["schema"] = "verifyd.native_call.v2"
        with self.assertRaises(ValueError):
            validate_data(data, now=1005)

    def test_reject_future_creation(self):
        with self.assertRaises(ValueError):
            build_invite(CALL_ID, "audio", 1200, now=1005)

    def test_reject_huge_expiry(self):
        data = fcm_data(self.invite)
        data["expires_at"] = "5000"
        with self.assertRaises(ValueError):
            validate_data(data, now=1005)

    def test_reject_mismatched_uuid(self):
        data = fcm_data(self.invite)
        data["call_uuid"] = "00000000-0000-0000-0000-000000000000"
        with self.assertRaises(ValueError):
            validate_data(data, now=1005)

    def test_reject_stale_data(self):
        with self.assertRaises(ValueError):
            validate_data(fcm_data(self.invite), now=1030)

    def test_reject_unknown_call_type(self):
        with self.assertRaises(ValueError):
            build_invite(CALL_ID, "emergency", 1000, now=1005)

    def test_stable_uuid(self):
        self.assertEqual(
            self.invite.call_uuid,
            build_invite(CALL_ID, "audio", 1000, now=1010).call_uuid,
        )

    def test_validate_expiry_before_deadline(self):
        self.assertEqual(validate_data(fcm_data(self.invite), now=1029), self.invite)

    def test_video_valid(self):
        i = build_invite(CALL_ID, "video", 1000, now=1005)
        self.assertEqual("video", validate_data(fcm_data(i), now=1005).call_type)


if __name__ == "__main__":
    unittest.main()
