"""Static safeties for the optional local-only Horizons staging page."""
import unittest
from pathlib import Path

PAGE = Path(__file__).resolve().parents[1] / 'phase7/horizons/TrustVoicePrivateGatewayLab.jsx.txt'

class StageFrontendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = PAGE.read_text(encoding='utf-8')

    def test_dev_flag_and_origin_gate(self):
        s = self.source
        self.assertIn('import.meta.env.DEV', s)
        self.assertIn("VITE_VERIFYD_PRIVATE_GATEWAY_STAGE_BROKER_ENABLED === 'true'", s)
        self.assertIn('STAGE_BROKER_ENABLED &&', s)
        self.assertIn('supportedOrigin()', s)

    def test_manual_echo_still_available(self):
        s = self.source
        self.assertIn('const ROOM_URL = \'ws://127.0.0.1:7880\';', s)
        self.assertIn('Join room and enable microphone', s)
        self.assertIn('Leave and stop microphone', s)
        self.assertIn('call-echo --confirm-private-echo', s)
        self.assertIn('room.localParticipant.setMicrophoneEnabled(true)', s)

    def test_browser_refuses_unexpected_gateway(self):
        s = self.source
        self.assertIn('grant.room !== \'verifyd-private-echo\'', s)
        self.assertIn('grant.url !== ROOM_URL', s)
        self.assertIn('grant.emergency_service_available !== false', s)
        self.assertIn("if (roomUrlOverride !== ROOM_URL)", s)

    def test_no_account_password_or_sip_admin_controls(self):
        s = self.source
        self.assertNotIn('setItem(', s)
        self.assertNotIn('LIVEKIT_API_SECRET', s)
        self.assertNotIn('CreateSIPParticipant', s)
        self.assertNotIn('/call-911', s)
        self.assertNotIn('api_secret', s)

if __name__ == '__main__': unittest.main()