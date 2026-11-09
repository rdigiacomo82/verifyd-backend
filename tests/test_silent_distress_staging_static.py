"""No network, no DB, no real emergency calls; read-only regression guards."""
import ast
from pathlib import Path
import unittest
ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / 'trust_voice_circle_safety.py'
EXPERIMENT = ROOT / 'experiments' / 'silent_distress_text'
class SilentDistressGuardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.py = BACKEND.read_text()
        cls.jsx = (EXPERIMENT / 'TrustVoiceCircleSafety.jsx.txt').read_text()
        cls.beta_diff = (EXPERIMENT / 'TrustVoiceBeta.jsx.diff').read_text()
        cls.settings_diff = (EXPERIMENT / 'TrustVoiceSettings.jsx.diff').read_text()
    def test_valid_backend_python(self):
        ast.parse(self.py)
    def test_explicit_backend_gate(self):
        self.assertIn('VERIFYD_CIRCLE_SILENT_DISTRESS_ENABLED', self.py)
        self.assertIn('silent_distress_unavailable', self.py)
        self.assertIn('silent_mode_supported', self.py)
        self.assertLess(self.py.index('if data.mode =='), self.py.index("    _schema()\n    sender_id = str(owner[\"id\"])\n    if (data.latitude"))
    def test_server_persists_and_restores_mode(self):
        self.assertIn('next_push_at,mode)', self.py)
        self.assertIn('"mode": row.get("mode")', self.py)
        self.assertIn('"mode": r.get("mode")', self.py)
    def test_recipient_auth_guard(self):
        self.assertIn('trust_circle_contains(str(r["sender_id"]), recipient_id)', self.py)
        self.assertIn('r.muted=FALSE', self.py)
    def test_ui_rejects_backend_downgrade(self):
        self.assertIn('capabilities?.silent_mode_supported !== true', self.jsx)
        self.assertIn('Contacts were NOT notified.', self.jsx)
    def test_sender_only_pin(self):
        block = self.jsx.split("if (active?.mode === 'silent') return",1)[1].split('  const shouldShow',1)[0]
        self.assertIn('Enter PIN',block)
        self.assertNotIn('BellRing',block)
        self.assertNotIn('onRequestEmergency911',block)
    def test_recipient_native_sms_not_api(self):
        self.assertIn('href="sms:911"', self.jsx)
        self.assertIn('only if 911 requests one',self.jsx)
        self.assertNotIn('dispatch_connected: true',self.jsx)
    def test_no_unauthorized_dispatch(self):
        for x in ['CreateSIPParticipant','/calls/911','rapidsos.com','Twilio']:
            self.assertNotIn(x,self.py)
    def test_beta_settings_wiring(self):
        self.assertIn('silentRequestNonce',self.beta_diff)
        self.assertIn('onRequestSilentDistress',self.settings_diff)
if __name__ == '__main__':unittest.main()