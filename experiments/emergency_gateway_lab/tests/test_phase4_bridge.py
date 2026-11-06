"""Offline configuration/security tests; DO NOT contact any SIP destination."""
from __future__ import annotations

import base64
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from bridge import phase4_bridge as b


def decode_segment(value: str):
    body = value.split('.')[1]
    return json.loads(base64.urlsafe_b64decode(body + '=' * (-len(body) % 4)))


class Phase4BridgeTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.root = Path(self.folder.name)
        self._orig_generated = b.GENERATED
        self._orig_creds = b.CREDS
        b.GENERATED = self.root / '.generated'
        b.CREDS = b.GENERATED / 'credentials.json'

    def tearDown(self):
        b.GENERATED, b.CREDS = self._orig_generated, self._orig_creds
        self.folder.cleanup()

    def test_init_creates_private_files_and_no_carrier(self):
        r = b.init_lab()
        self.assertEqual(r['carrier_connected'], False)
        self.assertFalse(r['real_933_call_placed'])
        self.assertTrue((b.GENERATED / 'livekit.yaml').exists())
        self.assertTrue((b.GENERATED / 'sip.yaml').exists())
        if __import__('os').name == 'posix':
            self.assertEqual(b.CREDS.stat().st_mode & 0o777, 0o600)

    def test_init_prevents_credential_overwrite(self):
        b.init_lab()
        first = b.CREDS.read_bytes()
        with self.assertRaises(RuntimeError): b.init_lab()
        self.assertEqual(first, b.CREDS.read_bytes())

    def test_join_token_mic_only_expires_5_min_and_no_sip_admin(self):
        b.init_lab()
        token = b.generate_join_token()
        d = decode_segment(token)
        self.assertLessEqual(d['exp']-d['iat'], 300)
        self.assertEqual(d['video']['room'], 'verifyd-private-echo')
        self.assertEqual(d['video']['canPublishSources'], ['microphone'])
        self.assertNotIn('sip', d)

    def test_fixed_echo_request_does_not_route_emergency(self):
        d = b.echo_request()
        self.assertEqual(d['sip_call_to'], '6000')
        self.assertEqual(d['trunk']['hostname'], '127.0.0.1:5070')
        self.assertEqual(d['room_name'], 'verifyd-private-echo')
        self.assertNotIn('911', json.dumps(d))
        self.assertNotIn('933', json.dumps(d))

    def test_refuses_unconfirmed_invites(self):
        b.init_lab()
        with self.assertRaises(PermissionError): b.call_echo(confirmed=False)

    def test_refuses_non_loopback_gateway(self):
        with patch.object(b, 'LOCAL_SIP_ENDPOINT', 'example.org:5070'):
            with self.assertRaises(RuntimeError): b._only_loopback()

    def test_creates_single_local_twirp_request_when_confirmed(self):
        b.init_lab()
        saved = []
        class FakeResponse:
            def __enter__(self): return self
            def __exit__(self, *_args): return False
            def read(self, *_args): return b'{"participant_identity":"local-sip-echo"}'
        def intercept(request, timeout=35):
            saved.append((request, timeout))
            return FakeResponse()
        with patch.object(b, 'urlopen', side_effect=intercept):
            result = b.call_echo(confirmed=True)
        self.assertTrue(result['ok'])
        self.assertEqual(len(saved), 1)
        req, timeout = saved[0]
        self.assertEqual(req.full_url, 'http://127.0.0.1:7880/twirp/livekit.SIP/CreateSIPParticipant')
        payload = json.loads(req.data)
        self.assertEqual(payload['sip_call_to'], '6000')
        self.assertEqual(payload['trunk']['hostname'], '127.0.0.1:5070')
        self.assertEqual(result['real_911_call_placed'], False)

    def test_pbxs_have_no_trunks_or_real_numbers(self):
        root = Path(b.__file__).resolve().parent
        dialplan = (root / 'asterisk/extensions.conf').read_text()
        sip = (root / 'asterisk/pjsip.conf').read_text()
        self.assertIn('exten => 6000,1,Answer()', dialplan)
        self.assertIn('Echo()', dialplan)
        self.assertNotIn('Dial(', dialplan)
        self.assertNotIn('Transfer(', dialplan)
        self.assertIn('bind=127.0.0.1:5070', sip)
        self.assertIn('match=127.0.0.1/32', sip)
        self.assertNotIn('type=registration', sip)

    def test_config_has_no_external_routing_or_public_listen(self):
        b.init_lab()
        a = (b.GENERATED / 'livekit.yaml').read_text()
        s = (b.GENERATED / 'sip.yaml').read_text()
        self.assertIn('node_ip: 127.0.0.1', a)
        self.assertIn('bind_addresses:\n  - 127.0.0.1', a)
        self.assertIn('listen_ip: 127.0.0.1', s)
        self.assertIn('use_external_ip: false', s)
        self.assertIn('rtp_port: 11000-11099', s)
        # Ephemeral one-call runner gets full CPU threshold, never production settings.
        self.assertIn('max_cpu_utilization: 1.0', s)

    def test_frontend_has_no_api_secret_or_trunk_controls(self):
        source=(Path(b.__file__).resolve().parent/'web/main.js').read_text()
        self.assertNotIn('api_secret', source)
        self.assertNotIn('CreateSIPParticipant', source)
        self.assertNotIn('tel:', source)
        self.assertIn('setMicrophoneEnabled(true)', source)

    def test_runtime_container_mounts_private_sip_config(self):
        root = Path(b.__file__).resolve().parent
        compose = (root / 'compose.yaml').read_text()
        self.assertIn('./.generated/sip.yaml:/sip/config.yaml:ro', compose)
        self.assertNotIn('SIP_CONFIG_FILE: /lab/sip.yaml', compose)
        modules = (root / 'asterisk/modules.conf').read_text()
        for name in ('chan_iax2.so', 'pbx_dundi.so', 'chan_unistim.so'):
            self.assertIn('noload => ' + name, modules)
        self.assertIn('./asterisk/modules.conf:/etc/asterisk/modules.conf:ro', compose)

    def test_compose_no_public_port_publication(self):
        c=(Path(b.__file__).resolve().parent/'compose.yaml').read_text()
        self.assertEqual(c.count('network_mode: host'),4)
        self.assertNotIn('\n    ports:', c)
        self.assertNotIn('restart: always', c)


if __name__ == '__main__': unittest.main()