"""No secrets, real phone numbers, precise coordinates or public emergency network."""
import asyncio
import os
import unittest
from contextlib import asynccontextmanager
from unittest.mock import patch

from fastapi.testclient import TestClient

from gateway.api import app, store
from gateway.lab_core import LabStore, LAB_DESTINATION, SESSION_LIFETIME_SECONDS
from gateway.sip_loopback import MockSIPProtocol, run_options_probe

TEST_KEY = 'lab_testing_only_key_abcdefghijklmnopqrstuvwxyz0123456789'


@asynccontextmanager
async def local_sip_server():
    loop = asyncio.get_running_loop()
    transport, _ = await loop.create_datagram_endpoint(MockSIPProtocol, local_addr=('127.0.0.1', 5099))
    try:
        yield
    finally:
        transport.close()


class LabTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        self.cred = {'x-verifyd-lab-key': TEST_KEY}
        self.vars = patch.dict(os.environ, {'VERIFYD_EMERGENCY_LAB_ENABLED': '1', 'VERIFYD_EMERGENCY_LAB_KEY': TEST_KEY})
        self.vars.start()

    def tearDown(self):
        self.vars.stop()

    def test_console_is_explicitly_lab_only(self):
        r = self.client.get('/lab/console')
        self.assertEqual(r.status_code, 200)
        self.assertIn('TEST ONLY', r.text)
        self.assertIn('NOT E911', r.text)
        self.assertNotIn(TEST_KEY, r.text)
        self.assertIn('no-store', r.headers.get('cache-control'))
        with patch.dict(os.environ, {'VERIFYD_EMERGENCY_LAB_ENABLED': '0'}):
            self.assertEqual(self.client.get('/lab/console').status_code, 404)

    def test_disabled_by_default(self):
        with patch.dict(os.environ, {'VERIFYD_EMERGENCY_LAB_ENABLED': '0'}):
            self.assertEqual(self.client.get('/lab/health', headers=self.cred).status_code, 404)
            self.assertEqual(self.client.post('/lab/sessions', json={}, headers=self.cred).status_code, 404)

    def test_strong_key_required(self):
        self.assertEqual(self.client.get('/lab/health').status_code, 401)
        with patch.dict(os.environ, {'VERIFYD_EMERGENCY_LAB_KEY': 'weak'}):
            self.assertEqual(self.client.get('/lab/health', headers=self.cred).status_code, 401)
        self.assertEqual(self.client.get('/lab/health', headers=self.cred).json()['real_911_enabled'], False)

    def test_refuses_real_calls_and_geo_data(self):
        for body in [
            {'number': '911'}, {'destination': '911'},
            {'destination': '+12125551234'}, {'sip_uri': 'sip:911@emergency.example'},
            {'latitude': 40.72, 'longitude': -73.98},
            {'synthetic_location': 'my_address'},
        ]:
            r = self.client.post('/lab/sessions', headers=self.cred, json=body)
            self.assertEqual(r.status_code, 422, body)

    def test_session_lifecycle_no_media_or_dispatch(self):
        r = self.client.post('/lab/sessions', headers=self.cred, json={})
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertEqual(data['destination'], LAB_DESTINATION)
        self.assertFalse(data['real_emergency_call'])
        self.assertFalse(data['media_connected'])
        self.assertIn('LAB ONLY', data['message'])
        sid = data['id']
        self.assertEqual(self.client.get(f'/lab/sessions/{sid}', headers=self.cred).status_code, 200)
        self.assertEqual(self.client.post(f'/lab/sessions/{sid}/close', headers=self.cred).json()['status'], 'closed')

    def test_websocket_lab_only(self):
        sid = self.client.post('/lab/sessions', headers=self.cred, json={}).json()['id']
        with self.client.websocket_connect(f'/lab/ws/{sid}') as ws:
            ws.send_json({'type': 'lab_auth', 'key': TEST_KEY})
            hello = ws.receive_json()
            self.assertEqual(hello['type'], 'lab_ready')
            ws.send_json({'type': 'status'})
            self.assertEqual(ws.receive_json()['session']['id'], sid)
            ws.send_json({'type': 'dial', 'number': '911'})
            self.assertEqual(ws.receive_json()['error'], 'test_actions_only')
            ws.send_json({'type': 'close'})
            self.assertEqual(ws.receive_json()['session']['status'], 'closed')

    def test_invalid_websocket_key_rejected(self):
        from starlette.websockets import WebSocketDisconnect
        sid = self.client.post('/lab/sessions', headers=self.cred, json={}).json()['id']
        with self.assertRaises(WebSocketDisconnect):
            with self.client.websocket_connect(f'/lab/ws/{sid}') as ws:
                ws.send_json({'type': 'lab_auth', 'key': 'wrong'})
                ws.receive_json()

    def test_bounded_session_count(self):
        lab = LabStore()
        from gateway.lab_core import MAX_ACTIVE_SESSIONS
        for _ in range(MAX_ACTIVE_SESSIONS): lab.create()
        with self.assertRaises(OverflowError): lab.create()

    def test_fixed_short_session_ttl(self):
        lab = LabStore()
        item = lab.create()
        self.assertLessEqual((item.expires_at - item.created_at).total_seconds(), SESSION_LIFETIME_SECONDS)

    def test_sip_no_network_emergency_dial(self):
        from gateway.sip_loopback import _options_request
        req = _options_request().decode()
        self.assertTrue(req.startswith('OPTIONS sip:lab@127.0.0.1:5099 SIP/2.0'))
        self.assertNotIn('INVITE', req)
        self.assertNotIn('911', req)

    def test_sip_options_loopback(self):
        async def run():
            async with local_sip_server():
                return await run_options_probe()
        self.assertTrue(asyncio.run(run()))

    def test_probe_returns_explicit_signaling_only(self):
        sid = self.client.post('/lab/sessions', headers=self.cred, json={}).json()['id']
        async def successful_probe():
            return True
        with patch('gateway.api.run_options_probe', side_effect=successful_probe):
            result = self.client.post(f'/lab/sessions/{sid}/sip-probe', headers=self.cred).json()
        self.assertEqual(result['sip_probe_status'], 'sip_options_acknowledged')
        self.assertFalse(result['media_connected'])
        self.assertFalse(result['real_emergency_call'])

    def test_closed_lab_session_rejects_sip_probe(self):
        sid = self.client.post('/lab/sessions', headers=self.cred, json={}).json()['id']
        self.client.post(f'/lab/sessions/{sid}/close', headers=self.cred)
        self.assertEqual(self.client.post(f'/lab/sessions/{sid}/sip-probe', headers=self.cred).status_code, 409)

    def test_unknown_session_no_leak(self):
        self.assertEqual(self.client.get('/lab/sessions/not_real', headers=self.cred).status_code, 404)


if __name__ == '__main__':
    unittest.main()