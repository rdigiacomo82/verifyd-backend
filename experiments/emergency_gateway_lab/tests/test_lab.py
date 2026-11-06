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

    def test_simulated_933_includes_only_fictional_placeholder_values(self):
        from gateway.simulated_933 import simulate_933_readback
        result = simulate_933_readback('provisioned')
        self.assertEqual(result['test_number_label'], '933')
        self.assertEqual(result['mode'], 'OFFLINE_SIMULATED_933')
        self.assertEqual(result['status'], 'simulated_ready')
        self.assertFalse(result['carrier_connected'])
        self.assertFalse(result['real_933_call_placed'])
        self.assertFalse(result['real_911_call_placed'])
        self.assertFalse(result['audio_connected'])
        self.assertFalse(result['real_emergency_service_validated'])
        self.assertIn('SIMULATED-', result['readback']['callback'])
        self.assertIn('SIMULATED-', result['readback']['address'])

    def test_simulated_933_reports_missing_callback_and_address(self):
        from gateway.simulated_933 import simulate_933_readback
        no_callback = simulate_933_readback('missing_callback')
        no_address = simulate_933_readback('missing_address')
        self.assertEqual(no_callback['status'], 'simulated_failure')
        self.assertIsNone(no_callback['readback']['callback'])
        self.assertEqual(no_address['status'], 'simulated_failure')
        self.assertIsNone(no_address['readback']['address'])

    def test_933_action_rejects_real_telephone_or_location_data(self):
        sid = self.client.post('/lab/sessions', headers=self.cred, json={}).json()['id']
        for body in [
            {'dial': '933'}, {'dial': '911'},
            {'callback': '+12125550123'},
            {'emergency_address': 'Some street address'},
            {'latitude': 40.0}, {'scenario': 'call_carrier'},
        ]:
            result = self.client.post(f'/lab/sessions/{sid}/simulate-933', headers=self.cred, json=body)
            self.assertEqual(result.status_code, 422, body)

    def test_933_lab_endpoint_requires_auth_and_enablement(self):
        sid = self.client.post('/lab/sessions', headers=self.cred, json={}).json()['id']
        path = f'/lab/sessions/{sid}/simulate-933'
        self.assertEqual(self.client.post(path, json={}).status_code, 401)
        with patch.dict(os.environ, {'VERIFYD_EMERGENCY_LAB_ENABLED': '0'}):
            self.assertEqual(self.client.post(path, json={}, headers=self.cred).status_code, 404)
        r = self.client.post(path, json={}, headers=self.cred)
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()['session_id'], sid)
        self.client.post(f'/lab/sessions/{sid}/close', headers=self.cred)
        self.assertEqual(self.client.post(path, json={}, headers=self.cred).status_code, 409)

    def test_933_websocket_mock_never_dials(self):
        sid = self.client.post('/lab/sessions', headers=self.cred, json={}).json()['id']
        with self.client.websocket_connect(f'/lab/ws/{sid}') as ws:
            ws.send_json({'type': 'lab_auth', 'key': TEST_KEY})
            self.assertEqual(ws.receive_json()['type'], 'lab_ready')
            ws.send_json({'type': 'simulate_933', 'scenario': 'unregistered'})
            result = ws.receive_json()
            self.assertEqual(result['type'], 'simulated_933_result')
            self.assertFalse(result['result']['carrier_connected'])
            self.assertEqual(result['result']['status'], 'simulated_failure')
            ws.send_json({'type': 'simulate_933', 'scenario': 'provisioned', 'callback': '+12125550123'})
            self.assertEqual(ws.receive_json()['error'], 'synthetic_scenario_only')
            ws.send_json({'type': 'dial', 'number': '933'})
            self.assertEqual(ws.receive_json()['error'], 'test_actions_only')
            ws.send_json({'type': 'close'})
            self.assertEqual(ws.receive_json()['type'], 'session_closed')

    def test_pcmu_codec_and_rtp_sample_validation(self):
        from gateway.private_sip_rtp import pcm16_to_mulaw, mulaw_to_pcm16, make_rtp_packet, decode_rtp_packet
        self.assertEqual(pcm16_to_mulaw(0), 0xff)
        self.assertEqual(mulaw_to_pcm16(0xff), 0)
        for n in (-20000, -10000, -2000, 2000, 10000, 20000):
            self.assertLess(abs(mulaw_to_pcm16(pcm16_to_mulaw(n)) - n), 800)
        packet = make_rtp_packet(sequence=0, ssrc=123, tone_hz=440)
        seq, ssrc, energy = decode_rtp_packet(packet)
        self.assertEqual((seq, ssrc), (0, 123))
        self.assertGreater(energy, 500)
        with self.assertRaises(ValueError):
            make_rtp_packet(sequence=100000, ssrc=123, tone_hz=440)
        with self.assertRaises(ValueError):
            make_rtp_packet(sequence=0, ssrc=123, tone_hz=900)
        with self.assertRaises(ValueError):
            decode_rtp_packet(b'\x80' * 12)

    def test_real_two_way_synthetic_rtp_on_loopback(self):
        from gateway.private_sip_rtp import run_private_audio_test
        result = asyncio.run(run_private_audio_test(packets=8))
        self.assertTrue(result['ok'], result)
        self.assertEqual(result['sip_dialog'], 'invite_200_ack_bye_completed')
        self.assertEqual(result['codec'], 'PCMU/8000')
        self.assertEqual(result['sdp_address'], '127.0.0.1')
        self.assertEqual(result['client_receives_lab_tone']['packets_received'], 8)
        self.assertEqual(result['lab_receives_client_tone']['packets_received'], 8)
        self.assertEqual(result['client_receives_lab_tone']['packet_loss'], 0)
        self.assertEqual(result['lab_receives_client_tone']['packet_loss'], 0)
        self.assertTrue(result['media_transport_connected'])
        self.assertFalse(result['livekit_connected'])
        self.assertFalse(result['carrier_connected'])
        self.assertFalse(result['real_933_call_placed'])
        self.assertFalse(result['real_911_call_placed'])

    def test_private_media_http_endpoint_auth_gate_and_body_restrictions(self):
        sid = self.client.post('/lab/sessions', headers=self.cred, json={}).json()['id']
        path = f'/lab/sessions/{sid}/private-audio-test'
        self.assertEqual(self.client.post(path, json={}).status_code, 401)
        with patch.dict(os.environ, {'VERIFYD_EMERGENCY_LAB_ENABLED': '0'}):
            self.assertEqual(self.client.post(path, json={}, headers=self.cred).status_code, 404)
        for body in ({'number': '911'}, {'number': '933'},
                     {'sip_uri': 'sip:911@example.com'}, {'latitude': 40.0},
                     {'scenario': 'real_caller'}, {'destination': '+12125550100'}):
            self.assertEqual(self.client.post(path, json=body, headers=self.cred).status_code, 422, body)
        result = self.client.post(path, json={'scenario': 'synthetic_two_way_pcmu'}, headers=self.cred)
        self.assertEqual(result.status_code, 200, result.text)
        self.assertTrue(result.json()['media_transport_connected'])
        self.assertFalse(result.json()['livekit_connected'])
        self.client.post(f'/lab/sessions/{sid}/close', headers=self.cred)
        self.assertEqual(self.client.post(path, json={}, headers=self.cred).status_code, 409)

    def test_private_media_websocket_rejects_dial_or_location_payloads(self):
        sid = self.client.post('/lab/sessions', headers=self.cred, json={}).json()['id']
        with self.client.websocket_connect(f'/lab/ws/{sid}') as ws:
            ws.send_json({'type': 'lab_auth', 'key': TEST_KEY})
            self.assertEqual(ws.receive_json()['type'], 'lab_ready')
            ws.send_json({'type': 'private_audio_test', 'target': '933'})
            self.assertEqual(ws.receive_json()['error'], 'synthetic_media_only')
            ws.send_json({'type': 'private_audio_test'})
            response = ws.receive_json()
            self.assertEqual(response['type'], 'private_audio_result')
            self.assertTrue(response['result']['media_transport_connected'])
            self.assertFalse(response['result']['real_911_call_placed'])
            ws.send_json({'type': 'close'})
            self.assertEqual(ws.receive_json()['type'], 'session_closed')

    def test_private_media_refuses_unapproved_payloads_from_udp_peer(self):
        from gateway.private_sip_rtp import RtpSink
        sink = RtpSink(peer=('127.0.0.1', 30000))
        sink.datagram_received(b'HELLO', ('203.0.113.10', 30000))
        sink.datagram_received(b'INVITE sip:911@127.0.0.1', ('127.0.0.1', 30000))
        self.assertEqual(sink.rejected, 2)
        self.assertEqual(sink.received, {})

    def test_unknown_session_no_leak(self):
        self.assertEqual(self.client.get('/lab/sessions/not_real', headers=self.cred).status_code, 404)


if __name__ == '__main__':
    unittest.main()