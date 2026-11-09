"""Offline safety tests; no production DB, secrets, SIP calls, or E911."""
from __future__ import annotations
import base64
import importlib.util
import json
import os
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

TEST_ID = 'f9e98d1e-ff2e-4d3a-bb3b-cd29becf505c'
MODULE = Path(__file__).resolve().parents[1] / 'trust_voice_private_gateway_staging.py'

# The real validator requires the production DB and secret. Stub ONLY in tests;
# the router imports the real validator in a configured backend deployment.
fake = types.ModuleType('trust_voice_messages')
def verify(authorization):
    if authorization != 'Bearer local-test-circle-session':
        from fastapi import HTTPException
        raise HTTPException(status_code=401, detail={'error': 'session_required'})
    return {'id': TEST_ID}
fake._identity_from_bearer = verify
with patch.dict(sys.modules, {'trust_voice_messages': fake}):
    spec = importlib.util.spec_from_file_location('phase7_staging_under_test', MODULE)
    stage = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(stage)
app = FastAPI()
app.include_router(stage.router)
client = TestClient(app)

VALID_ENV = {
    'VERIFYD_PRIVATE_GATEWAY_STAGE_ENABLED': '1',
    'VERIFYD_PRIVATE_GATEWAY_STAGE_ALLOW_IDS': TEST_ID,
    'VERIFYD_PRIVATE_GATEWAY_STAGE_ALLOW_LOCALHOST': '1',
    'VERIFYD_PRIVATE_GATEWAY_STAGE_LIVEKIT_URL': 'ws://127.0.0.1:7880',
    'VERIFYD_PRIVATE_GATEWAY_STAGE_LIVEKIT_KEY': 'isolated_stage_test_key',
    'VERIFYD_PRIVATE_GATEWAY_STAGE_LIVEKIT_SECRET': 'isolated-test-secret-at-least-32-characters-long',
}
HDR = {'authorization': 'Bearer local-test-circle-session'}

def decode_payload(token):
    raw = token.split('.')[1]
    return json.loads(base64.urlsafe_b64decode(raw+'='*(-len(raw)%4)))

class StageTests(unittest.TestCase):
    def test_feature_off_is_404_even_with_token(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(client.get('/trust-voice/private-gateway/staging-status',headers=HDR).status_code, 404)

    def test_needs_valid_revocable_circle_session(self):
        with patch.dict(os.environ, VALID_ENV, clear=True):
            self.assertEqual(client.get('/trust-voice/private-gateway/staging-status').status_code, 401)
            self.assertEqual(client.post('/trust-voice/private-gateway/staging-room-token',json={},headers={'authorization':'Bearer forged'}).status_code,401)

    def test_allowlist_deny_by_default(self):
        with patch.dict(os.environ, {**VALID_ENV, 'VERIFYD_PRIVATE_GATEWAY_STAGE_ALLOW_IDS':''}, clear=True):
            self.assertEqual(client.get('/trust-voice/private-gateway/staging-status',headers=HDR).status_code,403)
            self.assertEqual(client.post('/trust-voice/private-gateway/staging-room-token',headers=HDR,json={}).status_code,403)

    def test_status_is_not_health_check(self):
        with patch.dict(os.environ, VALID_ENV, clear=True):
            x=client.get('/trust-voice/private-gateway/staging-status',headers=HDR)
            self.assertEqual(x.status_code,200)
            self.assertTrue(x.json()['configured'])
            self.assertFalse(x.json()['network_verified'])
            self.assertFalse(x.json()['emergency_service_available'])

    def test_join_token_fixed_room_mic_only_three_minutes(self):
        with patch.dict(os.environ, VALID_ENV, clear=True):
            x=client.post('/trust-voice/private-gateway/staging-room-token',headers=HDR,json={})
            self.assertEqual(x.status_code,200)
            j=x.json()
            self.assertEqual(j['url'],'ws://127.0.0.1:7880')
            self.assertEqual(j['expires_in_seconds'],180)
            self.assertEqual(j['room'],'verifyd-private-echo')
            self.assertEqual(x.headers.get('Cache-Control'), 'no-store')
            v=decode_payload(j['token'])
            self.assertEqual(v['video']['canPublishSources'],['microphone'])
            self.assertFalse(v['video']['canPublishData'])
            self.assertLessEqual(v['exp']-v['iat'],180)
            self.assertNotIn(TEST_ID, j['token'])
            self.assertEqual(j['outbound_calls_enabled'], False)

    def test_payload_rejects_phone_emergency_room_and_location(self):
        with patch.dict(os.environ, VALID_ENV, clear=True):
            for bad in ({'destination':'911'}, {'phone_number':'933'}, {'room':'unsafe'}, {'latitude':40.0}, {'sip_uri':'sip:6001@other'}):
                x=client.post('/trust-voice/private-gateway/staging-room-token',headers=HDR,json=bad)
                self.assertEqual(x.status_code,422,bad)

    def test_missing_config_fails_closed(self):
        with patch.dict(os.environ, {**VALID_ENV,'VERIFYD_PRIVATE_GATEWAY_STAGE_LIVEKIT_SECRET':''}, clear=True):
            self.assertFalse(client.get('/trust-voice/private-gateway/staging-status',headers=HDR).json()['configured'])
            self.assertEqual(client.post('/trust-voice/private-gateway/staging-room-token',headers=HDR,json={}).status_code,503)

    def test_production_livekit_credentials_rejected(self):
        env={**VALID_ENV,'LIVEKIT_API_SECRET':VALID_ENV['VERIFYD_PRIVATE_GATEWAY_STAGE_LIVEKIT_SECRET']}
        with patch.dict(os.environ, env,clear=True):
            self.assertEqual(client.post('/trust-voice/private-gateway/staging-room-token',headers=HDR,json={}).status_code,503)

    def test_external_ws_rejected(self):
        with patch.dict(os.environ, {**VALID_ENV,'VERIFYD_PRIVATE_GATEWAY_STAGE_LIVEKIT_URL':'ws://outside.example:7880'}, clear=True):
            self.assertEqual(client.post('/trust-voice/private-gateway/staging-room-token',headers=HDR,json={}).status_code,503)

    def test_localhost_requires_explicit_operator_opt_in(self):
        with patch.dict(os.environ, {**VALID_ENV,'VERIFYD_PRIVATE_GATEWAY_STAGE_ALLOW_LOCALHOST':'0'}, clear=True):
            self.assertEqual(client.post('/trust-voice/private-gateway/staging-room-token',headers=HDR,json={}).status_code,503)

    def test_staging_url_cannot_equal_production(self):
        with patch.dict(os.environ, {**VALID_ENV,'VERIFYD_PRIVATE_GATEWAY_STAGE_LIVEKIT_URL':'wss://prod.example:443','LIVEKIT_URL':'wss://prod.example:443'}, clear=True):
            self.assertEqual(client.post('/trust-voice/private-gateway/staging-room-token',headers=HDR,json={}).status_code,503)

    def test_no_dial_api_exposed(self):
        paths=set(app.openapi()['paths'])
        self.assertEqual(paths,{'/trust-voice/private-gateway/staging-status','/trust-voice/private-gateway/staging-room-token'})
        self.assertNotIn('911',str(paths))

if __name__=='__main__': unittest.main()