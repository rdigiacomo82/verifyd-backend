"""Android FCM pilot API and transport checks without credentials or live push."""
import asyncio
import os
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from cryptography.fernet import Fernet
from fastapi import FastAPI
from fastapi.testclient import TestClient

import trust_voice_android_calls as android
import trust_voice_signaling as signaling

ENV = {
    'VERIFYD_TRUST_VOICE_SIGNALING_ENABLED': '1',
    'VERIFYD_ANDROID_CALLS_ENABLED': '1',
    'VERIFYD_ANDROID_CALLS_PILOT_HANDLES': 'pilot',
    'VERIFYD_ANDROID_FIREBASE_PROJECT_ID': 'verifyd-circle-test',
    'GOOGLE_APPLICATION_CREDENTIALS': '/test-only',
    'VERIFYD_NATIVE_CALL_TOKEN_ENCRYPTION_KEY': Fernet.generate_key().decode(),
}
INSTALLATION = '00000000-0000-4000-8000-000000000001'
IDENTITY = {'id': 'callee', 'handle': 'pilot'}
BODY = {
    'installation_id': INSTALLATION,
    'package_name': android.PACKAGE,
    'fcm_token': 'a' * 120,
    'opted_in': True,
}
CALL_ID = 'tvcall_' + 'a' * 32
PAIRINGS_URL = '/trust-voice/android-call-pairings'


class AndroidPilotTests(unittest.TestCase):
    def setUp(self):
        app = FastAPI()
        app.include_router(android.router)
        self.client = TestClient(app)

    def test_disabled_routes_do_not_touch_auth_or_registry(self):
        with patch.dict(os.environ, {
            'VERIFYD_TRUST_VOICE_SIGNALING_ENABLED': '1',
        }, clear=True), patch.object(android, '_auth') as auth:
            self.assertEqual(self.client.post(PAIRINGS_URL).status_code, 404)
            self.assertEqual(self.client.get(f'/trust-voice/android-calls/{CALL_ID}').status_code, 404)
            self.assertEqual(self.client.post(f'/trust-voice/android-calls/{CALL_ID}/answer').status_code, 404)
        auth.assert_not_called()

    def test_requires_persistent_unrevoked_pilot_session(self):
        with patch.dict(os.environ, ENV, clear=True):
            self.assertEqual(self.client.post(PAIRINGS_URL).status_code, 401)
            with patch.object(signaling, '_secret', return_value=b'test-secret'), \
                 patch.object(signaling, '_identity_by_id', return_value=IDENTITY), \
                 patch.object(signaling, '_persistent_session_row', return_value={
                     'identity_id': IDENTITY['id'], 'revoked_at': 'now',
                 }):
                token = signaling._sign({
                    'identity_id': IDENTITY['id'], 'session_id': 'revoked',
                    'purpose': 'trust_voice_session',
                })
                result = self.client.post(PAIRINGS_URL, headers={
                    'Authorization': f'Bearer {token}',
                })
            self.assertEqual(result.status_code, 401)

    def test_excludes_detector_even_if_configured(self):
        with patch.dict(os.environ, {
            'VERIFYD_ANDROID_CALLS_ALLOWED_PACKAGES': 'com.verifyd.app,com.verifyd.circle.pilot.dev',
        }, clear=True):
            self.assertEqual(android.allowed_packages(), {'com.verifyd.circle.pilot.dev'})

    def register_with_rows(self, rows):
        db = MagicMock()
        cursor = db.return_value.__enter__.return_value.cursor.return_value
        cursor.fetchone.side_effect = [
            {'identity_id': IDENTITY['id'], 'session_id': 'session'}, {'handle_lower': 'pilot'}, *rows,
        ]
        with patch.dict(os.environ, ENV, clear=True), \
             patch.object(android, 'ensure_schema'), patch.object(android, 'get_db', db):
            result = self.client.post(PAIRINGS_URL + '/redeem', json=BODY, headers={
                'Authorization': 'Bearer ap_' + 'a' * 43,
            })
        return result, cursor.execute.call_args_list

    def test_registration_encrypts_and_binds_to_authenticated_session(self):
        result, statements = self.register_with_rows([{'n': 0}, None, {'installation_id': INSTALLATION}])
        self.assertEqual(result.status_code, 200)
        sql, args = statements[-1].args
        self.assertIn('WHERE verifyd_android_call_devices.identity_id=EXCLUDED.identity_id', sql)
        self.assertEqual(args[:4], (INSTALLATION, android.PACKAGE, IDENTITY['id'], 'session'))
        self.assertEqual(Fernet(ENV['VERIFYD_NATIVE_CALL_TOKEN_ENCRYPTION_KEY']).decrypt(
            args[5].encode()).decode(), BODY['fcm_token'])
        self.assertNotIn(BODY['fcm_token'], result.text)

    def test_registration_refuses_token_owned_by_another_account(self):
        result, statements = self.register_with_rows([{'n': 0}, {
            'installation_id': INSTALLATION, 'identity_id': 'other',
        }])
        self.assertEqual(result.status_code, 409)
        self.assertFalse(any('INSERT INTO' in call.args[0] for call in statements))

    def test_registration_refuses_installation_owned_by_another_account(self):
        result, _ = self.register_with_rows([{'n': 0}, None, None])
        self.assertEqual(result.status_code, 409)

    def test_issued_code_is_short_lived_and_only_digest_is_stored(self):
        with patch.dict(os.environ, ENV, clear=True), \
             patch.object(android, '_auth', return_value=(IDENTITY, {'session_id': 'session'})), \
             patch.object(android, 'ensure_schema'), patch.object(android, 'get_db') as db:
            result = self.client.post('/trust-voice/android-call-pairings')
        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.headers['cache-control'], 'no-store')
        code = result.json()['code']
        self.assertIsNotNone(android.PAIR_CODE_PATTERN.fullmatch(code))
        self.assertEqual(result.json()['expires_in_seconds'], 120)
        sql, args = db.return_value.__enter__.return_value.cursor.return_value.execute.call_args.args
        self.assertIn('ON CONFLICT (session_id)', sql)
        self.assertEqual(args[:3], ('session', IDENTITY['id'], android._digest(code)))

    def test_redeem_claims_code_and_writes_device_in_one_transaction(self):
        code = 'ap_' + 'a' * 43
        db = MagicMock()
        cursor = db.return_value.__enter__.return_value.cursor.return_value
        cursor.fetchone.side_effect = [
            {'identity_id': IDENTITY['id'], 'session_id': 'session'},
            {'handle_lower': 'pilot'},
            None,
        ]
        url = '/trust-voice/android-call-pairings/redeem'
        with patch.dict(os.environ, ENV, clear=True), patch.object(android, 'ensure_schema'), \
             patch.object(android, 'get_db', db), patch.object(android, '_write_device') as write:
            result = self.client.post(url, json=BODY, headers={'Authorization': f'Bearer {code}'})
            replay = self.client.post(url, json=BODY, headers={'Authorization': f'Bearer {code}'})
        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.headers['cache-control'], 'no-store')
        self.assertEqual(replay.status_code, 401)
        write.assert_called_once()
        self.assertEqual(write.call_args.args[2:], (
            IDENTITY['id'], 'session', android._digest(result.json()['device_token']),
        ))
        sql, args = cursor.execute.call_args_list[0].args
        self.assertIn('DELETE FROM verifyd_android_call_pairings', sql)
        self.assertIn('p.expires_at > EXTRACT(EPOCH FROM clock_timestamp())', sql)
        self.assertIn('s.revoked_at IS NULL', sql)
        self.assertEqual(args[0], android._digest(code))

    def test_scoped_device_can_decline_but_cannot_answer_without_media(self):
        with patch.dict(os.environ, ENV, clear=True), \
             patch.object(android, '_paired_identity', return_value=IDENTITY['id']), \
             patch.object(signaling, 'native_call_snapshot', new_callable=AsyncMock,
                          return_value={'is_callee': True}), \
             patch.object(signaling, 'apply_call_action', new_callable=AsyncMock,
                          return_value={'type': 'call_declined'}) as action:
            url = f'/trust-voice/android-calls/{CALL_ID}'
            answer = self.client.post(url + '/answer')
            decline = self.client.post(url + '/decline')
        self.assertEqual(answer.status_code, 409)
        self.assertEqual(answer.json()['detail']['error'], 'native_media_not_ready')
        self.assertEqual(decline.status_code, 200)
        action.assert_awaited_once_with(CALL_ID, IDENTITY['id'], 'call_decline')

    def test_scoped_token_is_not_a_trust_voice_session_token(self):
        token = 'ac_' + 'a' * 43
        with self.assertRaises(signaling.HTTPException) as error:
            signaling._verify_token(token)
        self.assertEqual(error.exception.status_code, 401)

    def test_unlink_does_not_initialize_pilot_schema(self):
        url = '/trust-voice/android-call-devices/current'
        with patch.dict(os.environ, {}, clear=True), \
             patch.object(android, 'ensure_schema') as schema, \
             patch.object(android, 'get_db') as db:
            self.assertEqual(self.client.delete(url).status_code, 401)
            db.assert_not_called()
            db.return_value.__enter__.return_value.cursor.return_value.fetchone.return_value = {'registry': None}
            response = self.client.delete(url, headers={'Authorization': 'Bearer ac_' + 'a' * 43})
            self.assertEqual(response.status_code, 401)
        schema.assert_not_called()

    def dispatch(self, *, ringing, age):
        post = AsyncMock(return_value=MagicMock(status_code=200))
        client = MagicMock()
        client.return_value.__aenter__.return_value.post = post
        call = {'call_id': CALL_ID, 'call_type': 'audio', 'created_at': 2000 - age}
        with patch.dict(os.environ, ENV, clear=True), \
             patch.object(android.time, 'time', return_value=2000), \
             patch.object(android, 'active_devices', return_value=[{
                 'package_name': android.PACKAGE, 'fcm_token': BODY['fcm_token'],
             }]), patch.object(android, '_access_token', return_value='test-only'), \
             patch.object(android.httpx, 'AsyncClient', client):
            sent = asyncio.run(android.dispatch_invite(IDENTITY, call, AsyncMock(return_value=ringing)))
        return sent, post

    def test_fcm_uses_original_expiry_and_circle_package(self):
        sent, post = self.dispatch(ringing=True, age=10)
        self.assertTrue(sent)
        payload = post.call_args.kwargs['json']['message']
        self.assertEqual(payload['android'], {
            'priority': 'HIGH', 'ttl': '20s', 'restricted_package_name': android.PACKAGE,
        })
        self.assertEqual(payload['data']['expires_at'], '2020')
        self.assertEqual(payload['data']['call_id'], CALL_ID)
        self.assertNotIn('identity_id', payload['data'])
        self.assertNotIn('notification', payload)

    def test_fcm_does_not_send_cancelled_or_expired_invites(self):
        for ringing, age in [(False, 10), (True, 31)]:
            with self.subTest(ringing=ringing, age=age):
                sent, post = self.dispatch(ringing=ringing, age=age)
                self.assertFalse(sent)
                post.assert_not_awaited()
