"""Exercise the Android-to-browser call bridge without a phone or live credentials."""
import asyncio
import os
import time
import unittest
from unittest.mock import AsyncMock, patch

from fastapi import FastAPI, HTTPException, WebSocketDisconnect
from fastapi.testclient import TestClient

import trust_voice_android_calls as android
import trust_voice_signaling as voice

CALL = 'tvcall_' + 'a' * 32
URL = f'/trust-voice/android-calls/{CALL}/media'
ENV = {
    'VERIFYD_ANDROID_CALLS_ENABLED': '1',
    'VERIFYD_ANDROID_NATIVE_AUDIO_ENABLED': '1',
    'VERIFYD_TRUST_VOICE_SIGNALING_ENABLED': '1',
}


class AndroidMediaTests(unittest.TestCase):
    def setUp(self):
        app = FastAPI()
        app.include_router(android.router)
        self.client = TestClient(app)

    def test_bridge_disabled_by_default_before_auth(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(android, '_paired_identity') as auth:
            with self.assertRaises(WebSocketDisconnect):
                with self.client.websocket_connect(URL):
                    self.fail('Disabled socket was accepted')
        auth.assert_not_called()

    def test_only_callee_with_an_audio_invite_can_connect(self):
        for snapshot in (
            {'is_callee': False, 'state': 'ringing', 'call_type': 'audio'},
            {'is_callee': True, 'state': 'ringing', 'call_type': 'video'},
            {'is_callee': True, 'state': 'answered', 'call_type': 'audio'},
        ):
            with self.subTest(snapshot=snapshot), patch.dict(os.environ, ENV, clear=True), \
                 patch.object(android, '_paired_identity', return_value='callee'), \
                 patch.object(voice, 'native_call_snapshot', new_callable=AsyncMock, return_value=snapshot), \
                 patch.object(voice, 'apply_call_action', new_callable=AsyncMock) as action:
                with self.assertRaises(WebSocketDisconnect):
                    with self.client.websocket_connect(URL):
                        self.fail('Ineligible socket was accepted')
                action.assert_not_awaited()

    def test_signaling_uses_only_the_call_from_the_url(self):
        with patch.dict(os.environ, ENV, clear=True), \
             patch.object(android, '_paired_identity', return_value='callee'), \
             patch.object(voice, 'native_call_snapshot', new_callable=AsyncMock,
                          return_value={'is_callee': True, 'state': 'ringing', 'call_type': 'audio'}), \
             patch.object(voice, 'apply_call_action', new_callable=AsyncMock) as action, \
             patch.object(voice, 'relay_call_media', new_callable=AsyncMock) as relay:
            with self.client.websocket_connect(URL) as socket:
                socket.send_json({'type': 'call_answer'})
                socket.send_json({'type': 'webrtc_ice', 'call_id': 'other'})
                with self.assertRaises(WebSocketDisconnect):
                    socket.receive_json()
        self.assertEqual([a.args for a in action.await_args_list], [
            (CALL, 'callee', 'call_answer'), (CALL, 'callee', 'call_end'),
        ])
        self.assertIs(action.await_args_list[0].kwargs['android_socket'],
                      action.await_args_list[1].kwargs['android_socket'])
        relay.assert_not_awaited()

    def test_detector_requires_separate_flag_and_explicit_package(self):
        with patch.dict(os.environ, {
            'VERIFYD_ANDROID_DETECTOR_CALLS_ENABLED': '1',
            'VERIFYD_ANDROID_NATIVE_AUDIO_ENABLED': '1',
            'VERIFYD_ANDROID_CALLS_ALLOWED_PACKAGES': 'com.verifyd.app,com.verifyd.app.pilot.dev',
        }, clear=True):
            self.assertEqual(android.allowed_packages(), {'com.verifyd.app.pilot.dev'})
        with patch.dict(os.environ, {'VERIFYD_ANDROID_DETECTOR_CALLS_ENABLED': '1'}, clear=True):
            self.assertNotIn('com.verifyd.app', android.allowed_packages())

    def test_detector_pilot_only_receives_audio(self):
        devices = [{'package_name': 'com.verifyd.app.pilot.dev'}, {'package_name': 'com.verifyd.circle'}]
        with patch.object(android, 'active_devices', return_value=devices):
            self.assertEqual(android._devices_for_call('callee', 'audio'), devices)
            self.assertEqual(android._devices_for_call('callee', 'video'), devices[1:])


class MediaOwnershipTests(unittest.IsolatedAsyncioTestCase):
    async def test_android_answer_routes_browser_sdp_and_ice_to_one_socket(self):
        native = AsyncMock()
        browser = AsyncMock()
        callee_browser = AsyncMock()
        calls = {CALL: {
            'call_id': CALL, 'state': 'ringing', 'caller_id': 'caller',
            'callee_id': 'callee', 'call_type': 'audio', 'created_at': time.time(),
        }}
        offer = {'type': 'webrtc_offer', 'sdp': {'type': 'offer', 'sdp': 'v=0'}}
        answer = {'type': 'webrtc_answer', 'sdp': {'type': 'answer', 'sdp': 'v=0'}}
        candidate = {'type': 'webrtc_ice', 'candidate': {'candidate': 'candidate:1'}}
        with patch.object(voice, '_state_lock', asyncio.Lock()), \
             patch.object(voice, '_active_calls', calls), \
             patch.object(voice, '_connections', {'caller': {browser}, 'callee': {callee_browser}}), \
             patch.object(voice, '_connection_meta', {}), \
             patch.object(voice, '_webrtc_enabled', return_value=True), \
             patch.object(voice, '_identity_by_id', side_effect=lambda i: {'id': i}), \
             patch.object(voice, 'call_permission_decision', return_value=(True, 'ok')), \
             patch('trust_voice_blocking.identities_blocked', return_value=False), \
             patch.object(voice, '_record_call_answered'), patch.object(voice, '_record_call_terminal'):
            await voice.apply_call_action(CALL, 'callee', 'call_answer', android_socket=native)
            await voice.relay_call_media(CALL, 'caller', offer)
            await voice.relay_call_media(CALL, 'caller', candidate)
            await voice.relay_call_media(CALL, 'callee', answer, android_socket=native)
            with self.assertRaises(HTTPException):
                await voice.relay_call_media(CALL, 'callee', answer)
            with self.assertRaises(HTTPException):
                await voice.apply_call_action(CALL, 'callee', 'call_end', android_socket=AsyncMock())
            await voice._unregister('callee', callee_browser)
            self.assertIn(CALL, calls)
            await voice.apply_call_action(CALL, 'caller', 'call_end')
        self.assertNotIn(CALL, calls)
        self.assertEqual([c.args[0]['type'] for c in native.send_json.await_args_list],
                         ['call_answered', 'webrtc_offer', 'webrtc_ice', 'call_ended'])
        self.assertEqual([c.args[0]['type'] for c in browser.send_json.await_args_list],
                         ['call_answered', 'webrtc_answer', 'call_ended'])
        callee_browser.send_json.assert_awaited_once_with({
            'type': 'call_ended', 'call_id': CALL, 'reason': 'answered_on_another_device',
        })
