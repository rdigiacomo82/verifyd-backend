"""Native API/auth/state and mocked APNs transport; no live push or credentials."""
import asyncio
import base64
from contextlib import ExitStack, contextmanager
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric.utils import encode_dss_signature
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

import trust_voice_native_calls as native
import trust_voice_android_calls as android
import trust_voice_signaling as signal

CALL_ID = 'tvcall_' + 'a' * 32
IDENTITY = {'id': 'callee', 'handle': 'pilot'}
SESSION = {'identity_id': 'callee', 'session_id': 'session', 'purpose': 'trust_voice_session'}
ENV = {'VERIFYD_NATIVE_CALLS_ENABLED': '1', 'VERIFYD_NATIVE_CALLS_PILOT_HANDLES': 'pilot',
       'VERIFYD_TRUST_VOICE_SIGNALING_ENABLED': '1',
       'VERIFYD_APNS_TEAM_ID': '3623QELM75', 'VERIFYD_APNS_KEY_ID': 'ABCDEFGHIJ',
       'VERIFYD_APNS_PRIVATE_KEY_FILE': '/test-key',
       'VERIFYD_NATIVE_CALL_TOKEN_ENCRYPTION_KEY': base64.urlsafe_b64encode(b't'*32).decode()}
DEVICE = dict(installation_id='00000000-0000-4000-8000-000000000001',
              environment='sandbox', bundle_id=native.BUNDLE_ID, token='ab'*32,
              updated_at=100, identity_id='callee', session_id='session', disabled_at=None, token_hash='test-hash')


def call():
    return dict(call_id=CALL_ID, caller_id='caller', callee_id='callee',
                state='ringing', created_at=time.time(), call_type='audio')


@contextmanager
def authentication(row=None):
    with ExitStack() as stack:
        stack.enter_context(patch.dict(os.environ, ENV))
        stack.enter_context(patch.object(signal, '_secret', return_value=b'x'*40))
        stack.enter_context(patch.object(signal, '_identity_by_id', return_value=IDENTITY))
        stack.enter_context(patch.object(signal, '_persistent_session_row', return_value=(
            {'identity_id': 'callee', 'revoked_at': None} if row is None else row)))
        token = signal._sign(SESSION)
        yield {'Authorization': 'Bearer '+token}


class NativeAPI(unittest.TestCase):
    def test_circle_registration_is_off_by_default(self):
        body = dict(installation_id=DEVICE['installation_id'], token=DEVICE['token'],
                    environment='production', bundle_id='com.verifyd.circle', opted_in=True)
        with authentication() as auth, patch.dict(os.environ, {'VERIFYD_NATIVE_CALLS_ALLOWED_BUNDLE_IDS': native.BUNDLE_ID}):
            self.assertEqual(422, self.client.post('/trust-voice/native-call-devices', headers=auth, json=body).status_code)

    def test_circle_registration_uses_explicit_bundle_allowlist(self):
        conn = MagicMock()
        conn.cursor.return_value.fetchone.return_value = {'n': 0}
        body = dict(installation_id=DEVICE['installation_id'], token=DEVICE['token'],
                    environment='production', bundle_id='com.verifyd.circle', opted_in=True)
        with authentication() as auth, patch.dict(os.environ, {
            'VERIFYD_NATIVE_CALLS_ALLOWED_BUNDLE_IDS': native.BUNDLE_ID + ',com.verifyd.circle'
        }), patch.object(native, 'ensure_schema'), patch.object(native, 'get_db') as db:
            db.return_value.__enter__.return_value = conn
            response = self.client.post('/trust-voice/native-call-devices', headers=auth, json=body)
        self.assertEqual(200, response.status_code)
        insert = next(c.args for c in conn.cursor.return_value.execute.call_args_list if 'INSERT INTO' in c.args[0])
        self.assertEqual('com.verifyd.circle', insert[1][2])
        self.assertEqual('production', insert[1][1])

    def setUp(self):
        app = FastAPI()
        app.include_router(native.router)
        self.client = TestClient(app)
        self.body = dict(installation_id=DEVICE['installation_id'], token=DEVICE['token'],
                         environment='sandbox', bundle_id=native.BUNDLE_ID, opted_in=True)

    def test_disabled_registration_returns_404(self):
        with patch.dict(os.environ, {'VERIFYD_NATIVE_CALLS_ENABLED':'0',
                                    'VERIFYD_TRUST_VOICE_SIGNALING_ENABLED':'1'}):
            self.assertEqual(404, self.client.post('/trust-voice/native-call-devices', json=self.body).status_code)

    def test_missing_auth_is_rejected(self):
        with patch.dict(os.environ, ENV):
            self.assertEqual(401, self.client.post('/trust-voice/native-call-devices', json=self.body).status_code)

    def test_revoked_session_is_rejected(self):
        with authentication({'identity_id':'callee', 'revoked_at':'now'}) as auth:
            self.assertEqual(401, self.client.post('/trust-voice/native-call-devices', headers=auth, json=self.body).status_code)

    def test_session_db_failure_is_not_logout(self):
        with authentication() as auth, patch.object(signal, '_persistent_session_row', side_effect=signal.psycopg2.OperationalError()):
            self.assertEqual(503, self.client.post('/trust-voice/native-call-devices', headers=auth, json=self.body).status_code)

    def test_nonpilot_rejected(self):
        with authentication() as auth, patch.dict(os.environ, {'VERIFYD_NATIVE_CALLS_PILOT_HANDLES':'other'}):
            self.assertEqual(403, self.client.post('/trust-voice/native-call-devices', headers=auth, json=self.body).status_code)

    def test_legacy_token_rejected(self):
        with authentication() as auth, patch.object(signal, '_verify_token', return_value={'identity_id':'callee'}):
            self.assertEqual(401, self.client.post('/trust-voice/native-call-devices', headers=auth, json=self.body).status_code)

    def test_invalid_registration_fields(self):
        for field, value in [('token','zz'*32), ('environment','custom'), ('bundle_id','evil.app'),
                             ('opted_in',False), ('installation_id','not-uuid'), ('token','a'*65)]:
            with self.subTest(field=field, value=value), authentication() as auth:
                body = dict(self.body, **{field:value})
                self.assertEqual(422, self.client.post('/trust-voice/native-call-devices', headers=auth, json=body).status_code)

    def test_registration_scopes_to_authenticated_session(self):
        conn = MagicMock()
        conn.cursor.return_value.fetchone.return_value = {'n':0}
        with authentication() as auth, patch.object(native,'ensure_schema'), patch.object(native,'get_db') as db:
            db.return_value.__enter__.return_value = conn
            response = self.client.post('/trust-voice/native-call-devices', headers=auth, json=self.body)
        self.assertEqual(200,response.status_code)
        self.assertNotIn(DEVICE['token'],response.text)
        calls = conn.cursor.return_value.execute.call_args_list
        insert = next(c.args for c in calls if 'INSERT INTO' in c.args[0])
        self.assertEqual(('callee','session'),insert[1][3:5])
        self.assertNotIn(DEVICE['token'],insert[1])
        self.assertEqual(DEVICE['token'],native.Fernet(ENV['VERIFYD_NATIVE_CALL_TOKEN_ENCRYPTION_KEY'].encode()).decrypt(insert[1][6].encode()).decode())

    def test_device_cap(self):
        conn = MagicMock()
        conn.cursor.return_value.fetchone.return_value={'n':8}
        with authentication() as auth, patch.object(native,'ensure_schema'), patch.object(native,'get_db') as db:
            db.return_value.__enter__.return_value=conn
            self.assertEqual(409,self.client.post('/trust-voice/native-call-devices',headers=auth,json=self.body).status_code)
        self.assertFalse(any('INSERT INTO' in c.args[0] for c in conn.cursor.return_value.execute.call_args_list))

    def test_optout_after_feature_disabled(self):
        conn=MagicMock()
        with authentication() as auth, patch.dict(os.environ,{'VERIFYD_NATIVE_CALLS_ENABLED':'0'}), patch.object(native,'ensure_schema'), patch.object(native,'get_db') as db:
            db.return_value.__enter__.return_value=conn
            self.assertEqual(200,self.client.delete('/trust-voice/native-call-devices/'+DEVICE['installation_id'],headers=auth).status_code)
        args=conn.cursor.return_value.execute.call_args.args[1]
        self.assertEqual((DEVICE['installation_id'],'callee','session'),args)

    def test_call_route_requires_auth(self):
        with patch.dict(os.environ,ENV):
            self.assertEqual(401,self.client.post('/trust-voice/native-calls/'+CALL_ID+'/answer').status_code)


class CallState(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        signal._state_lock=asyncio.Lock()
        signal._active_calls.clear()
        signal._active_calls[CALL_ID]=call()
        self.stack=ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch('trust_voice_blocking.identities_blocked',return_value=False))
        self.stack.enter_context(patch.object(signal,'_identity_by_id',side_effect=lambda id: {'id':id}))
        self.stack.enter_context(patch.object(signal,'call_permission_decision',return_value=(True,'ok')))
        self.stack.enter_context(patch.object(signal,'_cancel_call_expiry'))
        self.record=self.stack.enter_context(patch.object(signal,'_record_call_answered'))
        self.terminal=self.stack.enter_context(patch.object(signal,'_record_call_terminal'))
        self.send=self.stack.enter_context(patch.object(signal,'_send_to_identity',new_callable=AsyncMock))

    async def test_answer_confirmed_and_broadcast(self):
        result=await signal.apply_call_action(CALL_ID,'callee','call_answer')
        self.assertEqual('call_answered',result['type'])
        self.assertEqual('answered',signal._active_calls[CALL_ID]['state'])
        self.assertEqual(2,self.send.await_count)

    async def test_simultaneous_answers_only_one_wins(self):
        results=await asyncio.gather(*(signal.apply_call_action(CALL_ID,'callee','call_answer') for _ in range(2)),return_exceptions=True)
        self.assertEqual(1,sum(isinstance(r,dict) for r in results))
        self.assertEqual(1,self.record.call_count)

    async def test_expired_answer_rejected_even_before_timer_runs(self):
        signal._active_calls[CALL_ID]['created_at']=time.time()-31
        with self.assertRaises(HTTPException) as ctx:
            await signal.apply_call_action(CALL_ID,'callee','call_answer')
        self.assertEqual(410,ctx.exception.status_code)
        self.record.assert_not_called()

    async def test_caller_cannot_answer(self):
        with self.assertRaises(HTTPException) as ctx:
            await signal.apply_call_action(CALL_ID,'caller','call_answer')
        self.assertEqual(403,ctx.exception.status_code)

    async def test_outsider_cannot_read_or_end(self):
        for fn,args in [(signal.native_call_snapshot,(CALL_ID,'outsider')),
                        (signal.apply_call_action,(CALL_ID,'outsider','call_end'))]:
            with self.assertRaises(HTTPException) as ctx:
                await fn(*args)
            self.assertEqual(404,ctx.exception.status_code)

    async def test_new_block_prevents_answer(self):
        with patch('trust_voice_blocking.identities_blocked',return_value=True):
            with self.assertRaises(HTTPException) as ctx:
                await signal.apply_call_action(CALL_ID,'callee','call_answer')
        self.assertEqual(403,ctx.exception.status_code)

    async def test_changed_permission_prevents_answer(self):
        with patch.object(signal,'call_permission_decision',return_value=(False,'privacy')):
            with self.assertRaises(HTTPException) as ctx:
                await signal.apply_call_action(CALL_ID,'callee','call_answer')
        self.assertEqual(403,ctx.exception.status_code)

    async def test_cancel_wins_over_pending_answer_validation(self):
        started=asyncio.Event()
        release=asyncio.Event()
        original=asyncio.to_thread
        async def delayed(fn,*args,**kwargs):
            if fn is signal.call_permission_decision:
                started.set()
                await release.wait()
            return await original(fn,*args,**kwargs)
        with patch.object(signal.asyncio,'to_thread',side_effect=delayed):
            task=asyncio.create_task(signal.apply_call_action(CALL_ID,'callee','call_answer'))
            await started.wait()
            await signal.apply_call_action(CALL_ID,'caller','call_end')
            release.set()
            with self.assertRaises(HTTPException):
                await task
        self.record.assert_not_called()

    async def test_decline_removes_call(self):
        result=await signal.apply_call_action(CALL_ID,'callee','call_decline')
        self.assertEqual('call_declined',result['type'])
        self.assertNotIn(CALL_ID,signal._active_calls)

    async def test_snapshot_uses_original_deadline(self):
        item=await signal.native_call_snapshot(CALL_ID,'callee')
        self.assertEqual(int(signal._active_calls[CALL_ID]['created_at']+30),item['expires_at'])


class APNs(unittest.IsolatedAsyncioTestCase):
    async def test_each_app_uses_its_own_registered_voip_topic(self):
        self.devices.return_value = [dict(DEVICE), dict(DEVICE, bundle_id='com.verifyd.circle', token='cd'*32,
                                                      environment='production')]
        with patch.dict(os.environ, {'VERIFYD_NATIVE_CALLS_ALLOWED_BUNDLE_IDS': native.BUNDLE_ID + ',com.verifyd.circle'}):
            self.assertTrue(await self.dispatch())
        self.assertEqual(2, len(self.requests))
        by_host = {r.url.host: r.headers['apns-topic'] for r in self.requests}
        self.assertEqual('com.verifyd.videoauth.voip', by_host['api.sandbox.push.apple.com'])
        self.assertEqual('com.verifyd.circle.voip', by_host['api.push.apple.com'])

    async def test_unknown_stored_app_cannot_select_an_apns_topic(self):
        self.devices.return_value = [dict(DEVICE, bundle_id='evil.app')]
        self.assertFalse(await self.dispatch())
        self.assertFalse(self.requests)

    async def asyncSetUp(self):
        self.stack=ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.dict(os.environ,ENV))
        self.devices=self.stack.enter_context(patch.object(native,'active_devices',return_value=[dict(DEVICE)]))
        self.stack.enter_context(patch.object(native,'_provider_token',return_value='test-provider-token'))
        self.disabled=self.stack.enter_context(patch.object(native,'_disable_invalid_token'))
        self.requests=[]
        self.status=200
        self.reason='BadDeviceToken'
        original=httpx.AsyncClient
        def respond(request):
            self.requests.append(request)
            return httpx.Response(self.status,json={'reason':self.reason})
        self.stack.enter_context(patch.object(native.httpx,'AsyncClient',side_effect=lambda **kw: original(transport=httpx.MockTransport(respond),**kw)))

    async def dispatch(self, item=None, current=True):
        return await native.dispatch_invite(IDENTITY,item or call(),AsyncMock(return_value=current))

    async def test_sandbox_headers_payload_and_deadline(self):
        item=call()
        self.assertTrue(await self.dispatch(item))
        req=self.requests[0]
        self.assertEqual('api.sandbox.push.apple.com',req.url.host)
        self.assertEqual(native.BUNDLE_ID+'.voip',req.headers['apns-topic'])
        self.assertEqual('voip',req.headers['apns-push-type'])
        self.assertEqual(str(int(item['created_at']+30)),req.headers['apns-expiration'])
        data=json.loads(req.content)
        self.assertEqual({'aps','schema','kind','call_id','call_uuid','call_type','expires_at'},set(data))

    async def test_production_host(self):
        self.devices.return_value=[dict(DEVICE,environment='production')]
        self.assertTrue(await self.dispatch())
        self.assertEqual('api.push.apple.com',self.requests[0].url.host)

    async def test_default_off_no_request(self):
        with patch.dict(os.environ,{'VERIFYD_NATIVE_CALLS_ENABLED':'0'}):
            self.assertFalse(await self.dispatch())
        self.assertFalse(self.requests)

    async def test_expired_no_request(self):
        item=call();item['created_at']-=31
        self.assertFalse(await self.dispatch(item))
        self.assertFalse(self.requests)

    async def test_cancelled_no_request(self):
        self.assertFalse(await self.dispatch(current=False))
        self.assertFalse(self.requests)

    async def test_revoked_during_dispatch_no_request(self):
        self.devices.side_effect=[[dict(DEVICE)],[]]
        self.assertFalse(await self.dispatch())
        self.assertFalse(self.requests)

    async def test_invalid_token_disabled(self):
        self.status=410
        self.assertFalse(await self.dispatch())
        self.disabled.assert_called_once()

    async def test_bad_device_token_disabled(self):
        self.status=400
        self.assertFalse(await self.dispatch())
        self.disabled.assert_called_once()

    async def test_credentials_error_keeps_device(self):
        self.status=403
        self.reason='InvalidProviderToken'
        self.assertFalse(await self.dispatch())
        self.disabled.assert_not_called()

    async def test_timeout_no_token_in_logs(self):
        def failed(**kwargs):
            raise httpx.ConnectTimeout('private-token-in-exception')
        with patch.object(native.httpx,'AsyncClient',side_effect=failed), self.assertLogs(native.log,level='WARNING') as logs:
            self.assertFalse(await self.dispatch())
        self.assertNotIn('private-token-in-exception',' '.join(logs.output))


class ProviderSigning(unittest.TestCase):
    def test_es256_signature_verifies(self):
        key=ec.generate_private_key(ec.SECP256R1())
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'test.p8'
            path.write_bytes(key.private_bytes(serialization.Encoding.PEM,serialization.PrivateFormat.PKCS8,serialization.NoEncryption()))
            with patch.dict(os.environ,{**ENV,'VERIFYD_APNS_PRIVATE_KEY_FILE':str(path)}):
                token=native._provider_token()
        header,claims,signature=token.split('.')
        raw=base64.urlsafe_b64decode(signature+'==')
        key.public_key().verify(encode_dss_signature(int.from_bytes(raw[:32],'big'),int.from_bytes(raw[32:],'big')),
                                (header+'.'+claims).encode(),ec.ECDSA(hashes.SHA256()))
        self.assertEqual('3623QELM75',json.loads(base64.urlsafe_b64decode(claims+'=='))['iss'])



class SignalingDispatch(unittest.IsolatedAsyncioTestCase):
    async def run_invite(self, reachable=False, sent=False, during_dispatch=None, provider=native):
        from fastapi import WebSocketDisconnect
        signal._state_lock=asyncio.Lock()
        signal._active_calls.clear()
        signal._last_invite_at.clear()
        outbound=[]
        class Socket:
            async def accept(self): pass
            async def receive_json(self):
                if getattr(self,'received',False): raise WebSocketDisconnect()
                self.received=True
                return {'type':'call_invite','target_handle':'pilot','call_type':'audio'}
            async def send_json(self,payload): outbound.append(payload)
        caller={'id':'caller','handle':'caller','handle_lower':'caller'}
        async def dispatch(identity,item,current):
            self.assertTrue(await current())
            history.assert_called_once()
            if during_dispatch:
                await during_dispatch(item)
            return sent
        with ExitStack() as stack:
            stack.enter_context(patch.object(signal,'_enabled',return_value=True))
            stack.enter_context(patch.object(signal,'_verify_token',return_value=dict(SESSION,identity_id='caller')))
            stack.enter_context(patch.object(signal,'_identity_by_id',return_value=caller))
            stack.enter_context(patch.object(signal,'_identity_by_handle',return_value=IDENTITY))
            stack.enter_context(patch.object(signal,'_public_identity',side_effect=lambda i:i))
            stack.enter_context(patch.object(signal,'_register',new_callable=AsyncMock))
            stack.enter_context(patch.object(signal,'_unregister',new_callable=AsyncMock))
            stack.enter_context(patch.object(signal,'_presence',new_callable=AsyncMock,return_value=False))
            stack.enter_context(patch.object(signal,'_send_to_identity',new_callable=AsyncMock,return_value=0))
            stack.enter_context(patch.object(signal,'_incoming_call_push_pilot_enabled',return_value=False))
            stack.enter_context(patch.object(signal,'call_permission_decision',return_value=(True,'ok')))
            for candidate in (native, android):
                stack.enter_context(patch.object(candidate, 'reachable',
                                                return_value=reachable and candidate is provider))
            stack.enter_context(patch.object(native,'pilot_allowed',return_value=False))
            push=stack.enter_context(patch.object(provider,'dispatch_invite',side_effect=dispatch))
            history=stack.enter_context(patch.object(signal,'_create_call_history'))
            stack.enter_context(patch.object(signal,'_schedule_call_expiry'))
            stack.enter_context(patch.object(signal,'_cancel_call_expiry'))
            terminal=stack.enter_context(patch.object(signal,'_record_call_terminal'))
            await signal.signaling_ws(Socket(),token='test')
        return outbound,push,terminal

    async def test_offline_without_native_registration_unchanged(self):
        events,push,_=await self.run_invite()
        self.assertEqual('offline',events[-1]['reason'])
        push.assert_not_called()

    async def test_offline_native_device_can_ring(self):
        events,push,_=await self.run_invite(reachable=True,sent=True)
        self.assertEqual('call_ringing',events[-1]['type'])
        self.assertTrue(next(iter(signal._active_calls.values()))['native_incoming_push'])
        push.assert_awaited_once()

    async def test_native_push_failure_falls_back_to_offline(self):
        events,_,terminal=await self.run_invite(reachable=True,sent=False)
        self.assertEqual('offline',events[-1]['reason'])
        self.assertFalse(signal._active_calls)
        self.assertEqual(('failed','offline'),terminal.call_args.args[1:])

    async def test_offline_android_device_can_ring(self):
        events, push, _ = await self.run_invite(reachable=True, sent=True, provider=android)
        self.assertEqual('call_ringing', events[-1]['type'])
        self.assertTrue(next(iter(signal._active_calls.values()))['android_incoming_push'])
        push.assert_awaited_once()

    async def test_android_push_failure_falls_back_to_offline(self):
        events, _, terminal = await self.run_invite(reachable=True, provider=android)
        self.assertEqual('offline', events[-1]['reason'])
        self.assertFalse(signal._active_calls)
        self.assertEqual(('failed', 'offline'), terminal.call_args.args[1:])

    async def test_no_stale_ringing_after_cancellation_during_push(self):
        async def cancel(item): signal._active_calls.pop(item['call_id'])
        events,_,_=await self.run_invite(reachable=True,sent=True,during_dispatch=cancel)
        self.assertNotIn('call_ringing',[e['type'] for e in events])

if __name__ == '__main__':
    unittest.main()
