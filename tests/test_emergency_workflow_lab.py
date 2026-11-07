"""Offline, no network/PSAP: tests the VeriFYD owned simulation API."""
import importlib.util
import os
import sys
import types
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

HERE = Path(__file__).resolve().parents[1]

@pytest.fixture()
def ctx(monkeypatch):
    monkeypatch.setenv('VERIFYD_EMERGENCY_WORKFLOW_LAB_ENABLED', '1')
    monkeypatch.setenv('VERIFYD_EMERGENCY_WORKFLOW_LAB_IDS', 'alice')
    stub = types.ModuleType('trust_voice_messages')
    def verify(auth):
        if auth == 'Bearer alice': return {'id':'alice'}
        if auth == 'Bearer bob': return {'id':'bob'}
        from fastapi import HTTPException
        raise HTTPException(status_code=401, detail='invalid_session')
    stub._identity_from_bearer = verify
    monkeypatch.setitem(sys.modules, 'trust_voice_messages', stub)
    name='emergency_workflow_lab_test'
    spec = importlib.util.spec_from_file_location(name,HERE/'verifyd_emergency_workflow_lab.py')
    mod=importlib.util.module_from_spec(spec)
    sys.modules[name]=mod
    spec.loader.exec_module(mod)
    app=FastAPI()
    app.include_router(mod.router)
    return TestClient(app),mod

BASE='/trust-voice/emergency-workflow-lab'
HEAD={'Authorization':'Bearer alice'}


def make(client,**overrides):
    body={'scenario':'medical','confirm_training_only':True}
    body.update(overrides)
    return client.post(BASE+'/incidents',json=body,headers=HEAD)


def test_default_disabled(ctx,monkeypatch):
    client,_=ctx
    monkeypatch.delenv('VERIFYD_EMERGENCY_WORKFLOW_LAB_ENABLED')
    assert client.get(BASE+'/status',headers=HEAD).status_code==404
    assert make(client).status_code==404


def test_empty_allowlist_fails_closed(ctx,monkeypatch):
    client,_=ctx
    monkeypatch.delenv('VERIFYD_EMERGENCY_WORKFLOW_LAB_IDS')
    assert client.get(BASE+'/status',headers=HEAD).status_code==404


def test_unauthenticated_rejected(ctx):
    client,_=ctx
    assert client.get(BASE+'/status').status_code==401
    assert client.post(BASE+'/incidents',json={'scenario':'medical','confirm_training_only':True}).status_code==401


def test_nonmember_denied(ctx):
    client,_=ctx
    assert client.get(BASE+'/status',headers={'Authorization':'Bearer bob'}).status_code==404


def test_status_never_claims_connection(ctx):
    client,_=ctx
    d=client.get(BASE+'/status',headers=HEAD).json()
    assert d['mode']=='TRAINING_SIMULATION_ONLY'
    assert d['service_available'] is False
    assert d['dispatch_connected'] is False
    assert d['real_911_call_placed'] is False
    assert d['real_911_text_sent'] is False
    assert d['emergency_provider_connected'] is False


def test_confirmation_required(ctx):
    client,_=ctx
    assert client.post(BASE+'/incidents',json={'scenario':'medical'},headers=HEAD).status_code==422
    assert client.post(BASE+'/incidents',json={'scenario':'medical','confirm_training_only':False},headers=HEAD).status_code==422


def test_extra_destination_rejected(ctx):
    client,_=ctx
    for name in ('destination','phone','address','location','psap','provider','recipient','real_911_call'):
        assert make(client,**{name:'911'}).status_code==422


def test_arbitrary_scenario_rejected(ctx):
    client,_=ctx
    assert make(client,scenario='911').status_code==422


def test_create_returns_simulation_only(ctx):
    client,_=ctx
    r=make(client)
    assert r.status_code==201
    d=r.json()
    assert d['service_type']=='TRAINING_SIMULATION_ONLY'
    assert d['dispatch_connected'] is False and d['psap_acknowledged'] is False
    assert d['public_emergency_message_delivered'] is False
    assert d['location']=='SIMULATED_LOCATION_ONLY'
    assert d['status']=='simulation_open'


def test_duplicate_active_rejected(ctx):
    client,_=ctx
    assert make(client).status_code==201
    assert make(client).status_code==409


def test_message_returns_obvious_fake_reply(ctx):
    client,_=ctx
    i=make(client).json()['id']
    r=client.post(f'{BASE}/incidents/{i}/messages',json={'message':'TEST ONLY: practice'},headers=HEAD)
    assert r.status_code==200
    d=r.json()
    assert d['status']=='simulation_acknowledged'
    assert any(x['sender']=='simulated_responder' and 'NOT 911' in x['message'] for x in d['messages'])
    assert not d['dispatch_connected']


def test_second_message_does_not_fake_new_ack(ctx):
    client,_=ctx
    i=make(client).json()['id']
    for _ in range(2):
        assert client.post(f'{BASE}/incidents/{i}/messages',json={'message':'TEST ONLY: practice'},headers=HEAD).status_code==200
    d=client.get(f'{BASE}/incidents/{i}',headers=HEAD).json()
    assert sum(x['sender']=='simulated_responder' for x in d['messages'])==1


def test_identity_isolation(ctx):
    client,mod=ctx
    i=make(client).json()['id']
    mod._simulations[i]['owner']='bob'
    for op in [client.get(f'{BASE}/incidents/{i}',headers=HEAD),client.post(f'{BASE}/incidents/{i}/close',headers=HEAD),client.post(f'{BASE}/incidents/{i}/messages',json={'message':'Hi'},headers=HEAD)]:
        assert op.status_code==404


def test_unknown_incident(ctx):
    client,_=ctx
    assert client.get(f'{BASE}/incidents/not-an-id',headers=HEAD).status_code==404
    assert client.get(f'{BASE}/incidents/{"a"*32}',headers=HEAD).status_code==404


def test_close_and_reject_later_messages(ctx):
    client,_=ctx
    i=make(client).json()['id']
    assert client.post(f'{BASE}/incidents/{i}/close',headers=HEAD).json()['status']=='simulation_closed'
    assert client.post(f'{BASE}/incidents/{i}/messages',json={'message':'Hi'},headers=HEAD).status_code==409
    assert make(client,scenario='fire').status_code==201


def test_message_extra_fields_denied(ctx):
    client,_=ctx
    i=make(client).json()['id']
    assert client.post(f'{BASE}/incidents/{i}/messages',json={'message':'hi','to':'911'},headers=HEAD).status_code==422


def test_message_length_limits(ctx):
    client,_=ctx
    i=make(client).json()['id']
    for s in ['', 'x'*401]:
        assert client.post(f'{BASE}/incidents/{i}/messages',json={'message':s},headers=HEAD).status_code==422
    assert client.post(f'{BASE}/incidents/{i}/messages',json={'message':'   '},headers=HEAD).status_code==422


def test_rate_limit_incidents(ctx):
    client,_=ctx
    for _ in range(8):
        i=make(client).json()['id']
        assert client.post(f'{BASE}/incidents/{i}/close',headers=HEAD).status_code==200
    assert make(client).status_code==429


def test_rate_limit_messages(ctx):
    client,_=ctx
    i=make(client).json()['id']
    statuses=[]
    for _ in range(24):
        statuses.append(client.post(f'{BASE}/incidents/{i}/messages',json={'message':'test'},headers=HEAD).status_code)
    assert 429 in statuses


def test_expiration_purges(ctx):
    client,mod=ctx
    i=make(client).json()['id']
    mod._simulations[i]['_created_ts']-=1801
    assert client.get(f'{BASE}/incidents/{i}',headers=HEAD).status_code==404


def test_no_egress_or_911_dial_capability():
    text=(HERE/'verifyd_emergency_workflow_lab.py').read_text()
    for banned in ('urlopen(', 'requests.post(', 'httpx.', 'smtplib', 'tel:911', 'CreateSIPParticipant', 'rapidsos.com', 'send_sms(', 'send_to_identity(', 'dial_911('):
        assert banned not in text


def test_no_production_default():
    s=(HERE/'verifyd_emergency_workflow_lab.py').read_text()
    assert 'VERIFYD_EMERGENCY_WORKFLOW_LAB_ENABLED' in s
    assert 'VERIFYD_EMERGENCY_WORKFLOW_LAB_IDS' in s
    assert 'TRAINING_SIMULATION_ONLY' in s


def test_current_training_resume(ctx):
    client,_=ctx
    assert client.get(BASE+'/incidents/current',headers=HEAD).json()['incident'] is None
    i=make(client).json()['id']
    current=client.get(BASE+'/incidents/current',headers=HEAD)
    assert current.status_code==200
    d=current.json()
    assert d['incident']['id']==i
    assert d['dispatch_connected'] is False
    client.post(f'{BASE}/incidents/{i}/close',headers=HEAD)
    assert client.get(BASE+'/incidents/current',headers=HEAD).json()['incident'] is None