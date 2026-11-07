"""Offline-only tests of E911 enrollment staging; no dial/provider connections."""
from __future__ import annotations
import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest
from cryptography.fernet import Fernet
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

PATH = Path(__file__).resolve().parents[1] / 'trust_voice_emergency_enrollment.py'
spec = importlib.util.spec_from_file_location('e911_staging_test_target', PATH)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


class MemoryDatabase:
    def __init__(self):
        self.rows = {}
        self.calls = []
        self.pending = None

    def __enter__(self): return self
    def __exit__(self, *a): return False
    def cursor(self): return self
    def execute(self, query, args=None):
        self.calls.append((query, args))
        if 'SELECT mobile_last4' in query:
            self.pending = ({'mobile_last4': self.rows[args[0]][1]} if args[0] in self.rows else None)
        elif 'INSERT INTO verifyd_e911_enrollment_staging' in query:
            self.rows[args[0]] = (args[1], args[2])
        elif 'DELETE FROM verifyd_e911_enrollment_staging' in query:
            self.rows.pop(args[0], None)

    def fetchone(self): return self.pending


@pytest.fixture()
def app(monkeypatch):
    module._SCHEMA_READY = False
    db = MemoryDatabase()
    monkeypatch.setattr(module, '_db', lambda: db)
    def authenticate(header):
        if header != 'Bearer valid-circle-test-session':
            raise HTTPException(status_code=401, detail={'error':'invalid_session'})
        return 'test-id-opaque-1'
    monkeypatch.setattr(module, '_identity', authenticate)
    monkeypatch.delenv('VERIFYD_E911_ENROLLMENT_STAGING_ENABLED', raising=False)
    monkeypatch.delenv('VERIFYD_E911_PROFILE_FERNET_KEY', raising=False)
    api = FastAPI()
    api.include_router(module.router)
    return TestClient(api), db


HEADER = {'Authorization': 'Bearer valid-circle-test-session'}
DATA = dict(address1='100 Example Street', address2='Apt 2', city='Albany', state='NY', zip='12207',
            mobile='(518) 555-0123', consent_to_store=True, acknowledges_not_911=True)


def test_no_auth_returns_401(app):
    api, _ = app
    assert api.get('/trust-voice/emergency-services/status').status_code == 401
    assert api.post('/trust-voice/emergency-services/enrollment', json=DATA).status_code == 401
    assert api.post('/trust-voice/emergency-services/calls/911').status_code == 401


def test_disabled_by_default_never_touches_database(app):
    api, db = app
    data = api.get('/trust-voice/emergency-services/status', headers=HEADER).json()
    assert data['can_place_911_call'] is False
    assert data['enrollment_available'] is False
    assert data['provider_connected'] is False
    assert data['enrollment_status'] == 'not_available'
    assert api.post('/trust-voice/emergency-services/enrollment', headers=HEADER, json=DATA).status_code == 503
    assert db.calls == []


def test_911_fails_closed_even_when_enrollment_enabled(app, monkeypatch):
    api, _ = app
    monkeypatch.setenv('VERIFYD_E911_ENROLLMENT_STAGING_ENABLED', '1')
    monkeypatch.setenv('VERIFYD_E911_PROFILE_FERNET_KEY', Fernet.generate_key().decode())
    r = api.post('/trust-voice/emergency-services/calls/911', headers=HEADER)
    assert r.status_code == 503
    assert r.json()['detail']['error'] == 'e911_not_implemented'
    assert api.post('/trust-voice/emergency-services/mobile/request-code', headers=HEADER).status_code == 503


def test_no_encryption_key_refuses_collection(app, monkeypatch):
    api, db = app
    monkeypatch.setenv('VERIFYD_E911_ENROLLMENT_STAGING_ENABLED', '1')
    r = api.post('/trust-voice/emergency-services/enrollment', headers=HEADER, json=DATA)
    assert r.status_code == 503
    assert r.json()['detail']['error'] == 'encrypted_storage_unavailable'
    assert not db.rows


def test_bad_crypto_key_refuses_collection(app, monkeypatch):
    api, db = app
    monkeypatch.setenv('VERIFYD_E911_ENROLLMENT_STAGING_ENABLED', '1')
    monkeypatch.setenv('VERIFYD_E911_PROFILE_FERNET_KEY','not-a-valid-key')
    r = api.post('/trust-voice/emergency-services/enrollment', headers=HEADER, json=DATA)
    assert r.status_code == 503
    assert not db.rows


def test_enrollment_is_encrypted_pending_and_deletable(app, monkeypatch):
    api, db = app
    key = Fernet.generate_key()
    monkeypatch.setenv('VERIFYD_E911_ENROLLMENT_STAGING_ENABLED', '1')
    monkeypatch.setenv('VERIFYD_E911_PROFILE_FERNET_KEY', key.decode())
    before=api.get('/trust-voice/emergency-services/status', headers=HEADER).json()
    assert before['enrollment_status'] == 'not_registered'
    response = api.post('/trust-voice/emergency-services/enrollment', headers=HEADER, json=DATA)
    assert response.status_code == 202, response.text
    body=response.json()
    assert body['enrollment_status'] == 'verification_pending'
    assert body['can_place_911_call'] is False
    enc, last4 = db.rows['test-id-opaque-1']
    assert b'Example Street' not in enc and b'555-0123' not in enc
    saved = json.loads(Fernet(key).decrypt(enc).decode())
    assert saved['address1'] == '100 Example Street'
    assert saved['mobile'] == '+15185550123'
    assert last4 == '0123'
    status = api.get('/trust-voice/emergency-services/status', headers=HEADER).json()
    assert status['mobile_last4'] == '0123'
    assert 'Example Street' not in json.dumps(status)
    assert api.delete('/trust-voice/emergency-services/enrollment', headers=HEADER).status_code == 200
    assert db.rows == {}


@pytest.mark.parametrize('change', [
    {'state':'XX'}, {'zip':'1234'}, {'mobile':'000'}, {'mobile':'+1518555123'},
    {'consent_to_store':False}, {'acknowledges_not_911':False},
    {'address1':'x'}, {'address2':'x'*130}, {'city':'Z'},
    {'state':'New York'}, {'zip':'abcde'},
])
def test_invalid_form_rejected(app, monkeypatch, change):
    api, db=app
    monkeypatch.setenv('VERIFYD_E911_ENROLLMENT_STAGING_ENABLED','1')
    monkeypatch.setenv('VERIFYD_E911_PROFILE_FERNET_KEY',Fernet.generate_key().decode())
    r=api.post('/trust-voice/emergency-services/enrollment',headers=HEADER,json={**DATA,**change})
    assert r.status_code == 422
    assert db.rows == {}


def test_additional_fields_rejected(app):
    api, db = app
    r=api.post('/trust-voice/emergency-services/enrollment',headers=HEADER,json={**DATA, 'dial_number':'911'})
    assert r.status_code == 422
    assert not db.rows


def test_submitting_different_address_still_never_enables_911(app,monkeypatch):
    api, db=app
    monkeypatch.setenv('VERIFYD_E911_ENROLLMENT_STAGING_ENABLED','1')
    monkeypatch.setenv('VERIFYD_E911_PROFILE_FERNET_KEY',Fernet.generate_key().decode())
    for address in ('100 Example Street','101 Example Street'):
        resp=api.post('/trust-voice/emergency-services/enrollment',headers=HEADER,json={**DATA,'address1':address})
        assert resp.status_code==202
        assert resp.json()['can_place_911_call'] is False
    assert api.get('/trust-voice/emergency-services/status',headers=HEADER).json()['can_place_911_call'] is False