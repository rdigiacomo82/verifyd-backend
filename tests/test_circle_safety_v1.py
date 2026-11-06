"""Isolated Circle Safety route tests: no live database, internet or services."""
import importlib.util
import os
import sys
import types
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch, Mock

from fastapi import HTTPException
from pydantic import ValidationError

MOD = Path(__file__).resolve().parents[1] / 'trust_voice_circle_safety.py'

# Load source independently of production Postgres/Redis dependencies.
for name, attr in (
    ('database', {'get_db': lambda: None}),
    ('trust_voice_blocking', {'identities_blocked': lambda a, b: False}),
    ('trust_voice_permissions', {'trust_circle_contains': lambda a, b: True}),
    ('trust_voice_signaling', {'_identity_by_id': lambda who: {'id': who, 'handle': who, 'status': 'active'}}),
    ('trust_voice_messages', {'_identity_from_bearer': lambda auth: {'id': 'sender', 'handle': 'sender', 'status': 'active'}}),
):
    fake = types.ModuleType(name)
    for key, value in attr.items(): setattr(fake, key, value)
    sys.modules.setdefault(name, fake)

spec = importlib.util.spec_from_file_location('trust_voice_circle_safety', MOD)
safety = importlib.util.module_from_spec(spec)
sys.modules['trust_voice_circle_safety'] = safety
spec.loader.exec_module(safety)


class FakeConnection:
    def __init__(self, cursor): self.c = cursor; self.committed = False
    def __enter__(self): return self
    def __exit__(self, exc_type, exc_value, traceback):
        self.committed = exc_type is None
        return False
    def cursor(self): return self.c


class StopCursor:
    def __init__(self):
        self.queries = []
        self.record = {
            'incident_id': 'incident1', 'sender_id': 'sender', 'status': 'active',
            'expires_at': datetime.now(timezone.utc) + timedelta(minutes=20),
            'pin_fail_count': 0, 'pin_locked_until': None,
        }
    def execute(self, sql, params=()): self.queries.append((sql, params))
    def fetchone(self): return self.record
    def fetchall(self): return []


class CircleSafetyUnitTests(unittest.TestCase):
    def test_feature_closed_by_default(self):
        with patch.dict(os.environ, {'VERIFYD_CIRCLE_SAFETY_ENABLED': '0'}):
            with self.assertRaises(HTTPException) as c:
                safety.safety_settings(authorization='Bearer fake')
            self.assertEqual(c.exception.status_code, 404)
            self.assertEqual(safety.run_safety_dispatch_once(), 0)

    def test_geodesic_distance_and_coordinate_validation(self):
        self.assertLess(safety._distance_m(40.7, -73.9, 40.7, -73.9), 1)
        self.assertTrue(110000 < safety._distance_m(0, 0, 1, 0) < 112000)
        with self.assertRaises(HTTPException): safety._geo(float('nan'), -73)
        with self.assertRaises(HTTPException): safety._geo(91, 0)

    def test_pin_is_salted_and_not_stored_in_plaintext(self):
        salt1, salt2 = 'ab' * 16, 'cd' * 16
        dig1 = safety._hash_pin('583926', salt1)
        dig2 = safety._hash_pin('583926', salt2)
        self.assertNotEqual(dig1, dig2)
        self.assertNotIn('583926', dig1)
        self.assertTrue(safety._check_pin('583926', {'pin_salt':salt1, 'pin_digest':dig1}))
        self.assertFalse(safety._check_pin('123456', {'pin_salt':salt1, 'pin_digest':dig1}))
        with self.assertRaises(ValidationError): safety.PinChange(pin='123')
        with self.assertRaises(ValidationError): safety.PinChange(pin='AB1234')

    def test_eligible_range_contacts_only_returns_consented_members(self):
        location_module = types.ModuleType('trust_voice_location')
        location_module._enabled = lambda: True
        location_module._schema = lambda: None

        class EligibleCursor:
            def execute(self, sql, params=()): pass
            def fetchall(self): return [
                {'member_identity_id': 'approved'},
                {'member_identity_id': 'not_approved'},
            ]

        with patch.dict(sys.modules, {'trust_voice_location': location_module}), \
             patch.object(safety, '_owner', return_value={'id': 'watcher'}), \
             patch.object(safety, '_schema'), \
             patch.object(safety, 'get_db', return_value=FakeConnection(EligibleCursor())), \
             patch.object(safety, '_identity_by_id', side_effect=lambda ident: {'id': ident, 'handle': ident, 'status': 'active'}), \
             patch.object(safety, '_location_granted', side_effect=lambda ident, viewer: ident == 'approved'):
            eligible = safety.eligible_range_contacts(authorization='Bearer fake')
        self.assertEqual([m['identity_id'] for m in eligible['eligible']], ['approved'])
        self.assertTrue(eligible['location_enabled'])
        self.assertEqual(eligible['eligible'][0]['precision'], 'precise')

    def test_eligible_list_requires_existing_location_feature(self):
        location_module = types.ModuleType('trust_voice_location')
        location_module._enabled = lambda: False
        location_module._schema = Mock()
        with patch.dict(sys.modules, {'trust_voice_location': location_module}), \
             patch.object(safety, '_owner', return_value={'id': 'watcher'}), \
             patch.object(safety, '_schema'):
            result = safety.eligible_range_contacts(authorization='Bearer fake')
        self.assertEqual(result, {'eligible': [], 'location_enabled': False})
        location_module._schema.assert_not_called()

    def test_invalid_zone_cannot_monitor_without_separate_permission(self):
        with patch.object(safety, '_owner', return_value={'id': 'watcher'}), \
             patch.object(safety, '_schema'), \
             patch.object(safety, '_location_granted', return_value=False):
            with self.assertRaises(HTTPException) as c:
                safety.set_zone('target', safety.ZoneUpdate(center_lat=40,center_lon=-73,radius_m=500), None)
            self.assertEqual(c.exception.status_code, 403)

    def test_distress_cannot_be_started_without_pin(self):
        with patch.object(safety, '_owner', return_value={'id': 'sender'}), \
             patch.object(safety, '_schema'), patch.object(safety, '_settings', return_value={}):
            with self.assertRaises(HTTPException) as c:
                safety.start_distress(safety.DistressStart(latitude=None,longitude=None), None)
            self.assertEqual(c.exception.status_code, 409)

    def test_failed_stop_pin_attempt_commits_lockout_counter(self):
        cursor = StopCursor(); connection = FakeConnection(cursor)
        with patch.object(safety, '_owner', return_value={'id': 'sender'}), \
             patch.object(safety, '_schema'), \
             patch.object(safety, '_settings', return_value={'pin_salt':'ab'*16,'pin_digest':'bad'}), \
             patch.object(safety, 'get_db', return_value=connection):
            with self.assertRaises(HTTPException) as c:
                safety.stop_distress('incident1', safety.DistressStop(pin='123456'), None)
            self.assertEqual(c.exception.status_code, 403)
            self.assertTrue(connection.committed, 'Failed PIN must be recorded before 403, not rolled back')
            self.assertTrue(any('pin_fail_count=%s' in sql for sql,_ in cursor.queries))

    def test_queued_distress_push_dropped_after_stop_or_mute(self):
        class Cursor:
            def execute(self, sql, params=()): self.query = sql
            def fetchone(self): return None  # stopped or muted recipient

        with patch.object(safety, '_enabled', return_value=True), \
             patch.object(safety, '_schema'), \
             patch.object(safety, 'get_db', return_value=FakeConnection(Cursor())):
            self.assertFalse(safety._safety_push_authorized('recipient', 'circle_distress', 'incident'))
            self.assertFalse(safety._safety_push_authorized('recipient', 'circle_distress_clear', 'incident'))

    def test_queued_range_push_dropped_after_permission_revoked(self):
        class Cursor:
            def execute(self, sql, params=()): pass
            def fetchone(self): return {'actor_id': 'target'}

        with patch.object(safety, '_enabled', return_value=True), \
             patch.object(safety, '_schema'), \
             patch.object(safety, 'get_db', return_value=FakeConnection(Cursor())), \
             patch.object(safety, '_location_granted', return_value=False):
            self.assertFalse(safety._safety_push_authorized('watcher', 'circle_range_exit', 'alert'))

    def test_safety_push_does_not_block_location_and_call_threads(self):
        with patch.object(safety._PUSH_SLOTS, 'acquire', return_value=True), \
             patch.object(safety._PUSH_POOL, 'submit') as submit:
            result = safety._push('recipient', kind='circle_distress',
                                  title='Alert', body='Test', event_id='123')
        self.assertTrue(result)
        submit.assert_called_once()

    def test_safety_push_is_bounded_when_capacity_exhausted(self):
        with patch.object(safety._PUSH_SLOTS, 'acquire', return_value=False), \
             patch.object(safety._PUSH_POOL, 'submit') as submit:
            result = safety._push('recipient', kind='circle_range_exit',
                                  title='Alert', body='Test', event_id='456')
        self.assertFalse(result)
        submit.assert_not_called()

    def test_distress_has_hard_maximum_one_hour_and_recipient_mute(self):
        source = MOD.read_text()
        self.assertIn("NOW()+INTERVAL '1 hour'", source)
        self.assertIn("i.expires_at>NOW()", source)
        self.assertIn("SET muted=TRUE,muted_at=NOW()", source)
        self.assertIn("next_push_at=NOW()+INTERVAL '5 minutes'", source)
        self.assertIn("idx_verifyd_safety_one_active_incident", source)
        self.assertIn("status='expired',ended_at=NOW(),latitude=NULL", source)


if __name__ == '__main__': unittest.main()