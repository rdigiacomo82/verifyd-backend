"""Safety regressions for opt-in adult Therapy & Adaptive Practice (mocked database)."""
import importlib.util
import os
import pathlib
import sys
import types
from contextlib import contextmanager
from unittest.mock import patch
import unittest

from fastapi import HTTPException

MODULE = pathlib.Path(__file__).resolve().parents[1] / 'trust_voice_therapy_practice.py'


def load():
    database = types.ModuleType('database')
    database.get_db = lambda: (_ for _ in ()).throw(AssertionError('DB should not be read'))
    messages = types.ModuleType('trust_voice_messages')
    messages._identity_from_bearer = lambda header: {'status':'active','id':header.removeprefix('Bearer ') if header else 'client'}
    permissions = types.ModuleType('trust_voice_permissions')
    permissions.ensure_trust_circle_schema=lambda:None
    blocking = types.ModuleType('trust_voice_blocking')
    blocking.identities_blocked=lambda a,b:False
    name='isolated_therapy_'+str(id(database))
    spec=importlib.util.spec_from_file_location(name,MODULE)
    module=importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules,{'database':database,'trust_voice_messages':messages,
                                 'trust_voice_permissions':permissions,'trust_voice_blocking':blocking,name:module}):
        spec.loader.exec_module(module)
    return module


class Cursor:
    def __init__(self,returns):self.answers=iter(returns);self.executed=[];self.current=None
    def execute(self,query,args=None):self.executed.append((query,args));self.current=next(self.answers)
    def fetchone(self):return self.current
    def fetchall(self):return self.current


class FakeDatabase:
    def __init__(self,answers):self.cursor_data=Cursor(answers)
    def cursor(self):return self.cursor_data
    def __enter__(self):return self
    def __exit__(self,*args):return False


class TherapyAccessTests(unittest.TestCase):
    def test_default_disabled_without_database(self):
        with patch.dict(os.environ,{},clear=True):
            m=load()
            with self.assertRaises(HTTPException) as err:m.health('Bearer pro')
            self.assertEqual(err.exception.status_code,404)

    def test_professional_allowlist_is_fail_closed(self):
        with patch.dict(os.environ,{'VERIFYD_CIRCLE_THERAPY_PRACTICE_ENABLED':'1'},clear=True):
            m=load()
            with self.assertRaises(HTTPException) as err:m.professional_contacts('Bearer pro')
            self.assertEqual(err.exception.status_code,403)
            self.assertFalse(m._is_professional('pro'))

    def test_manual_allowlist_is_specific(self):
        with patch.dict(os.environ,{'VERIFYD_CIRCLE_THERAPY_PRACTICE_ENABLED':'1',
                                    'VERIFYD_CIRCLE_THERAPY_PROFESSIONAL_IDS':'pro,other'},clear=True):
            m=load()
            self.assertTrue(m._is_professional('pro'))
            self.assertFalse(m._is_professional('patient'))

    def test_pilot_invitation_requires_adult_confirmation(self):
        with patch.dict(os.environ,{'VERIFYD_CIRCLE_THERAPY_PRACTICE_ENABLED':'1'},clear=True):
            m=load()
            with self.assertRaises(HTTPException) as err:m.accept('invite',m.AcceptInvite(adult_attestation=False),'Bearer patient')
            self.assertEqual(err.exception.status_code,422)

    def test_invalid_duplicates_and_self_invite_prevented(self):
        with patch.dict(os.environ,{'VERIFYD_CIRCLE_THERAPY_PRACTICE_ENABLED':'1',
                                    'VERIFYD_CIRCLE_THERAPY_PROFESSIONAL_IDS':'pro'},clear=True):
            m=load();m._schema=lambda:None
            for ids in (['same','same'],['pro']):
                with self.assertRaises(HTTPException) as err:m.invite('group',m.InviteClients(client_ids=ids),'Bearer pro')
                self.assertEqual(err.exception.status_code,422)

    def test_selected_recipients_required_and_invalid_studio_ids_fail_closed(self):
        with patch.dict(os.environ,{'VERIFYD_CIRCLE_THERAPY_PRACTICE_ENABLED':'1',
                                    'VERIFYD_CIRCLE_THERAPY_PROFESSIONAL_IDS':'pro'},clear=True):
            m=load();m._schema=lambda:None
            payload=m.PracticeTaskCreate(title='Speech practice',instructions='Choose a prompt',recipient_ids=['a','a'])
            with self.assertRaises(HTTPException) as err:m.create_task('group',payload,'Bearer pro')
            self.assertEqual(err.exception.status_code,422)
            payload=m.PracticeTaskCreate(title='Speech practice',instructions='Choose a prompt',recipient_ids=['a'],activities=['nonsense'])
            with self.assertRaises(HTTPException) as err:m.create_task('group',payload,'Bearer pro')
            self.assertEqual(err.exception.status_code,422)

    def test_removed_or_nonmember_cannot_open_practice(self):
        with patch.dict(os.environ,{'VERIFYD_CIRCLE_THERAPY_PRACTICE_ENABLED':'1',
                                    'VERIFYD_CIRCLE_THERAPY_PROFESSIONAL_IDS':'pro'},clear=True):
            m=load()
            for status in ('removed',None):
                cursor=Cursor([{'program_id':'a','professional_id':'pro','status':status}])
                with self.assertRaises(HTTPException) as err:m._membership(cursor,'a','client')
                self.assertEqual(err.exception.status_code,404)

    def test_revoked_professional_blocks_client_access(self):
        with patch.dict(os.environ,{'VERIFYD_CIRCLE_THERAPY_PRACTICE_ENABLED':'1'},clear=True):
            m=load()
            cur=Cursor([{'program_id':'a','professional_id':'revoked','status':'active'}])
            with self.assertRaises(HTTPException) as err:m._membership(cur,'a','client')
            self.assertEqual(err.exception.status_code,404)

    def test_client_cannot_submit_before_assignment(self):
        with patch.dict(os.environ,{'VERIFYD_CIRCLE_THERAPY_PRACTICE_ENABLED':'1'},clear=True):
            m=load();m._schema=lambda:None
            m._membership=lambda *args: {'professional_id':'pro'}
            m._visible_task=lambda *args: (_ for _ in ()).throw(HTTPException(404,detail={'error':'practice_task_unavailable'}))
            m.get_db=lambda:FakeDatabase([])
            with self.assertRaises(HTTPException) as err:m.save_response('group','task',m.PracticeResponse(response='I completed it',action='submit'),'Bearer client')
            self.assertEqual(err.exception.status_code,404)

    def test_locked_submission_cannot_be_changed(self):
        with patch.dict(os.environ,{'VERIFYD_CIRCLE_THERAPY_PRACTICE_ENABLED':'1'},clear=True):
            m=load();m._schema=lambda:None
            m._membership=lambda *args: {'professional_id':'pro'}
            m._visible_task=lambda *args: {'task_id':'a'}
            fake=FakeDatabase([None, {'status':'submitted'}]);m.get_db=lambda:fake
            with self.assertRaises(HTTPException) as err:m.save_response('group','a',m.PracticeResponse(response='Changed',action='save'),'Bearer client')
            self.assertEqual(err.exception.status_code,409)
            self.assertEqual(len(fake.cursor_data.executed),2)

    def test_review_is_professional_only(self):
        with patch.dict(os.environ,{'VERIFYD_CIRCLE_THERAPY_PRACTICE_ENABLED':'1'},clear=True):
            m=load();m._schema=lambda:None
            m.get_db=lambda:FakeDatabase([])
            with self.assertRaises(HTTPException) as err:m.review('group','task','client',m.PracticeReview(feedback='Great effort'),'Bearer nonpro')
            self.assertEqual(err.exception.status_code,403)


if __name__=='__main__':unittest.main()