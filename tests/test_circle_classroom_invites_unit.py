"""Fail-closed invitation and per-assignment audience tests; no live API or database."""
import importlib.util
import os
import pathlib
import sys
import types
import unittest
from contextlib import contextmanager
from unittest.mock import patch
from fastapi import HTTPException

MODULE_PATH = pathlib.Path(__file__).resolve().parents[1] / "trust_voice_classroom_invitations.py"


def load_module():
    database = types.ModuleType('database')
    def never_connect():
        raise AssertionError('Unexpected database query')
    database.get_db = never_connect
    classroom=types.ModuleType('trust_voice_classrooms')
    classroom._identity=lambda header: 'teacher_one'
    classroom._schema=lambda: None
    classroom._teacher=lambda *a: None
    classroom._approved_teacher=lambda *a: None
    classroom._teacher_allowed=lambda *a: True
    permissions=types.ModuleType('trust_voice_permissions')
    permissions.ensure_trust_circle_schema=lambda: None
    blocks=types.ModuleType('trust_voice_blocking')
    blocks.identities_blocked=lambda *a: False
    spec=importlib.util.spec_from_file_location('test_isolated_invites_'+str(id(database)), MODULE_PATH)
    module=importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {"database":database,"trust_voice_classrooms":classroom,
                                  "trust_voice_permissions":permissions,"trust_voice_blocking":blocks,
                                  spec.name:module}):
        spec.loader.exec_module(module)
        module.AudienceSelection.model_rebuild()
    return module


class FakeCursor:
    def __init__(self, answers):
        self.answers=iter(answers)
        self.queries=[]
        self.current=None
    def execute(self, query, params=None):
        self.queries.append((query,params))
        self.current=next(self.answers)
    def fetchone(self): return self.current


class InvitationSecurityTests(unittest.TestCase):
    def test_invitation_defaults_off_and_no_query(self):
        with patch.dict(os.environ,{},clear=True):
            m=load_module()
            with self.assertRaises(HTTPException) as err:
                m.incoming_class_invitations('Bearer a')
            self.assertEqual(err.exception.status_code,404)

    def test_duplicate_recipients_denied_before_db(self):
        with patch.dict(os.environ,{"VERIFYD_CIRCLE_CLASSROOM_INVITES_ENABLED":"1"},clear=True):
            m=load_module(); m._schema=lambda: None
            with self.assertRaises(HTTPException) as err:
                m.send_class_invitations('class1',m.InviteSelection(student_ids=['one','one']),'Bearer a')
            self.assertEqual(err.exception.status_code,422)

    def test_accept_requires_adult_attestation(self):
        with patch.dict(os.environ,{"VERIFYD_CIRCLE_CLASSROOM_INVITES_ENABLED":"1"},clear=True):
            m=load_module()
            with self.assertRaises(HTTPException) as err:
                m.accept_class_invitation('invite1',m.AcceptInvitation(adult_attestation=False),'Bearer a')
            self.assertEqual(err.exception.status_code,422)

    def test_audience_disabled_without_both_flags(self):
        with patch.dict(os.environ,{"VERIFYD_CIRCLE_CLASSROOM_AUDIENCE_ENABLED":"1"},clear=True):
            m=load_module()
            self.assertFalse(m.audience_enabled())
            self.assertTrue(m.audience_allows(FakeCursor([]),'a','s'))

    def test_audience_all_by_default(self):
        with patch.dict(os.environ,{"VERIFYD_CIRCLE_CLASSROOM_INVITES_ENABLED":"1", "VERIFYD_CIRCLE_CLASSROOM_AUDIENCE_ENABLED":"1"},clear=True):
            m=load_module()
            self.assertTrue(m.audience_allows(FakeCursor([None]),'a','s'))
            self.assertTrue(m.audience_allows(FakeCursor([{'mode':'all'}]),'a','s'))

    def test_selected_audience_enforced_by_identity(self):
        with patch.dict(os.environ,{"VERIFYD_CIRCLE_CLASSROOM_INVITES_ENABLED":"1", "VERIFYD_CIRCLE_CLASSROOM_AUDIENCE_ENABLED":"1"},clear=True):
            m=load_module()
            self.assertFalse(m.audience_allows(FakeCursor([{'mode':'selected'},None]),'a','not_enrolled'))
            cur=FakeCursor([{'mode':'selected'},{'?':1}])
            self.assertTrue(m.audience_allows(cur,'a','student_two'))
            self.assertEqual(cur.queries[1][1],('a','student_two'))

    def test_missing_selected_recipients_rejected(self):
        with patch.dict(os.environ,{"VERIFYD_CIRCLE_CLASSROOM_INVITES_ENABLED":"1", "VERIFYD_CIRCLE_CLASSROOM_AUDIENCE_ENABLED":"1"},clear=True):
            m=load_module()
            with self.assertRaises(HTTPException) as err:
                m.update_assignment_audience('class','assignment',m.AudienceSelection(mode='selected',student_ids=[]),'Bearer a')
            self.assertEqual(err.exception.status_code,422)

    def test_duplicate_selected_recipients_rejected(self):
        with patch.dict(os.environ,{"VERIFYD_CIRCLE_CLASSROOM_INVITES_ENABLED":"1", "VERIFYD_CIRCLE_CLASSROOM_AUDIENCE_ENABLED":"1"},clear=True):
            m=load_module()
            with self.assertRaises(HTTPException) as err:
                m.update_assignment_audience('class','assignment',m.AudienceSelection(mode='selected',student_ids=['s1','s1']),'Bearer a')
            self.assertEqual(err.exception.status_code,422)

    def test_non_teacher_cannot_access_selected_audience(self):
        with patch.dict(os.environ,{"VERIFYD_CIRCLE_CLASSROOM_INVITES_ENABLED":"1", "VERIFYD_CIRCLE_CLASSROOM_AUDIENCE_ENABLED":"1"},clear=True):
            m=load_module()
            def deny_teacher(_): raise HTTPException(403,detail={'error':'teacher_registration_required'})
            m._approved_teacher=deny_teacher
            with self.assertRaises(HTTPException) as err:
                m.get_assignment_audience('class','assignment','Bearer a')
            self.assertEqual(err.exception.status_code,403)

    def test_cannot_change_published_assignment_audience(self):
        with patch.dict(os.environ,{"VERIFYD_CIRCLE_CLASSROOM_INVITES_ENABLED":"1", "VERIFYD_CIRCLE_CLASSROOM_AUDIENCE_ENABLED":"1"},clear=True):
            m=load_module();m._schema=lambda:None
            class Context:
                def __enter__(self):return self
                def __exit__(self,*a):return False
                def cursor(self):return FakeCursor([{'status':'published'}])
            m.get_db=lambda:Context()
            with self.assertRaises(HTTPException) as err:
                m.update_assignment_audience('class','assignment',m.AudienceSelection(mode='all'),'Bearer a')
            self.assertEqual(err.exception.status_code,409)

if __name__=='__main__':unittest.main()