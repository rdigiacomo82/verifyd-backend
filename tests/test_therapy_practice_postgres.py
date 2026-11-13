"""End-to-end Therapy pilot isolation against disposable PostgreSQL only.

TEST_PG_DSN must identify an ephemeral test database, never production.
"""
import importlib.util
import os
import pathlib
import sys
import types
import uuid
import unittest
from contextlib import contextmanager
from unittest.mock import patch

from fastapi import HTTPException

if os.getenv('TEST_PG_DSN'):
    import psycopg2
    from psycopg2.extras import RealDictCursor
else:
    psycopg2=None

ROOT=pathlib.Path(__file__).resolve().parents[1]

@contextmanager
def db():
    conn=psycopg2.connect(os.environ['TEST_PG_DSN'],cursor_factory=RealDictCursor)
    try:
        yield conn
        conn.commit()
    except BaseException:
        conn.rollback()
        raise
    finally:
        conn.close()


def auth(header):
    if not header or not header.startswith('Bearer edu-'):
        raise HTTPException(401,detail={'error':'session_required'})
    return {'id':header[len('Bearer '):],'status':'active'}


def prepare_trust():
    with db() as conn:
        cur=conn.cursor()
        cur.execute('''CREATE TABLE IF NOT EXISTS verifyd_trust_circle_members (
            owner_identity_id TEXT NOT NULL,member_identity_id TEXT NOT NULL,
            contact_type TEXT NOT NULL DEFAULT 'person',display_label TEXT NOT NULL DEFAULT '',
            sort_order INTEGER NOT NULL DEFAULT 0,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            PRIMARY KEY(owner_identity_id,member_identity_id))''')


@unittest.skipUnless(bool(os.getenv('TEST_PG_DSN')),'Disposable PostgreSQL connection required')
class TherapyPostgresIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with db() as conn:
            conn.cursor().execute('''CREATE TABLE IF NOT EXISTS verifyd_identities(
                id TEXT PRIMARY KEY,handle TEXT,status TEXT NOT NULL DEFAULT 'active')''')
        ids=['edu-'+uuid.uuid4().hex for _ in range(4)]
        cls.pro,cls.client,cls.other,cls.stranger=ids
        with db() as conn:
            cur=conn.cursor()
            for index, identity in enumerate(ids):
                cur.execute('INSERT INTO verifyd_identities(id,handle,status) VALUES (%s,%s,\'active\')',
                            (identity,'therapytest'+str(index)+uuid.uuid4().hex[:8]))
        database=types.ModuleType('database');database.get_db=db
        messages=types.ModuleType('trust_voice_messages');messages._identity_from_bearer=auth
        permissions=types.ModuleType('trust_voice_permissions');permissions.ensure_trust_circle_schema=prepare_trust
        blocking=types.ModuleType('trust_voice_blocking');blocking.identities_blocked=lambda *args:False
        cls.import_patch=patch.dict(sys.modules,{'database':database,'trust_voice_messages':messages,
            'trust_voice_permissions':permissions,'trust_voice_blocking':blocking})
        cls.import_patch.start()
        cls.env_patch=patch.dict(os.environ,{
            'VERIFYD_CIRCLE_THERAPY_PRACTICE_ENABLED':'1',
            'VERIFYD_CIRCLE_THERAPY_PROFESSIONAL_IDS':cls.pro})
        cls.env_patch.start()
        name='test_therapy_instance_'+uuid.uuid4().hex
        spec=importlib.util.spec_from_file_location(name,ROOT/'trust_voice_therapy_practice.py')
        cls.module=importlib.util.module_from_spec(spec)
        sys.modules[name]=cls.module
        spec.loader.exec_module(cls.module)
        cls.import_name=name
        cls.module._schema()

    @classmethod
    def tearDownClass(cls):
        try:
            with db() as conn:
                cur=conn.cursor()
                cur.execute('DELETE FROM verifyd_trust_circle_members WHERE owner_identity_id=%s',(cls.pro,))
                cur.execute('DELETE FROM verifyd_identities WHERE id=ANY(%s)',([cls.pro,cls.client,cls.other,cls.stranger],))
        finally:
            cls.env_patch.stop();cls.import_patch.stop();sys.modules.pop(cls.import_name,None)

    def bearer(self,user): return 'Bearer '+user

    def test_full_invitation_selection_submission_and_review(self):
        m=self.module
        # Unapproved accounts cannot create a professional practice group.
        with self.assertRaises(HTTPException) as denied:
            m.create_program(m.ProgramCreate(title='Unauthorized'),self.bearer(self.client))
        self.assertEqual(denied.exception.status_code,403)
        created=m.create_program(m.ProgramCreate(title='Communication Practice A'),self.bearer(self.pro))
        program=created['program_id']
        with db() as conn:
            cur=conn.cursor()
            for identity in (self.client,self.other):
                cur.execute('''INSERT INTO verifyd_trust_circle_members(owner_identity_id,member_identity_id)
                    VALUES(%s,%s) ON CONFLICT DO NOTHING''',(self.pro,identity))
        sent=m.invite(program,m.InviteClients(client_ids=[self.client,self.other]),self.bearer(self.pro))
        self.assertEqual(sent['sent'],2)
        with self.assertRaises(HTTPException):
            m.invite(program,m.InviteClients(client_ids=[self.stranger]),self.bearer(self.pro))
        for user in (self.client,self.other):
            inbox=m.inbox(self.bearer(user))['invitations']
            invite=next(row for row in inbox if row['program_id']==program)
            with self.assertRaises(HTTPException):
                m.accept(invite['invitation_id'],m.AcceptInvite(adult_attestation=False),self.bearer(user))
            accepted=m.accept(invite['invitation_id'],m.AcceptInvite(adult_attestation=True),self.bearer(user))
            self.assertTrue(accepted['joined']);self.assertFalse(accepted['requires_second_approval'])
        self.assertEqual(len(m.members(program,self.bearer(self.pro))['members']),2)
        task=m.create_task(program,m.PracticeTaskCreate(
            title='Choice board trial',instructions='Use the phrase board to describe your practice.',
            mode='choices',activities=[],recipient_ids=[self.client]),self.bearer(self.pro))['task_id']
        m.publish_task(program,task,self.bearer(self.pro))
        self.assertEqual(len(m.list_tasks(program,self.bearer(self.client))['tasks']),1)
        self.assertEqual(len(m.list_tasks(program,self.bearer(self.other))['tasks']),0)
        with self.assertRaises(HTTPException) as inaccessible:
            m.save_response(program,task,m.PracticeResponse(action='submit',response='I did it.'),self.bearer(self.other))
        self.assertEqual(inaccessible.exception.status_code,404)
        saved=m.save_response(program,task,m.PracticeResponse(action='save',response='I used the phrase board.'),self.bearer(self.client))
        self.assertEqual(saved['work']['status'],'draft')
        submitted=m.save_response(program,task,m.PracticeResponse(action='submit',response='I practiced and need more time.'),self.bearer(self.client))
        self.assertEqual(submitted['work']['status'],'submitted')
        with self.assertRaises(HTTPException) as locked:
            m.save_response(program,task,m.PracticeResponse(action='save',response='Changed'),self.bearer(self.client))
        self.assertEqual(locked.exception.status_code,409)
        self.assertEqual(len(m.submissions(program,task,self.bearer(self.pro))['submissions']),1)
        returned=m.review(program,task,self.client,m.PracticeReview(feedback='Thanks for your effort. Try again when ready.'),self.bearer(self.pro))
        self.assertEqual(returned['work']['status'],'returned')
        self.assertFalse(returned['clinical_grade'])
        work=m.my_response(program,task,self.bearer(self.client))['work']
        self.assertEqual(work['status'],'returned')
        self.assertIn('Try again',work['feedback'])
        m.remove_member(program,self.client,self.bearer(self.pro))
        self.assertFalse(any(row['program_id']==program for row in m.programs(self.bearer(self.client))['programs']))
        with self.assertRaises(HTTPException):
            m.my_response(program,task,self.bearer(self.client))
        # Re-inviting the same test client must not reactivate past assignments.
        self.assertEqual(m.invite(program,m.InviteClients(client_ids=[self.client]),self.bearer(self.pro))['sent'],1)
        again=next(row for row in m.inbox(self.bearer(self.client))['invitations'] if row['program_id']==program)
        self.assertTrue(m.accept(again['invitation_id'],m.AcceptInvite(adult_attestation=True),self.bearer(self.client))['joined'])
        self.assertNotIn(task,[row['task_id'] for row in
            m.list_tasks(program,self.bearer(self.client))['tasks']])
        with self.assertRaises(HTTPException) as hidden:
            m.my_response(program,task,self.bearer(self.client))
        self.assertEqual(hidden.exception.status_code,404)
        with self.assertRaises(HTTPException) as unauthorized:
            m.save_response(program,task,m.PracticeResponse(response='Legacy task',action='save'),self.bearer(self.client))
        self.assertEqual(unauthorized.exception.status_code,404)

if __name__=='__main__':unittest.main()