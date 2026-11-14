"""Professional applications, approvals and revocations — disposable PostgreSQL ONLY.

This suite never connects to the production VeriFYD database.
"""
from __future__ import annotations
import importlib.util
import os
import pathlib
import sys
import types
import unittest
import uuid
from contextlib import contextmanager
from unittest.mock import patch
from fastapi import HTTPException

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
except ImportError:
    psycopg2=None

ROOT=pathlib.Path(__file__).resolve().parents[1]

@contextmanager
def database():
    conn=psycopg2.connect(os.environ['TEST_PG_DSN'],cursor_factory=RealDictCursor)
    try:
        yield conn;conn.commit()
    except BaseException:
        conn.rollback();raise
    finally:conn.close()


def auth(header):
    if not header or not header.startswith('Bearer test-'):
        raise HTTPException(401,detail={'error':'session_required'})
    return {'id':header[len('Bearer '):],'status':'active'}

@unittest.skipUnless(bool(os.getenv('TEST_PG_DSN')) and psycopg2, 'Disposable PostgreSQL required')
class TherapyProfessionalRegistrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reviewer='test-'+uuid.uuid4().hex
        cls.applicant='test-'+uuid.uuid4().hex
        cls.outsider='test-'+uuid.uuid4().hex
        cls.unverified='test-'+uuid.uuid4().hex
        with database() as conn:
            cur=conn.cursor()
            cur.execute('''CREATE TABLE IF NOT EXISTS verifyd_identities(
                id TEXT PRIMARY KEY,status TEXT NOT NULL DEFAULT 'active',
                email_verified BOOLEAN NOT NULL DEFAULT FALSE,handle TEXT NOT NULL)''')
            for index,user in enumerate((cls.reviewer,cls.applicant,cls.outsider,cls.unverified)):
                cur.execute('''INSERT INTO verifyd_identities(id,status,email_verified,handle)
                    VALUES (%s,'active',%s,%s)''',(user,index!=3,'test_handle'+str(index)))
        mdb=types.ModuleType('database');mdb.get_db=database
        msgs=types.ModuleType('trust_voice_messages');msgs._identity_from_bearer=auth
        cls.module_patch=patch.dict(sys.modules,{'database':mdb,'trust_voice_messages':msgs})
        cls.module_patch.start()
        cls.env_patch=patch.dict(os.environ,{
            'VERIFYD_CIRCLE_THERAPY_REGISTRATION_ENABLED':'1',
            'VERIFYD_CIRCLE_THERAPY_ADMIN_IDS':cls.reviewer,
            'VERIFYD_CIRCLE_THERAPY_PRACTICE_ENABLED':'0',
            'VERIFYD_CIRCLE_THERAPY_PROFESSIONAL_IDS':''})
        cls.env_patch.start()
        spec=importlib.util.spec_from_file_location('trust_voice_therapy_professionals',ROOT/'trust_voice_therapy_professionals.py')
        cls.pro=importlib.util.module_from_spec(spec)
        sys.modules['trust_voice_therapy_professionals']=cls.pro
        spec.loader.exec_module(cls.pro)
        cls.pro._schema()
        perms=types.ModuleType('trust_voice_permissions');perms.ensure_trust_circle_schema=lambda:None
        block=types.ModuleType('trust_voice_blocking');block.identities_blocked=lambda *args:False
        cls.extra_patch=patch.dict(sys.modules,{'trust_voice_permissions':perms,'trust_voice_blocking':block})
        cls.extra_patch.start()
        sp=importlib.util.spec_from_file_location('therapy_practice_test_module',ROOT/'trust_voice_therapy_practice.py')
        cls.therapy=importlib.util.module_from_spec(sp)
        sys.modules['therapy_practice_test_module']=cls.therapy
        sp.loader.exec_module(cls.therapy)

    @classmethod
    def tearDownClass(cls):
        try:
            with database() as conn:
                conn.cursor().execute('DELETE FROM verifyd_identities WHERE id=ANY(%s)',(
                    [cls.reviewer,cls.applicant,cls.outsider,cls.unverified],))
        finally:
            cls.extra_patch.stop();cls.env_patch.stop();cls.module_patch.stop()
            sys.modules.pop('trust_voice_therapy_professionals',None)
            sys.modules.pop('therapy_practice_test_module',None)

    def token(self,user):return 'Bearer '+user
    def app(self,**overrides):
        values=dict(display_name='Adult Test Professional',
            professional_role='Speech-Language Pathologist',specialty='Assistive communication',
            organization='Fictional Test Center',license_jurisdiction='New York, USA',
            licensing_board='Fictional Test Licensing Board',license_number='TEST-123456',
            npi=None,adult_attestation=True,no_patient_data_acknowledgment=True,
            truthful_information_acknowledgment=True)
        values.update(overrides)
        return self.pro.Application(**values)

    def test_00_closed_registration_fails_closed(self):
        with patch.dict(os.environ,{'VERIFYD_CIRCLE_THERAPY_REGISTRATION_ENABLED':'0'}):
            with self.assertRaises(HTTPException) as exc:
                self.pro.my_application(self.token(self.applicant))
            self.assertEqual(exc.exception.status_code,404)

    def test_01_email_verification_and_attestations_required(self):
        with self.assertRaises(HTTPException) as exc:
            self.pro.apply(self.app(),self.token(self.unverified))
        self.assertEqual(exc.exception.status_code,403)
        with self.assertRaises(HTTPException) as exc:
            self.pro.apply(self.app(adult_attestation=False),self.token(self.applicant))
        self.assertEqual(exc.exception.status_code,422)

    def test_02_apply_never_grants_access_and_private_review(self):
        self.assertFalse(self.therapy._is_professional(self.applicant))
        result=self.pro.apply(self.app(),self.token(self.applicant))
        self.assertEqual(result['status'],'pending')
        self.assertFalse(self.therapy._is_professional(self.applicant))
        with self.assertRaises(HTTPException) as exc:
            self.pro.apply(self.app(),self.token(self.applicant))
        self.assertEqual(exc.exception.status_code,409)
        with self.assertRaises(HTTPException) as exc:
            self.pro.admin_applications(authorization=self.token(self.outsider))
        self.assertEqual(exc.exception.status_code,403)
        pending=self.pro.admin_applications(authorization=self.token(self.reviewer))['applications']
        self.assertEqual(len(pending),1)
        self.assertEqual(pending[0]['identity_id'],self.applicant)
        self.assertNotIn('license_number',str(self.pro.my_application(self.token(self.outsider))))

    def test_03_only_manual_independent_review_approves(self):
        with self.assertRaises(HTTPException) as exc:
            self.pro.decide(self.applicant,self.pro.Decision(action='approve'),self.token(self.reviewer))
        self.assertEqual(exc.exception.status_code,422)
        with self.assertRaises(HTTPException) as exc:
            self.pro.decide(self.applicant,self.pro.Decision(action='approve',credential_verified=True,
                adult_test_only_confirmed=True),self.token(self.outsider))
        self.assertEqual(exc.exception.status_code,403)
        with self.assertRaises(HTTPException) as exc:
            self.pro.decide(self.reviewer,self.pro.Decision(action='approve',credential_verified=True,
                adult_test_only_confirmed=True),self.token(self.reviewer))
        self.assertEqual(exc.exception.status_code,403)
        result=self.pro.decide(self.applicant,self.pro.Decision(action='approve',credential_verified=True,
            adult_test_only_confirmed=True),self.token(self.reviewer))
        self.assertEqual(result['status'],'approved')
        self.assertFalse(result['practice_enabled'])
        self.assertTrue(self.therapy._is_professional(self.applicant))
        status=self.pro.my_application(self.token(self.applicant))
        self.assertEqual(status['application']['status'],'approved')
        self.assertFalse(status['practice_enabled'])
        self.assertGreaterEqual(status['unread_notifications'],2)
        items=self.pro.notifications(self.token(self.applicant))['notifications']
        approval=next(i for i in items if i['kind']=='approved')
        foreign=self.pro.read_notification(approval['notice_id'],self.token(self.outsider))
        self.assertFalse(foreign['read'])
        read=self.pro.read_notification(approval['notice_id'],self.token(self.applicant))
        self.assertTrue(read['read'])

    def test_04_revocation_blocks_access_and_writes_audit(self):
        outcome=self.pro.decide(self.applicant,self.pro.Decision(action='revoke',reason='Pilot access withdrawn'),self.token(self.reviewer))
        self.assertEqual(outcome['status'],'revoked')
        self.assertFalse(self.therapy._is_professional(self.applicant))
        with self.assertRaises(HTTPException) as exc:
            self.pro.apply(self.app(),self.token(self.applicant))
        self.assertEqual(exc.exception.status_code,409)
        with database() as conn:
            cur=conn.cursor()
            cur.execute('''SELECT next_status,credential_check_attested FROM verifyd_therapy_professional_decisions
                WHERE applicant_id=%s ORDER BY recorded_at''',(self.applicant,))
            rows=cur.fetchall()
        self.assertEqual([r['next_status'] for r in rows],['approved','revoked'])

if __name__=='__main__': unittest.main()