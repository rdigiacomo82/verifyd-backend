"""Disposable PostgreSQL only. Never use a production DSN."""
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
    psycopg2 = None
ROOT=pathlib.Path(__file__).resolve().parents[1]

@contextmanager
def db():
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

def load(name,relative):
    spec=importlib.util.spec_from_file_location(name,ROOT/relative)
    module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    spec.loader.exec_module(module)
    return module

@unittest.skipUnless(bool(os.getenv('TEST_PG_DSN')) and psycopg2,'Requires disposable PostgreSQL')
class ArcadePostgresTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.teacher='test-'+uuid.uuid4().hex
        cls.student='test-'+uuid.uuid4().hex
        cls.other='test-'+uuid.uuid4().hex
        cls.class_id='class-'+uuid.uuid4().hex
        with db() as conn:
            cur=conn.cursor()
            cur.execute("CREATE TABLE IF NOT EXISTS verifyd_identities(id TEXT PRIMARY KEY,status TEXT NOT NULL DEFAULT 'active',handle TEXT NOT NULL)")
            for user in (cls.teacher,cls.student,cls.other):
                cur.execute("INSERT INTO verifyd_identities (id,status,handle) VALUES(%s,'active',%s)",(user,user))
            cur.execute('''CREATE TABLE IF NOT EXISTS verifyd_circle_classes(
                class_id TEXT PRIMARY KEY, teacher_id TEXT REFERENCES verifyd_identities(id),
                title TEXT NOT NULL DEFAULT '',subject TEXT NOT NULL DEFAULT '',grade_band TEXT NOT NULL DEFAULT 'Adult')''')
            cur.execute('''CREATE TABLE IF NOT EXISTS verifyd_circle_class_members(
                class_id TEXT REFERENCES verifyd_circle_classes(class_id),
                student_id TEXT REFERENCES verifyd_identities(id),status TEXT NOT NULL,
                PRIMARY KEY(class_id,student_id))''')
            cur.execute('INSERT INTO verifyd_circle_classes(class_id,teacher_id) VALUES(%s,%s)',(cls.class_id,cls.teacher))
            for user in (cls.student,cls.other):
                cur.execute("INSERT INTO verifyd_circle_class_members VALUES (%s,%s,'approved')",(cls.class_id,user))
        database=types.ModuleType('database');database.get_db=db
        messages=types.ModuleType('trust_voice_messages');messages._identity_from_bearer=auth
        classes=types.ModuleType('trust_voice_classrooms')
        classes._schema=lambda:None
        def member(cur,class_id,user):
            cur.execute('''SELECT c.teacher_id,m.status FROM verifyd_circle_classes c
                LEFT JOIN verifyd_circle_class_members m ON m.class_id=c.class_id AND m.student_id=%s
                WHERE c.class_id=%s''',(user,class_id))
            row=cur.fetchone()
            if not row or (row['teacher_id']!=user and row['status']!='approved'):raise HTTPException(404)
            return row
        def teacher(cur,class_id,user):
            row=member(cur,class_id,user)
            if row['teacher_id']!=user:raise HTTPException(404)
            return row
        classes._member=member;classes._teacher=teacher
        cls.module_patch=patch.dict(sys.modules,{'database':database,'trust_voice_messages':messages,'trust_voice_classrooms':classes})
        cls.module_patch.start()
        cls.env_patch=patch.dict(os.environ,{
            'VERIFYD_CIRCLE_RANKING_ENABLED':'1','VERIFYD_CIRCLE_ARCADE_RANK_ENABLED':'1',
            'VERIFYD_CIRCLE_CLASSROOM_GAME_ASSIGNMENTS_ENABLED':'1'})
        cls.env_patch.start()
        cls.rank=load('trust_voice_ranks','trust_voice_ranks.py')
        cls.arcade=load('trust_voice_arcade_ranked','trust_voice_arcade_ranked.py')
        cls.arcade.schema()

    @classmethod
    def tearDownClass(cls):
        try:
            with db() as conn:
                conn.cursor().execute('DELETE FROM verifyd_identities WHERE id=ANY(%s)',([cls.teacher,cls.student,cls.other],))
        finally:
            cls.env_patch.stop();cls.module_patch.stop()
            sys.modules.pop('trust_voice_ranks',None);sys.modules.pop('trust_voice_arcade_ranked',None)

    def bearer(self,user):return 'Bearer '+user
    def finish(self,game,level,user,assignment_id=None):
        m=self.arcade
        payload=m.Begin(game=game,level=level,assignment_id=assignment_id)
        s=m.begin(payload,authorization=self.bearer(user))
        for round_idx in range(m.ROUND_COUNTS[level]):
            answer=m.expected(game,level,round_idx,s['seed'])
            if game=='route':
                size,_=answer
                answer=list(range(size))+[(i+1)*size+size-1 for i in range(size-1)]
            result=m.round_complete(s['session_id'],m.RoundProof(round=round_idx,proof=answer),authorization=self.bearer(user))
            self.assertTrue(result['correct'])
        self.assertTrue(result['completed'])
        return s,result

    def test_00_flags_fail_closed(self):
        with patch.dict(os.environ,{'VERIFYD_CIRCLE_ARCADE_RANK_ENABLED':'0'}):
            with self.assertRaises(HTTPException) as err:
                self.arcade.begin(self.arcade.Begin(game='pattern',level=0),authorization=self.bearer(self.student))
            self.assertEqual(err.exception.status_code,404)

    def test_01_all_five_game_types_and_levels_one_time_only(self):
        for g in self.arcade.GAMES:
            for level,points in enumerate((10,25,50)):
                _,done=self.finish(g,level,self.student)
                self.assertEqual(done['points_added'],points)
                _,again=self.finish(g,level,self.student)
                self.assertEqual(again['points_added'],0)
                self.assertTrue(again['already_earned'])
        earned=self.arcade.catalog(authorization=self.bearer(self.student))['earned_challenges']
        self.assertEqual(len(earned),15)
        self.assertEqual(self.rank.my_rank(self.bearer(self.student))['points'],(10+25+50)*5)

    def test_02_teacher_assignment_only_selected_student_and_recorded(self):
        m=self.arcade
        with self.assertRaises(HTTPException):
            m.assign_to_class(self.class_id,m.TeacherGameAssignment(title='Not allowed',game='logic',level=1,recipient_ids=[self.student]),self.bearer(self.student))
        assigned=m.assign_to_class(self.class_id,m.TeacherGameAssignment(title='Period A logic',game='logic',level=1,recipient_ids=[self.student]),self.bearer(self.teacher))
        aid=assigned['assignment_id']
        self.assertEqual(len(m.game_assignments(self.class_id,self.bearer(self.student))['assignments']),1)
        self.assertEqual(len(m.game_assignments(self.class_id,self.bearer(self.other))['assignments']),0)
        with self.assertRaises(HTTPException) as denied:
            m.begin(m.Begin(game='logic',level=1,assignment_id=aid),self.bearer(self.other))
        self.assertEqual(denied.exception.status_code,404)
        _,done=self.finish('logic',1,self.student,assignment_id=aid)
        self.assertEqual(done['points_added'],0) # Already received global unique reward in prior test
        self.assertTrue(m.game_assignments(self.class_id,self.bearer(self.student))['assignments'][0]['completed'])
        self.assertEqual(m.assignment_results(self.class_id,aid,self.bearer(self.teacher))['recipients'][0]['student_id'],self.student)
        with self.assertRaises(HTTPException):
            m.assignment_results(self.class_id,aid,self.bearer(self.other))

    def test_03_wrong_round_and_other_account_cannot_finish(self):
        m=self.arcade
        s=m.begin(m.Begin(game='word',level=0),self.bearer(self.other))
        with self.assertRaises(HTTPException) as denied:
            m.round_complete(s['session_id'],m.RoundProof(round=0,proof=0),self.bearer(self.student))
        self.assertEqual(denied.exception.status_code,404)
        result=m.round_complete(s['session_id'],m.RoundProof(round=0,proof=-1),self.bearer(self.other))
        self.assertFalse(result['correct'])
        with self.assertRaises(HTTPException) as outofsequence:
            m.round_complete(s['session_id'],m.RoundProof(round=1,proof=0),self.bearer(self.other))
        self.assertEqual(outofsequence.exception.status_code,409)
        with db() as conn:
            cur=conn.cursor();cur.execute('SELECT COUNT(*) AS n FROM verifyd_circle_rank_awards WHERE identity_id=%s',(self.other,));self.assertEqual(cur.fetchone()['n'],0)

if __name__=='__main__':unittest.main()