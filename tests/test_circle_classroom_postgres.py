"""Isolated PostgreSQL integration: teacher approval, private assignments, grading.

TEST_PG_DSN must be an ephemeral test database. No production connection.
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

if os.getenv("TEST_PG_DSN"):
    import psycopg2
    import psycopg2.extras
else:
    psycopg2 = None

ROOT=pathlib.Path(__file__).resolve().parents[1]

@contextmanager
def db():
    conn=psycopg2.connect(os.environ["TEST_PG_DSN"],cursor_factory=psycopg2.extras.RealDictCursor)
    try:
        yield conn
        conn.commit()
    except BaseException:
        conn.rollback()
        raise
    finally:
        conn.close()

def auth(header):
    from fastapi import HTTPException
    if not header or not header.startswith("Bearer edu-"):
        raise HTTPException(401, detail={"error":"session_required"})
    return {"id":header.split(" ",1)[1],"status":"active"}

def load(name,file):
    spec=importlib.util.spec_from_file_location(name,ROOT/file)
    obj=importlib.util.module_from_spec(spec)
    sys.modules[name]=obj
    spec.loader.exec_module(obj)
    return obj

@unittest.skipUnless(os.getenv("TEST_PG_DSN"),"Needs isolated PostgreSQL")
class ClassroomPostgresTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with db() as conn:
            conn.cursor().execute("""CREATE TABLE IF NOT EXISTS verifyd_identities
                (id TEXT PRIMARY KEY,status TEXT NOT NULL DEFAULT 'active')""")
        cls.teacher="edu-"+uuid.uuid4().hex
        cls.student="edu-"+uuid.uuid4().hex
        cls.other="edu-"+uuid.uuid4().hex
        cls.stranger="edu-"+uuid.uuid4().hex
        with db() as conn:
            cur=conn.cursor()
            for identity in (cls.teacher,cls.student,cls.other,cls.stranger):
                cur.execute("INSERT INTO verifyd_identities(id,status) VALUES (%s,'active')",(identity,))
        database=types.ModuleType("database")
        database.get_db=db
        messages=types.ModuleType("trust_voice_messages")
        messages._identity_from_bearer=auth
        cls.modules=patch.dict(sys.modules,{"database":database,"trust_voice_messages":messages})
        cls.modules.start()
        cls.flags=patch.dict(os.environ,{
            "VERIFYD_CIRCLE_CLASSROOM_ENABLED":"1",
            "VERIFYD_CIRCLE_CLASSROOM_TEACHER_IDS":cls.teacher})
        cls.flags.start()
        cls.c=load("trust_voice_classrooms","trust_voice_classrooms.py")
        cls.personal=load("trust_voice_personal_coursework","trust_voice_personal_coursework.py")
        cls.c._schema()
        cls.personal.schema()

    @classmethod
    def tearDownClass(cls):
        try:
            with db() as conn:
                conn.cursor().execute("DELETE FROM verifyd_identities WHERE id=ANY(%s)",
                    ([cls.teacher,cls.student,cls.other,cls.stranger],))
        finally:
            cls.flags.stop()
            cls.modules.stop()
            sys.modules.pop("trust_voice_classrooms",None)
            sys.modules.pop("trust_voice_personal_coursework",None)

    def bearer(self,identity):
        return "Bearer "+identity

    def test_01_role_approval_and_class_join(self):
        from fastapi import HTTPException
        with self.assertRaises(HTTPException):
            self.c.create_class(self.c.ClassCreate(title="Unauthorized Class",subject="Science",grade_band="3-5"),
                                authorization=self.bearer(self.student))
        created=self.c.create_class(self.c.ClassCreate(title="Science Circle",subject="Science",grade_band="3-5"),
                                    authorization=self.bearer(self.teacher))
        self.__class__.class_id=created["class_id"]
        self.assertGreaterEqual(len(created["join_code"]),20)
        joined=self.c.request_to_join(self.c.JoinRequest(join_code=created["join_code"]),
                                      authorization=self.bearer(self.student))
        self.assertEqual(joined["status"],"pending")
        with self.assertRaises(HTTPException):
            self.c.list_assignments(self.class_id,authorization=self.bearer(self.student))
        self.c.approve_member(self.class_id,self.student,authorization=self.bearer(self.teacher))
        self.assertEqual(len(self.c.my_classes(authorization=self.bearer(self.student))["classes"]),1)
        with self.assertRaises(HTTPException):
            self.c.members(self.class_id,authorization=self.bearer(self.student))

    def test_02_assignment_and_grade_no_rank_points(self):
        from fastapi import HTTPException
        made=self.c.create_assignment(self.class_id,self.c.AssignmentCreate(
            title="Solar system notes",instructions="Write three facts",points_possible=20),
            authorization=self.bearer(self.teacher))
        assignment=made["assignment_id"]
        self.__class__.assignment=assignment
        self.assertEqual(self.c.list_assignments(self.class_id,authorization=self.bearer(self.student))["assignments"],[])
        self.c.publish_assignment(self.class_id,assignment,authorization=self.bearer(self.teacher))
        self.assertEqual(len(self.c.list_assignments(self.class_id,authorization=self.bearer(self.student))["assignments"]),1)
        with self.assertRaises(HTTPException):
            self.c.save_or_turn_in(self.class_id,assignment,self.c.Work(response="secret",action="turn_in"),
                                   authorization=self.bearer(self.stranger))
        saved=self.c.save_or_turn_in(self.class_id,assignment,
                                    self.c.Work(response="draft response",action="save"),
                                    authorization=self.bearer(self.student))
        self.assertEqual(saved["work"]["status"],"draft")
        submitted=self.c.save_or_turn_in(self.class_id,assignment,
                                       self.c.Work(response="Three accurate facts",action="turn_in"),
                                       authorization=self.bearer(self.student))
        self.assertEqual(submitted["work"]["status"],"submitted")
        with self.assertRaises(HTTPException):
            self.c.save_or_turn_in(self.class_id,assignment,
                self.c.Work(response="changed",action="save"),authorization=self.bearer(self.student))
        with self.assertRaises(HTTPException):
            self.c.review_submission(self.class_id,assignment,self.student,
                self.c.Review(points_awarded=21),authorization=self.bearer(self.teacher))
        with self.assertRaises(HTTPException):
            self.c.review_submission(self.class_id,assignment,self.student,
                self.c.Review(points_awarded=19),authorization=self.bearer(self.student))
        review=self.c.review_submission(self.class_id,assignment,self.student,
            self.c.Review(points_awarded=19,feedback="Clear explanation"),
            authorization=self.bearer(self.teacher))
        self.assertEqual(review["trust_rank_points_added"],0)
        self.assertEqual(review["review"]["points_awarded"],19)
        self.assertEqual(self.c.my_work(self.class_id,assignment,
            authorization=self.bearer(self.student))["work"]["feedback"],"Clear explanation")
        self.assertEqual(len(self.c.teacher_submissions(self.class_id,assignment,
            authorization=self.bearer(self.teacher))["submissions"]),1)

    def test_03_private_journal_isolation(self):
        from fastapi import HTTPException
        made=self.personal.create_private(self.personal.PersonalCreate(
            title="My private assignment",assignment="Practice an example"),
            authorization=self.bearer(self.other))
        key=made["coursework_id"]
        self.assertEqual(self.personal.list_private(authorization=self.bearer(self.teacher))["private_work"],[])
        with self.assertRaises(HTTPException):
            self.personal.update_private(key,self.personal.PersonalUpdate(response="attempt"),
                authorization=self.bearer(self.student))
        self.personal.update_private(key,self.personal.PersonalUpdate(
            response="My answer",action="complete"),authorization=self.bearer(self.other))
        self.assertEqual(self.personal.list_private(authorization=self.bearer(self.other))["private_work"][0]["status"],"completed")
        with self.assertRaises(HTTPException):
            self.personal.delete_private(key,authorization=self.bearer(self.teacher))
        self.assertTrue(self.personal.delete_private(key,authorization=self.bearer(self.other))["deleted"])

if __name__=="__main__":
    unittest.main()
