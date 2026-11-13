"""Isolated PostgreSQL integration: teacher approval, private assignments, grading.

TEST_PG_DSN must be an ephemeral test database. No production connection.
"""
import importlib.util
import os
import pathlib
import sys
import types
import uuid
from datetime import datetime, timedelta, timezone
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
        cls.tools=load("trust_voice_classroom_tools","trust_voice_classroom_tools.py")
        cls.c._schema()
        cls.personal.schema()
        cls.tools.schema()

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
            sys.modules.pop("trust_voice_classroom_tools",None)

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
        updated=self.c.edit_assignment(self.class_id, assignment,
            self.c.AssignmentEdit(title="Planet notes revised",
                instructions="Write five accurate facts",due_at=None),
            authorization=self.bearer(self.teacher))
        self.assertEqual(updated["assignment"]["title"],"Planet notes revised")
        self.assertEqual(updated["assignment"]["instructions"],"Write five accurate facts")
        self.assertEqual(self.c.list_assignments(self.class_id,authorization=self.bearer(self.student))["assignments"],[])
        self.c.publish_assignment(self.class_id,assignment,authorization=self.bearer(self.teacher))
        with self.assertRaises(HTTPException) as locked:
            self.c.edit_assignment(self.class_id,assignment,
                self.c.AssignmentEdit(title="Tamper published",instructions="Not allowed"),
                authorization=self.bearer(self.teacher))
        self.assertEqual(locked.exception.status_code,409)
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



    def test_03a_scheduled_classwork_remains_hidden(self):
        from fastapi import HTTPException
        when=datetime.now(timezone.utc)+timedelta(minutes=5)
        due=when+timedelta(days=1)
        made=self.c.create_assignment(self.class_id,self.c.AssignmentCreate(
            title="Scheduled science quiz",instructions="Five simple questions",due_at=due),
            authorization=self.bearer(self.teacher))
        key=made["assignment_id"]
        result=self.c.schedule_assignment(self.class_id,key,self.c.ScheduleAssignment(
            publish_at=when),authorization=self.bearer(self.teacher))
        self.assertEqual(result["status"],"scheduled")
        teacher=self.c.list_assignments(self.class_id,authorization=self.bearer(self.teacher))
        self.assertIn(key,[x["assignment_id"] for x in teacher["assignments"]])
        student=self.c.list_assignments(self.class_id,authorization=self.bearer(self.student))
        self.assertNotIn(key,[x["assignment_id"] for x in student["assignments"]])
        todo=self.tools.student_todo(self.class_id,authorization=self.bearer(self.student))
        self.assertNotIn(key,[x["assignment_id"] for x in todo["assignments"]])
        with self.assertRaises(HTTPException) as err:
            self.c.save_or_turn_in(self.class_id,key,self.c.Work(response="Not yet"),
                authorization=self.bearer(self.student))
        self.assertEqual(err.exception.status_code,409)
        with db() as conn:
            conn.cursor().execute("""UPDATE verifyd_circle_class_assignments
                SET publish_at=NOW()-INTERVAL '5 seconds' WHERE assignment_id=%s""",(key,))
        student=self.c.list_assignments(self.class_id,authorization=self.bearer(self.student))
        self.assertIn(key,[x["assignment_id"] for x in student["assignments"]])
        todo=self.tools.student_todo(self.class_id,authorization=self.bearer(self.student))
        self.assertIn(key,[x["assignment_id"] for x in todo["assignments"]])
        worked=self.c.save_or_turn_in(self.class_id,key,self.c.Work(response="Ready",
            action="save"),authorization=self.bearer(self.student))
        self.assertEqual(worked["work"]["status"],"draft")

    def test_04_classroom_tools_privacy_and_rubrics(self):
        from fastapi import HTTPException
        made=self.c.create_assignment(self.class_id,self.c.AssignmentCreate(
            title="Planetary comparison",instructions="Compare two planets",points_possible=10),
            authorization=self.bearer(self.teacher))
        ident=made["assignment_id"]
        criteria=self.tools.Rubric(criteria=[
            self.tools.RubricCriterion(label="Scientific facts",max_points=6),
            self.tools.RubricCriterion(label="Clarity",max_points=4)])
        with self.assertRaises(HTTPException):
            self.tools.create_rubric(self.class_id,ident,criteria,
                authorization=self.bearer(self.student))
        self.assertEqual(self.tools.create_rubric(self.class_id,ident,criteria,
            authorization=self.bearer(self.teacher))["total_points"],10)
        with self.assertRaises(HTTPException):
            self.tools.get_rubric(self.class_id,ident,authorization=self.bearer(self.student))
        self.c.publish_assignment(self.class_id,ident,authorization=self.bearer(self.teacher))
        self.assertEqual(len(self.tools.get_rubric(self.class_id,ident,
            authorization=self.bearer(self.student))["rubric"]),2)
        with self.assertRaises(HTTPException):
            self.tools.create_rubric(self.class_id,ident,criteria,
                authorization=self.bearer(self.teacher))

        announcement=self.tools.post_announcement(self.class_id,self.tools.Announcement(body="Welcome"),
            authorization=self.bearer(self.teacher))
        self.assertTrue(announcement["posted"])
        self.assertEqual(len(self.tools.list_announcements(self.class_id,
            authorization=self.bearer(self.student))["announcements"]),1)
        with self.assertRaises(HTTPException):
            self.tools.list_announcements(self.class_id,authorization=self.bearer(self.stranger))
        self.assertGreaterEqual(len(self.tools.student_todo(self.class_id,
            authorization=self.bearer(self.student))["assignments"]),1)
        gradebook=self.tools.teacher_gradebook(self.class_id,authorization=self.bearer(self.teacher))
        self.assertGreaterEqual(len(gradebook["gradebook"]),2)
        self.assertIn(self.student,[x["student_id"] for x in gradebook["gradebook"]])
        with self.assertRaises(HTTPException):
            self.tools.teacher_gradebook(self.class_id,authorization=self.bearer(self.student))

        rotated=self.tools.rotate_join_code(self.class_id,authorization=self.bearer(self.teacher))
        self.assertTrue(rotated["previous_code_invalidated"])
        self.assertTrue(self.tools.remove_member(self.class_id,self.student,
            authorization=self.bearer(self.teacher))["removed"])
        with self.assertRaises(HTTPException):
            self.c.list_assignments(self.class_id,authorization=self.bearer(self.student))

    def test_05_revoked_teacher_cannot_read_student_roster(self):
        from fastapi import HTTPException
        # The operator may revoke educator access after a class is created.
        # Previously GET /members omitted this check and leaked student IDs.
        rows = self.c.members(self.class_id, authorization=self.bearer(self.teacher))["members"]
        self.assertTrue(any(row["student_id"] == self.student for row in rows))
        with patch.dict(os.environ, {"VERIFYD_CIRCLE_CLASSROOM_TEACHER_IDS": ""}):
            with self.assertRaises(HTTPException) as error:
                self.c.members(self.class_id, authorization=self.bearer(self.teacher))
            self.assertEqual(error.exception.status_code, 403)

    def test_06_student_deletion_and_teacher_class_deletion(self):
        # Rejoin after prior test removed membership: no data from a student
        # may remain after an explicit deletion request.
        with db() as conn:
            cur=conn.cursor()
            cur.execute("""UPDATE verifyd_circle_class_members SET status='approved'
                WHERE class_id=%s AND student_id=%s""",(self.class_id,self.student))
        result=self.c.leave_class_and_erase_my_work(self.class_id,
            authorization=self.bearer(self.student))
        self.assertTrue(result["my_classwork_deleted"])
        with db() as conn:
            cur=conn.cursor()
            cur.execute("""SELECT COUNT(*) AS n FROM verifyd_circle_class_submissions s
                JOIN verifyd_circle_class_assignments a USING (assignment_id)
                WHERE a.class_id=%s AND s.student_id=%s""",(self.class_id,self.student))
            self.assertEqual(cur.fetchone()["n"],0)
        removed=self.c.delete_teacher_class(self.class_id,
            authorization=self.bearer(self.teacher))
        self.assertTrue(removed["deleted"])
        with db() as conn:
            cur=conn.cursor()
            cur.execute("SELECT COUNT(*) AS n FROM verifyd_circle_classes WHERE class_id=%s",(self.class_id,))
            self.assertEqual(cur.fetchone()["n"],0)


    def test_06_adult_educator_self_registration_and_revocation(self):
        from fastapi import HTTPException
        with patch.dict(os.environ, {
            "VERIFYD_CIRCLE_CLASSROOM_SELF_REGISTER_ENABLED": "1",
            "VERIFYD_CIRCLE_CLASSROOM_TEACHER_IDS": "",
        }):
            self.assertEqual(
                self.c.my_teacher_profile(authorization=self.bearer(self.other))["status"],
                "not_registered",
            )
            with self.assertRaises(HTTPException):
                self.c.create_class(self.c.ClassCreate(
                    title="Premature class", subject="Science", grade_band="Adult"),
                    authorization=self.bearer(self.other))
            base = dict(display_name="Adult Tutor", teaching_role="Tutor",
                        organization="", subjects="Science", credentials="Self-declared",
                        accept_terms=True)
            with self.assertRaises(HTTPException):
                self.c.register_teacher(self.c.TeacherRegistration(
                    **base, adult_attestation=False), authorization=self.bearer(self.other))
            result = self.c.register_teacher(self.c.TeacherRegistration(
                **base, adult_attestation=True), authorization=self.bearer(self.other))
            self.assertEqual(result["status"], "self_registered")
            self.assertFalse(result["independently_verified"])
            self.assertTrue(
                self.c.my_teacher_profile(authorization=self.bearer(self.other))["can_create_classes"])
            with self.assertRaises(HTTPException):
                self.c.create_class(self.c.ClassCreate(
                    title="Unapproved K12", subject="Science", grade_band="3-5"),
                    authorization=self.bearer(self.other))
            created = self.c.create_class(self.c.ClassCreate(
                title="Adult tutoring", subject="Science", grade_band="Adult"),
                authorization=self.bearer(self.other))
            with self.assertRaises(HTTPException):
                self.c.request_to_join(self.c.JoinRequest(join_code=created["join_code"]),
                    authorization=self.bearer(self.student))
            self.assertEqual(
                self.c.request_to_join(self.c.JoinRequest(
                    join_code=created["join_code"], adult_attestation=True),
                    authorization=self.bearer(self.student))["status"], "pending")
            self.assertTrue(created["join_code"])
            self.assertEqual(self.c.my_classes(
                authorization=self.bearer(self.other))["classes"][0]["my_role"], "teacher")
            with db() as conn:
                conn.cursor().execute("UPDATE verifyd_circle_teacher_profiles SET status='suspended' WHERE identity_id=%s", (self.other,))
            with self.assertRaises(HTTPException):
                self.c.members(created["class_id"], authorization=self.bearer(self.other))
            with self.assertRaises(HTTPException):
                self.c.create_class(self.c.ClassCreate(
                    title="Suspended class", subject="Science", grade_band="Adult"),
                    authorization=self.bearer(self.other))
            with db() as conn:
                conn.cursor().execute("UPDATE verifyd_circle_teacher_profiles SET status='active' WHERE identity_id=%s", (self.other,))

if __name__=="__main__":
    unittest.main()
