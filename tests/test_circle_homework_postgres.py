"""PostgreSQL integration for optional Circle Homework + ranks.

Runs only with TEST_PG_DSN (isolated CI database). Never contacts production.
"""
import importlib.util
import os
import sys
import threading
import types
import unittest
import uuid
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

if os.getenv("TEST_PG_DSN"):
    import psycopg2
    import psycopg2.extras
else:
    psycopg2 = None

ROOT = Path(__file__).resolve().parents[1]


@contextmanager
def _db():
    conn = psycopg2.connect(os.environ["TEST_PG_DSN"], cursor_factory=psycopg2.extras.RealDictCursor)
    try:
        yield conn
        conn.commit()
    except BaseException:
        conn.rollback()
        raise
    finally:
        conn.close()


def _auth(header):
    if not header or not header.startswith("Bearer itest-"):
        from fastapi import HTTPException
        raise HTTPException(401, detail="session_required")
    user = header.split(" ", 1)[1].removeprefix("itest-")
    return {"id": user, "handle": user, "status": "active"}


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, ROOT / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@unittest.skipUnless(os.getenv("TEST_PG_DSN"), "Needs isolated TEST_PG_DSN")
class CircleHomeworkPostgresTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with _db() as conn:
            conn.cursor().execute("""
                CREATE TABLE IF NOT EXISTS verifyd_identities (
                    id TEXT PRIMARY KEY, status TEXT NOT NULL DEFAULT 'active'
                )
            """)
        db = types.ModuleType("database")
        db.get_db = _db
        auth = types.ModuleType("trust_voice_messages")
        auth._identity_from_bearer = _auth
        cls._patch = patch.dict(sys.modules, {"database": db, "trust_voice_messages": auth})
        cls._patch.start()
        cls.ranks = _load("trust_voice_ranks", "trust_voice_ranks.py")
        sys.modules["trust_voice_ranks"] = cls.ranks
        cls.hw = _load("trust_voice_homework_integration", "trust_voice_homework.py")
        cls._flags = patch.dict(os.environ, {
            "VERIFYD_CIRCLE_RANKING_ENABLED": "1",
            "VERIFYD_CIRCLE_HOMEWORK_ENABLED": "1",
        })
        cls._flags.start()
        cls.ranks._schema()
        cls.hw._schema()
        cls.u1, cls.u2 = ("itest-"+uuid.uuid4().hex, "itest-"+uuid.uuid4().hex)
        with _db() as conn:
            cur=conn.cursor()
            cur.execute("INSERT INTO verifyd_identities (id,status) VALUES (%s,'active'),(%s,'active')",
                        (cls.u1, cls.u2))

    @classmethod
    def tearDownClass(cls):
        try:
            with _db() as conn:
                conn.cursor().execute("DELETE FROM verifyd_identities WHERE id IN (%s,%s)", (cls.u1,cls.u2))
        finally:
            cls._flags.stop()
            cls._patch.stop()
            sys.modules.pop("trust_voice_ranks",None)

    def auth1(self):
        return "Bearer itest-"+self.u1

    def auth2(self):
        return "Bearer itest-"+self.u2

    def answers(self, id, correct=3):
        item=self.hw.ASSIGNMENTS[id]
        answer={}
        for i,q in enumerate(item["questions"]):
            if i<correct:
                answer[q["id"]]=q["answer"]
            else:
                answer[q["id"]]=next(c["id"] for c in q["choices"] if c["id"]!=q["answer"])
        return answer

    def test_01_draft_status_and_account_isolation(self):
        item_id="k2-math-add"
        out=self.hw.save_draft(item_id,self.hw.HomeworkDraft(
            answers={"q1":"b"},reflection="My private thought"),
            authorization=self.auth1())
        self.assertEqual(out["work"]["status"],"in_progress")
        me=self.hw.my_work(self.auth1())
        other=self.hw.my_work(self.auth2())
        self.assertEqual(me["summary"]["in_progress"],1)
        self.assertEqual(other["summary"]["in_progress"],0)
        self.assertFalse(any("My private thought" in str(w) for w in other["work"]))

    def test_02_complete_and_one_time_credit(self):
        item_id="k2-math-add"
        out=self.hw.submit(item_id,self.hw.HomeworkSubmit(
            answers=self.answers(item_id,correct=2), reflection="I used counting"),
            authorization=self.auth1())
        self.assertEqual(out["work"]["status"],"completed")
        self.assertEqual(out["points_added"],60)
        self.assertEqual(out["result"]["correct"],2)
        rank=self.ranks._snapshot(self.u1,owner=True)
        self.assertEqual(rank["points"],60)
        self.assertEqual(rank["unique_games"],0)
        with self.assertRaises(self.hw.HTTPException) as error:
            self.hw.submit(item_id,self.hw.HomeworkSubmit(answers=self.answers(item_id)),
                authorization=self.auth1())
        self.assertEqual(error.exception.status_code,409)
        self.assertEqual(self.ranks._snapshot(self.u1,owner=True)["points"],60)
        # Another legitimate account has its own completion.
        other=self.hw.submit(item_id,self.hw.HomeworkSubmit(answers=self.answers(item_id)),
            authorization=self.auth2())
        self.assertEqual(other["points_added"],60)

    def test_03_failed_stays_in_progress(self):
        item_id="g35-math-fractions"
        out=self.hw.submit(item_id,self.hw.HomeworkSubmit(
            answers=self.answers(item_id,correct=1)),authorization=self.auth1())
        self.assertEqual(out["work"]["status"],"in_progress")
        self.assertEqual(out["points_added"],0)

    def test_04_public_number_opt_in(self):
        before=self.ranks._snapshot(self.u1)
        self.assertIsNone(before["public_number"])
        self.ranks.settings(self.ranks.RankSettings(show_public_number=True),authorization=self.auth1())
        after=self.ranks._snapshot(self.u1)
        self.assertEqual(after["public_number"],60)
        self.ranks.settings(self.ranks.RankSettings(show_public_number=False),authorization=self.auth1())
        self.assertIsNone(self.ranks._snapshot(self.u1)["public_number"])

    def test_05_simultaneous_submit_awards_once(self):
        assignment_id="g912-math-linear"
        errors,done=[],[]
        def run():
            try:
                done.append(self.hw.submit(assignment_id,self.hw.HomeworkSubmit(
                    answers=self.answers(assignment_id)),authorization=self.auth1()))
            except Exception as exc:
                errors.append(exc)
        a=threading.Thread(target=run);b=threading.Thread(target=run)
        a.start();b.start();a.join(timeout=25);b.join(timeout=25)
        self.assertFalse(a.is_alive() or b.is_alive())
        self.assertEqual(len(done),1)
        self.assertEqual(len(errors),1)
        self.assertEqual(getattr(errors[0],"status_code",None),409)
        self.assertEqual(done[0]["points_added"],60)
        with _db() as conn:
            cur=conn.cursor()
            cur.execute("""SELECT COUNT(*) AS n FROM verifyd_circle_rank_awards
                WHERE identity_id=%s AND challenge_id=%s""",
                (self.u1,f"homework:{assignment_id}"))
            self.assertEqual(cur.fetchone()["n"],1)


if __name__=="__main__":
    unittest.main()
