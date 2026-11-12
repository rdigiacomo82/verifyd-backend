"""Isolated PostgreSQL integration: game awards, sessions and owner isolation.

Runs only when TEST_PG_DSN is configured by the GitHub Actions service.
Never reaches live VeriFYD users or production PostgreSQL.
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

ROOT = pathlib.Path(__file__).resolve().parents[1]

@contextmanager
def db():
    conn = psycopg2.connect(os.environ["TEST_PG_DSN"],cursor_factory=psycopg2.extras.RealDictCursor)
    try:
        yield conn
        conn.commit()
    except BaseException:
        conn.rollback()
        raise
    finally:
        conn.close()

def owner(header):
    from fastapi import HTTPException
    if not header or not header.startswith("Bearer game-itest-"):
        raise HTTPException(401,detail="session_required")
    return {"id":header.split(" ",1)[1].removeprefix("game-itest-"), "status":"active"}

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,ROOT/path)
    obj=importlib.util.module_from_spec(spec)
    sys.modules[name]=obj
    spec.loader.exec_module(obj)
    return obj

@unittest.skipUnless(os.getenv("TEST_PG_DSN"),"Only run against isolated test PostgreSQL")
class VerifiedGamePostgresTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with db() as conn:
            conn.cursor().execute("CREATE TABLE IF NOT EXISTS verifyd_identities(id TEXT PRIMARY KEY,status TEXT NOT NULL DEFAULT 'active')")
        dbmod=types.ModuleType("database")
        dbmod.get_db=db
        auth=types.ModuleType("trust_voice_messages")
        auth._identity_from_bearer=owner
        cls.patchmods=patch.dict(sys.modules,{"database":dbmod,"trust_voice_messages":auth})
        cls.patchmods.start()
        cls.ranks=load("trust_voice_ranks","trust_voice_ranks.py")
        sys.modules["trust_voice_ranks"]=cls.ranks
        cls.games=load("circle_verified_games_test", "trust_voice_verified_games.py")
        cls.flags=patch.dict(os.environ,{"VERIFYD_CIRCLE_RANKING_ENABLED":"1","VERIFYD_CIRCLE_VERIFIED_GAMES_ENABLED":"1"})
        cls.flags.start()
        cls.ranks._schema()
        cls.games.schema()
        cls.a="game-itest-"+uuid.uuid4().hex
        cls.b="game-itest-"+uuid.uuid4().hex
        with db() as conn:
            conn.cursor().execute("INSERT INTO verifyd_identities(id,status) VALUES (%s,'active'),(%s,'active')",(cls.a,cls.b))

    @classmethod
    def tearDownClass(cls):
        try:
            with db() as conn:
                conn.cursor().execute("DELETE FROM verifyd_identities WHERE id IN (%s,%s)",(cls.a,cls.b))
        finally:
            cls.flags.stop()
            cls.patchmods.stop()
            sys.modules.pop("trust_voice_ranks",None)
            sys.modules.pop("circle_verified_games_test",None)

    def token(self,person):
        return "Bearer game-itest-"+person

    def play_match(self,session,person,variant):
        while True:
            with db() as conn:
                cur=conn.cursor()
                cur.execute("SELECT state FROM verifyd_circle_game_sessions WHERE session_id=%s",(session,))
                state=dict(cur.fetchone()["state"])
            indexes={}
            for i,s in enumerate(state["deck"]):
                if i not in state["matched"]:
                    indexes.setdefault(s,[]).append(i)
            result=None
            for slots in indexes.values():
                for index in slots:
                    result=self.games.move(session,self.games.Move(choice=index),authorization=self.token(person))
            return result

    def test_01_server_only_completion_and_no_duplicate_awards(self):
        user=self.a
        first=self.games.start(self.games.Start(game="circle-match",variant=2),authorization=self.token(user))
        self.assertTrue(first["server_verified"])
        self.assertEqual(first["state"]["cards"],[None]*4)
        done=self.play_match(first["session_id"],user,2)
        self.assertTrue(done["state"]["finished"])
        self.assertEqual(done["points_added"],100)
        again=self.games.start(self.games.Start(game="circle-match",variant=2),authorization=self.token(user))
        second=self.play_match(again["session_id"],user,2)
        self.assertEqual(second["points_added"],0)
        with db() as conn:
            cur=conn.cursor()
            cur.execute("SELECT count(*) AS num FROM verifyd_circle_rank_awards WHERE identity_id=%s AND kind='game'",(user,))
            self.assertEqual(cur.fetchone()["num"],1)

    def test_02_owner_isolation_and_one_time_session(self):
        session=self.games.start(self.games.Start(game="circle-match",variant=4),authorization=self.token(self.a))
        from fastapi import HTTPException
        with self.assertRaises(HTTPException) as error:
            self.games.move(session["session_id"],self.games.Move(choice=0),authorization=self.token(self.b))
        self.assertEqual(error.exception.status_code,409)
        self.play_match(session["session_id"],self.a,4)
        with self.assertRaises(HTTPException) as error:
            self.games.move(session["session_id"],self.games.Move(choice=1),authorization=self.token(self.a))
        self.assertEqual(error.exception.status_code,409)

    def test_03_star_quest_level_unlocks(self):
        from fastapi import HTTPException
        with self.assertRaises(HTTPException) as error:
            self.games.start(self.games.Start(game="star-quest",variant=1),authorization=self.token(self.b))
        self.assertEqual(error.exception.status_code,403)
        session=self.games.start(self.games.Start(game="star-quest",variant=0),authorization=self.token(self.b))
        for r in range(5):
            target,_=self.games.star_round(0,r)
            result=self.games.move(session["session_id"],self.games.Move(choice=target),authorization=self.token(self.b))
        self.assertTrue(result["state"]["finished"])
        self.assertEqual(result["points_added"],100)
        next_game=self.games.start(self.games.Start(game="star-quest",variant=1),authorization=self.token(self.b))
        self.assertTrue(next_game["server_verified"])

if __name__=="__main__":
    unittest.main()
