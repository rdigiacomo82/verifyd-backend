"""Fast, isolated contract tests for four new ranked mission games.

Never connects to Render/Postgres. Exercises the real route handlers using an
in-memory fake transaction and verifies only the server can grant points.
"""
import importlib.util
import json
import sys
import types
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from fastapi import HTTPException

MODULE = Path(__file__).resolve().parents[1] / "trust_voice_mission_arcade.py"

class FakeStore:
    def __init__(self, points=0):
        self.points = points
        self.sessions = {}
        self.awards = set()
        self.cursor_obj = FakeCursor(self)
    def __enter__(self):
        return self
    def __exit__(self, *args):
        return False
    def cursor(self):
        return self.cursor_obj

class FakeCursor:
    def __init__(self, store):
        self.store = store
        self.one = None
    def fetchone(self):
        return self.one
    def execute(self, sql, params=()):
        normalized = " ".join(sql.split()).lower()
        if "select coalesce(sum(points),0)" in normalized:
            self.one = {"points": self.store.points}
        elif "select count(*) as n from verifyd_circle_ranked_mission_sessions" in normalized:
            self.one = {"n": len(self.store.sessions)}
        elif "insert into verifyd_circle_ranked_mission_sessions" in normalized:
            sid,uid,game,level,blob,token=params
            self.store.sessions[sid] = {
                "identity_id":uid,"game":game,"level":level,
                "questions_json":json.loads(blob),"step":0,"correct_count":0,
                "step_token":token,"expires_at":datetime.now(timezone.utc)+timedelta(days=1),
                "completed_at":None,
            }
            self.one = None
        elif "from verifyd_circle_ranked_mission_sessions" in normalized and "for update" in normalized:
            sid,uid=params
            data=self.store.sessions.get(sid)
            self.one=data.copy() if data is not None and data["identity_id"]==uid else None
        elif "update verifyd_circle_ranked_mission_sessions" in normalized:
            step,correct_count,token,done,sid = params
            record=self.store.sessions[sid]
            record.update(step=step,correct_count=correct_count,step_token=token,
                          completed_at=datetime.now(timezone.utc) if done else None)
            self.one=None
        elif "insert into verifyd_circle_rank_awards" in normalized:
            award_id,uid,challenge,points,ref=params
            key=(uid,challenge)
            if key in self.store.awards: self.one=None
            else:
                self.store.awards.add(key)
                self.store.points+=points
                self.one={"award_id":award_id}
        else:
            raise AssertionError("Unexpected SQL in mocked API: "+normalized[:150])

@pytest.fixture
def api(monkeypatch):
    store=FakeStore()
    db=types.ModuleType("database")
    db.get_db=lambda:store
    msg=types.ModuleType("trust_voice_messages")
    msg._identity_from_bearer=lambda _:{"id":"test-id","status":"active"}
    rank=types.ModuleType("trust_voice_ranks")
    rank._schema=lambda:None
    rank.enabled=lambda:True
    monkeypatch.setitem(sys.modules,"database",db)
    monkeypatch.setitem(sys.modules,"trust_voice_messages",msg)
    monkeypatch.setitem(sys.modules,"trust_voice_ranks",rank)
    monkeypatch.setenv("VERIFYD_CIRCLE_MISSION_ARCADE_ENABLED","1")
    spec=importlib.util.spec_from_file_location("_mission_arcade_test",MODULE)
    module=importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules,spec.name,module)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module,"_schema",lambda:None)
    return module,store

@pytest.mark.parametrize("game", ["mars-rescue","media-detective","debate-duel","design-lab"])
@pytest.mark.parametrize("level",range(3))
def test_server_questions_are_valid_and_private(api,game,level):
    module,_=api
    questions=module._questions(game,level)
    assert len(questions)==4
    for n,q in enumerate(questions):
        assert len(q["choices"])==3
        assert len(set(q["choices"]))==3
        assert q["correct"] in (0,1,2)
        public=module._public_question(q,n)
        assert "correct" not in public
        assert "explanation" not in public

def test_points_only_after_validated_pass(api):
    m,store=api
    item=m.start(m.Start(game="mars-rescue",level=0),authorization="Bearer fake")
    assert item["ranked"] and item["question"].get("correct") is None
    record=store.sessions[item["session_id"]]
    token=item["step_token"]
    results=[]
    for step in range(4):
        result=m.answer(item["session_id"],
            m.Answer(step=step,choice=record["questions_json"][step]["correct"],step_token=token),
            authorization="Bearer fake")
        results.append(result)
        token=result["next_step_token"]
        assert result["points_added"]== (10 if step==3 else 0)
    assert results[-1]["passed"] is True
    assert store.points==10
    assert len(store.awards)==1

def test_repeated_win_does_not_double_award(api):
    m,store=api
    for attempt in range(2):
        item=m.start(m.Start(game="mars-rescue",level=0),authorization="Bearer fake")
        token=item["step_token"]
        for step,q in enumerate(store.sessions[item["session_id"]]["questions_json"]):
            result=m.answer(item["session_id"],
                m.Answer(step=step,choice=q["correct"],step_token=token),
                authorization="Bearer fake")
            token=result["next_step_token"]
        assert result["points_added"]==(10 if attempt==0 else 0)
        if attempt: assert result["already_earned"] is True
    assert store.points==10

def test_wrong_answers_do_not_earn_rank(api):
    m,store=api
    item=m.start(m.Start(game="media-detective",level=0),authorization="Bearer fake")
    token=item["step_token"]
    for step,q in enumerate(store.sessions[item["session_id"]]["questions_json"]):
        wrong=(q["correct"]+1)%3
        result=m.answer(item["session_id"],m.Answer(step=step,choice=wrong,step_token=token),
                        authorization="Bearer fake")
        token=result["next_step_token"]
    assert result["passed"] is False
    assert result["points_added"]==0
    assert not store.awards

def test_locked_game_enforced_on_server(api):
    m,store=api
    with pytest.raises(HTTPException) as error:
        m.start(m.Start(game="debate-duel",level=0),authorization="Bearer fake")
    assert error.value.status_code==403
    assert error.value.detail["required_points"]==150

def test_replayed_step_rejected(api):
    m,store=api
    item=m.start(m.Start(game="mars-rescue",level=0),authorization="Bearer fake")
    token=item["step_token"]
    m.answer(item["session_id"],m.Answer(step=0,choice=0,step_token=token),authorization="Bearer fake")
    with pytest.raises(HTTPException) as error:
        m.answer(item["session_id"],m.Answer(step=0,choice=0,step_token=token),authorization="Bearer fake")
    assert error.value.status_code==409

def test_off_flag_disables_ranked_session(api,monkeypatch):
    m,_=api
    monkeypatch.delenv("VERIFYD_CIRCLE_MISSION_ARCADE_ENABLED")
    assert m.catalog()["enabled"] is False
    with pytest.raises(HTTPException) as error:
        m.start(m.Start(game="mars-rescue",level=0),authorization="Bearer fake")
    assert error.value.status_code==404
