"""Isolated contract tests for per-account Community Board acknowledgements.

No database, no HTTP, no real user identities, and no production changes.
"""
import importlib.util
import sys
import types
from datetime import datetime, timezone
from pathlib import Path

import pytest
from fastapi import HTTPException

PATH = Path(__file__).resolve().parents[1] / "trust_voice_community_consents.py"


class MemoryDB:
    def __init__(self):
        self.accepted = {}
        self.cur = Cursor(self)
    def __enter__(self): return self
    def __exit__(self,*args): return False
    def cursor(self): return self.cur

class Cursor:
    def __init__(self,db):
        self.db=db
        self.row=None
    def execute(self,query,params=()):
        q=" ".join(query.lower().split())
        if q.startswith("create table"):
            self.row=None
        elif q.startswith("select 1 from verifyd_voice_community_acknowledgements"):
            who,key,version=params
            self.row={"?column?":1} if (who,key,version) in self.db.accepted else None
        elif q.startswith("insert into verifyd_voice_community_acknowledgements"):
            who,key,version=params
            self.db.accepted.setdefault((who,key,version),datetime.now(timezone.utc))
            self.row=None
        elif q.startswith("select acknowledged_at from verifyd_voice_community_acknowledgements"):
            who,key,version=params
            when=self.db.accepted.get((who,key,version))
            self.row={"acknowledged_at":when} if when else None
        else:
            raise AssertionError("Unexpected query: "+q)
    def fetchone(self):return self.row

@pytest.fixture()
def setup(monkeypatch):
    db=MemoryDB()
    data=types.ModuleType("database");data.get_db=lambda:db
    messages=types.ModuleType("trust_voice_messages")
    def actor(header):
        who=(header or "").removeprefix("Bearer ").strip()
        return {"id":who,"status":"inactive" if who=="inactive" else "active"}
    messages._identity_from_bearer=actor
    monkeypatch.setitem(sys.modules,"database",data)
    monkeypatch.setitem(sys.modules,"trust_voice_messages",messages)
    monkeypatch.setenv("VERIFYD_TRUST_VOICE_COMMUNITY_ENABLED","1")
    monkeypatch.delenv("VERIFYD_CIRCLE_BOARD_ACK_ENFORCEMENT_ENABLED",raising=False)
    spec=importlib.util.spec_from_file_location("_circle_ack_tests",PATH)
    module=importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules,spec.name,module)
    spec.loader.exec_module(module)
    return module,db

def test_first_open_requires_acceptance(setup):
    mod,_=setup
    state=mod.get_acknowledgements(authorization="Bearer user1")
    assert state["acknowledged"]=={"rules":False,"public_notice":False}
    assert state["versions"]["rules"]==mod.RULES_VERSION

def test_rules_accepted_once_and_server_recorded(setup):
    mod,db=setup
    payload=mod.Acknowledge(notice_key="rules",version=mod.RULES_VERSION)
    result=mod.acknowledge(payload,authorization="Bearer user1")
    assert result["acknowledged"] is True
    assert result["acknowledged_at"]
    mod.acknowledge(payload,authorization="Bearer user1")
    assert len(db.accepted)==1
    state=mod.get_acknowledgements(authorization="Bearer user1")
    assert state["acknowledged"]["rules"] is True
    assert state["acknowledged"]["public_notice"] is False

def test_notice_is_separate_from_rules(setup):
    mod,db=setup
    mod.acknowledge(mod.Acknowledge(notice_key="public_notice",
        version=mod.PUBLIC_NOTICE_VERSION),authorization="Bearer user1")
    state=mod.get_acknowledgements(authorization="Bearer user1")
    assert state["acknowledged"]=={"rules":False,"public_notice":True}
    assert len(db.accepted)==1

def test_accounts_have_separate_acknowledgements(setup):
    mod,db=setup
    mod.acknowledge(mod.Acknowledge(notice_key="rules",
        version=mod.RULES_VERSION),authorization="Bearer user1")
    state=mod.get_acknowledgements(authorization="Bearer user2")
    assert state["acknowledged"]["rules"] is False
    assert len(db.accepted)==1

def test_policy_version_change_does_not_inherit_old_acceptance(setup):
    mod,db=setup
    mod.acknowledge(mod.Acknowledge(notice_key="rules",
        version=mod.RULES_VERSION),authorization="Bearer user1")
    mod.CURRENT_VERSIONS["rules"]="community-rules-future-test"
    state=mod.get_acknowledgements(authorization="Bearer user1")
    assert state["acknowledged"]["rules"] is False

def test_stale_version_rejected(setup):
    mod,_=setup
    with pytest.raises(HTTPException) as e:
        mod.acknowledge(mod.Acknowledge(notice_key="rules",version="old"),
                        authorization="Bearer user1")
    assert e.value.status_code==409

def test_enforcement_is_disabled_during_frontend_rollout(setup):
    mod,_=setup
    mod.verify_post_acknowledgements("user1",is_public=True)

def test_enabled_enforcement_requires_acknowledgements(setup,monkeypatch):
    mod,_=setup
    monkeypatch.setenv("VERIFYD_CIRCLE_BOARD_ACK_ENFORCEMENT_ENABLED","1")
    with pytest.raises(HTTPException) as e:
        mod.verify_post_acknowledgements("user1",is_public=False)
    assert e.value.status_code==403
    mod.acknowledge(mod.Acknowledge(notice_key="rules",version=mod.RULES_VERSION),
                    authorization="Bearer user1")
    mod.verify_post_acknowledgements("user1",is_public=False)
    with pytest.raises(HTTPException) as e2:
        mod.verify_post_acknowledgements("user1",is_public=True)
    assert e2.value.status_code==403
    mod.acknowledge(mod.Acknowledge(notice_key="public_notice",
                                    version=mod.PUBLIC_NOTICE_VERSION),authorization="Bearer user1")
    mod.verify_post_acknowledgements("user1",is_public=True)

def test_inactive_user_cannot_accept(setup):
    mod,_=setup
    with pytest.raises(HTTPException) as e:
        mod.acknowledge(mod.Acknowledge(notice_key="rules",version=mod.RULES_VERSION),
                        authorization="Bearer inactive")
    assert e.value.status_code==401

def test_feature_off_blocks_acknowledgements(setup,monkeypatch):
    mod,_=setup
    monkeypatch.delenv("VERIFYD_TRUST_VOICE_COMMUNITY_ENABLED")
    with pytest.raises(HTTPException) as e:
        mod.get_acknowledgements(authorization="Bearer user1")
    assert e.value.status_code==503
