"""Isolated contract checks for Universal Handle Registry Phase 2.

No real database, no external users and no migration side effects.
"""
import importlib.util
import os
import sys
import types
from pathlib import Path

import pytest
from fastapi import HTTPException

PATH=Path(__file__).resolve().parents[1] / "trust_voice_handle_registry_phase2.py"

class FakeStore:
    def __init__(self):
        self.registry={}
        self.privacy={}
        self.audit={}
        self.identities=[
            {"id":"vfyd_usr_one","handle_lower":"first_user","status":"active"},
            {"id":"vfyd_usr_two","handle_lower":"second_user","status":"active"}
        ]
        self.cursor_obj=FakeCursor(self)
    def __enter__(self):return self
    def __exit__(self,*args):return False
    def cursor(self):return self.cursor_obj

class FakeCursor:
    def __init__(self,store):self.store=store;self.one=None;self.rows=None
    def execute(self,sql,params=()):
        q=" ".join(sql.lower().split())
        s=self.store
        if q.startswith("create table") or q.startswith("create index"):
            self.one=None;self.rows=None
        elif q.startswith("select identity_id, handle_lower, allocation_state from verifyd_universal_handle_registry"):
            identity,handle=params
            self.rows=[{"identity_id":k,"handle_lower":v,"allocation_state":"existing_account"}
                for k,v in s.registry.items() if k==identity or v==handle]
        elif q.startswith("insert into verifyd_universal_handle_registry"):
            identity,namespace,handle=params
            if identity in s.registry or handle in s.registry.values():
                self.one=None
            else:
                s.registry[identity]=handle
                self.one={"identity_id":identity}
        elif q.startswith("insert into verifyd_universal_handle_audit"):
            identity,handle,identity2,handle2,actor=params
            assert identity==identity2 and handle==handle2
            s.audit.setdefault(identity+":"+handle,actor);self.one=None
        elif q.startswith("insert into verifyd_universal_handle_privacy"):
            uid=params[0]
            s.privacy.setdefault(uid,[False]*5);self.one=None
        elif q.startswith("select r.identity_id,r.handle_lower,r.allocation_state"):
            uid=params[0]
            if uid not in s.registry:
                self.one=None
            else:
                self.one=dict(identity_id=uid,handle_lower=s.registry[uid],allocation_state="existing_account",
                    public_discovery=s.privacy[uid][0],partner_discovery=s.privacy[uid][1],
                    public_call_route=s.privacy[uid][2],public_message_route=s.privacy[uid][3],
                    public_email_route=s.privacy[uid][4])
        elif q.startswith("select id,handle_lower,status from verifyd_identities"):
            self.rows=s.identities
        elif q.startswith("select identity_id,handle_lower from verifyd_universal_handle_registry"):
            self.rows=[dict(identity_id=k,handle_lower=v) for k,v in s.registry.items()]
        else:
            raise AssertionError("Unrecognized SQL: "+q)
    def fetchone(self):return self.one
    def fetchall(self):return self.rows or []

@pytest.fixture
def ctx(monkeypatch):
    db=FakeStore()
    database=types.ModuleType("database");database.get_db=lambda:db
    messages=types.ModuleType("trust_voice_messages")
    def identity(auth):
        who=auth.replace("Bearer ","") if auth else ""
        match=next((x for x in db.identities if x["id"]==who),None)
        if not match:
            raise HTTPException(401,detail="bad_auth")
        return match
    messages._identity_from_bearer=identity
    monkeypatch.setitem(sys.modules,"database",database)
    monkeypatch.setitem(sys.modules,"trust_voice_messages",messages)
    monkeypatch.setenv("VERIFYD_UNIVERSAL_HANDLE_REGISTRY_ENABLED","1")
    monkeypatch.delenv("VERIFYD_UNIVERSAL_HANDLE_REGISTRY_MAINTENANCE_ALLOWED",raising=False)
    spec=importlib.util.spec_from_file_location("_registry_phase2",PATH)
    module=importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules,spec.name,module)
    spec.loader.exec_module(module)
    return module,db

def test_default_off(ctx,monkeypatch):
    mod,store=ctx
    monkeypatch.setenv("VERIFYD_UNIVERSAL_HANDLE_REGISTRY_ENABLED","0")
    with pytest.raises(HTTPException) as exc:
        mod.my_registry_record(authorization="Bearer vfyd_usr_one")
    assert exc.value.status_code==404
    assert store.registry=={}

def test_seeds_legacy_identity_without_public_contact(ctx):
    mod,store=ctx
    record=mod.my_registry_record(authorization="Bearer vfyd_usr_one")
    assert record["identity_id"]=="vfyd_usr_one"
    assert record["handle"]=="@first_user"
    assert record["qualified_address"]=="acct:first_user@vfvid.com"
    assert record["discovery"]=={"public":False,"partners":False}
    assert record["routing"]=={"public_calls":False,"public_messages":False,"public_email":False}
    assert record["permissions_editable"] is False
    assert len(store.audit)==1

def test_reads_idempotent(ctx):
    mod,store=ctx
    mod.my_registry_record(authorization="Bearer vfyd_usr_one")
    mod.my_registry_record(authorization="Bearer vfyd_usr_one")
    assert len(store.registry)==len(store.audit)==len(store.privacy)==1

def test_account_isolation(ctx):
    mod,store=ctx
    record1=mod.my_registry_record(authorization="Bearer vfyd_usr_one")
    record2=mod.my_registry_record(authorization="Bearer vfyd_usr_two")
    assert record1["identity_id"]!=record2["identity_id"]
    assert len(store.registry)==2
    assert len(store.audit)==2

def test_no_ownership_takeover(ctx):
    mod,store=ctx
    store.registry["attacker"]="first_user"
    with pytest.raises(HTTPException) as exc:
        mod.my_registry_record(authorization="Bearer vfyd_usr_one")
    assert exc.value.status_code==409
    assert store.registry["attacker"]=="first_user"

def test_maintenance_is_gated(ctx):
    mod,_=ctx
    with pytest.raises(HTTPException) as exc:
        mod._bootstrap_legacy_accounts(dry_run=True)
    assert exc.value.status_code==403

def test_maintenance_dry_run_no_writes(ctx,monkeypatch):
    mod,store=ctx
    monkeypatch.setenv("VERIFYD_UNIVERSAL_HANDLE_REGISTRY_MAINTENANCE_ALLOWED","1")
    result=mod._bootstrap_legacy_accounts(dry_run=True)
    assert result["active_legacy_accounts"]==2
    assert result["new_records"]==2
    assert store.audit==store.registry==store.privacy=={}

def test_maintenance_seeds_with_separate_authorization(ctx,monkeypatch):
    mod,store=ctx
    monkeypatch.setenv("VERIFYD_UNIVERSAL_HANDLE_REGISTRY_MAINTENANCE_ALLOWED","1")
    result=mod._bootstrap_legacy_accounts(dry_run=False)
    assert result["new_records"]==2
    assert len(store.audit)==2
    assert set(store.audit.values())=={"authorized_maintenance"}
    assert mod._bootstrap_legacy_accounts(dry_run=False)["new_records"]==0

def test_legacy_handle_mismatch_refused(ctx):
    mod,store=ctx
    store.registry["vfyd_usr_one"]="other_name"
    with pytest.raises(HTTPException) as exc:
        mod.my_registry_record(authorization="Bearer vfyd_usr_one")
    assert exc.value.status_code==409

def test_health_never_claims_public_discovery(ctx):
    mod,_=ctx
    assert mod.health()["public_discovery_enabled"] is False
    assert mod.health()["handle_transfers_enabled"] is False
