"""Server-authoritative Trust Voice identity blocking. A block suppresses direct contact both ways."""
import os
from datetime import datetime, timezone
from typing import Optional
from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field
from database import get_db

router = APIRouter(prefix="/trust-voice", tags=["Trust Voice Blocking"])

def blocking_enabled() -> bool:
    return os.environ.get("VERIFYD_CIRCLE_BLOCKING_ENABLED", "0").strip().lower() in ("1", "true", "yes", "on")

def require_blocking_enabled():
    if not blocking_enabled():
        raise HTTPException(status_code=503, detail={"error": "blocking_not_enabled"})

def ensure_block_schema():
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("""CREATE TABLE IF NOT EXISTS verifyd_voice_blocks (
            blocker_identity_id TEXT NOT NULL,
            blocked_identity_id TEXT NOT NULL,
            created_at TEXT NOT NULL,
            PRIMARY KEY (blocker_identity_id, blocked_identity_id),
            CHECK (blocker_identity_id <> blocked_identity_id)
        )""")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_verifyd_voice_blocks_target ON verifyd_voice_blocks (blocked_identity_id, blocker_identity_id)")

def identities_blocked(a: str, b: str) -> bool:
    if not blocking_enabled() or not a or not b or a == b:
        return False
    ensure_block_schema()
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("""SELECT 1 FROM verifyd_voice_blocks
            WHERE (blocker_identity_id=%s AND blocked_identity_id=%s)
               OR (blocker_identity_id=%s AND blocked_identity_id=%s)
            LIMIT 1""", (a,b,b,a))
        return bool(cur.fetchone())

def _actor(auth):
    from trust_voice_signaling import _identity_from_http_authorization
    return _identity_from_http_authorization(auth)

def _target(handle):
    from trust_voice_signaling import _identity_by_handle, _normalize_handle
    user = _identity_by_handle(_normalize_handle(handle))
    if not user:
        raise HTTPException(404, detail={"error":"handle_not_found"})
    return user

class BlockInput(BaseModel):
    handle: str = Field(min_length=3, max_length=25)

@router.get("/blocked-users")
def blocked_users(authorization: Optional[str]=Header(default=None)):
    identity = _actor(authorization)
    require_blocking_enabled()
    ensure_block_schema()
    with get_db() as conn:
        cur=conn.cursor()
        cur.execute("""SELECT b.blocked_identity_id AS identity_id, i.handle, i.display_name, b.created_at
            FROM verifyd_voice_blocks b
            LEFT JOIN verifyd_identities i ON i.id=b.blocked_identity_id
            WHERE b.blocker_identity_id=%s ORDER BY b.created_at DESC""",(str(identity["id"]),))
        rows=[dict(row) for row in (cur.fetchall() or [])]
    return {"ok":True,"blocked_users":[{"identity_id":r["identity_id"],"handle":"@"+str(r.get("handle") or ""),"display_name":r.get("display_name") or "","created_at":r["created_at"]} for r in rows]}

@router.post("/blocked-users")
async def block_user(payload:BlockInput, authorization:Optional[str]=Header(default=None)):
    actor=_actor(authorization)
    require_blocking_enabled()
    target=_target(payload.handle)
    a,b=str(actor["id"]),str(target["id"])
    if a==b:
        raise HTTPException(400,detail={"error":"cannot_block_self"})
    ensure_block_schema()
    with get_db() as conn:
        cur=conn.cursor()
        cur.execute("""INSERT INTO verifyd_voice_blocks (blocker_identity_id,blocked_identity_id,created_at)
            VALUES (%s,%s,%s) ON CONFLICT DO NOTHING""",(a,b,datetime.now(timezone.utc).isoformat()))
    # Terminate active same-process calls immediately, notifying both peers.
    # Cross-worker teardown requires a distributed event bus in a later hardening pass.
    from trust_voice_signaling import (
        _active_calls, _state_lock, _cancel_call_expiry,
        _record_call_terminal, _send_to_identity,
    )
    async with _state_lock:
        affected = [(call_id, dict(call)) for call_id,call in list(_active_calls.items())
            if {str(call.get("caller_id") or ""),str(call.get("callee_id") or "")} == {a,b}]
        for call_id,_ in affected:
            _active_calls.pop(call_id,None)
    for call_id,call in affected:
        _cancel_call_expiry(call_id)
        _record_call_terminal(call_id,"ended","blocked")
        for who in (a,b):
            await _send_to_identity(who,{"type":"call_ended","call_id":call_id,"reason":"ended"})
    return {"ok":True,"blocked":True,"handle":"@"+str(target.get("handle") or "")}

@router.delete("/blocked-users/{handle}")
def unblock_user(handle:str,authorization:Optional[str]=Header(default=None)):
    actor=_actor(authorization)
    require_blocking_enabled()
    target=_target(handle)
    ensure_block_schema()
    with get_db() as conn:
        cur=conn.cursor()
        cur.execute("DELETE FROM verifyd_voice_blocks WHERE blocker_identity_id=%s AND blocked_identity_id=%s",
          (str(actor["id"]),str(target["id"])))
    return {"ok":True,"blocked":False}
