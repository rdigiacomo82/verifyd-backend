"""List and revoke persistent Circle sessions without exposing bearer credentials."""
from typing import Optional
from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel
from database import get_db
from trust_voice_signaling import _token_from_authorization, _verify_token, ensure_persistent_session_schema, _now_iso

router = APIRouter(prefix="/trust-voice", tags=["Trust Voice Devices"])

def _session_context(auth):
    token = _token_from_authorization(auth)
    payload = _verify_token(token)
    identity_id = str(payload.get("identity_id") or "")
    if not identity_id:
        raise HTTPException(401,detail={"error":"invalid_session"})
    return identity_id,str(payload.get("session_id") or "")

@router.get("/device-sessions")
def device_sessions(authorization:Optional[str]=Header(default=None)):
    identity_id,current=_session_context(authorization)
    ensure_persistent_session_schema()
    with get_db() as conn:
        cur=conn.cursor()
        cur.execute("""SELECT session_id,created_at,last_seen_at FROM verifyd_voice_sessions
            WHERE identity_id=%s AND revoked_at IS NULL ORDER BY created_at DESC""",(identity_id,))
        rows=[dict(row) for row in (cur.fetchall() or [])]
    return {"ok":True,"devices":[{"session_id":row["session_id"],"created_at":row["created_at"],"last_seen_at":row["last_seen_at"],"is_current":row["session_id"]==current,"name":"Signed-in device"} for row in rows],
            "sessions":[{"session_id":row["session_id"],"created_at":row["created_at"],"last_seen_at":row["last_seen_at"],"is_current":row["session_id"]==current,"name":"Signed-in device"} for row in rows]}
class RevokeDevice(BaseModel):
    session_id: str

@router.post("/device-sessions/revoke")
def revoke_device(body:RevokeDevice, authorization:Optional[str]=Header(default=None)):
    identity_id,current=_session_context(authorization)
    if body.session_id==current:
        raise HTTPException(409,detail={"error":"use_sign_out_for_current_device"})
    ensure_persistent_session_schema()
    with get_db() as conn:
        cur=conn.cursor()
        cur.execute("""UPDATE verifyd_voice_sessions SET revoked_at=%s,revoked_reason='user_device_revoke'
            WHERE session_id=%s AND identity_id=%s AND revoked_at IS NULL""",(_now_iso(),body.session_id,identity_id))
        count=cur.rowcount
    if not count:
        raise HTTPException(404,detail={"error":"device_session_not_found"})
    return {"ok":True,"revoked":True}

@router.delete("/device-sessions/{session_id}")
def revoke_device_delete(session_id:str,authorization:Optional[str]=Header(default=None)):
    return revoke_device(RevokeDevice(session_id=session_id),authorization)
