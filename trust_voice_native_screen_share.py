from __future__ import annotations

import hashlib
import json
import os
import uuid
from datetime import timedelta
from typing import Literal, Optional

from fastapi import APIRouter, Header, HTTPException
from livekit import api
from pydantic import BaseModel, Field

from database import get_db
from trust_voice_group_video import (
    _active_session,
    _configured as _livekit_configured,
    _enabled as _group_video_enabled,
    _livekit_key,
    _livekit_secret,
    _livekit_url,
    _room_exists,
)
from trust_voice_groups import _require_group_member
from trust_voice_messages import _identity_from_bearer
from trust_voice_signaling import ensure_call_history_schema

router = APIRouter(prefix="/trust-voice", tags=["Trust Voice Native Screen Share"])
FEATURE_VERSION = "0.1.0"
TOKEN_MINUTES = 20


class DirectConnectionRequest(BaseModel):
    role: Literal["publisher", "viewer"]
    client_instance_id: Optional[str] = Field(default=None, max_length=128)


class GroupConnectionRequest(BaseModel):
    client_instance_id: Optional[str] = Field(default=None, max_length=128)


def _enabled() -> bool:
    raw = os.environ.get("VERIFYD_TRUST_VOICE_NATIVE_SCREEN_SHARE_ENABLED")
    if raw is None:
        return _group_video_enabled()
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _require_available() -> None:
    if not _enabled():
        raise HTTPException(status_code=404, detail={"error": "not_found"})
    if not _livekit_configured():
        raise HTTPException(
            status_code=503,
            detail={
                "error": "native_screen_share_not_configured",
                "message": "Native screen sharing is not configured yet.",
            },
        )


def _direct_room_name(call_id: str) -> str:
    digest = hashlib.sha256(str(call_id).encode("utf-8")).hexdigest()[:32]
    return f"tvss_{digest}"


def _aux_identity(prefix: str, identity_id: str, scope_id: str, client_instance_id: Optional[str]) -> str:
    nonce = (client_instance_id or uuid.uuid4().hex).strip()[:128]
    digest = hashlib.sha256(
        f"{identity_id}|{scope_id}|{nonce}".encode("utf-8")
    ).hexdigest()[:36]
    return f"{prefix}_{digest}"


def _display_name(identity: dict) -> str:
    return str(
        identity.get("display_name")
        or identity.get("handle")
        or "VeriFYD Circle User"
    )[:120]


def _issue_token(
    *,
    room_name: str,
    participant_identity: str,
    identity: dict,
    scope_type: str,
    scope_id: str,
    role: Literal["publisher", "viewer"],
) -> str:
    can_publish = role == "publisher"
    metadata = json.dumps(
        {
            "kind": "verifyd_native_screen_share_v1",
            "scope_type": scope_type,
            "scope_id_hash": hashlib.sha256(scope_id.encode("utf-8")).hexdigest()[:20],
            "owner_identity_hash": hashlib.sha256(
                str(identity.get("id") or "").encode("utf-8")
            ).hexdigest()[:20],
            "owner_livekit_identity": "tvp_" + hashlib.sha256(
                str(identity.get("id") or "").encode("utf-8")
            ).hexdigest()[:32],
            "role": role,
        },
        separators=(",", ":"),
    )

    grants = api.VideoGrants(
        room_join=True,
        room=room_name,
        can_publish=can_publish,
        can_subscribe=role == "viewer",
        can_publish_data=False,
        can_publish_sources=["screen_share"] if can_publish else None,
    )

    return (
        api.AccessToken(_livekit_key(), _livekit_secret())
        .with_identity(participant_identity)
        .with_name(_display_name(identity))
        .with_metadata(metadata)
        .with_grants(grants)
        .with_ttl(timedelta(minutes=TOKEN_MINUTES))
        .to_jwt()
    )


def _direct_call_for_identity(call_id: str, identity_id: str) -> dict:
    ensure_call_history_schema()
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT call_id, caller_identity_id, callee_identity_id, call_type,
                   status, started_at, answered_at, ended_at
            FROM verifyd_voice_calls
            WHERE call_id=%s
            LIMIT 1
            """,
            (call_id,),
        )
        row = cur.fetchone()

    if not row:
        raise HTTPException(status_code=404, detail={"error": "call_not_found"})
    call = dict(row)
    if identity_id not in {
        str(call.get("caller_identity_id") or ""),
        str(call.get("callee_identity_id") or ""),
    }:
        raise HTTPException(status_code=403, detail={"error": "not_call_participant"})
    if str(call.get("call_type") or "audio") != "video":
        raise HTTPException(
            status_code=409,
            detail={"error": "video_call_required", "message": "Start a video call before sharing your screen."},
        )
    if str(call.get("status") or "") != "answered":
        raise HTTPException(
            status_code=409,
            detail={"error": "call_not_connected", "message": "The video call is not connected."},
        )
    return call


@router.get("/native-screen-share/health")
def native_screen_share_health():
    return {
        "status": "ok",
        "feature": "trust_voice_native_screen_share",
        "version": FEATURE_VERSION,
        "enabled": _enabled(),
        "configured": _livekit_configured(),
        "transport": "livekit_aux_screen_track_v1",
        "direct_call_audio_video_untouched": True,
        "publisher_sources": ["screen_share"],
        "token_minutes": TOKEN_MINUTES,
    }


@router.post("/calls/{call_id}/native-screen-share/connection")
def direct_native_screen_share_connection(
    call_id: str,
    body: DirectConnectionRequest,
    authorization: Optional[str] = Header(default=None),
):
    _require_available()
    identity = _identity_from_bearer(authorization)
    identity_id = str(identity.get("id") or "")
    _direct_call_for_identity(call_id, identity_id)

    room_name = _direct_room_name(call_id)
    participant_identity = _aux_identity(
        "tvssp" if body.role == "publisher" else "tvssv",
        identity_id,
        call_id,
        body.client_instance_id,
    )
    token = _issue_token(
        room_name=room_name,
        participant_identity=participant_identity,
        identity=identity,
        scope_type="direct_call",
        scope_id=call_id,
        role=body.role,
    )
    return {
        "ok": True,
        "transport": "native_livekit_v1",
        "server_url": _livekit_url(),
        "participant_token": token,
        "participant_identity": participant_identity,
        "token_expires_in_seconds": TOKEN_MINUTES * 60,
        "role": body.role,
    }


@router.post("/groups/{conversation_id}/video/native-screen-share/connection")
async def group_native_screen_share_connection(
    conversation_id: str,
    body: GroupConnectionRequest,
    authorization: Optional[str] = Header(default=None),
):
    _require_available()
    identity = _identity_from_bearer(authorization)
    identity_id = str(identity.get("id") or "")
    _require_group_member(conversation_id, identity_id)

    session = _active_session(conversation_id)
    if not session:
        raise HTTPException(
            status_code=409,
            detail={"error": "no_active_group_video", "message": "There is no active group video call to share into."},
        )
    room_name = str(session.get("room_name") or "")
    if not room_name or not await _room_exists(room_name):
        raise HTTPException(
            status_code=409,
            detail={"error": "no_active_group_video", "message": "This group video call has ended."},
        )

    scope_id = str(session.get("session_id") or conversation_id)
    participant_identity = _aux_identity(
        "tvgssp",
        identity_id,
        scope_id,
        body.client_instance_id,
    )
    token = _issue_token(
        room_name=room_name,
        participant_identity=participant_identity,
        identity=identity,
        scope_type="group_video",
        scope_id=scope_id,
        role="publisher",
    )
    return {
        "ok": True,
        "transport": "native_livekit_v1",
        "server_url": _livekit_url(),
        "participant_token": token,
        "participant_identity": participant_identity,
        "token_expires_in_seconds": TOKEN_MINUTES * 60,
        "role": "publisher",
        "session_id": session.get("session_id") or "",
        "group_id": conversation_id,
    }
