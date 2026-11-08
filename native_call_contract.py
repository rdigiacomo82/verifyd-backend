"""VeriFYD native incoming-call push protocol v1.

Pure payload formatting only: NO device registry, FCM/APNs credentials, delivery,
or production routing. The existing WebSocket call state remains authoritative.
"""
from __future__ import annotations

import re
import time
import uuid
from dataclasses import dataclass
from typing import Mapping

SCHEMA = "verifyd.native_call.v1"
KIND = "incoming_call"
INVITE_TTL_SECONDS = 30
CALL_ID_PATTERN = re.compile(r"^tvcall_[0-9a-f]{32}$", re.ASCII)
_NAMESPACE = uuid.UUID("fba684f5-f324-4e32-9329-63b0f6632691")


@dataclass(frozen=True)
class CallInvite:
    call_id: str
    call_uuid: str
    call_type: str
    expires_at: int


def build_invite(call_id: str, call_type: str, created_at: float, now: float | None = None) -> CallInvite:
    """Build with the *original* 30s deadline, never a fresh 30s."""
    if not CALL_ID_PATTERN.fullmatch(call_id):
        raise ValueError("Invalid VeriFYD call ID")
    if call_type not in {"audio", "video"}:
        raise ValueError("Unsupported call type")
    current = time.time() if now is None else float(now)
    deadline = float(created_at) + INVITE_TTL_SECONDS
    if not (current < deadline and 0 < deadline - current <= INVITE_TTL_SECONDS):
        raise ValueError("Call has expired or timestamps are invalid")
    return CallInvite(
        call_id=call_id,
        call_uuid=str(uuid.uuid5(_NAMESPACE, call_id)),
        call_type=call_type,
        expires_at=int(deadline),
    )


def fcm_data(invite: CallInvite) -> dict[str, str]:
    """FCM HTTP v1 message.data strings; transport must use HIGH priority."""
    return {
        "schema": SCHEMA,
        "kind": KIND,
        "call_id": invite.call_id,
        "call_uuid": invite.call_uuid,
        "call_type": invite.call_type,
        "expires_at": str(invite.expires_at),
    }


def apns_voip_payload(invite: CallInvite) -> dict[str, object]:
    """APNs VoIP body. Headers must specify push-type=voip, short expiration."""
    return {"aps": {"content-available": 1}, **fcm_data(invite)}


def validate_data(data: Mapping[str, str], now: int | None = None) -> CallInvite:
    """Reject stale or invalid invitations before native UI is requested.

    The native app MUST authenticate and recheck the server's live call status
    before accepting/declining/media setup. Push receipt grants no authority.
    """
    if data.get("schema") != SCHEMA or data.get("kind") != KIND:
        raise ValueError("Unsupported native call payload")
    call_id = data.get("call_id", "")
    call_type = data.get("call_type", "")
    if not CALL_ID_PATTERN.fullmatch(call_id) or call_type not in {"audio", "video"}:
        raise ValueError("Invalid call")
    expected_uuid = str(uuid.uuid5(_NAMESPACE, call_id))
    if data.get("call_uuid") != expected_uuid:
        raise ValueError("Invalid CallKit identifier")
    try:
        deadline = int(data.get("expires_at", ""))
    except (TypeError, ValueError) as exc:
        raise ValueError("Invalid expiry") from exc
    current = int(time.time()) if now is None else int(now)
    if not current < deadline <= current + INVITE_TTL_SECONDS:
        raise ValueError("Expired or implausibly long call invitation")
    return CallInvite(call_id, expected_uuid, call_type, deadline)
