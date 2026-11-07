"""VeriFYD Emergency Workflow LAB — account-authenticated TRAINING ONLY.

NO 911, PSAP, RapidSOS, Text-to-911, SMS, SIP routing or outbound webhook.
In-memory synthetic incidents are deliberately non-durable. Do not use this
router to claim emergency coverage. Auth uses existing revocable Circle sessions.
"""
from __future__ import annotations

import os
import re
import secrets
import threading
import time
from collections import defaultdict
from typing import Literal

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, ConfigDict, Field

router = APIRouter(prefix="/trust-voice/emergency-workflow-lab", tags=["Emergency Workflow LAB — NO DISPATCH"])

SCENARIOS = ("medical", "fire", "safety", "other")
MAX_LIFETIME_S = 1800
MAX_MESSAGES = 24
MAX_ACTIVE_PER_MEMBER = 1
MAX_NEW_PER_HOUR = 8
_simulations: dict[str, dict] = {}
_created_at: dict[str, list[float]] = defaultdict(list)
_lock = threading.RLock()


def _allowed(authorization: str | None) -> str:
    if os.environ.get("VERIFYD_EMERGENCY_WORKFLOW_LAB_ENABLED", "").strip().lower() not in ("1", "true", "yes"):
        raise HTTPException(status_code=404, detail="not_found")
    permitted = {s.strip() for s in os.environ.get("VERIFYD_EMERGENCY_WORKFLOW_LAB_IDS", "").split(",") if s.strip()}
    if not permitted:
        raise HTTPException(status_code=404, detail="not_found")
    # Existing identity resolver checks persistent-session revocation before returning identity.
    from trust_voice_messages import _identity_from_bearer
    identity = _identity_from_bearer(authorization)
    identity_id = str(identity.get("id") or identity.get("identity_id") or "").strip()
    if not identity_id or identity_id not in permitted:
        raise HTTPException(status_code=404, detail="not_found")
    return identity_id


def _purge(now: float) -> None:
    for key, value in list(_simulations.items()):
        if now - value["_created_ts"] > MAX_LIFETIME_S:
            del _simulations[key]
    for owner in list(_created_at):
        _created_at[owner] = [ts for ts in _created_at[owner] if now - ts < 3600]
        if not _created_at[owner]:
            del _created_at[owner]


def _view(incident: dict) -> dict:
    return {
        "id": incident["id"],
        "scenario": incident["scenario"],
        "status": incident["status"],
        "created_at_unix": int(incident["_created_ts"]),
        "messages": list(incident["messages"]),
        "location": "SIMULATED_LOCATION_ONLY",
        "service_type": "TRAINING_SIMULATION_ONLY",
        "dispatch_connected": False,
        "public_emergency_message_delivered": False,
        "psap_acknowledged": False,
    }


def _get(incident_id: str, owner: str) -> dict:
    if not re.fullmatch(r"[0-9a-f]{32}", incident_id):
        raise HTTPException(status_code=404, detail="not_found")
    incident = _simulations.get(incident_id)
    if not incident or incident["owner"] != owner:
        raise HTTPException(status_code=404, detail="not_found")
    return incident


class CreateScenario(BaseModel):
    model_config = ConfigDict(extra="forbid")
    scenario: Literal["medical", "fire", "safety", "other"]
    confirm_training_only: Literal[True]


class TrainingMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    message: str = Field(min_length=1, max_length=400)


@router.get("/status")
def lab_status(authorization: str | None = Header(default=None)):
    _allowed(authorization)
    return {
        "enabled": True,
        "mode": "TRAINING_SIMULATION_ONLY",
        "service_available": False,
        "dispatch_connected": False,
        "real_911_call_placed": False,
        "real_911_text_sent": False,
        "emergency_provider_connected": False,
        "message": "Practice only. No emergency services will receive anything.",
    }


@router.post("/incidents", status_code=201)
def create_incident(payload: CreateScenario, authorization: str | None = Header(default=None)):
    owner = _allowed(authorization)
    now = time.time()
    with _lock:
        _purge(now)
        if len(_created_at.get(owner, ())) >= MAX_NEW_PER_HOUR:
            raise HTTPException(status_code=429, detail="training_rate_limit")
        if any(v["owner"] == owner and v["status"] != "simulation_closed" for v in _simulations.values()):
            raise HTTPException(status_code=409, detail="close_current_training_first")
        incident_id = secrets.token_hex(16)
        incident = {
            "id": incident_id, "owner": owner, "scenario": payload.scenario,
            "status": "simulation_open", "_created_ts": now,
            "messages": [{
                "sender": "training_system", "message": "TRAINING ONLY: No 911 center, dispatcher, or trusted contact has been notified. Use fictitious information only."
            }],
        }
        _simulations[incident_id] = incident
        _created_at[owner].append(now)
        return _view(incident)


@router.get("/incidents/{incident_id}")
def read_incident(incident_id: str, authorization: str | None = Header(default=None)):
    owner = _allowed(authorization)
    with _lock:
        _purge(time.time())
        return _view(_get(incident_id, owner))


@router.post("/incidents/{incident_id}/messages")
def post_training_message(incident_id: str, payload: TrainingMessage, authorization: str | None = Header(default=None)):
    owner = _allowed(authorization)
    content = payload.message.strip()
    if not content:
        raise HTTPException(status_code=422, detail="message_required")
    with _lock:
        _purge(time.time())
        incident = _get(incident_id, owner)
        if incident["status"] == "simulation_closed":
            raise HTTPException(status_code=409, detail="training_closed")
        if len(incident["messages"]) >= MAX_MESSAGES:
            raise HTTPException(status_code=429, detail="training_message_limit")
        incident["messages"].append({"sender": "member_demo", "message": content})
        if incident["status"] == "simulation_open":
            incident["status"] = "simulation_acknowledged"
            incident["messages"].append({
                "sender": "simulated_responder",
                "message": "SIMULATED REPLY — THIS IS NOT 911: Your practice message appeared in the training room. No real emergency responder has received it."
            })
        return _view(incident)


@router.post("/incidents/{incident_id}/close")
def close_training(incident_id: str, authorization: str | None = Header(default=None)):
    owner = _allowed(authorization)
    with _lock:
        _purge(time.time())
        incident = _get(incident_id, owner)
        incident["status"] = "simulation_closed"
        return _view(incident)