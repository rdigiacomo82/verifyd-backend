"""Standalone local-only FastAPI lab app. DO NOT mount in production main.py.

Every route is disabled unless explicitly enabled with a strong lab-only key.
No real 911/PSTN number, media, location, or dispatch route is implemented.
"""
from __future__ import annotations

import asyncio
import hmac
import os
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, ConfigDict, Field
from .lab_core import LabStore
from .sip_loopback import run_options_probe
from .simulated_933 import Scenario, simulate_933_readback

app = FastAPI(title='VeriFYD Emergency Gateway — LAB ONLY', docs_url=None, redoc_url=None,
              description='Test signaling only; not a 911 calling service.')
store = LabStore()


def _guard(key: str | None) -> None:
    if os.getenv('VERIFYD_EMERGENCY_LAB_ENABLED', '0') != '1':
        raise HTTPException(status_code=404, detail='lab_unavailable')
    expected = os.getenv('VERIFYD_EMERGENCY_LAB_KEY', '')
    if len(expected) < 32 or not key or not hmac.compare_digest(expected, key):
        raise HTTPException(status_code=401, detail='lab_key_required')


class Simulate933Body(BaseModel):
    model_config = ConfigDict(extra='forbid')
    scenario: Scenario = 'provisioned'


class StartBody(BaseModel):
    model_config = ConfigDict(extra='forbid')
    # No free-form destinations, SIP addresses, phone numbers, or coordinates.
    synthetic_location: str = Field(default='SYNTHETIC_LAB_LOCATION', pattern='^SYNTHETIC_LAB_LOCATION$')


@app.get('/lab/console', response_class=HTMLResponse)
def console():
    # The HTML never includes a credential and contains only lab-only actions.
    if os.getenv('VERIFYD_EMERGENCY_LAB_ENABLED', '0') != '1':
        raise HTTPException(status_code=404, detail='lab_unavailable')
    file = Path(__file__).with_name('lab_console.html')
    return HTMLResponse(file.read_text(encoding='utf-8'), headers={
        'Cache-Control': 'no-store',
        'Content-Security-Policy': "default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; connect-src 'self' ws://127.0.0.1:8099; base-uri 'none'; frame-ancestors 'none'",
        'X-Content-Type-Options': 'nosniff',
    })


@app.get('/lab/health')
def health(x_verifyd_lab_key: str | None = Header(default=None)):
    _guard(x_verifyd_lab_key)
    return {'ok': True, 'mode': 'synthetic_test_only', 'real_911_enabled': False,
            'sip_invite_enabled': False, 'audio_enabled': False}


@app.post('/lab/sessions')
def start(body: StartBody, x_verifyd_lab_key: str | None = Header(default=None)):
    _guard(x_verifyd_lab_key)
    try:
        item = store.create(location_label=body.synthetic_location)
    except OverflowError:
        raise HTTPException(status_code=429, detail='lab_capacity_reached')
    return item.public()


def _session_or_404(sid: str):
    obj = store.get(sid)
    if not obj:
        raise HTTPException(status_code=404, detail='lab_session_not_found')
    return obj


@app.get('/lab/sessions/{sid}')
def status(sid: str, x_verifyd_lab_key: str | None = Header(default=None)):
    _guard(x_verifyd_lab_key)
    return _session_or_404(sid).public()


@app.post('/lab/sessions/{sid}/sip-probe')
async def sip_probe(sid: str, x_verifyd_lab_key: str | None = Header(default=None)):
    _guard(x_verifyd_lab_key)
    item = _session_or_404(sid)
    if item.status == 'closed':
        raise HTTPException(status_code=409, detail='lab_session_closed')
    result = await run_options_probe()
    return store.set_probe(sid, 'sip_options_acknowledged' if result else 'sip_probe_unavailable').public()


@app.post('/lab/sessions/{sid}/simulate-933')
def simulate_933(sid: str, body: Simulate933Body,
                 x_verifyd_lab_key: str | None = Header(default=None)):
    _guard(x_verifyd_lab_key)
    item = _session_or_404(sid)
    if item.status == 'closed':
        raise HTTPException(status_code=409, detail='lab_session_closed')
    result = simulate_933_readback(body.scenario)
    result['session_id'] = item.id
    return result


@app.post('/lab/sessions/{sid}/close')
def close(sid: str, x_verifyd_lab_key: str | None = Header(default=None)):
    _guard(x_verifyd_lab_key)
    try:
        return store.close(sid).public()
    except KeyError:
        raise HTTPException(status_code=404, detail='lab_session_not_found')


@app.websocket('/lab/ws/{sid}')
async def ws_lab(websocket: WebSocket, sid: str):
    # Browser WebSocket clients cannot supply arbitrary headers; the first
    # frame must authenticate. This is NOT a production user-auth mechanism.
    if os.getenv('VERIFYD_EMERGENCY_LAB_ENABLED', '0') != '1':
        await websocket.close(code=1008)
        return
    await websocket.accept()
    try:
        hello = await asyncio.wait_for(websocket.receive_json(), timeout=3)
        if not isinstance(hello, dict) or hello.get('type') != 'lab_auth':
            await websocket.close(code=1008)
            return
        try:
            _guard(hello.get('key'))
        except HTTPException:
            await websocket.close(code=1008)
            return
        obj = store.get(sid)
        if obj is None:
            await websocket.send_json({'type': 'error', 'error': 'session_not_found'})
            await websocket.close(code=1008)
            return
        await websocket.send_json({'type': 'lab_ready', 'session': obj.public()})
        while True:
            message = await websocket.receive_json()
            if not isinstance(message, dict):
                await websocket.send_json({'type': 'error', 'error': 'invalid_message'})
            elif message.get('type') == 'status':
                await websocket.send_json({'type': 'status', 'session': _session_or_404(sid).public()})
            elif message.get('type') == 'sip_probe':
                if _session_or_404(sid).status == 'closed':
                    await websocket.send_json({'type': 'error', 'error': 'lab_session_closed'})
                    continue
                result = await run_options_probe()
                item = store.set_probe(sid, 'sip_options_acknowledged' if result else 'sip_probe_unavailable')
                await websocket.send_json({'type': 'sip_probe_result', 'session': item.public()})
            elif message.get('type') == 'simulate_933':
                try:
                    # Reject even unused caller, address and routing fields.
                    # A simulated test must not silently accept real identities.
                    if set(message) - {'type', 'scenario'}:
                        raise ValueError('Only synthetic scenario fields allowed')
                    session = _session_or_404(sid)
                    if session.status == 'closed':
                        await websocket.send_json({'type': 'error', 'error': 'lab_session_closed'})
                        continue
                    scenario = message.get('scenario', 'provisioned')
                    result = simulate_933_readback(scenario)
                    result['session_id'] = session.id
                    await websocket.send_json({'type': 'simulated_933_result', 'result': result})
                except ValueError:
                    await websocket.send_json({'type': 'error', 'error': 'synthetic_scenario_only'})
            elif message.get('type') == 'close':
                await websocket.send_json({'type': 'session_closed', 'session': store.close(sid).public()})
                await websocket.close(code=1000)
                return
            else:
                await websocket.send_json({'type': 'error', 'error': 'test_actions_only'})
    except (WebSocketDisconnect, asyncio.TimeoutError):
        return