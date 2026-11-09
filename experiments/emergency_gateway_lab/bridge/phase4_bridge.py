"""Local-only LiveKit <-> Asterisk echo demo. NEVER used for public 911/933.

This is intentionally not integrated into production VeriFYD Circle or Render.
Only an operator on the same Linux host can request the single local echo extension.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import hmac
import json
import os
from pathlib import Path
import secrets
import socket
import sys
import time
from typing import Any
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent
GENERATED = ROOT / '.generated'
CREDS = GENERATED / 'credentials.json'
ROOM = 'verifyd-private-echo'
KEY = 'verifyd_local_echo_lab'
LIVEKIT_HOST = '127.0.0.1'
LIVEKIT_URL = f'http://{LIVEKIT_HOST}:7880'
LOCAL_SIP_ENDPOINT = '127.0.0.1:5070'
ONLY_EXTENSION = '6000'


def _json_str(value: str) -> str:
    """Quote YAML scalars as JSON strings to avoid special-character interpolation."""
    return json.dumps(value)


def _private_file(path: Path, content: str) -> None:
    path.write_text(content, encoding='utf-8')
    if os.name == 'posix':
        path.chmod(0o600)


def init_lab() -> dict[str, str]:
    """Generate fresh local credentials once; never overwrite an existing lab's credentials."""
    GENERATED.mkdir(exist_ok=True)
    if os.name == 'posix':
        GENERATED.chmod(0o700)
    if CREDS.exists():
        raise RuntimeError('Lab already initialized: preserving existing local credentials; do not regenerate while containers are running.')
    secret = secrets.token_urlsafe(48)
    cfg: dict[str, str] = {'api_key': KEY, 'api_secret': secret}
    _private_file(CREDS, json.dumps(cfg, indent=2) + '\n')
    livekit = f'''# Generated private lab config. DO NOT USE AS PRODUCTION CONFIG.
port: 7880
bind_addresses:
  - 127.0.0.1
redis:
  address: 127.0.0.1:6379
rtc:
  udp_port: 7882
  use_external_ip: false
  node_ip: 127.0.0.1
keys:
  {KEY}: {_json_str(secret)}
log_level: info
'''
    sip = f'''# Local-only SIP service, no public trunk or 933/911 routing.
api_key: {KEY}
api_secret: {_json_str(secret)}
ws_url: ws://127.0.0.1:7880
redis:
  address: 127.0.0.1:6379
sip_port: 5060
listen_ip: 127.0.0.1
rtp_port: 11000-11099
use_external_ip: false
# 1.0 is a lab-only one-call threshold for resource-constrained ephemeral runners.
# Never apply this value to production SIP deployments.
max_cpu_utilization: 1.0
log_level: info
'''
    _private_file(GENERATED / 'livekit.yaml', livekit)
    _private_file(GENERATED / 'sip.yaml', sip)
    return {'ok': True, 'mode': 'PRIVATE_LINUX_LIVEKIT_ECHO_LAB', 'room': ROOM,
            'sip_target': f'LOCAL_ECHO_EXTENSION_{ONLY_EXTENSION}',
            'livekit_connected': False, 'carrier_connected': False,
            'real_933_call_placed': False, 'real_911_call_placed': False}


def _creds() -> dict[str, str]:
    if not CREDS.is_file():
        raise RuntimeError('Run `python bridge/phase4_bridge.py init` first. Do not use production API secrets.')
    value = json.loads(CREDS.read_text(encoding='utf-8'))
    if value.get('api_key') != KEY or len(value.get('api_secret', '')) < 32:
        raise RuntimeError('Invalid local lab credentials.')
    return value


def _jwt(payload: dict[str, Any], secret: str) -> str:
    def encode(data: dict[str, Any]) -> str:
        return base64.urlsafe_b64encode(json.dumps(data, separators=(',', ':'), sort_keys=True).encode()).decode().rstrip('=')
    head = encode({'alg': 'HS256', 'typ': 'JWT'})
    body = encode(payload)
    to_sign = f'{head}.{body}'.encode()
    mac = base64.urlsafe_b64encode(hmac.new(secret.encode(), to_sign, hashlib.sha256).digest()).decode().rstrip('=')
    return f'{head}.{body}.{mac}'


def generate_join_token() -> str:
    cred = _creds()
    now = int(time.time())
    return _jwt({'iss': cred['api_key'], 'sub': 'local-web-test-'+secrets.token_hex(5),
                 'iat': now, 'nbf': now-2, 'exp': now+300,
                 'video': {'room': ROOM, 'roomJoin': True, 'canPublish': True,
                           'canPublishSources': ['microphone'], 'canSubscribe': True}}, cred['api_secret'])


def echo_request() -> dict[str, Any]:
    """The extension, host and room are non-parameterized constants: no PSTN routing."""
    return {'trunk': {'hostname': LOCAL_SIP_ENDPOINT, 'destination_country': 'US'},
            'sip_number': '6001', 'sip_call_to': ONLY_EXTENSION, 'room_name': ROOM,
            'participant_identity': 'local-sip-echo',
            'participant_name': 'LOCAL ECHO TEST — NOT EMERGENCY',
            'wait_until_answered': True}


def _only_loopback() -> None:
    host, port = LOCAL_SIP_ENDPOINT.rsplit(':', 1)
    if host != '127.0.0.1' or int(port) != 5070 or LIVEKIT_HOST != '127.0.0.1':
        raise RuntimeError('Non-loopback host is prohibited in private lab.')
    if echo_request()['sip_call_to'] != '6000':
        raise RuntimeError('Only internal echo extension 6000 is permitted.')


def call_echo(*, confirmed: bool = False, timeout: int = 35) -> dict[str, Any]:
    """LiveKit Twirp CreateSIPParticipant toward Asterisk localhost ONLY."""
    _only_loopback()
    if not confirmed:
        raise PermissionError('Add --confirm-private-echo to place the local-only test call.')
    cred = _creds()
    now = int(time.time())
    admin_token = _jwt({'iss': cred['api_key'], 'sub': 'local-lab-operator', 'iat': now,
                        'nbf': now-2, 'exp': now+90, 'sip': {'call': True}}, cred['api_secret'])
    data = json.dumps(echo_request(), separators=(',', ':')).encode('utf-8')
    request = Request(f'{LIVEKIT_URL}/twirp/livekit.SIP/CreateSIPParticipant', data=data,
                      headers={'Authorization': f'Bearer {admin_token}', 'Content-Type': 'application/json'}, method='POST')
    with urlopen(request, timeout=timeout) as resp:
        body = resp.read(32768)
        result = json.loads(body)
    return {'ok': True, 'mode': 'LOCAL_ONLY_LIVEKIT_TO_ASTERISK_ECHO',
            'room': ROOM, 'internal_extension': ONLY_EXTENSION,
            'sip_participant': result, 'carrier_connected': False,
            'real_933_call_placed': False, 'real_911_call_placed': False}


def doctor() -> dict[str, Any]:
    _only_loopback()
    available = {}
    for name, port in [('redis', 6379), ('livekit', 7880), ('sip_signaling', 5060), ('asterisk', 5070)]:
        if name == 'sip_signaling' or name == 'asterisk':
            # UDP has no connection-based readiness. Status is checked via Docker logs.
            available[name] = 'UDP — inspect compose logs'
            continue
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.4)
            available[name] = sock.connect_ex(('127.0.0.1', port)) == 0
    return {'lab_config_initialized': CREDS.is_file(), 'services': available,
            'linux_host_network_required': True, 'carrier_connected': False,
            'real_933_call_placed': False, 'real_911_call_placed': False}


def main() -> None:
    parser = argparse.ArgumentParser(description='VeriFYD LAB ONLY. Private SIP echo, NEVER 911/933.')
    parser.add_argument('command', choices=['init', 'doctor', 'token', 'show-fixed-target', 'call-echo'])
    parser.add_argument('--confirm-private-echo', action='store_true')
    args = parser.parse_args()
    try:
        if args.command == 'init': result = init_lab()
        elif args.command == 'doctor': result = doctor()
        elif args.command == 'token':
            print('Short-lived browser token (expires after 5 minutes; do not share):')
            print(generate_join_token())
            return
        elif args.command == 'show-fixed-target': result = echo_request()
        else: result = call_echo(confirmed=args.confirm_private_echo)
        print(json.dumps(result, indent=2))
    except Exception as err:
        print(f'Private Lab Error: {err}', file=sys.stderr)
        raise SystemExit(1)


if __name__ == '__main__': main()