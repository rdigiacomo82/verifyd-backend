"""Non-destructive public readiness check; no real identities or locations."""
import json
import sys
from urllib.error import HTTPError
from urllib.request import Request, urlopen

BASE = 'https://verifyd-backend.onrender.com/trust-voice/location'

def call(method, route, body=None, headers=None):
    payload = json.dumps(body).encode('utf-8') if body is not None else None
    request = Request(BASE + route, data=payload, method=method, headers={
        **({'Content-Type': 'application/json'} if payload else {}),
        **(headers or {})
    })
    try:
        with urlopen(request, timeout=40) as response:
            return response.status, dict(response.headers), response.read().decode()[:400]
    except HTTPError as error:
        return error.code, dict(error.headers), error.read().decode()[:400]

status, headers, body = call('GET', '/status')
print('GET status:', status, body)
if status != 200:
    sys.exit('Location readiness endpoint failed')
try:
    data = json.loads(body)
except ValueError:
    sys.exit('Location readiness did not return JSON')
if data != {'enabled': True, 'ready': True}:
    sys.exit('Location backend not yet enabled or schema initialization failed')

for method, route, payload, expected in [
    ('GET','/sharing',None,401),
    ('GET','/visible',None,401),
    ('POST','/native-token',None,401),
    ('DELETE','/sharing',None,401),
    ('POST','/native-position', {'latitude':0.0,'longitude':0.0,'time':None,'accuracy':None}, 401),
]:
    code, response_headers, response = call(method, route, payload)
    print(method, route, code)
    if code != expected:
        sys.exit(f'Expected {expected} from {method} {route}, got {code}: {response[:160]}')

code, hs, response = call('OPTIONS','/sharing',None,{
    'Origin':'https://vfvid.com',
    'Access-Control-Request-Method':'PUT',
    'Access-Control-Request-Headers':'authorization,content-type',
})
cors_headers = {str(k).lower(): v for k,v in hs.items()}
print('CORS preflight:',code, 'allow-origin', cors_headers.get('access-control-allow-origin'))
if code != 200 or cors_headers.get('access-control-allow-origin') != 'https://vfvid.com':
    sys.exit('Location frontend cannot pass cross-origin preflight')
print('PASS: feature enabled, database ready, unauthenticated access denied, CORS allowed')
