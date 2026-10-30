"""Protect persistent sessions from temporary PostgreSQL pool exhaustion."""
from __future__ import annotations
import ast
import base64
import hashlib
import hmac
import json
import pathlib
import time
import unittest
from types import SimpleNamespace

ROOT=pathlib.Path(__file__).resolve().parents[1]
SIGNAL=ROOT/"trust_voice_signaling.py"
DB=ROOT/"database.py"
SECRET=b"S"*42


class HttpError(Exception):
    def __init__(self,status_code,detail):
        self.status_code=status_code
        self.detail=detail
        super().__init__(str(detail))


class DbFailure(Exception):
    pass


def b64(b):
    return base64.urlsafe_b64encode(b).decode().rstrip("=")


def make_token():
    payload={"purpose":"trust_voice_session","identity_id":"test-owner",
             "session_id":"tvsession_test"}
    encoded=b64(json.dumps(payload).encode("utf8"))
    sig=b64(hmac.new(SECRET,encoded.encode("ascii"),hashlib.sha256).digest())
    return encoded+"."+sig


def function_with_db(lookup):
    tree=ast.parse(SIGNAL.read_text(encoding="utf-8"))
    found=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="_verify_token"]
    assert len(found)==1
    n=found[0]
    n.returns=None
    for arg in n.args.args:arg.annotation=None
    # Use real signature checks with a pure fake DB to exercise both
    # temporary connectivity errors and legitimately invalid sessions.
    env=dict(
        hmac=hmac,hashlib=hashlib,json=json,time=time,
        _secret=lambda:SECRET,
        _b64url_decode=lambda s:base64.urlsafe_b64decode(s+"="*(-len(s)%4)),
        _persistent_session_row=lookup,
        HTTPException=HttpError,
        psycopg2=SimpleNamespace(Error=DbFailure,pool=SimpleNamespace(PoolError=DbFailure)),
    )
    exec(compile(ast.Module(body=[n],type_ignores=[]),str(SIGNAL),"exec"),env)
    return env["_verify_token"]


class SessionReliabilityTests(unittest.TestCase):
    def test_pool_exhaustion_returns_retriable_503_not_logout_401(self):
        def unavailable(_):
            raise DbFailure("connection pool exhausted")
        verify=function_with_db(unavailable)
        with self.assertRaises(HttpError) as ctx:
            verify(make_token())
        self.assertEqual(ctx.exception.status_code,503)
        self.assertEqual(ctx.exception.detail["error"],"session_check_unavailable")
        self.assertNotIn("invalid_session",ctx.exception.detail["error"])

    def test_revoked_session_stays_invalid(self):
        verify=function_with_db(lambda _:dict(revoked_at="2026-10-30",identity_id="test-owner"))
        with self.assertRaises(HttpError) as ctx:
            verify(make_token())
        self.assertEqual(ctx.exception.status_code,401)

    def test_missing_session_stays_invalid(self):
        verify=function_with_db(lambda _:None)
        with self.assertRaises(HttpError) as ctx:
            verify(make_token())
        self.assertEqual(ctx.exception.status_code,401)

    def test_good_session_is_accepted(self):
        verify=function_with_db(lambda _:dict(revoked_at=None,identity_id="test-owner"))
        self.assertEqual(verify(make_token())["identity_id"],"test-owner")

    def test_bad_signature_is_still_rejected(self):
        verify=function_with_db(lambda _:dict(revoked_at=None,identity_id="test-owner"))
        with self.assertRaises(HttpError) as ctx:
            verify(make_token()[:-7]+"INVALID")
        self.assertEqual(ctx.exception.status_code,401)

    def test_websocket_handshake_distinguishes_invalid_token_from_busy_db(self):
        src=SIGNAL.read_text(encoding="utf8")
        handshake=src[src.index('async def signaling_ws('):src.index('identity_id = (',src.index('async def signaling_ws('))]
        self.assertIn('code=4401 if exc.status_code == 401 else 1013',handshake)
        self.assertIn('except (psycopg2.Error, psycopg2.pool.PoolError):',handshake)
        self.assertIn('await websocket.close(code=1013)',handshake)

    def test_pool_tolerates_short_peak_in_defaults(self):
        src=DB.read_text(encoding="utf8")
        self.assertIn('DB_POOL_MAX = _db_env_int(\n    "VERIFYD_DB_POOL_MAX",\n    16,',src)
        self.assertIn('DB_POOL_WAIT_SECONDS = _db_env_float(\n    "VERIFYD_DB_POOL_WAIT_SECONDS",\n    8.0,',src)
        self.assertIn('"_db_pool_gate"'.strip('"'),src)
        self.assertIn("_get_db_pool().putconn",src)

if __name__=="__main__":unittest.main()
