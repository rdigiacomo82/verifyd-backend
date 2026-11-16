"""Pure cryptographic contract tests; no external service, DB, FCM or production settings."""
import base64
import os
import secrets
import sys
import types
import unittest
from pathlib import Path
from uuid import uuid4
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# Pure contract tests use no database connection.
sys.modules.setdefault("database", types.SimpleNamespace(get_db=lambda: None))
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from fastapi import HTTPException
from trust_voice_android_bridge import (
    PACKAGE, enabled, message_to_sign, parse_public_key, verify_proof,
    _require_enabled, _unb64,
)


def b64(data): return base64.urlsafe_b64encode(data).decode().rstrip('=')


class BridgeContract(unittest.TestCase):
    def setUp(self):
        self.key = ec.generate_private_key(ec.SECP256R1())
        raw = self.key.public_key().public_bytes(serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo)
        self.pub = b64(raw)
        self.digest = parse_public_key(self.pub)[1]
        self.iid = str(uuid4())
        self.intent = 'nbi_' + secrets.token_urlsafe(24)
        self.nonce = secrets.token_urlsafe(32)

    def sign(self, **changes):
        data = dict(intent_id=self.intent, challenge=self.nonce,
                    installation_id=self.iid, package_name=PACKAGE)
        data.update(changes)
        message = message_to_sign(data['intent_id'], data['challenge'], data['installation_id'], data['package_name'],self.digest)
        return b64(self.key.sign(message,ec.ECDSA(hashes.SHA256())))

    def prove(self, signature, **changes):
        data = dict(signature_b64=signature,public_key_spki_b64=self.pub,
                    intent_id=self.intent,challenge=self.nonce,
                    installation_id=self.iid,package_name=PACKAGE)
        data.update(changes)
        return verify_proof(**data)

    def test_valid(self): self.assertEqual(self.prove(self.sign()),self.digest)
    def test_wrong_nonce(self):
        with self.assertRaises(ValueError): self.prove(self.sign(),challenge=secrets.token_urlsafe(32))
    def test_wrong_installation(self):
        with self.assertRaises(ValueError): self.prove(self.sign(),installation_id=str(uuid4()))
    def test_wrong_package(self):
        with self.assertRaises(ValueError): self.prove(self.sign(),package_name='com.verifyd.circle')
    def test_wrong_key(self):
        other=ec.generate_private_key(ec.SECP256R1()).public_key().public_bytes(serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo)
        with self.assertRaises(ValueError): self.prove(self.sign(), public_key_spki_b64=b64(other))
    def test_malformed_key(self):
        with self.assertRaises(ValueError): parse_public_key('not-a-key')
    def test_wrong_curve(self):
        raw=ec.generate_private_key(ec.SECP384R1()).public_key().public_bytes(serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo)
        with self.assertRaises(ValueError):parse_public_key(b64(raw))
    def test_noncanonical_b64(self):
        with self.assertRaises(ValueError): _unb64('AQ=', minimum=1, maximum=10)
    def test_invalid_intent(self):
        with self.assertRaises(ValueError): self.prove(self.sign(),intent_id='nbi_invalid')
    def test_default_off(self):
        with patch.dict(os.environ,{'VERIFYD_ANDROID_NATIVE_BRIDGE_ENABLED':'0'}):
            self.assertFalse(enabled())
            with self.assertRaises(HTTPException) as ex: _require_enabled()
            self.assertEqual(ex.exception.status_code,404)
    def test_malformed_base64url_length_rejected_cleanly(self):
        # Schema-length-valid malformed input must not become an HTTP 500.
        with self.assertRaises(ValueError): parse_public_key('A' * 101)

    def test_no_embedded_web_session(self):
        self.assertNotIn('Bearer', message_to_sign(self.intent,self.nonce,self.iid,PACKAGE,self.digest).decode())

if __name__ == '__main__':unittest.main()
