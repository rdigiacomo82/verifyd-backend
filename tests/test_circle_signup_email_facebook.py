"""Regression checks for signup auto-session, OTP email and Facebook profiles.

No real emails, real user credentials, database connections, or external calls.
"""
import ast
import importlib.util
import pathlib
import re
import sys
import types
import unittest
from unittest.mock import patch
from urllib.parse import urlsplit,urlunsplit,parse_qs
from fastapi import HTTPException

ROOT = pathlib.Path(__file__).resolve().parents[1]


def extract_function(filepath, name, globals_dict):
    tree=ast.parse((ROOT/filepath).read_text(encoding="utf-8"))
    func=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
    exec(compile(ast.Module(body=[func],type_ignores=[]),str(ROOT/filepath),"exec"),globals_dict)
    return globals_dict[name]


class FacebookProfiles(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source=ast.parse((ROOT/"trust_voice_social_links.py").read_text(encoding="utf-8"))
        hosts=next(n for n in source.body if isinstance(n,ast.Assign)
                   and any(isinstance(t,ast.Name) and t.id=="HOSTS" for t in n.targets))
        env={"HOSTS":ast.literal_eval(hosts.value),"urlsplit":urlsplit,"urlunsplit":urlunsplit,
             "parse_qs":parse_qs,"re":re,"HTTPException":HTTPException}
        cls.normalize=staticmethod(extract_function("trust_voice_social_links.py","_normalize_url",env))

    def test_facebook_numeric_id_kept_with_tracking_dropped(self):
        self.assertEqual(self.normalize("facebook","https://m.facebook.com/profile.php?id=123456789&ref=share"),
                         "https://www.facebook.com/profile.php?id=123456789")

    def test_vanity_and_people_profiles(self):
        self.assertEqual(self.normalize("facebook","https://web.facebook.com/SomeUser/?ref=copy"),
                         "https://web.facebook.com/SomeUser")
        self.assertEqual(self.normalize("facebook","https://www.facebook.com/people/Example-Name/123456789"),
                         "https://www.facebook.com/people/Example-Name/123456789")
        self.assertEqual(self.normalize("facebook","https://fb.com/ExamplePage"),"https://fb.com/ExamplePage")

    def test_external_domains_and_share_redirects_rejected(self):
        for url in (
            "http://www.facebook.com/SomeUser",
            "https://www.facebook.com.evil.example/SomeUser",
            "https://facebook.com@evil.example/SomeUser",
            "https://www.facebook.com/profile.php?id=javascript:alert(1)",
            "https://www.facebook.com/profile.php?id=123&id=456",
            "https://www.facebook.com/share/p/1234567",
            "https://www.facebook.com/reel/123",
            "https://www.facebook.com/profile.php?id=",
        ):
            with self.subTest(url=url),self.assertRaises(HTTPException):
                self.normalize("facebook",url)

    def test_other_network_restrictions_unchanged(self):
        self.assertEqual(self.normalize("instagram","https://instagram.com/valid.handle?hl=en"),
                         "https://instagram.com/valid.handle")
        with self.assertRaises(HTTPException):
            self.normalize("instagram","https://instagram.com/someone/post/123")


class SignupSession(unittest.TestCase):
    def test_claim_creates_normal_revocable_token(self):
        called=[]
        def signer(p):
            called.append(p)
            return "signed-test-token"
        fake=types.ModuleType("trust_voice_signaling")
        fake._secret=lambda: called.append("signer_verified")
        fake._create_persistent_session=lambda i: "session-for-"+i
        fake._sign=signer
        with patch.dict(sys.modules,{"trust_voice_signaling":fake}):
            f=extract_function("trust_voice_identity.py","_issue_verified_claim_session",
                               {"time":types.SimpleNamespace(time=lambda:1234567890),
                                "secrets":types.SimpleNamespace(token_hex=lambda n:"c"*n)})
            response=f({"id":"vfyd_usr_test","handle_lower":"handle"})
        self.assertEqual(response["token"],"signed-test-token")
        self.assertEqual(response["persistent"],True)
        self.assertEqual(called[-1]["purpose"],"trust_voice_session")
        self.assertEqual(called[-1]["identity_id"],"vfyd_usr_test")
        self.assertEqual(called[-1]["session_id"],"session-for-vfyd_usr_test")
        self.assertIn("signer_verified",called)

    def test_signup_proof_precedes_session_and_old_login_is_preserved(self):
        source=(ROOT/"trust_voice_identity.py").read_text()
        start=source.index("def claim_identity(")
        code=source[start:]
        self.assertLess(code.index("verify_otp("),code.index("_issue_verified_claim_session("))
        self.assertIn('"signin_required": True',code)
        old=(ROOT/"trust_voice_signaling.py").read_text()
        self.assertIn('"/session/verify"',old)


class OtpEmail(unittest.TestCase):
    def test_multipart_accessible_email_without_real_delivery(self):
        spec=importlib.util.spec_from_file_location("testemailer",ROOT/"emailer.py")
        module=importlib.util.module_from_spec(spec)
        sys.modules[spec.name]=module
        spec.loader.exec_module(module)
        with patch.object(module,"_send",side_effect=lambda p: p) as send:
            payload=module.send_otp_email("test@example.invalid","123456")
        self.assertEqual(payload["subject"],"Your VeriFYD verification code")
        self.assertEqual(payload["from"],"VeriFYD Circle <support@vfvid.com>")
        self.assertIn("123456",payload["text"])
        self.assertIn("123456",payload["html"])
        self.assertIn("10 minutes",payload["text"])
        self.assertNotIn("AI Video Detection",payload["html"])
        self.assertEqual(payload["to"],["test@example.invalid"])


if __name__=="__main__":
    unittest.main()
