"""Fail-closed malware-quarantine and file-type regression tests."""
import importlib.util
import os
import pathlib
import sys
import types
import unittest
from unittest.mock import patch

ROOT=pathlib.Path(__file__).resolve().parents[1]

class QuarantineSafetyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        database=types.ModuleType("database")
        database.get_db=lambda: None
        classrooms=types.ModuleType("trust_voice_classrooms")
        for attr in ("_identity","_schema","_teacher","_member","_assignment"):
            setattr(classrooms,attr,lambda *a,**k: None)
        cls.modules=patch.dict(sys.modules,{"database":database,"trust_voice_classrooms":classrooms})
        cls.modules.start()
        spec=importlib.util.spec_from_file_location("quarantine_test",ROOT/"trust_voice_classroom_quarantine.py")
        cls.q=importlib.util.module_from_spec(spec)
        sys.modules[spec.name]=cls.q
        spec.loader.exec_module(cls.q)

    @classmethod
    def tearDownClass(cls):
        sys.modules.pop("quarantine_test",None)
        cls.modules.stop()

    def test_disallowed_extensions_and_mismatched_signatures(self):
        from fastapi import HTTPException
        for name,raw in (
            ("malware.exe",b"MZvalid"),("faked.pdf",b"this is not a pdf"),
            ("wrong.jpg",b"fake jpeg"),("wrong.png",b"no png"),
            ("wrong.docx",b"not a zip"),("wrong.txt",b"bad\x00text")
        ):
            with self.subTest(name=name),self.assertRaises(HTTPException):
                self.q.check_signature(name,raw)

    def test_known_file_signatures_pass_but_are_not_clean(self):
        cases=(
            ("sample.pdf",b"%PDF-1.4\n","application/pdf"),
            ("sample.jpg",b"\xff\xd8\xff00","image/jpeg"),
            ("sample.png",b"\x89PNG\r\n\x1a\n","image/png"),
            ("sample.docx",b"PK\x03\x04x","application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
            ("sample.txt","Hello from learner".encode(),"text/plain")
        )
        for name,raw,expected in cases:
            with self.subTest(name=name):
                self.assertEqual(self.q.check_signature(name,raw),expected)

    def test_untrusted_names_are_normalized(self):
        self.assertEqual(self.q.safe_name("../../do-not-open.pdf"),"do-not-open.pdf")
        self.assertEqual(self.q.safe_name("folder\\malware.exe.pdf"),"malware.exe.pdf")

    def test_quarantine_disabled_by_default_and_no_download_routes(self):
        with patch.dict(os.environ,{"VERIFYD_CIRCLE_CLASSROOM_ENABLED":"0",
                                     "VERIFYD_CIRCLE_CLASSROOM_QUARANTINE_UPLOAD_ENABLED":"1"}):
            self.assertFalse(self.q.upload_enabled())
        with patch.dict(os.environ,{"VERIFYD_CIRCLE_CLASSROOM_ENABLED":"1",
                                     "VERIFYD_CIRCLE_CLASSROOM_QUARANTINE_UPLOAD_ENABLED":"0"}):
            self.assertFalse(self.q.upload_enabled())
        with patch.dict(os.environ,{"VERIFYD_CIRCLE_CLASSROOM_ENABLED":"1",
                                     "VERIFYD_CIRCLE_CLASSROOM_QUARANTINE_UPLOAD_ENABLED":"1"}):
            self.assertTrue(self.q.upload_enabled())
            self.assertFalse(self.q.health()["malware_scanner_active"])
            self.assertFalse(self.q.health()["download_available"])
        self.assertFalse(any("download" in route.path or "preview" in route.path
                             for route in self.q.router.routes))

    def test_separate_bucket_is_required(self):
        from fastapi import HTTPException
        with patch.dict(os.environ,{"R2_ACCOUNT_ID":"x",
            "R2_CLASSROOM_ACCESS_KEY_ID":"y","R2_CLASSROOM_SECRET_ACCESS_KEY":"z",
            "R2_CLASSROOM_QUARANTINE_BUCKET":"same-bucket","R2_TRUST_VOICE_BUCKET":"same-bucket"}):
            with self.assertRaises(HTTPException) as ex:
                self.q.storage()
            self.assertEqual(ex.exception.status_code,503)

if __name__=="__main__":
    unittest.main()
