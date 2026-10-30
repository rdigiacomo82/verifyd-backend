"""Offline regression tests: safe member-triggered recovery of past received files."""
from __future__ import annotations
import ast
from pathlib import Path
import re
import unittest

EMAIL=Path(__file__).resolve().parents[1]/"trust_voice_circle_email.py"
MID="tvcmail_"+"a"*32
PID="2a0c9ce0-3112-4728-976e-47ddcd16a318"


class FakeHTTP(Exception):
    def __init__(self,code,detail=None):
        self.status_code=code
        self.detail=detail


class FakeCursor:
    def __init__(self,state):
        self.state=state
        self.last=""
        self.calls=[]
    def execute(self,sql,params):
        self.last=sql
        self.calls.append((sql,params))
    def fetchone(self):
        if "SELECT e.provider_email_id" in self.last:
            return {"provider_email_id":PID} if self.state['owner'] else None
        if "SELECT m.mail_id" in self.last:
            return {"mail_id":MID} if self.state['owner'] else None
        return None


class FakeDB:
    def __init__(self,st):self.cursor0=FakeCursor(st)
    def __enter__(self):return self
    def __exit__(self,*a):return False
    def cursor(self):return self.cursor0


def run_endpoint(configured=True,owner=True,already=False,provider_files=True):
    tree=ast.parse(EMAIL.read_text(encoding='utf8'))
    nodes=[x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=="recover_inbound_attachments"]
    assert len(nodes)==1
    fn=nodes[0]
    fn.decorator_list=[]
    fn.returns=None
    fn.args.defaults=[]
    for arg in fn.args.args:arg.annotation=None
    class RemoveImports(ast.NodeTransformer):
        def visit_ImportFrom(self,node):
            if node.module in ("trust_voice_circle_email_inbound","trust_voice_circle_email_transport"):
                return ast.Pass()
            return node
    fn=RemoveImports().visit(fn)
    ast.fix_missing_locations(fn)
    st=dict(owner=owner,already=already,provider_files=provider_files,retrieved=0,stored=0)
    db=FakeDB(st)
    def listing(_db,ids):
        return {MID:[{'file_id':'tvcfile_'+'c'*32,'filename':'sample.pdf',
                       'source':'external','malware_scanned':False}]} if st['already'] else {}
    def fetch(_pid,advertised):
        st['retrieved']+=1
        return ([{'filename':'sample.pdf','bytes':b'%PDF-test','content_type':'application/pdf'}],0) if provider_files else ([],0)
    def persist(_db,**kwargs):
        st['stored']+=1
        st['already']=True
        return ['circle-email/test']
    env=dict(
        feature=lambda:None,
        enabled=lambda k:configured,
        _storage_configured=lambda:True,
        fail=lambda code,msg,status=400: (_ for _ in ()).throw(FakeHTTP(status,{'error':code})),
        re=re,
        _identity_from_bearer=lambda _: {'id':'owner-1'},
        schema=lambda:None,
        ensure_inbound_schema=lambda:None,
        get_db=lambda:db,
        files_for=listing,
        _RESEND_ID=re.compile("^[a-zA-Z0-9-]{15,90}$"),
        fetch_inbound_attachment_files=fetch,
        persist_inbound_attachments=persist,
        utc=lambda:"2026-10-30T15:00:00+00:00",
        _get_attachment_client=lambda:None,
        TRUST_VOICE_BUCKET="test"
    )
    exec(compile(ast.Module(body=[fn],type_ignores=[]),str(EMAIL),"exec"),env)
    return env["recover_inbound_attachments"],st,db


class TestRecovery(unittest.TestCase):
    def test_staged_feature_is_closed_until_activated(self):
        call,st,db=run_endpoint(configured=False)
        with self.assertRaises(FakeHTTP) as ctx:
            call(MID, type("Input",(),{'acknowledge_unscanned':True})(), None)
        self.assertEqual(ctx.exception.status_code,503)
        self.assertEqual(st['retrieved'],0)

    def test_requires_explicit_unscanned_acknowledgement(self):
        call,st,db=run_endpoint()
        with self.assertRaises(FakeHTTP) as ctx:
            call(MID, type("Input",(),{'acknowledge_unscanned':False})(), None)
        self.assertEqual(ctx.exception.status_code,409)
        self.assertEqual(st['retrieved'],0)

    def test_cross_user_mail_cannot_recover_files(self):
        call,st,db=run_endpoint(owner=False)
        with self.assertRaises(FakeHTTP) as ctx:
            call(MID, type("Input",(),{'acknowledge_unscanned':True})(), None)
        self.assertEqual(ctx.exception.status_code,404)
        self.assertEqual(st['retrieved'],0)

    def test_retrieve_once_and_preserve_private_storage_metadata(self):
        call,st,db=run_endpoint()
        result=call(MID, type("Input",(),{'acknowledge_unscanned':True})(), None)
        self.assertEqual(result['status'],'recovered')
        self.assertEqual(st['retrieved'],1)
        self.assertEqual(st['stored'],1)
        self.assertEqual(result['attachments'][0]['filename'],'sample.pdf')
        self.assertFalse(result['attachments'][0]['malware_scanned'])
        result2=call(MID,type("Input",(),{'acknowledge_unscanned':True})(),None)
        self.assertEqual(result2['status'],'already_available')
        self.assertEqual(st['retrieved'],1)

    def test_no_eligible_provider_files_does_not_change_mail(self):
        call,st,db=run_endpoint(provider_files=False)
        result=call(MID, type("Input",(),{'acknowledge_unscanned':True})(), None)
        self.assertEqual(result['status'],'none_available')
        self.assertEqual(st['stored'],0)

    def test_private_route_only_and_network_fetch_outside_db_transaction(self):
        src=EMAIL.read_text(encoding='utf8')
        part=src[src.index("def recover_inbound_attachments("):
                 src.index("@router.post('/circle-email/messages/{mail_id}/action')")]
        self.assertIn('m.owner_id=%s',part)
        self.assertIn("m.delivery='received_external'",part)
        self.assertIn('r.owner_id=m.owner_id AND r.thread_id=m.thread_id',part)
        self.assertIn("m.sender_id=('external:' || r.external_recipient)",part)
        self.assertIn("r.revoked_at IS NULL",part)
        self.assertIn('VERIFYD_CIRCLE_EMAIL_INBOUND_ATTACHMENT_RECOVERY_ENABLED',part)
        self.assertIn('fetch_inbound_attachment_files(provider_id',part)
        self.assertIn('persist_inbound_attachments(',part)
        self.assertLess(part.index("fetched,omitted=fetch_inbound_attachment_files"),
                        part.index("c.execute(\"\"\"SELECT m.mail_id"))
        self.assertNotIn("DELETE FROM",part)

if __name__=="__main__":
    unittest.main()
