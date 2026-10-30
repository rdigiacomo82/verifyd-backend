"""Isolated bulk Circle email authorization and transaction regression tests."""
from __future__ import annotations

import ast
import pathlib
import re
import unittest
from types import SimpleNamespace

MAIL=pathlib.Path(__file__).resolve().parents[1]/'trust_voice_circle_email.py'
FILE_ID='tvcmail_'+'a'*32
OTHER_ID='tvcmail_'+'b'*32


class MailFailure(Exception):
    def __init__(self,code,msg,status=400):
        self.code=code
        self.status=status
        super().__init__(msg)


class FakeCursor:
    def __init__(self,rows):
        self.rows=rows
        self.commands=[]
    def execute(self,sql,values):
        self.commands.append((sql,values))
    def fetchall(self):
        # Database SELECT filters by owner_id; emulate that scoping exactly.
        owner,ids=self.commands[-1][1]
        return [r.copy() for r in self.rows
                if r['owner_id']==owner and r['mail_id'] in ids]


class FakeConnection:
    def __init__(self,rows):
        self.c=FakeCursor(rows)
    def __enter__(self):return self
    def __exit__(self,*args):return False
    def cursor(self):return self.c


def isolated(rows):
    tree=ast.parse(MAIL.read_text(encoding='utf-8'))
    nodes=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='bulk_action']
    assert len(nodes)==1
    node=nodes[0];node.decorator_list=[];node.returns=None
    for arg in (*node.args.args,*node.args.kwonlyargs):
        arg.annotation=None
    db=FakeConnection(rows)
    env=dict(
        Optional=__import__('typing').Optional,
        Header=lambda **kw:None,
        feature=lambda:None,
        _identity_from_bearer=lambda _: {'id':'owner-1'},
        schema=lambda:None,
        get_db=lambda:db,
        re=re,
        fail=lambda code,msg,status=400: (_ for _ in ()).throw(MailFailure(code,msg,status)),
        utc=lambda:'2026-10-30T00:00:00+00:00',
        _restore_folder=lambda _c,owner,row:
            ('sent' if row.get('previous_folder')=='sent' or row.get('sender_id')==owner else 'inbox'),
    )
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(MAIL),'exec'),env)
    return env['bulk_action'],db


def mail(mid=FILE_ID,owner='owner-1',folder='inbox',sender='external:s@gmail.com'):
    return dict(mail_id=mid,folder=folder,sender_id=sender,owner_id=owner)


class BulkTests(unittest.TestCase):
    def test_two_owned_messages_trash_together(self):
        action,db=isolated([mail(),mail(OTHER_ID,folder='sent',sender='owner-1')])
        data=action(SimpleNamespace(action='trash',mail_ids=[FILE_ID,OTHER_ID]),None)
        self.assertEqual(data['updated'],2)
        self.assertEqual(sum("SET folder='trash'" in sql for sql,args in db.c.commands),2)
        self.assertTrue(all('owner_id=%s' in sql for sql,args in db.c.commands))

    def test_restore_only_trashed_and_keep_sent_copy_in_sent(self):
        action,db=isolated([mail(folder='trash'),mail(OTHER_ID,folder='trash',sender='owner-1')])
        data=action(SimpleNamespace(action='restore',mail_ids=[FILE_ID,OTHER_ID]),None)
        self.assertEqual(data['updated'],2)
        updates=[x for x in db.c.commands if 'SET folder=%s' in x[0]]
        self.assertEqual([x[1][0] for x in updates],['inbox','sent'])

    def test_cross_account_message_cannot_be_bulk_deleted(self):
        action,db=isolated([mail(),mail(OTHER_ID,owner='other')])
        with self.assertRaises(MailFailure) as err:
            action(SimpleNamespace(action='trash',mail_ids=[FILE_ID,OTHER_ID]),None)
        self.assertEqual(err.exception.status,404)
        self.assertEqual(len(db.c.commands),1)  # no UPDATE committed

    def test_duplicate_ids_and_bad_ids_fail_before_query(self):
        action,db=isolated([mail()])
        for ids in ([FILE_ID,FILE_ID],['evil'],['tvcmail_'+'F'*32]):
            with self.subTest(ids=ids):
                with self.assertRaises(MailFailure):
                    action(SimpleNamespace(action='trash',mail_ids=ids),None)
        self.assertEqual(len(db.c.commands),0)

    def test_safety_contract_and_bounded_model(self):
        s=MAIL.read_text(encoding='utf-8')
        self.assertIn('class MailBulkUpdate(BaseModel):',s)
        self.assertIn("action: Literal['trash','restore','move','move_inbox','archive','unarchive','read','unread']",s)
        self.assertIn('Field(min_length=1, max_length=150)',s)
        self.assertIn("@router.post('/circle-email/messages/bulk-action')",s)
        self.assertIn('owner_id=%s AND mail_id=ANY(%s) FOR UPDATE',s)
        self.assertNotIn('DELETE FROM verifyd_circle_email_messages',s)
        self.assertNotIn('delete_object(',s[s.index("def bulk_action("):s.index("@router.post('/circle-email/files')")])


if __name__=='__main__':
    unittest.main()
