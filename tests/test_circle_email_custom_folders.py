"""Isolated custom mailbox-folder security and routing tests."""
from __future__ import annotations
import ast
import pathlib
import re
import unittest
from types import SimpleNamespace

ROOT=pathlib.Path(__file__).resolve().parents[1]
SOURCE=ROOT/'trust_voice_circle_email.py'

class MailError(Exception):
    def __init__(self, code, message, status=400):
        self.code=code
        self.status=status
        super().__init__(message)

class Cursor:
    def __init__(self, exists=False):
        self.exists=exists
        self.commands=[]
    def execute(self, sql, args):
        self.commands.append((sql,args))
    def fetchone(self):
        return {"?column?":1} if self.exists else None

def helpers():
    tree=ast.parse(SOURCE.read_text(encoding='utf-8'))
    names={'_folder_name','_custom_folder_id','_require_owned_folder','_restore_folder'}
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    assert len(nodes)==4
    for n in nodes:
        n.returns=None
        for arg in n.args.args:arg.annotation=None
    env=dict(
        re=re,
        fail=lambda code,msg,status=400: (_ for _ in ()).throw(MailError(code,msg,status)),
    )
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(SOURCE),'exec'),env)
    return env

class CustomFolderTests(unittest.TestCase):
    def test_names_are_normalized_and_bounded(self):
        fn=helpers()['_folder_name']
        self.assertEqual(fn('  Medical   Records  '),'Medical Records')
        for name in ('',' ','inbox','Inbox','Sent','Trash','<script>','x'*41):
            with self.subTest(name=name),self.assertRaises(MailError):fn(name)
        self.assertEqual(fn('SLP / Family'.replace('/','-')),'SLP - Family')

    def test_folder_ids_cannot_inject_sql(self):
        fn=helpers()['_custom_folder_id']
        self.assertTrue(fn('tvcmfolder_'+'a'*32))
        for bad in ('inbox','tvcmfolder_'+'F'*32,'custom:tvcmfolder_'+'a'*32,'tvcmfolder_'+'a'*32+' OR 1=1'):
            self.assertFalse(fn(bad))

    def test_ownership_lookup_scoped_to_user(self):
        fn=helpers()['_require_owned_folder']
        db=Cursor()
        with self.assertRaises(MailError) as err:fn(db,'ownerA','tvcmfolder_'+'a'*32)
        self.assertEqual(err.exception.status,404)
        self.assertEqual(db.commands[-1][1],('ownerA','tvcmfolder_'+'a'*32))

    def test_restore_returns_to_existing_custom_folder_only(self):
        fn=helpers()['_restore_folder']
        db=Cursor(exists=True)
        row=dict(previous_folder='custom:tvcmfolder_'+'a'*32,sender_id='external:x@example.com')
        self.assertEqual(fn(db,'ownerA',row),row['previous_folder'])
        self.assertEqual(db.commands[-1][1],('ownerA','tvcmfolder_'+'a'*32))
        self.assertEqual(fn(Cursor(exists=False),'ownerA',row),'inbox')
        self.assertEqual(fn(Cursor(),'ownerA',dict(previous_folder='sent',sender_id='ownerA')),'sent')

    def test_db_schema_and_owner_scoping(self):
        src=SOURCE.read_text(encoding='utf-8')
        self.assertIn('CREATE TABLE IF NOT EXISTS verifyd_circle_email_folders',src)
        self.assertIn('UNIQUE(owner_id,name_lower)',src)
        self.assertIn('ADD COLUMN IF NOT EXISTS previous_folder TEXT',src)
        self.assertIn('WHERE owner_id=%s AND folder_id=%s',src)
        self.assertIn("SET folder='inbox'",src)
        self.assertIn("@router.get('/circle-email/folders')",src)
        self.assertIn("@router.post('/circle-email/folders')",src)
        self.assertIn("@router.patch('/circle-email/folders/{folder_id}')",src)
        self.assertIn("@router.delete('/circle-email/folders/{folder_id}')",src)

    def test_bulk_move_only_from_inbox_or_custom_folder(self):
        src=SOURCE.read_text(encoding='utf-8')
        self.assertIn("action: Literal['trash','restore','move','move_inbox','archive','unarchive','read','unread']",src)
        self.assertIn("if row['folder'] not in ('inbox',) and not row['folder'].startswith('custom:'):",src)
        self.assertIn('COUNT(m.mail_id) FILTER (WHERE m.is_read=FALSE)',src)
        self.assertIn("folder LIKE %s) AND is_read=FALSE\",(uid,'custom:%')",src)
        self.assertIn('m.sender_id=m.owner_id AND f.owner_id=m.owner_id',src)
        self.assertIn('previous_folder=%s,deleted_at=%s',src)

if __name__=='__main__':unittest.main()
