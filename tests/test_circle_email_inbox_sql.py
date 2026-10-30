"""Regression for Inbox 500: psycopg2 percent placeholders in unread query."""
from __future__ import annotations
import ast
from contextlib import contextmanager
from pathlib import Path
import unittest

SRC=Path(__file__).resolve().parents[1]/"trust_voice_circle_email.py"

class SQLCursor:
    def __init__(self):
        self.calls=[]
    def execute(self,sql,args):
        # psycopg2's param-style renderer treats any bare % as a formatting
        # operator. This used to raise IndexError: tuple index out of range
        # when the query embedded the literal LIKE 'custom:%'.
        self.assert_parameterized(sql,args)
        self.calls.append((sql,args))
    def assert_parameterized(self,sql,args):
        remaining=sql.replace('%s','')
        if '%' in remaining:
            raise AssertionError("SQL contains unescaped literal percent")
        if sql.count('%s')!=len(args):
            raise AssertionError("SQL placeholder count differs from arguments")
    def fetchall(self):
        return []
    def fetchone(self):
        return {'n':4}

class SQLDb:
    def __init__(self):
        self.c=SQLCursor()
    def __enter__(self):return self
    def __exit__(self,*args):return False
    def cursor(self):return self.c

def isolated_inbox():
    t=ast.parse(SRC.read_text(encoding='utf-8'))
    nodes=[n for n in t.body if isinstance(n,ast.FunctionDef) and n.name=="inbox"]
    assert len(nodes)==1
    n=nodes[0]
    n.decorator_list=[]
    n.returns=None
    n.args.defaults=[]
    for a in n.args.args:a.annotation=None
    db=SQLDb()
    env=dict(feature=lambda:None,_identity_from_bearer=lambda token:{'id':'owner-one'},
        schema=lambda:None,get_db=lambda:db,files_for=lambda db,ids:{},
        public_mail=lambda row:row, _require_owned_folder=lambda c,owner,folder:None)
    exec(compile(ast.Module(body=[n],type_ignores=[]),str(SRC),'exec'),env)
    return env['inbox'],db.c

class MailInboxSqlRegression(unittest.TestCase):
    def test_inbox_and_custom_folder_load_without_unescaped_percent(self):
        view,cursor=isolated_inbox()
        for folder in ('inbox','sent','spam','trash','starred','custom:tvcmfolder_'+'a'*32):
            with self.subTest(folder=folder):
                result=view(folder,'',None)
                self.assertEqual(result['unread'],4)
                self.assertEqual(result['messages'],[])
        count=[(s,a) for s,a in cursor.calls if 'SELECT COUNT(*) AS n' in s]
        self.assertEqual(len(count),6)
        for sql,args in count:
            self.assertIn('folder LIKE %s',sql)
            self.assertEqual(args,('owner-one','custom:%'))
            self.assertNotIn("folder LIKE 'custom:%'",sql)

    def test_old_raw_percent_literal_cannot_reappear(self):
        src=SRC.read_text(encoding='utf8')
        self.assertNotIn("folder LIKE 'custom:%'",src)
        self.assertIn("(uid,'custom:%')",src)

if __name__=='__main__':unittest.main()
