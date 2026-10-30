"""Owner-scoped Circle mail Archive / All Mail / mark unread regression tests."""
from __future__ import annotations
import ast
from pathlib import Path
import re
import unittest

FILE=Path(__file__).resolve().parents[1]/"trust_voice_circle_email.py"
MAIL_ID="tvcmail_"+"a"*32


class HTTPError(Exception):
    def __init__(self,status,detail=None):
        self.status_code=status
        self.detail=detail


class Cursor:
    def __init__(self,row=None,rows=None):
        self.row=row or {"mail_id":MAIL_ID,"owner_id":"ownerA",
                         "sender_id":"someone-else","folder":"inbox","previous_folder":None}
        self.rows=rows
        self.executed=[]
    def execute(self,sql,params):
        self.executed.append((sql,params))
    def fetchone(self):
        return self.row
    def fetchall(self):
        return self.rows if self.rows is not None else [self.row]


class Db:
    def __init__(self,cur):self.cur=cur
    def __enter__(self):return self
    def __exit__(self,*_):return False
    def cursor(self):return self.cur


def extract_function(name,cur):
    tree=ast.parse(FILE.read_text(encoding="utf-8"))
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
    fn.decorator_list=[]
    fn.returns=None
    for arg in (*fn.args.posonlyargs,*fn.args.args,*fn.args.kwonlyargs):
        arg.annotation=None
    env=dict(Header=lambda default=None:None, feature=lambda:None,_identity_from_bearer=lambda a:{"id":"ownerA"},
             schema=lambda:None,get_db=lambda:Db(cur),
             fail=lambda code,msg,status=400: (_ for _ in ()).throw(HTTPError(status,code)),
             _restore_folder=lambda c,owner,row:row.get("previous_folder") or "inbox",
             utc=lambda:"now",re=re,_require_owned_folder=lambda *a:True)
    exec(compile(ast.Module(body=[fn],type_ignores=[]),str(FILE),"exec"),env)
    return env[name]


class MailStandardActionsTests(unittest.TestCase):
    def test_archive_single_remembers_origin_folder(self):
        cur=Cursor(row=dict(mail_id=MAIL_ID,owner_id="ownerA",sender_id="someone",
                            folder="custom:tvcmfolder_"+"f"*32,previous_folder=None))
        fn=extract_function("action",cur)
        result=fn(MAIL_ID,type("Payload",(),{"action":"archive"})(),None)
        self.assertEqual(result,{"ok":True})
        writes=[x for x in cur.executed if "SET folder='archive'" in x[0]]
        self.assertEqual(len(writes),1)
        self.assertEqual(writes[0][1],(cur.row["folder"],"ownerA",MAIL_ID))
        self.assertIn("owner_id=%s",writes[0][0])

    def test_unarchive_restores_origin(self):
        cur=Cursor(row=dict(mail_id=MAIL_ID,owner_id="ownerA",sender_id="someone",
                            folder="archive",previous_folder="inbox"))
        fn=extract_function("action",cur)
        fn(MAIL_ID,type("Payload",(),{"action":"unarchive"})(),None)
        writes=[x for x in cur.executed if "SET folder=%s" in x[0]]
        self.assertEqual(len(writes),1)
        self.assertEqual(writes[0][1],("inbox","ownerA",MAIL_ID))
        self.assertIn("previous_folder=NULL",writes[0][0])

    def test_bulk_archive_is_owner_scoped(self):
        cur=Cursor()
        fn=extract_function("bulk_action",cur)
        res=fn(type("Payload",(),{"action":"archive","mail_ids":[MAIL_ID],"folder_id":None})(),None)
        self.assertEqual(res["updated"],1)
        writes=[x for x in cur.executed if "SET folder='archive'" in x[0]]
        self.assertEqual(len(writes),1)
        self.assertEqual(writes[0][1],("inbox","ownerA",MAIL_ID))
        self.assertIn("owner_id=%s",writes[0][0])

    def test_bulk_mark_unread_is_owner_scoped(self):
        cur=Cursor()
        fn=extract_function("bulk_action",cur)
        res=fn(type("Payload",(),{"action":"unread","mail_ids":[MAIL_ID],"folder_id":None})(),None)
        self.assertEqual(res["updated"],1)
        writes=[x for x in cur.executed if "SET is_read=%s" in x[0]]
        self.assertEqual(writes[0][1],(False,"ownerA",MAIL_ID))
        self.assertIn("owner_id=%s",writes[0][0])

    def test_bulk_foreign_user_not_found_without_partial_changes(self):
        cur=Cursor(rows=[])
        fn=extract_function("bulk_action",cur)
        with self.assertRaises(HTTPError) as error:
            fn(type("Payload",(),{"action":"archive","mail_ids":[MAIL_ID],"folder_id":None})(),None)
        self.assertEqual(error.exception.status_code,404)
        self.assertFalse(any("UPDATE " in x[0] for x in cur.executed))

    def test_archive_skips_spam_and_trash(self):
        for folder in ("spam","trash","archive"):
            with self.subTest(folder=folder):
                cur=Cursor(row=dict(mail_id=MAIL_ID,owner_id="ownerA",sender_id="someone",
                                   folder=folder,previous_folder=None))
                fn=extract_function("action",cur)
                fn(MAIL_ID,type("Payload",(),{"action":"archive"})(),None)
                self.assertFalse(any("UPDATE " in sql for sql,params in cur.executed))

    def test_all_mail_folder_excludes_spam_and_trash(self):
        src=FILE.read_text(encoding="utf-8")
        self.assertIn("if folder=='all':",src)
        self.assertIn("folder NOT IN ('trash','spam') AND deleted_at IS NULL",src)
        self.assertIn("if folder not in {'inbox','sent','spam','trash','starred','archive','all'}",src)
        self.assertIn("Literal['trash','restore','move','move_inbox','archive','unarchive','read','unread']",src)


if __name__=="__main__":
    unittest.main()
