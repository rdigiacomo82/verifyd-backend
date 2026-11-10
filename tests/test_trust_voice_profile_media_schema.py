"""Regression coverage for process-local profile-media schema initialization."""
import ast
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from threading import Barrier, Event, Lock
from types import SimpleNamespace
import unittest


def schema_initializer(get_db):
    source = Path(__file__).resolve().parents[1] / "trust_voice_profile_media.py"
    parsed = ast.parse(source.read_text(encoding="utf-8"))
    nodes = [
        node
        for node in parsed.body
        if (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id in {"_schema_lock", "_schema_ready"}
                for target in node.targets
            )
        ) or (isinstance(node, ast.FunctionDef) and node.name == "_ensure_schema")
    ]
    env = {"get_db": get_db, "Lock": Lock}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), env)
    return env["_ensure_schema"]


class ProfileMediaSchemaTests(unittest.TestCase):
    def test_concurrent_requests_initialize_once_and_later_requests_skip_db(self):
        opened = []
        statements = []
        workers = 8
        start = Barrier(workers)
        initializing = Event()
        finish = Event()

        @contextmanager
        def get_db():
            opened.append(True)
            initializing.set()
            if not finish.wait(timeout=5):
                raise RuntimeError("test did not release initialization")
            yield SimpleNamespace(cursor=lambda: SimpleNamespace(execute=statements.append))

        ensure_schema = schema_initializer(get_db)

        def request():
            start.wait(timeout=5)
            ensure_schema()

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(request) for _ in range(workers)]
            try:
                self.assertTrue(initializing.wait(timeout=5))
            finally:
                finish.set()
            for future in futures:
                future.result(timeout=5)

        ensure_schema()
        self.assertEqual(len(opened), 1)
        self.assertEqual(len(statements), 2)

    def test_failed_commit_does_not_mark_schema_ready(self):
        opened = []
        statements = []

        @contextmanager
        def get_db():
            opened.append(True)
            yield SimpleNamespace(cursor=lambda: SimpleNamespace(execute=statements.append))
            if len(opened) == 1:
                raise RuntimeError("commit failed")

        ensure_schema = schema_initializer(get_db)
        with self.assertRaisesRegex(RuntimeError, "commit failed"):
            ensure_schema()
        ensure_schema()
        ensure_schema()

        self.assertEqual(len(opened), 2)
        self.assertEqual(len(statements), 4)


if __name__ == "__main__":
    unittest.main()
