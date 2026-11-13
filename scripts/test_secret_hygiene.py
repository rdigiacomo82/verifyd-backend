import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_secret_hygiene import classify

class SecretGuardTests(unittest.TestCase):
    def test_blocks_dotenv_and_keyfiles(self):
        for name in ("apps/api/.env", "api/.env.production", "private/id_ed25519", "tls/private.key"):
            with self.subTest(name=name):
                self.assertIn("sensitive_filename", classify(name, b"TEST=1"))

    def test_private_keys_blocked(self):
        self.assertIn("private_key", classify("fixture.txt", b"-----BEGIN PRIVATE KEY-----\nFAKE\n"))

    def test_normal_code_and_examples_pass(self):
        self.assertEqual([], classify(".env.example", b"DATABASE_URL=\n"))
        self.assertEqual([], classify("main.py", b"x = os.environ.get('DATABASE_URL')"))
if __name__ == "__main__":
    unittest.main()
