"""Integration wiring checks, deliberately no production DB import."""
import ast
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / 'main.py'

class IntegrationStaticTests(unittest.TestCase):
    def test_main_compiles(self):
        ast.parse(MAIN.read_text(encoding='utf-8'))

    def test_router_is_mounted_once(self):
        s = MAIN.read_text(encoding='utf-8')
        self.assertEqual(s.count('app.include_router(private_gateway_staging_router)'), 1)
        self.assertEqual(s.count('from trust_voice_private_gateway_staging import router as private_gateway_staging_router'), 1)

    def test_cors_is_gated_not_broad_public(self):
        s = MAIN.read_text(encoding='utf-8')
        self.assertIn('VERIFYD_PRIVATE_GATEWAY_STAGE_ENABLED', s)
        self.assertIn('http://127.0.0.1:3000', s)
        self.assertIn('allow_origins=', s)
        self.assertNotIn('allow_origins=["*"]',s)

if __name__=='__main__': unittest.main()