"""Regressions: game points never influence Circle achievement ranks."""
import ast
import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
RANK = ROOT.joinpath("trust_voice_ranks.py").read_text(encoding="utf-8")
TREE = ast.parse(RANK)
TIERS_NODE = next(node for node in TREE.body if isinstance(node, ast.Assign)
                  and any(isinstance(t, ast.Name) and t.id=="TIERS" for t in node.targets))
TIERS = ast.literal_eval(TIERS_NODE.value)
TIER_FN = next(node for node in TREE.body if isinstance(node, ast.FunctionDef) and node.name=="tier_for")
namespace={"TIERS":TIERS}
exec(compile(ast.Module(body=[TIER_FN], type_ignores=[]), "trust_voice_ranks.py", "exec"),namespace)

class LearningOnlyRankTests(unittest.TestCase):
    def test_games_not_required_to_unlock_tiers(self):
        self.assertEqual(namespace["tier_for"](150,3,0)["level"],1)
        self.assertEqual(namespace["tier_for"](500,8,0)["level"],2)
        self.assertEqual(namespace["tier_for"](1100,16,0)["level"],3)

    def test_gaming_without_learning_not_ranked(self):
        self.assertEqual(namespace["tier_for"](0,0,10000)["level"],0)

    def test_no_game_challenges_allowed(self):
        self.assertIn("CHALLENGES = {}  # Game points disabled",RANK)
        self.assertIn("FILTER (WHERE kind='challenge') AS unique_count",RANK)
        self.assertIn("FILTER (WHERE kind='challenge'),0) AS points",RANK)

    def test_unlocks_require_completed_homework_only(self):
        text=ROOT.joinpath("trust_voice_learning_unlocks.py").read_text(encoding="utf-8")
        self.assertIn("status='completed'",text)
        self.assertIn('"base_games_remain_available": True',text)
        self.assertIn('"game_points_added_to_trust_rank": False',text)

if __name__=="__main__":
    unittest.main()
