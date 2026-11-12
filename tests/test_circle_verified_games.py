"""Independent, no-production-DB unit tests for server-verified game logic."""
import importlib.util
import pathlib
import sys
import types
import unittest
from unittest.mock import patch

ROOT = pathlib.Path(__file__).resolve().parents[1]


class GameStateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.modules = patch.dict(sys.modules, {
            "database": types.SimpleNamespace(get_db=lambda: None),
            "trust_voice_ranks": types.SimpleNamespace(_owner=lambda _: {"id": "test"}, _schema=lambda: None),
        })
        cls.modules.start()
        spec = importlib.util.spec_from_file_location("verified_games_under_test", ROOT / "trust_voice_verified_games.py")
        cls.game = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = cls.game
        spec.loader.exec_module(cls.game)

    @classmethod
    def tearDownClass(cls):
        sys.modules.pop("verified_games_under_test", None)
        cls.modules.stop()

    def test_fully_validated_circle_match(self):
        for pairs in (2,4,6):
            state = self.game.create_initial_state("circle-match", pairs)
            indexes = {}
            for i, symbol in enumerate(state["deck"]):
                indexes.setdefault(symbol, []).append(i)
            self.assertEqual(len(indexes), pairs)
            for n, selections in enumerate(indexes.values()):
                self.assertFalse(self.game.apply_choice("circle-match", pairs, state, selections[0]))
                self.assertEqual(self.game.apply_choice("circle-match", pairs, state, selections[1]), n == pairs-1)
            self.assertEqual(state["attempts"], pairs)
            self.assertEqual(len(state["matched"]), 2*pairs)

    def test_hidden_deck(self):
        state = self.game.create_initial_state("circle-match", 2)
        self.assertEqual(self.game.public_state("circle-match", 2, state)["cards"], [None]*4)
        self.game.apply_choice("circle-match", 2, state, 0)
        shown = self.game.public_state("circle-match", 2, state)["cards"]
        self.assertEqual(shown[0], state["deck"][0])
        self.assertTrue(all(x is None for x in shown[1:]))

    def test_star_quest_all_levels_and_five_rounds(self):
        for tier in range(3):
            state = self.game.create_initial_state("star-quest", tier)
            for r in range(5):
                target, choices = self.game.star_round(tier, r)
                self.assertEqual(choices.count(target),1)
                self.assertEqual(len(choices),3+tier)
                self.assertEqual(self.game.apply_choice("star-quest", tier, state, target), r==4)
            self.assertEqual(state["round"],5)

    def test_invalid_moves_and_game_types_are_rejected(self):
        from fastapi import HTTPException
        with self.assertRaises(HTTPException): self.game.create_initial_state("star-quest", 4)
        match = self.game.create_initial_state("circle-match", 2)
        with self.assertRaises(HTTPException): self.game.apply_choice("circle-match",2,match,False)
        with self.assertRaises(HTTPException): self.game.apply_choice("circle-match",2,match,99)
        self.game.apply_choice("circle-match",2,match,0)
        with self.assertRaises(HTTPException): self.game.apply_choice("circle-match",2,match,0)
        star = self.game.create_initial_state("star-quest",0)
        with self.assertRaises(HTTPException): self.game.apply_choice("star-quest",0,star,"not-an-option")

    def test_mismatch_does_not_award_and_next_move_resets(self):
        state = self.game.create_initial_state("circle-match", 2)
        deck = state["deck"]
        first, second = next((i,j) for i in range(4) for j in range(i+1,4) if deck[i] != deck[j])
        self.game.apply_choice("circle-match",2,state,first)
        self.assertFalse(self.game.apply_choice("circle-match",2,state,second))
        self.assertTrue(state["mismatch"])
        another = next(i for i in range(4) if i not in (first,second))
        self.game.apply_choice("circle-match",2,state,another)
        self.assertFalse(state["mismatch"])
        self.assertEqual(state["selected"],[another])

    def test_public_api_is_feature_flagged(self):
        from fastapi import HTTPException
        with patch.dict("os.environ",{"VERIFYD_CIRCLE_VERIFIED_GAMES_ENABLED":"0"}):
            self.assertFalse(self.game.enabled())
            with self.assertRaises(HTTPException): self.game.catalog()
        with patch.dict("os.environ",{"VERIFYD_CIRCLE_VERIFIED_GAMES_ENABLED":"1"}):
            self.assertTrue(self.game.enabled())
            self.assertEqual(len(self.game.catalog()["games"]),2)


if __name__ == "__main__":
    unittest.main()
