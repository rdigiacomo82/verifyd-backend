"""Deterministic bonus Arcade integrity tests; no app secrets or production DB."""
import os,sys,unittest
sys.path.insert(0,os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))
from trust_voice_arcade_bonus import BONUS_GAMES,BONUS_IDS,DIFFICULTY_MIN_POINTS,public_round,build_round,verify_bonus_round,required_points

class BonusIntegrityTests(unittest.TestCase):
 def test_all_games_levels_and_rounds_have_private_answers(self):
  for game in BONUS_IDS:
   for level in (0,1,2):
    for step in range((4,5,6)[level]):
     with self.subTest(game=game,level=level,step=step):
      server=build_round(game,level,step,'fixed-server-session-seed')
      client=public_round(game,level,step,'fixed-server-session-seed')
      self.assertNotIn('correct',client)
      self.assertEqual(len(client['options']),4)
      self.assertEqual(len(set(o['id'] for o in client['options'])),4)
      self.assertEqual(len(set(o['label'] for o in client['options'])),4)
      self.assertTrue(verify_bonus_round(game,level,step,'fixed-server-session-seed',server['correct']))
      self.assertFalse(verify_bonus_round(game,level,step,'fixed-server-session-seed','9999'))
      self.assertFalse(verify_bonus_round(game,level,step,'fixed-server-session-seed',int(server['correct'])))
 def test_unlock_monotone_and_distinct(self):
  self.assertEqual(len(BONUS_GAMES),6)
  for game in BONUS_IDS:
   points=[required_points(game,l) for l in range(3)]
   self.assertEqual(points,sorted(points))
   self.assertEqual(points[0],BONUS_GAMES[game]['unlock_points'])
  with self.assertRaises(ValueError):required_points('not-a-game',0)
 def test_different_rounds_can_change(self):
  for game in BONUS_IDS:
   prompts=[build_round(game,1,r,'other-seed')['prompt'] for r in range(5)]
   self.assertGreater(len(set(prompts)),1,(game,prompts))
if __name__=='__main__':unittest.main()