import json, tempfile, unittest
from pathlib import Path
from scripts.run_stage10d_r12f_r2 import run
class R12FR2(unittest.TestCase):
 @classmethod
 def setUpClass(c): c.t=tempfile.TemporaryDirectory();c.o=Path(c.t.name)/'r';run(c.o)
 @classmethod
 def tearDownClass(c): c.t.cleanup()
 def j(c,n): return json.loads((c.o/n).read_text())
 def test_firewall(c): c.assertFalse(any(c.j('stage-10d-r12f-r2-week5-firewall.json').values()))
 def test_model_frozen(c): c.assertFalse(c.j('stage-10d-r12f-r2-player-model-freeze.json')['refit_in_R12F_R2'])
 def test_scalar_range(c): c.assertTrue(2<=c.j('stage-10d-r12f-r2-bo3-volume-scalar.json')['avg_games_per_bo3']<=3)
 def test_scalar_series(c): c.assertGreater(c.j('stage-10d-r12f-r2-bo3-volume-scalar.json')['n_series'],0)
 def test_weekly_formula(c): c.assertEqual(c.j('stage-10d-r12f-r2-week5-prediction-accounting.json')['max_abs_error'],0)
 def test_objective(c): c.assertEqual(c.j('stage-10d-r12f-r2-objective-accounting.json')['max_abs_error'],0)
 def test_roster(c): c.assertTrue((c.o/'stage-10d-r12f-r2-week5-roster-a.csv').exists())
 def test_dashboard(c): c.assertTrue(c.j('stage-10d-r12f-r2-dashboard-data-parity.json')['ROSTER_A_exact_match'])
 def test_optimizer_payload_has_player_and_opponent_champion_picks(c):
  lineup=json.loads((Path(__file__).resolve().parents[1]/'dashboard/generated/current/matchup_lineups.json').read_text())
  week=next(w for w in lineup['weeks'] if w['round_name']=='Round 5 (Split 3)')
  player=week['lineups'][0]['players'][0]
  c.assertTrue(player['champion_options']); c.assertTrue(player['opponent_players']); c.assertTrue(all(x['champion_options'] for x in player['opponent_players']))
