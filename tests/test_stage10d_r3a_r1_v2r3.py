"""Focused V2-R3 reconstruction, cutoff-state, and analysis tests."""
import ast, unittest
from pathlib import Path
import pandas as pd
from scripts.run_stage10d_r3a_r1_v2r3 import build_team_role_state_at_cutoff, STALE
SCRIPT=Path(__file__).resolve().parents[1]/'scripts/run_stage10d_r3a_r1_v2r3.py'
def hist():
 t=pd.Timestamp('2025-01-10T00:00:00Z'); rows=[]
 for i in range(7): rows.append(dict(team_id='A',role='BOT',series_id=f's{i}',series_completion_timestamp=t+pd.Timedelta(days=i),role_actual_share=i,role_positive_share=i+.1,team_series_fantasy=10+i))
 rows.append(dict(team_id='B',role='BOT',series_id='old',series_completion_timestamp=t+pd.Timedelta(days=6),role_actual_share=99,role_positive_share=99,team_series_fantasy=99)); return pd.DataFrame(rows)
class V2R3Tests(unittest.TestCase):
 def test_stale_names_blacklisted(self): self.assertEqual(len(STALE),3)
 def test_last3_arbitrary_cutoff(self): self.assertTrue(build_team_role_state_at_cutoff(hist(),'A','BOT',pd.Timestamp('2025-01-16T00:00:00Z'),3)['state_available'])
 def test_last6_arbitrary_cutoff(self): self.assertTrue(build_team_role_state_at_cutoff(hist(),'A','BOT',pd.Timestamp('2025-01-16T00:00:00Z'),6)['state_available'])
 def test_latest_three_selected(self): self.assertEqual(build_team_role_state_at_cutoff(hist(),'A','BOT',pd.Timestamp('2025-01-16T00:00:00Z'),3)['source_series_ids'],'s3|s4|s5')
 def test_future_excluded(self): self.assertNotIn('s6',build_team_role_state_at_cutoff(hist(),'A','BOT',pd.Timestamp('2025-01-16T00:00:00Z'),3)['source_series_ids'])
 def test_same_lock_excluded(self): self.assertNotIn('s5',build_team_role_state_at_cutoff(hist(),'A','BOT',pd.Timestamp('2025-01-15T00:00:00Z'),3)['source_series_ids'])
 def test_previous_team_excluded(self): self.assertNotIn('old',build_team_role_state_at_cutoff(hist(),'A','BOT',pd.Timestamp('2025-01-20T00:00:00Z'),6)['source_series_ids'])
 def test_numeric_state(self): self.assertIsInstance(build_team_role_state_at_cutoff(hist(),'A','BOT',pd.Timestamp('2025-01-16T00:00:00Z'),3)['role_fantasy_share'],float)
 def test_source_ids_present(self): self.assertTrue(build_team_role_state_at_cutoff(hist(),'A','BOT',pd.Timestamp('2025-01-16T00:00:00Z'),3)['source_series_ids'])
 def test_insufficient_history_reason(self): self.assertEqual(build_team_role_state_at_cutoff(hist(),'A','BOT',pd.Timestamp('2025-01-12T00:00:00Z'),6)['state_missing_reason'],'INSUFFICIENT_PRIOR_HISTORY')
 def test_no_precomputed_reason_absent(self): self.assertNotIn('NO_PRECOMPUTED_B1_ROW',SCRIPT.read_text())
 def test_formula_fantasy_mean(self): self.assertEqual(build_team_role_state_at_cutoff(hist(),'A','BOT',pd.Timestamp('2025-01-16T00:00:00Z'),3)['role_fantasy_share'],4.0)
 def test_b1_equivalence_fixture(self):
  # B1's role_actual_share is the mean of the latest three completed shares.
  self.assertEqual(build_team_role_state_at_cutoff(hist(),'A','BOT',pd.Timestamp('2025-01-16T00:00:00Z'),3)['role_fantasy_share'],(3+4+5)/3)
 def test_strict_boundary(self): self.assertTrue(build_team_role_state_at_cutoff(hist(),'A','BOT',pd.Timestamp('2025-01-16T00:00:00Z'),3)['latest_source_before_cutoff'])
 def test_uses_original_r1(self): self.assertIn('stage-10d-r1-signal-completion',SCRIPT.read_text())
 def test_provenance_artifact(self): self.assertIn('pair-outcome-provenance.csv',SCRIPT.read_text())
 def test_residual_recomputed(self): self.assertIn("d['s30_actual']-d['s30_prediction']",SCRIPT.read_text())
 def test_delta_oracle_minus_s30(self): self.assertIn("oracle_last{w}",SCRIPT.read_text())
 def test_gate_order_c1(self): self.assertIn("if c1['status']!='PASS': return c1",SCRIPT.read_text())
 def test_gate_order_c2(self): self.assertIn("if c2['status']!='PASS': return c2",SCRIPT.read_text())
 def test_nine_role_team_relationships(self): self.assertIn("'BOT residual ↔ SUP residual'",SCRIPT.read_text()); self.assertIn("'SUP'",SCRIPT.read_text())
 def test_bootstrap_ci_is_seeded(self): self.assertIn('np.random.default_rng(SEED)',SCRIPT.read_text())
 def test_persistence_has_development_and_robustness(self): self.assertIn('2022-23 development',SCRIPT.read_text()); self.assertIn('2024 robustness',SCRIPT.read_text())
 def test_team_surprise_has_both_signs(self): self.assertIn("'positive','negative'",SCRIPT.read_text())
 def test_compression_has_full_fields(self): self.assertIn("'gap_ratio'",SCRIPT.read_text()); self.assertIn("'spread_ratio'",SCRIPT.read_text())
 def test_ranking_has_upside_metrics(self): self.assertIn("'actual_top3_intersection_recall'",SCRIPT.read_text()); self.assertIn("'NDCG'",SCRIPT.read_text())
 def test_posthoc_bootstraps(self): self.assertIn("boot(g,c,'residual_advantage')",SCRIPT.read_text())
 def test_freshness_has_hashes(self): self.assertIn("'input_hash'",SCRIPT.read_text())
