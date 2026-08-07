import unittest
from unittest.mock import patch
from pathlib import Path
import os
import json

from data_pipeline.official_prices import (
    reconstruct_price,
    resolve_price,
    PriceProvenance,
    calculate_next_budget,
    resolve_participation
)
from fantasy_prediction.historical_simulator import (
    HistoricalWeek,
    MarketPlayer,
    RosterDecision,
    SyntheticPriceModel,
    simulate_competition
)

class TestStage6RIntegration(unittest.TestCase):
    def test_reconstructed_price_participant(self):
        # 0.747528 * 15.0 + 0.239998 * 10 + 0.015874 = 11.21292 + 2.39998 + 0.015874 = 13.628774 -> 13.6
        self.assertAlmostEqual(reconstruct_price(15.0, 10.0, "PARTICIPATED"), 13.6)
        self.assertAlmostEqual(reconstruct_price(15.0, 10.0, True), 13.6)

    def test_reconstructed_price_dnp_holds_previous_price(self):
        self.assertEqual(reconstruct_price(15.0, 0.0, "DID_NOT_PARTICIPATE"), 15.0)
        self.assertEqual(reconstruct_price(15.0, 0.0, False), 15.0)

    def test_reconstructed_price_zero_score_participant_not_dnp(self):
        # 0.747528 * 15.0 + 0.239998 * 0 + 0.015874 = 11.21292 + 0.015874 = 11.228794 -> 11.2
        self.assertAlmostEqual(reconstruct_price(15.0, 0.0, "PARTICIPATED"), 11.2)
        self.assertAlmostEqual(reconstruct_price(15.0, 0.0, True), 11.2)

    def test_reconstructed_price_rounding_to_tenth(self):
        self.assertAlmostEqual(reconstruct_price(15.0, 8.5, True), 13.3)

    def test_reconstructed_price_no_unsupported_absolute_clamp(self):
        # High score, high price -> should go above 32.0
        self.assertAlmostEqual(reconstruct_price(30.0, 100.0, True), 46.4)
        # Low score, low price -> should go below 5.0
        self.assertAlmostEqual(reconstruct_price(5.0, -10.0, True), 1.4)

    def test_official_price_overrides_reconstructed_price(self):
        price, provenance = resolve_price(official_snapshot_price=16.0, reconstructed_price=14.5)
        self.assertEqual(price, 16.0)
        self.assertEqual(provenance, PriceProvenance.OFFICIAL_SNAPSHOT)

    def test_embedded_previous_price_precedence(self):
        price, provenance = resolve_price(
            official_embedded_previous_price=15.5,
            reconstructed_price=14.5
        )
        self.assertEqual(price, 15.5)
        self.assertEqual(provenance, PriceProvenance.OFFICIAL_EMBEDDED_PREVIOUS_PRICE)

    def test_resolve_price_unavailable(self):
        price, provenance = resolve_price()
        self.assertIsNone(price)
        self.assertEqual(provenance, PriceProvenance.UNAVAILABLE)

    def test_participation_three_states(self):
        # games > 0 -> PARTICIPATED
        self.assertEqual(resolve_participation(2), "PARTICIPATED")
        self.assertEqual(resolve_participation("1"), "PARTICIPATED")
        
        # games == 0 with known valid field -> DID_NOT_PARTICIPATE
        self.assertEqual(resolve_participation(0), "DID_NOT_PARTICIPATE")
        self.assertEqual(resolve_participation("0"), "DID_NOT_PARTICIPATE")
        
        # games missing/null/invalid -> UNKNOWN
        self.assertEqual(resolve_participation(None), "UNKNOWN")
        self.assertEqual(resolve_participation(""), "UNKNOWN")
        self.assertEqual(resolve_participation("abc"), "UNKNOWN")
        self.assertEqual(resolve_participation("None"), "UNKNOWN")
        
        # UNKNOWN fallback in reconstruct_price behaves as PARTICIPATED (fail-closed)
        # reconstructed: 0.747528 * 15.0 + 0.239998 * 10 + 0.015874 = 13.6
        self.assertAlmostEqual(reconstruct_price(15.0, 10.0, "UNKNOWN"), 13.6)

    def test_round1_round2_price_reproduction(self):
        repo_root = Path(__file__).resolve().parents[1]
        path = repo_root / ".agent-runs/player-model-v2-stage-6e-pricing-budget-audit-20260807/stage-6e-r1-r2-official-transition.csv"
        cw_path = repo_root / ".agent-runs/player-model-v2-stage-6e-pricing-budget-audit-20260807/stage-6e-player-price-crosswalk.csv"
        
        import pandas as pd
        import numpy as np
        
        df = pd.read_csv(path).merge(pd.read_csv(cw_path), on='pro_player_id', how='inner')
        p12 = df[df['role'] != 'coach']
        
        preds = []
        for _, row in p12.iterrows():
            prev_price = float(row['price_r1'])
            score = float(row['last_round_score_r2'])
            # simulate participation determination (score > 0.0 means participated)
            did_participate = score > 0.0
            reconstructed = reconstruct_price(prev_price, score, did_participate)
            preds.append(reconstructed)
            
        preds = np.array(preds)
        actuals = p12['price_r2'].values
        mae = np.mean(np.abs(actuals - preds))
        max_err = np.max(np.abs(actuals - preds))
        
        # Verify the errors match expected bounds for R1->R2 reconstruction (MAE ~0.007, max_err ~0.1)
        self.assertLessEqual(mae, 0.01)
        self.assertLessEqual(max_err, 0.11)

    def test_round2_round3_player_price_error_bounds(self):
        repo_root = Path(__file__).resolve().parents[1]
        path = repo_root / ".agent-runs/player-model-v2-stage-6e-pricing-budget-audit-20260807/stage-6e-r2-r3-official-transition.csv"
        cw_path = repo_root / ".agent-runs/player-model-v2-stage-6e-pricing-budget-audit-20260807/stage-6e-player-price-crosswalk.csv"
        
        import pandas as pd
        import numpy as np
        
        df = pd.read_csv(path).merge(pd.read_csv(cw_path), on='pro_player_id', how='inner')
        p23 = df[df['role'] != 'coach']
        
        preds = []
        for _, row in p23.iterrows():
            prev_price = float(row['price_r2'])
            score = float(row['last_round_score_r3'])
            did_participate = score > 0.0
            reconstructed = reconstruct_price(prev_price, score, did_participate)
            preds.append(reconstructed)
            
        preds = np.array(preds)
        actuals = p23['price_r3'].values
        mae = np.mean(np.abs(actuals - preds))
        max_err = np.max(np.abs(actuals - preds))
        
        # Verify the errors match expected bounds for R2->R3 reconstruction (MAE ~0.257, max_err ~0.4)
        self.assertLessEqual(mae, 0.26)
        self.assertLessEqual(max_err, 0.41)

    def test_inspired_dnp_price_hold(self):
        self.assertEqual(reconstruct_price(15.0, 0.0, False), 15.0)

    def test_zven_dnp_price_hold(self):
        self.assertEqual(reconstruct_price(18.0, 0.0, False), 18.0)

    def test_apa_dnp_price_hold(self):
        self.assertEqual(reconstruct_price(12.0, 0.0, False), 12.0)

    def test_budget_round1_to_round2_109_1(self):
        # Starting budget: 100.0. Roster cost: 99.0. Unspent: 1.0. Next-round held value: 108.1.
        self.assertAlmostEqual(calculate_next_budget(1.0, 108.1), 109.1)

    def test_budget_round2_to_round3_118_7(self):
        # Starting budget: 109.1. Roster cost: 106.4. Unspent: 2.7. Next-round held value: 116.0
        self.assertAlmostEqual(calculate_next_budget(2.7, 116.0), 118.7)

    def test_budget_path_dependent_no_reset(self):
        b2 = calculate_next_budget(1.0, 108.1)
        # Spent 106.4 of b2.
        unspent2 = b2 - 106.4
        b3 = calculate_next_budget(unspent2, 116.0)
        self.assertAlmostEqual(b3, 118.7)

    def test_historical_simulator_uses_shared_pricing_contract(self):
        model = SyntheticPriceModel()
        self.assertAlmostEqual(model.update(15.0, 10.0, True), 13.6)

    @patch('data_pipeline.official_prices.calculate_next_budget')
    def test_historical_simulator_uses_shared_budget_contract(self, mock_calculate):
        mock_calculate.return_value = 105.0
        
        roles = ("top", "jgl", "mid", "bot", "sup", "coach")
        h_week = HistoricalWeek(
            week=1,
            stage_round="Round 1",
            market=tuple(MarketPlayer(role, role, "Team", 10.0) for role in roles),
            actual_points={role: 10.0 for role in roles}
        )
        
        def selector(target, prices, budget):
            return RosterDecision(roles, {})
            
        simulate_competition([h_week], selector, SyntheticPriceModel())
        self.assertTrue(mock_calculate.called)

    def test_dashboard_does_not_duplicate_price_formula(self):
        repo_root = Path(__file__).resolve().parents[1]
        with open(repo_root / "data_pipeline/export_dashboard_data.py", "r") as f:
            content = f.read()
            self.assertNotIn("0.747528 * previous_price", content)

    def test_root_stage_scripts_classified(self):
        repo_root = Path(__file__).resolve().parents[1]
        inventory_path = repo_root / ".agent-runs/player-model-v2-stage-6r-remediation-20260807/stage-6r-remediation-root-classification.json"
        self.assertTrue(os.path.exists(inventory_path))
        with open(inventory_path, "r") as f:
            data = json.load(f)
            classifications = {item["name"]: item["classification"] for item in data}
            self.assertIn("fix_tests.py", classifications)
            self.assertIn("generate_final.py", classifications)

if __name__ == '__main__':
    unittest.main()
