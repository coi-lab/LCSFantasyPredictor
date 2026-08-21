import json
import tempfile
import unittest
from pathlib import Path
import pandas as pd
from scripts.run_stage10d_r12f import REGISTRY, run

class R12FR1(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(); cls.out = Path(cls.temp.name) / "r12f-r1"; run(cls.out)
    @classmethod
    def tearDownClass(cls): cls.temp.cleanup()
    def data(self, name): return json.loads((self.out / name).read_text())
    def test_firewall_is_intact(self): self.assertFalse(any(self.data("stage-10d-r12f-r1-week5-firewall.json").values()))
    def test_per_game_model_remains_frozen(self): self.assertFalse(self.data("stage-10d-r12f-r1-player-model-freeze.json")["refit_in_R12F_R1"])
    def test_registry_has_required_schema(self):
        expected = {"league_model", "season", "split", "stage", "playoffs_flag", "date_start", "date_end", "best_of", "source_authority", "source_title", "rule_scope", "notes"}
        self.assertTrue(expected.issubset(pd.read_csv(REGISTRY).columns))
    def test_required_bo3_stage_rules_are_present(self):
        rules = pd.read_csv(REGISTRY)
        for season, split in ((2016, "Summer"), (2017, "Spring"), (2017, "Summer"), (2024, "Summer"), (2025, "Split 3"), (2026, "Summer")):
            self.assertTrue(((rules.season == season) & (rules.split == split) & (rules.best_of == 3)).any())
    def test_no_game_count_is_used_for_format_assignment(self): self.assertNotIn("games_played", pd.read_csv(REGISTRY).columns)
    def test_training_coverage_block_is_explicit(self):
        report = self.data("stage-10d-r12f-r1-validator-report.json")
        self.assertEqual(report["verdict"], "BLOCKED_BY_BO3_TRAINING_COVERAGE"); self.assertEqual(len(report["missing_immutable_oracles_elixir_files"]), 2)
    def test_registry_audit_is_written(self): self.assertTrue((self.out / "stage-10d-r12f-r1-stage-rule-registry-audit.csv").exists())
    def test_manifest_exists(self): self.assertTrue((self.out / "manifest-sha256.json").exists())
