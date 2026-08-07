"""Stage 7 integration tests: verify the sealed 2026 reconstructed fantasy simulation evidence."""

from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / ".agent-runs" / "player-model-v2-stage-7-2026-reconstructed-fantasy-simulation-20260807"
G0_DIR = ROOT / "data" / "predictions" / "player_model_v2" / "candidates" / "G0"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_hash(payload: object) -> str:
    s = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


class TestStage7SimulationEvidence(unittest.TestCase):
    """Check that all required Stage 7 evidence files exist and are internally consistent."""

    def test_run_directory_exists(self) -> None:
        self.assertTrue(RUN_DIR.is_dir(), f"Run directory not found: {RUN_DIR}")

    def test_scope_file_declares_exposed_retrospective(self) -> None:
        scope = json.loads((RUN_DIR / "stage-7-scope.json").read_text())
        self.assertEqual(scope["stage"], "7")
        self.assertIn("EXPOSED RETROSPECTIVE", scope["evaluation_type"])

    def test_pre_leaderboard_result_exists_and_sealed(self) -> None:
        result_path = RUN_DIR / "stage-7-pre-leaderboard-result.json"
        sha_path = RUN_DIR / "stage-7-pre-leaderboard-result.sha256"
        self.assertTrue(result_path.is_file())
        self.assertTrue(sha_path.is_file())
        # Verify SHA matches file on disk
        expected_sha = sha_path.read_text().strip().split()[0]
        actual_sha = _sha256(result_path)
        self.assertEqual(expected_sha, actual_sha)

    def test_pre_leaderboard_result_has_eleven_weeks(self) -> None:
        result = json.loads((RUN_DIR / "stage-7-pre-leaderboard-result.json").read_text())
        self.assertEqual(len(result["weeks"]), 11)
        week_nums = [w["week"] for w in result["weeks"]]
        self.assertEqual(week_nums, list(range(1, 12)))

    def test_cumulative_points_monotonically_increasing(self) -> None:
        result = json.loads((RUN_DIR / "stage-7-pre-leaderboard-result.json").read_text())
        cumulative = [w["cumulative_points_with_champion_bonus"] for w in result["weeks"]]
        for i in range(1, len(cumulative)):
            self.assertGreater(cumulative[i], cumulative[i - 1],
                               f"Week {i+1} cumulative not greater than week {i}")

    def test_final_cumulative_matches_week_sum(self) -> None:
        result = json.loads((RUN_DIR / "stage-7-pre-leaderboard-result.json").read_text())
        reported = result["cumulative_points_with_champion_bonus"]
        last_week = result["weeks"][-1]["cumulative_points_with_champion_bonus"]
        self.assertAlmostEqual(reported, last_week, places=2)

    def test_budget_chain_is_consistent(self) -> None:
        """Week N+1 starting budget must equal week N next budget."""
        result = json.loads((RUN_DIR / "stage-7-pre-leaderboard-result.json").read_text())
        weeks = result["weeks"]
        for i in range(len(weeks) - 1):
            self.assertAlmostEqual(
                weeks[i]["next_budget"],
                weeks[i + 1]["starting_budget"],
                places=2,
                msg=f"Budget chain broken between week {weeks[i]['week']} and {weeks[i+1]['week']}",
            )

    def test_week_1_starts_at_100_gold(self) -> None:
        result = json.loads((RUN_DIR / "stage-7-pre-leaderboard-result.json").read_text())
        self.assertAlmostEqual(result["weeks"][0]["starting_budget"], 100.0, places=2)

    def test_leaderboard_access_gate_exists(self) -> None:
        gate = json.loads((RUN_DIR / "stage-7-leaderboard-access-gate.json").read_text())
        self.assertEqual(gate["status"], "AUTHORIZED")
        # Gate must contain the hash that matches the sealed result
        pre_sha = _sha256(RUN_DIR / "stage-7-pre-leaderboard-result.json")
        self.assertEqual(gate["pre_leaderboard_hash"], pre_sha)

    def test_determinism_check_passed(self) -> None:
        det = json.loads((RUN_DIR / "stage-7-determinism-comparison.json").read_text())
        self.assertTrue(det["determinism_passed"])
        self.assertEqual(det["validation_runs_count"], 2)
        self.assertEqual(det["discrepancies"], [])

    def test_eleven_sealed_lineups_exist_with_sha256(self) -> None:
        lineup_files = sorted(RUN_DIR.glob("stage-7-period-*-sealed-lineup.json"))
        sha_files = sorted(RUN_DIR.glob("stage-7-period-*-sealed-lineup.sha256"))
        self.assertEqual(len(lineup_files), 11, f"Expected 11 sealed lineups, got {len(lineup_files)}")
        self.assertEqual(len(sha_files), 11)
        # Verify each SHA
        for lf, sf in zip(lineup_files, sha_files):
            expected = sf.read_text().strip().split()[0]
            actual = _sha256(lf)
            self.assertEqual(expected, actual, f"SHA mismatch for {lf.name}")

    def test_eleven_cutoff_audits_exist(self) -> None:
        audits = sorted(RUN_DIR.glob("stage-7-period-*-cutoff-audit.json"))
        self.assertEqual(len(audits), 11)
        for audit_path in audits:
            audit = json.loads(audit_path.read_text())
            self.assertIn("target_cutoff", audit)
            self.assertTrue(audit["point_in_time_safety_verified"])

    def test_eleven_player_projection_csvs_exist(self) -> None:
        csvs = sorted(RUN_DIR.glob("stage-7-period-*-player-projections.csv"))
        self.assertEqual(len(csvs), 11)

    def test_eleven_champion_projection_csvs_exist(self) -> None:
        csvs = sorted(RUN_DIR.glob("stage-7-period-*-champion-projections.csv"))
        self.assertEqual(len(csvs), 11)

    def test_eleven_realized_points_files_exist(self) -> None:
        realized = sorted(RUN_DIR.glob("stage-7-period-*-realized-points.json"))
        self.assertEqual(len(realized), 11)

    def test_g0_result_summary_exists(self) -> None:
        summary_path = G0_DIR / "stage7-result-summary.json"
        self.assertTrue(summary_path.is_file())
        summary = json.loads(summary_path.read_text())
        self.assertEqual(summary["competition"], "2026_split_1")
        self.assertEqual(summary["fit_alpha"], 10.0)
        self.assertIn("cumulative_points_achieved", summary)
        self.assertIn("rank_comparison", summary)

    def test_g0_result_summary_consistent_with_pre_leaderboard(self) -> None:
        summary = json.loads((G0_DIR / "stage7-result-summary.json").read_text())
        result = json.loads((RUN_DIR / "stage-7-pre-leaderboard-result.json").read_text())
        self.assertAlmostEqual(
            summary["cumulative_points_achieved"],
            result["cumulative_points_with_champion_bonus"],
            places=2,
        )

    def test_all_sealed_lineups_have_six_roster_slots(self) -> None:
        for lf in sorted(RUN_DIR.glob("stage-7-period-*-sealed-lineup.json")):
            data = json.loads(lf.read_text())
            self.assertEqual(len(data["roster"]), 6,
                             f"Expected 6 roster slots in {lf.name}, got {len(data['roster'])}")

    def test_g0_provenance_files_still_intact(self) -> None:
        """Stage 7 must not alter the frozen G0 provenance files."""
        with (G0_DIR / "interaction-policy.json").open("r") as f:
            policy = json.load(f)
        with (G0_DIR / "champion-predictor-specification.json").open("r") as f:
            champ_spec = json.load(f)
        self.assertEqual(
            _canonical_hash(policy),
            "17e1d07677c587287cc94a690e6e53f98ae75672b29fa24d9cc747dff83fe890",
        )
        self.assertEqual(
            _canonical_hash(champ_spec),
            "83acf980ee71e6b8d0fca077b24d1e57fe2273dbf5cb88927614f22b304f2621",
        )


if __name__ == "__main__":
    unittest.main()
