"""Fail-closed shared-lock preflight tests for Stage 8E."""
from __future__ import annotations

import csv
import json
import unittest
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCHEDULE = ROOT / "data/processed/player_model_v2/stage_6a_m4_m5_context/historical_prelock_series_schedule.csv"
PARTITION = ROOT / "data/processed/player_model_v2/stage_3e_03/partitions/development_2022_2023.csv"
CONTRACT = ROOT / "data/predictions/player_model_v2/evaluation/stage-8e-information-availability-contract.json"
EXCLUSIONS = ROOT / "data/predictions/player_model_v2/evaluation/stage-8e-structural-schedule-exclusions.json"


class TestStage8EHeadToHeadPreflight(unittest.TestCase):
    def test_schedule_identity_is_structural_known_prelock_and_outcomes_forbidden(self) -> None:
        contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
        self.assertEqual(contract["schedule_identity_classification"], "KNOWN_PRELOCK_STRUCTURAL_INFORMATION")
        self.assertFalse(contract["publication_timestamp_required_for_schedule_identity"])
        self.assertIn("match outcomes", contract["forbidden_retrospective_information"])

    def test_reciprocal_rows_share_one_lock(self) -> None:
        groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
        with SCHEDULE.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row["season"] in {"2022.0", "2023.0"}:
                    groups[(row["prediction_period_id"], row["series_id"])].append(row)
        for rows in groups.values():
            self.assertEqual(len({row["target_cutoff"] for row in rows}), 1)

    def test_stale_lock_assignments_are_excluded_before_shared_lock_construction(self) -> None:
        active: dict[str, set[str]] = defaultdict(set)
        with PARTITION.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                active[row["prediction_period_id"]].add(row["team_id"])
        groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
        with SCHEDULE.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row["season"] in {"2022.0", "2023.0"}:
                    groups[(row["prediction_period_id"], row["series_id"])].append(row)
        exclusions = {
            (item["prediction_period_id"], item["series_id"])
            for item in json.loads(EXCLUSIONS.read_text(encoding="utf-8"))["exclusions"]
        }
        failures = 0
        for (period, _), rows in groups.items():
            teams = {rows[0]["team_id"], rows[0]["opponent_team_id"]}
            if not teams.issubset(active[period]) and (period, rows[0]["series_id"]) not in exclusions:
                failures += 1
        self.assertEqual(len(exclusions), 7)
        self.assertEqual(failures, 0)


if __name__ == "__main__":
    unittest.main()
