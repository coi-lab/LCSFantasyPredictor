"""Fail-closed tests for the frozen 2026 evaluation contract."""

from __future__ import annotations

import copy
import unittest

from fantasy_prediction.freeze_manifest import (
    FreezeManifestError,
    assert_ready_for_frozen_run,
    candidate_hash,
    load_manifest,
    validate_manifest,
)


class FreezeManifestTests(unittest.TestCase):
    def setUp(self) -> None:
        self.manifest = load_manifest()

    def test_blocked_canonical_manifest_is_complete_and_fingerprints_match(self) -> None:
        validate_manifest(self.manifest)
        self.assertEqual(self.manifest["status"], "BLOCKED_FROZEN_2026")
        self.assertEqual(self.manifest["frozen_evaluation"]["completed_runs"], 0)

    def test_blocked_manifest_cannot_authorize_2026(self) -> None:
        with self.assertRaisesRegex(FreezeManifestError, "not READY_FOR_FROZEN_2026"):
            assert_ready_for_frozen_run(self.manifest)

    def test_rejects_family_reordering(self) -> None:
        changed = copy.deepcopy(self.manifest)
        changed["candidate"]["families_in_order"].reverse()
        with self.assertRaisesRegex(FreezeManifestError, "frozen order"):
            validate_manifest(changed, verify_files=False)

    def test_rejects_official_claims_for_synthetic_market(self) -> None:
        changed = copy.deepcopy(self.manifest)
        changed["market_and_oracle_status"]["official_regret"] = "VERIFIED"
        with self.assertRaisesRegex(FreezeManifestError, "NOT_VERIFIED"):
            validate_manifest(changed, verify_files=False)

    def test_rejects_completed_2026_without_prior_hash(self) -> None:
        changed = copy.deepcopy(self.manifest)
        changed["frozen_evaluation"]["completed_runs"] = 1
        with self.assertRaisesRegex(FreezeManifestError, "prior candidate hash"):
            validate_manifest(changed, verify_files=False)

    def test_rejects_absolute_hash_input_path(self) -> None:
        changed = copy.deepcopy(self.manifest)
        changed["inputs"]["fingerprints"][0]["path"] = "/tmp/baseline.json"
        with self.assertRaisesRegex(FreezeManifestError, "repository-relative"):
            validate_manifest(changed, verify_files=False)

    def test_partial_candidate_hash_is_deterministic_but_not_recorded(self) -> None:
        self.assertEqual(candidate_hash(self.manifest), candidate_hash(self.manifest))
        self.assertIsNone(self.manifest["candidate_hash"])


if __name__ == "__main__":
    unittest.main()
