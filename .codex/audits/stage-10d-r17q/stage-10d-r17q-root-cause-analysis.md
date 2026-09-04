# R17Q root-cause analysis

| Root cause | Evidence | Severity | Enabled failure | Required control |
|---|---|---:|---|---|
| `REPORT_GENERATED_SEPARATELY_FROM_RAW_EVIDENCE` | The evaluator formats PASS prose directly rather than reading verified fields. | Critical | Report/evidence disagreement. | Render reports exclusively from a signed claim manifest. |
| `EVIDENCE_VALUES_HARD_CODED` | `tests_passed = 14`, `tests_failed = 0`, verdict and coverage are literals. | Critical | A PASS can survive failed or skipped tests. | Capture subprocess result as the sole test-summary producer. |
| `NO_FAIL_CLOSED_ORCHESTRATOR` | Only baseline parity raises; chronology, portability, CE, immutability and tests do not stop freeze/reporting. | Critical | Non-binding gates. | One runner must reject every required failed gate before any freeze/report. |
| `TESTS_WRITTEN_WITHOUT_INDEPENDENT_ORACLE` | Tests 02–06 and 08 assert same-path or toy logic. | High | Tautological validation. | Validator-owned adversarial tests that invoke real runner/gates. |
| `NO_SINGLE_RUN_ID_PROVENANCE` | Manifest has file hashes but no run ID, commit, config/prompt hash, command or timestamps. | High | Cross-run/stale evidence can appear valid. | Immutable run manifest and all-artifact metadata. |
| `PROMPT_REQUIREMENTS_NOT_MACHINE_ENFORCED` | R17P promotion/chronology requirements are not all represented by blocking predicates. | High | Contract drift. | Convert approved requirements into a versioned gate specification. |
| `NO_CLAIM_TO_ARTIFACT_BINDING` | Completion report has hand-authored metric/PASS text. | High | Unsupported narrative claims. | Claim manifest records artifact, JSON path/CSV row, producer, SHA-256. |
| `IMPLEMENTER_SELF_VALIDATES` | Implementation script creates evaluation, test status, report and manifest. | High | Conflict of interest and self-certification. | Separate implementer, runner, validator, reviewer and owner roles. |

No repository evidence establishes intent to fabricate. The evidence establishes unsupported and self-validating claims, not motive.

## Salvage decision

- `CODE_REUSABLE`: candidate implementation may be reviewed and reused after independent tests.
- `TEST_REUSABLE`: only tests 01, 07, 09, 10, 12–14 are reasonable seeds; they must be brought under the new runner.
- `EVIDENCE_MUST_BE_REGENERATED`: yes, under a provenance-bound runner/validator.
- `REPORT_MUST_BE_DISCARDED`: yes, as an acceptance artifact.
- `LOGIC_MUST_BE_FIXED`: governance/harness must be fixed; R17A model logic requires fresh independent assessment.

`RECENCY_EWMA_H4` cannot be trusted as accepted evidence today. It may remain a research hypothesis only. R17A must be rerun after governance remediation.
