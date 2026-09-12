# Stage 10D-R17A-R4 independent review and schedule disposition

Review date: 2026-09-12. Reviewer recommendation; the human owner retains acceptance authority.

## Review verdict

`R17A_R4_REMEDIATION_REJECTED_IMPLEMENTATION_DEFECT`

`R17A_MODEL_PROMOTION = NOT_AUTHORIZED`

`R17B_START = NOT_AUTHORIZED`

Remaining-block classification: `IMPLEMENTATION_DEFECT`. Historical schedule unavailability is also supported by the local recovery audit, but cannot explain away the absent historical CE implementation or incomplete tests. The validator's single failure is reproducible; it is not proof that all five repairs are complete.

## Provenance

`EXACT_COMMIT_PROVENANCE = PASS` for the 14 declared source inventory entries.

- Run: `928e4be9-2c23-401d-8853-480a892dc73d`, stage `STAGE_10D_R17A_R4`.
- Run commit: `e1aec38a27cee75cfda54a9b637ca3bc72b089bb`.
- Current HEAD: `8b761092060b3e94942bb14f86f0147a2a360573`.
- Local `origin/main`: `87556307828ca0f500b3774b354b8540d4d35010` (remote-tracking ref; no network fetch).
- Recorded execution interval: `2026-09-12T16:07:23Z`–`2026-09-12T16:09:10Z`.
- Bundle: `.agent-runs/stage_10d_r17a_r4-928e4be9-2c23-401d-8853-480a892dc73d`.
- AGY report located at `.codex/audits/stage-10d-r17a-r4-remediation-report.md`.

| Identity | Independently recomputed SHA-256 |
| --- | --- |
| Stage config | `647de3384e8214904aad0b44326b40522523c5342a97fc50594542b37ab7f2c5` |
| Contract | `c986cd3c38f4ec97a9b0031f74f5b6c073ab2925e1ef1ef6c76bcad1a2dfdbe8` |
| Evaluator | `09795cd4da7caa17b0d9021ede39a5d8d0c59dfd9e69b348ed7ad6e9ac44b728` |
| R4 tests | `d1d6da6a051554de67959857b620041592ff83274e55add8d2c0de8f49e728ee` |
| Frozen policy | `d5a2972360486dade1f5197460fb367417b11a825a9e32a58c8a2d87fedcbbd9` |

Every inventory path exists in the recorded Git tree. SHA-256 of `git show <run-commit>:<path>` matches both recorded committed/executed hashes and current disk bytes, including policy, harness, validator, runner, raw loader, canonical producer, CE and recovered components. The ingestion dependency is actually `data_pipeline/ingest.py`, not the example `fantasy_prediction/ingest.py`. Full per-path hashes are in the companion review evidence directory.

Independent temporary-repository tests verified rejection of tracked modification without moving HEAD and an untracked required source. In-memory checks verified rejection of incorrect validation dependency bytes and an incorrect recorded executed hash. Evaluator lines 420 and 1500 call the same verifier before and after evaluation; a modification still present at the final check raises before successful completion. This is before/after verification, not continuous monitoring of transient changes that are restored between checks. No mismatched execution bytes were found in this run. Its inventory should not be construed as a complete transitive import or immutable input inventory: for example, the portability market snapshot and sealed reproducible S30 state are consumed but absent from its immutable input list.

## Five P1 finding table

| Finding | Result | Direct evidence | Remaining limitation |
| --- | --- | --- | --- |
| 1. Exact-commit closure | PASS, declared inventory scope | All 14 committed-byte comparisons; independent negative probes; evaluator pre/post checks; validator Git-object comparison | No continuous source monitoring; R4 test 01 itself only exercises missing commit/path negatives |
| 2. CE opponent lineage | PASS for removal of fallback and blocked disposition | Evaluator lines 940–991; all 1,513 emitted rows match the modeling table's period/team/lock sequence; all opponents/source fields empty | Missing status is assigned unconditionally, not resolved by an implemented schedule loader; no player key beyond positional row_id |
| 3. Authoritative CE integration | FAIL | Historical section only constructs lineage and a BLOCKED JSON object; its `authoritative_integration_verified` is hard-coded true | No historical baseline/winner CE call path exists; test 09 spies on the library directly |
| 4. Target-free portability | FAIL overall; real S30 inference and target contradiction regression repaired | Entry point lines 319–401 calls `predict_s30_v2`; test 07 exercises target/market/schedule/empty-market rejections | Clean case passes `schedule_data=[]`; a timestamp literal is substituted for schedule provenance; empty/malformed schedules are accepted |
| 5. Artifact-bound tests | FAIL, partial repair | Tests 04/05 read folds/metrics/eligibility; development-MAE and ineligibility mutations execute; harness derives summary after subprocess completion | Secondary mutation is unused; bootstrap checks only a method label; protected-path test checks count/existence and a boolean instead of required-set equality and before/after hashes |

`AUTHORITATIVE_CE_CODE_PATH = FAIL`

`REAL_HISTORICAL_CE_EVALUATION = BLOCKED`

`PORTABILITY_REMEDIATION = FAIL`

`ARTIFACT_BOUND_TEST_REMEDIATION = FAIL`

### Concrete unresolved defects

1. **P1: Historical CE implementation is absent, while evidence asserts verification.** `scripts/run_stage10d_r17a_r4_evaluation.py:940` through 991 has no schedule input resolution or conditional historical inference. It always emits BLOCKED and sets `authoritative_integration_verified = True` at line 988. The evaluator's sole actual `predict_ce` call at line 601 checks 2026 runtime S30 parity, using an empty schedule. It cannot establish fold-specific historical baseline/winner CE execution. `tests/test_stage10d_r17a_r4_recency.py:371` tests the existing library, not an R4 historical entry point. The frozen CE/S30/FE identifiers are correct, and `predict_ce` really delegates to S30 and FE; neither establishes the missing evaluator path. Even supplying authentic schedules would not make this implementation evaluate historical CE.

2. **P1: Portability schedule evidence is not exercised.** Evaluator lines 997–1008 set a schedule timestamp to the market capture time and pass an empty schedule. The entry point rejects only `schedule_data is None`; non-list objects are silently converted to an empty list at line 371. Real S30 predictions are produced, but no authentic scheduled opponent is consumed or its source timestamp bound to the data. Empty-market, injected-target and explicit post-lock timestamp cases reject as claimed, and the policy rejects `target_columns_present=1` with `target_columns_removed=true`. Those improvements do not prove schedule-complete portability. Existing July 2026 raw official snapshots contain recoverable schedules for an honest portability fixture; they do not qualify the 2024/2025 historical evaluation.

3. **P1: Artifact mutation claims remain overstated.** Test 05 creates `mutated_2025_sec` at line 203 and never uses it; it calls the unchanged selector on unchanged inputs. It also does not read the emitted secondary-validation CSV. This repeats the central R3 test defect. Test 06 reads emitted bootstrap JSON only conditionally and checks a method string rather than emitted multiplicity evidence; its numerical exercise is a separate synthetic two-period example. Test 10 checks ten existing paths and a declared `PRODUCTION_UNCHANGED` value, without comparing the policy's exact required path set or actual before/after hashes. The separate harness governance checks help protection but do not make these R4 tests demonstrate their reported artifact-bound invariants.

## Validator reproduction

The configured R4 test command ran with the repository environment active, `PYTHONDONTWRITEBYTECODE=1`, and `EVIDENCE_ROOT` explicitly pointing to the original R4 bundle:

```bash
.venv/bin/python -m unittest tests/test_stage10d_r17a_r4_recency.py -v
```

Exit 0; **11 tests passed in 37.113 seconds**, no skips. Passing tests do not resolve the coverage defects above.

```bash
python -m unittest tests.test_evidence_harness tests.test_evidence_policy_governance tests.test_evidence_harness_protected_paths tests.test_harness_integration -v
```

Exit 0; **40 tests passed in 4.012 seconds**. The full repository suite and compileall were not run for this read-only focused review.

The requested validator CLI calls `replay`, which writes `ci-replay-validation.json`. To honor the stronger requirement not to alter the original bundle, I copied it byte-for-byte to `/tmp/r4-independent-review-bundle`, then invoked:

```bash
.venv/bin/python scripts/validate_stage_evidence.py --evidence-root /tmp/r4-independent-review-bundle
```

Exit **1**; exact output:

```text
Validation Status: BLOCKED
Validation Failures:
  - blocking gate failure GATE_CE_INTEGRATION
```

I also called `scripts.evidence_harness.validate(repository_root, original_bundle)` directly. It returned `valid=false`, `status=BLOCKED`, and exactly the same one-item failure list without writing to the bundle. Original bundle hashes were checked after review. No evaluator rerun or evidence repair occurred.

## Schedule lineage audit

```text
ROW_COUNT_EXPECTED = 1513
ROW_COUNT_OBSERVED = 1513
ROWS_WITH_AUTHENTIC_PRELOCK_SCHEDULE = 0
ROWS_MISSING_AUTHENTIC_PRELOCK_SCHEDULE = 1513
ROWS_USING_RESULT_FALLBACK = 0
ROWS_USING_SYNTHETIC_SCHEDULE = 0
```

Population: 675 rows in 2024, 838 in 2025, covering 48 prediction periods. I independently filtered the immutable modeling table by lock year and compared all emitted period/team/lock values in order. No rows were dropped. Authentic coverage for the required historical rows is **zero** in this bundle. There are no real historical CE predictions; the zero fallback/synthetic counts describe the blocked historical path, not completed CE inference.

## Historical source recovery audit

Search covered current and ignored repository content, archived `.agent-runs`, schedule tables, snapshots, caches, acquisition logs, generated lineup paths and all reachable Git history. The path search produced 73 historical object candidates, including trees; 62 were blobs whose bytes were read and hashed. Removed-path candidates included six old dashboard matchup blobs and a deleted evidence ZIP. Earliest reachable Git author/committer timestamp is 2026-07-21, after every required historical lock. Git timestamps therefore supply no pre-lock proof for 2024/2025. Historical revision timestamps embedded in archived content were considered separately.

| Candidate source | Provenance and period | Qualification |
| --- | --- | --- |
| `data/processed/player_model_v2/stage_3d/schedule_context.csv` | 1,162 OE result-derived series; all `POSTEVENT_CONTEXT_ONLY`; all revision IDs/timestamps null | No availability timestamp; does not qualify |
| `stage_4c_context{,_02,_03}/historical_schedule_context.csv` under the same processed root | All three have zero rows; archived snapshot index has `qualified_snapshots=[]`; acquisition log states no external sources accessed | No schedule evidence |
| `stage_6a_m4_m5_context/historical_prelock_series_schedule.csv` | 2,194 rows, all `STRUCTURAL_RECONCILIATION_ONLY`; no publication/revision timestamps | A filename containing “prelock” is not provenance |
| Adjacent team-period schedule and event crosswalk | 1,003 structural aggregate rows; crosswalk has 2,194 structural rows and 30 null evidence classes | No authentic availability proof |
| Archived Stage 8E canonical-series schedules | Source explicitly names the structural schedule; other rerun records use `oe_structural` identifiers | Derived structural opponents, not captured historical schedules |
| `data/raw/official_market_snapshots/*` and recovered `prelock-schedules*.jsonl` | Official market capture metadata and reciprocal matchup identities; captures begin July 2026 | Credible for relevant 2026 locks, outside R17A's historical population |
| Stage 8E-S1 archived Leaguepedia API cache, qualification ledger and source manifests | Historical revision payloads preserved; sampled actual payload includes 2023 revision timestamp and unresolved `AutoMatches`; archived qualification inventory records 23 unresolved transclusions, 4 missing pages, 1 missing pre-lock revision, zero qualified series | No exact historical pairing available from those payloads; no modern template expansion allowed |
| Stage 8E-S1 Liquipedia/Internet Archive fallback records and caches | Liquipedia revision 617023, 2023-01-26, unresolved `#lst`; Riot snapshot 20230123115430 is a client-rendered shell. These are archived local research records, not fresh external verification | No captured pairing; also not a recovered 2024/2025 schedule |
| Removed dashboard matchup blobs in Git | Generated lineup data, committed in 2026; no 2024/2025 pre-lock capture chain | Does not qualify |
| Removed ZIP blob `bda62c3c726bdf6bcb54a5df79f9d3db95e905b0` | Schedule-adjusted-form design/evaluation bundle from August 2026. Inspected contained upcoming-matchup and coverage tables: the former contains 2026 OE IDs, the latter asserts historical coverage without a source publication chain | Aggregate “known_pre_lock_schedule” counts do not establish authentic provenance |
| Raw OE/GOL result data and old prediction/history/lineup outputs | Results, retrospective features, or later generated artifacts; no independently verified pre-lock publication attached | Cannot substitute for captured schedules |

The companion `schedule-audit.json`, `archive-audit.json` and `history-content-audit.json` record exact source paths, content hashes and inspected timestamp samples. For example, the official July 24 snapshot SHA-256 is `e7f8597de4f6a2c893ca6b5685f47043091c7c8efa4f4e0552702b8566d07f3a`; its metadata names `https://api.lcsofficial.gg/market` and capture `2026-07-24T13:19:15.588988Z`, preceding its July 25 lock but not any 2024/2025 lock.

No qualifying historical source was recovered locally. This is a bounded repository/history audit, not proof that no external archive exists. No new external research was performed. In particular, archived prior research's zero-coverage pilot does not by itself prove impossibility for every 2024/2025 source.

## Scientific disposition

Retain `RECENCY_5`; no candidate promotion is justified. A single blocked gate plus passing tests does not establish implementation acceptance. If a subsequent corrected review confirms authentic historical provenance cannot be recovered, `RETAIN_RECENCY_5_AND_CLOSE_R17A_NO_PROMOTION` is scientifically defensible; repeatedly rerunning unchanged inputs is not. This review cannot silently alter the frozen 2024/2025 contract or replace schedule knowledge with results. R17B depends even more directly on trustworthy opponent information and remains unauthorized.

## Next node

`STAGE_10D_R17A_R4_R1_TARGETED_REMEDIATION`

Repair only the concrete defects above: implement the historical schedule-validation/authoritative CE entry point and test that entry point with explicitly synthetic test fixtures while keeping real missing rows BLOCKED; bind portability to actual schedule content and timestamps and reject missing/malformed schedules; make the secondary-data mutation, bootstrap audit and protected-path tests consume and verify the claimed artifacts. Correct unsupported verification claims in a fresh committed run. Preserve existing evidence. Do not manufacture historical schedules or authorize R17B.

No next-node implementation was performed. Review notes are the only repository additions.
