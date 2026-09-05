# R7C-R1 authoritative implementation closure

Scope: the owner's follow-up requests exact CE/S30/FE identifier binding and
stronger bootstrap multiplicity evidence. This is a governance-only closure.

Frozen R17A policy version 2 binds these exact file-and-function identifiers:

- `fantasy_prediction/ce_model.py:predict_ce`
- `fantasy_prediction/recovered_components.py:predict_s30_v2`
- `fantasy_prediction/recovered_components.py:predict_delta_e`

The CE wrapper imports and calls the latter two functions. The validator receives
the independently resolved policy and compares all three fields with exact
equality. Missing policy bindings fail closed. No artifact-supplied policy is
used. The registry anchors policy SHA-256
`d5a2972360486dade1f5197460fb367417b11a825a9e32a58c8a2d87fedcbbd9`.
All previous artifact, claim, gate, invariant, protected-path, report and status
requirements remain unchanged.

Bootstrap evidence must now contain `sampled_draw_trace` and
`consumed_cluster_counts`: parallel nonempty arrays of cluster-ID lists and
consumed-count maps. At least one audited draw must repeat a cluster. Every
count map must exactly equal the counts reconstructed from its draw. For example,
`[["a", "a", "b"]]` requires `[{"a": 2, "b": 1}]`; collapsing `a` to one
consumption is rejected. This is an artifact audit, not a recomputation of all B
resamples or runtime instrumentation proving which functions executed.

Focused verification command:

```
python -m unittest tests/test_evidence_harness.py tests/test_evidence_policy_governance.py tests/test_evidence_harness_protected_paths.py -v
```

Development verification: 36 tests, zero failures. New regressions check actual
repository function definitions, each substituted/missing CE identifier after
manifest resealing, policy-driven comparison, and absent, malformed or collapsed
bootstrap audit records. The complete positive governance fixture still returns
`PENDING_INDEPENDENT_REVIEW`.

Next step: independent review of this closure. R17A was not rerun.
