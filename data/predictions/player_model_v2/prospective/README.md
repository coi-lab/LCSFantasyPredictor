# Prospective T3 versus S30 tracking

For each future lock, create one immutable pre-lock record containing the T3 and S30 prediction tables, market prices, both selected rosters, and SHA-256 hashes. Mark it `PRELOCK_FROZEN` before results are known. After results resolve, append outcomes and derived metrics under `POSTLOCK_SCORED`; do not alter pre-lock model outputs.

T3_240d remains the validated checkpoint and S30 is an operational challenger. This directory contains no fabricated future observations and has no automatic-promotion rule.
