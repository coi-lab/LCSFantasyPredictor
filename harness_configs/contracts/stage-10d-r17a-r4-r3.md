# Stage 10D-R17A-R4-R3 — Codex Direct Remediation

This stage preserves the frozen R17A scientific disposition and repairs only
the evidence boundary.  The canonical `stage-10d-r17a-source-closure.json` is
the complete sealed execution closure; the validator independently recomputes
its static set and verifies runtime containment.  Historical CE and future
portability receive schedules only from `load_authenticated_schedule`; absent
historical pre-lock sources remain `MISSING_AUTHENTIC_PRELOCK_SCHEDULE` and
therefore keep `GATE_CE_INTEGRATION` truthfully blocked.
