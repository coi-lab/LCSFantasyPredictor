"""Support-protected B2Z allocation mechanics for the R5B research challenger."""
from __future__ import annotations

import numpy as np
import pandas as pd

GAMMA_GRID = (0.25, 0.50, 0.75, 1.00, 1.25)
NON_SUPPORT_ROLES = frozenset(("TOP", "JGL", "MID", "BOT"))


def neutralize_non_support(group: pd.DataFrame, raw_column: str = "raw_B2Z_delta") -> pd.DataFrame:
    """Return support-protected, non-support zero-sum allocation deltas.

    A period with fewer than two structurally supported non-SUPPORT rows is
    deliberately returned unchanged.  This makes SUP neither a recipient nor
    a balancing bucket and preserves the S30 team total exactly.
    """
    out = group.copy()
    out["SUP_protected"] = out.role.eq("SUP")
    eligible = out.structural_support.astype(bool) & out.role.isin(NON_SUPPORT_ROLES)
    count = int(eligible.sum())
    out["team_period_supported_non_sup_count"] = count
    out["team_period_fallback"] = count < 2
    out["neutralized_non_sup_delta"] = 0.0
    if count >= 2:
        values = pd.to_numeric(out.loc[eligible, raw_column], errors="coerce").fillna(0.0)
        centered = values - values.mean()
        # Exact floating residue is removed within the same non-SUP set.
        centered = centered - centered.sum() / len(centered)
        out.loc[eligible, "neutralized_non_sup_delta"] = centered.to_numpy()
    return out


def apply_gamma(frame: pd.DataFrame, gamma: float) -> pd.DataFrame:
    """Apply a frozen allocation strength after non-SUPPORT neutralization."""
    if gamma not in GAMMA_GRID:
        raise ValueError("gamma must be one of the frozen R5B grid values")
    out = frame.copy()
    out["selected_gamma"] = float(gamma)
    out["prediction_delta"] = out.neutralized_non_sup_delta * float(gamma)
    out.loc[out.role.eq("SUP"), "prediction_delta"] = 0.0
    out["B2Z_NS_prediction"] = out.S30_prediction + out.prediction_delta
    totals = out.groupby(["prediction_period_id", "team_id"])["B2Z_NS_prediction"].transform("sum")
    out["B2Z_NS_share"] = out.B2Z_NS_prediction / totals.replace(0, np.nan)
    return out
