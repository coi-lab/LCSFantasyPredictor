"""Real historical-price observation index and prior model for LCS Fantasy Player Model V2."""

from __future__ import annotations

import glob
import hashlib
import math
import re
from pathlib import Path
from typing import Any, Sequence
import numpy as np
import pandas as pd

from fantasy_prediction.model_v2_statistics import (
    compute_recency_weights,
    compute_effective_sample_size,
    format_statistic_result,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OFFICIAL_DIR = PROJECT_ROOT / "data" / "raw" / "official_market_snapshots"
DEFAULT_DASHBOARD_JSON = PROJECT_ROOT / "dashboard" / "generated" / "current" / "dashboard_data.json"

PROHIBITED_FUTURE_PRICE_COLUMNS = {
    "next_price",
    "post_week_price",
    "future_price",
    "next_market_price",
    "post_lock_price",
}


def assert_no_future_price_features(df_or_dict: pd.DataFrame | dict[str, Any]) -> None:
    """Reusable matrix assertion rejecting future-price columns or key aliases."""
    if isinstance(df_or_dict, pd.DataFrame):
        cols = set(df_or_dict.columns)
    elif isinstance(df_or_dict, dict):
        cols = set(df_or_dict.keys())
    else:
        return
    forbidden = cols.intersection(PROHIBITED_FUTURE_PRICE_COLUMNS)
    if forbidden:
        raise ValueError(f"Security assertion failure: prohibited future-price field(s) detected: {forbidden}")


def parse_week_integer(week_val: Any) -> int | None:
    """Parse integer week number cleanly from int, float, or string like 'Spring W9' / '9'."""
    if week_val is None or pd.isna(week_val):
        return None
    if isinstance(week_val, (int, float)):
        return int(week_val)
    val_str = str(week_val).strip()
    match = re.search(r"\d+", val_str)
    if match:
        return int(match.group(0))
    return None


def load_price_observations(
    official_dir: Path = DEFAULT_OFFICIAL_DIR,
    dashboard_path: Path = DEFAULT_DASHBOARD_JSON,
    constrained_match_history: pd.DataFrame | None = None,
    max_year: int = 2024,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Load cutoff-timestamped official observations.

    ``dashboard_path`` and ``constrained_match_history`` remain in the public
    signature for caller compatibility.  Dashboard histories are modeled,
    post-week values and have no established competition/split/fantasy-week
    mapping, so Phase A intentionally does not open or join that source.
    """
    records: list[dict[str, Any]] = []
    exclusion_counts: dict[str, int] = {
        "missing_official_timestamps": 0,
        "dashboard_join_unestablished": 0,
        "prohibited_future_year": 0,
    }

    # 1. Flatten Official Market Snapshots
    if official_dir.exists():
        csv_paths = sorted(glob.glob(str(official_dir / "*.csv")))
        for path in csv_paths:
            filename = Path(path).name
            filename_years = [int(value) for value in re.findall(r"(?:19|20)\d{2}", filename)]
            # A selection run must decide eligibility from the filename before
            # opening. Unknown-year and post-selection captures are not opened.
            if not filename_years or any(year > max_year for year in filename_years):
                exclusion_counts["prohibited_future_year"] += 1
                continue

            with open(path, "rb") as f_bin:
                file_hash = hashlib.sha256(f_bin.read()).hexdigest()

            df = pd.read_csv(path)
            assert_no_future_price_features(df)
            snapshot_id = Path(path).stem

            for row in df.itertuples():
                m_opens = pd.to_datetime(getattr(row, "market_opens_at", None), utc=True)
                c_utc = pd.to_datetime(getattr(row, "captured_at_utc", None), utc=True)

                # Official availability: max(market_opens_at, captured_at_utc). Both timestamps are required!
                if pd.notna(m_opens) and pd.notna(c_utc):
                    avail = max(m_opens, c_utc)
                else:
                    exclusion_counts["missing_official_timestamps"] += 1
                    continue

                if avail.year > max_year:
                    exclusion_counts["prohibited_future_year"] += 1
                    continue

                pro_id = str(getattr(row, "pro_player_id", getattr(row, "round_player_id", "")))
                pname = str(getattr(row, "summoner_name", getattr(row, "player", ""))).strip()

                records.append({
                    "observation_id": f"official_{snapshot_id}_{pro_id}_{pname}",
                    "player_id": pro_id,
                    "player": pname,
                    "role": str(getattr(row, "role", "")).casefold(),
                    "league": str(getattr(row, "league", "LCS")).upper(),
                    "team": str(getattr(row, "team_name", getattr(row, "team_code", ""))),
                    "season": str(getattr(row, "season", avail.year)),
                    "split": str(getattr(row, "split", "")),
                    "snapshot_id": snapshot_id,
                    "price": float(getattr(row, "price", 0.0)),
                    "available_at": avail,
                    "source_class": "official_snapshot",
                    "source_quality": 1.0,
                    "split_opening_flag": bool(getattr(row, "is_split_start_price", False)),
                    "source_path": str(path),
                    "source_hash": file_hash,
                })

    # Dashboard price history is explicitly disabled until a real fantasy-week
    # mapping and lock-time provenance are available.  In particular, fantasy
    # week numbers must never be equated to ISO calendar week numbers.
    if dashboard_path.exists():
        exclusion_counts["dashboard_join_unestablished"] = 1

    obs_df = pd.DataFrame.from_records(records)
    if not obs_df.empty:
        assert_no_future_price_features(obs_df)
    return obs_df, exclusion_counts


def build_role_price_percentiles(observations: pd.DataFrame) -> pd.DataFrame:
    """Calculate snapshot-scoped role percentiles with average ranks: (average_rank - 0.5) / N."""
    if observations.empty:
        return pd.DataFrame()

    df = observations.copy()
    assert_no_future_price_features(df)

    percentiles: list[float] = []
    for row in df.itertuples():
        snap_role_obs = df.loc[
            df["snapshot_id"].eq(row.snapshot_id) & df["role"].eq(row.role)
        ]
        prices = snap_role_obs["price"].to_numpy(dtype=float)
        n = len(prices)
        if n <= 1:
            percentiles.append(0.5)
        else:
            p_val = float(row.price)
            less_count = np.sum(prices < p_val)
            eq_count = np.sum(prices == p_val)
            avg_rank = less_count + 1.0 + (eq_count - 1.0) / 2.0
            percentiles.append(float((avg_rank - 0.5) / n))

    df["role_price_percentile"] = percentiles
    return df


def build_historical_price_prior(
    price_index: pd.DataFrame,
    player: str,
    role: str,
    league: str,
    team: str,
    target_split: str,
    cutoff: pd.Timestamp,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build point-in-time historical price prior features with strict availability and detailed provenance envelope."""
    cutoff_ts = pd.to_datetime(cutoff, utc=True)

    empty_res = format_statistic_result(
        value=0.5,
        cutoff=cutoff_ts,
        source_count=0,
        effective_count=0.0,
        max_timestamp=None,
        provenance_class="fallback_price_prior",
        available=False,
        fallback_reason="no_valid_obs_before_cutoff",
    )
    empty_res.update({
        "previous_split_closing_percentile": 0.5,
        "previous_year_latest_percentile": 0.5,
        "recency_decayed_career_percentile": 0.5,
        "official_source_fraction": 0.0,
        "estimated_source_fraction": 0.0,
        "role_change_flag": False,
        "league_change_flag": False,
        "team_change_flag": False,
        "source_path": None,
        "source_hash": None,
    })

    if price_index.empty:
        return empty_res

    assert_no_future_price_features(price_index)

    valid_obs = price_index.loc[price_index["available_at"].lt(cutoff_ts)].copy()
    if valid_obs.empty:
        return empty_res

    player_obs = valid_obs.loc[
        valid_obs["player"].str.casefold().eq(player.casefold())
    ].copy()

    if player_obs.empty:
        return empty_res

    if "role_price_percentile" not in player_obs.columns:
        valid_obs = build_role_price_percentiles(valid_obs)
        player_obs = valid_obs.loc[valid_obs["player"].str.casefold().eq(player.casefold())].copy()

    half_life = config.get("default", {}).get("half_life_days", 365.0) if config else 365.0

    role_factors = np.where(player_obs["role"].eq(role.casefold()), 1.0, 0.5)
    league_factors = np.where(player_obs["league"].eq(league.upper()), 1.0, 0.75)
    source_qualities = player_obs["source_quality"].to_numpy(dtype=float) * role_factors * league_factors

    weights = compute_recency_weights(
        player_obs["available_at"],
        cutoff_ts,
        half_life_days=half_life,
        source_qualities=source_qualities,
    )
    eff_count = compute_effective_sample_size(weights)

    percentiles = player_obs["role_price_percentile"].to_numpy(dtype=float)
    valid_w = weights > 0
    if not valid_w.any():
        return empty_res

    career_prior = float(np.average(percentiles[valid_w], weights=weights[valid_w]))
    max_ts = player_obs.loc[valid_w, "available_at"].max()

    latest_split_obs = player_obs.loc[player_obs["split"].ne(target_split)].sort_values("available_at", ascending=False)
    prev_split_pct = float(latest_split_obs.iloc[0]["role_price_percentile"]) if not latest_split_obs.empty else float(player_obs.sort_values("available_at", ascending=False).iloc[0]["role_price_percentile"])

    prev_year = str(cutoff_ts.year - 1)
    prev_year_obs = player_obs.loc[player_obs["season"].eq(prev_year)].sort_values("available_at", ascending=False)
    prev_year_pct = float(prev_year_obs.iloc[0]["role_price_percentile"]) if not prev_year_obs.empty else float(player_obs.sort_values("available_at", ascending=False).iloc[0]["role_price_percentile"])

    latest_obs = player_obs.sort_values("available_at", ascending=False).iloc[0]

    official_frac = float(np.sum(player_obs["source_class"].eq("official_snapshot")) / len(player_obs))
    estimated_frac = float(1.0 - official_frac)

    res = format_statistic_result(
        value=career_prior,
        cutoff=cutoff_ts,
        source_count=len(player_obs),
        effective_count=eff_count,
        max_timestamp=max_ts,
        provenance_class="historical_price_prior",
        available=True,
        fallback_reason=None,
    )
    res.update({
        "previous_split_closing_percentile": prev_split_pct,
        "previous_year_latest_percentile": prev_year_pct,
        "recency_decayed_career_percentile": career_prior,
        "official_source_fraction": official_frac,
        "estimated_source_fraction": estimated_frac,
        "role_change_flag": bool(latest_obs["role"] != role.casefold()),
        "league_change_flag": bool(latest_obs["league"] != league.upper()),
        "team_change_flag": bool(latest_obs["team"] != team),
        "source_path": str(latest_obs["source_path"]),
        "source_hash": str(latest_obs["source_hash"]),
        "most_recent_evidence_age_days": round((cutoff_ts - max_ts).total_seconds() / 86400.0, 2),
    })
    return res
