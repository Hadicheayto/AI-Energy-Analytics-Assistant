"""
Energy analytics — pure functions, no LLM involved anywhere in this file.

Every function here takes explicit parameters (a DataFrame plus filters) and
returns a plain dict of numbers/strings — never a pre-formatted sentence.
This is deliberate: the Tool Layer (next phase) wraps these with LLM-facing
signatures, and the LLM composes the final natural-language answer FROM
these structured results. If a function returned a string here, the LLM
would effectively be "doing math in prose," which is exactly what the
assignment forbids.

Inputs:
  - Most functions take the unified site-total daily table produced by
    data_processing.build_site_total_daily(): columns organization, site,
    date, kwh, meters_reporting, meters_expected, is_complete_day.
  - Peak demand / baseload / load factor need INTERVAL-level data (only
    available for the last ~3 months per our extraction scope), so they
    take the output of data_processing.load_detailed_long() for a site's
    total meters, pre-summed across meters into one site-level interval
    series (see `sum_detailed_to_site_series` below).

Every function is defensive about missing/partial data: if the requested
period isn't covered, it returns a dict with an explicit "error" or
"available": False key rather than raising or silently returning zero.
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd


# --------------------------------------------------------------------- utils

def _filter_site(df: pd.DataFrame, organization: str, site: str | None = None) -> pd.DataFrame:
    """
    site=None aggregates across ALL sites in the given organization — summed
    in Python (groupby + sum), never left for the LLM to add up itself. This
    is what lets us answer organization-level questions (e.g. "Best Resorts
    Hotels" as a whole, which has two sites: Alpha Hotel and Beta Resort).
    """
    org_df = df[df["organization"] == organization]
    if site is None:
        agg = org_df.groupby("date", as_index=False)["kwh"].sum(min_count=1)
        agg["organization"] = organization
        agg["site"] = "(all sites in organization)"
        return agg
    return org_df[org_df["site"] == site].copy()


def _filter_date_range(df: pd.DataFrame, start: date, end: date, date_col: str = "date") -> pd.DataFrame:
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    return df[(df[date_col] >= start_ts) & (df[date_col] <= end_ts)].copy()


def sum_detailed_to_site_series(detailed_df: pd.DataFrame) -> pd.Series:
    """
    Collapse a long-format detailed DataFrame (multiple meters) into one
    site-level interval series: sum of all meters' values at each timestamp.
    Returns a pandas Series indexed by timestamp, values in kWh per interval.
    """
    if detailed_df.empty:
        return pd.Series(dtype=float)
    return detailed_df.groupby("timestamp")["value"].sum(min_count=1)


# ---------------------------------------------------------------- analytics

def total_and_average(
    daily_df: pd.DataFrame, organization: str, site: str | None, start: date, end: date
) -> dict:
    """Total and average daily consumption over [start, end], inclusive. site=None aggregates the whole organization."""
    df = _filter_date_range(_filter_site(daily_df, organization, site), start, end)
    if df.empty:
        return {"available": False, "reason": "no data in requested range"}

    n_days_expected = (end - start).days + 1
    n_days_present = len(df)
    return {
        "available": True,
        "organization": organization,
        "site": site if site else "(entire organization)",
        "start": str(start),
        "end": str(end),
        "total_kwh": round(float(df["kwh"].sum(skipna=True)), 2),
        "average_daily_kwh": round(float(df["kwh"].mean(skipna=True)), 2),
        "days_present": n_days_present,
        "days_expected": n_days_expected,
        "complete_period": n_days_present == n_days_expected,
    }


def period_over_period_change(
    daily_df: pd.DataFrame,
    organization: str,
    site: str | None,
    period_end: date,
    period_length_days: int,
) -> dict:
    """
    Compare [period_end - period_length_days + 1, period_end] against the
    immediately preceding period of the same length. site=None aggregates
    the whole organization (all its sites summed). Works for both
    week-over-week (period_length_days=7) and month-over-month
    (period_length_days=28/30/actual calendar month — caller's choice)
    by construction.
    """
    current_start = period_end - timedelta(days=period_length_days - 1)
    previous_end = current_start - timedelta(days=1)
    previous_start = previous_end - timedelta(days=period_length_days - 1)

    current = total_and_average(daily_df, organization, site, current_start, period_end)
    previous = total_and_average(daily_df, organization, site, previous_start, previous_end)

    if not current["available"] or not previous["available"]:
        return {
            "available": False,
            "reason": "missing data in current or previous period",
            "current": current,
            "previous": previous,
        }

    abs_delta = current["total_kwh"] - previous["total_kwh"]
    pct_delta = (abs_delta / previous["total_kwh"] * 100) if previous["total_kwh"] else None

    return {
        "available": True,
        "organization": organization,
        "site": site if site else "(entire organization)",
        "current_period": {"start": str(current_start), "end": str(period_end), "total_kwh": current["total_kwh"]},
        "previous_period": {"start": str(previous_start), "end": str(previous_end), "total_kwh": previous["total_kwh"]},
        "absolute_change_kwh": round(abs_delta, 2),
        "percent_change": round(pct_delta, 2) if pct_delta is not None else None,
        "both_periods_complete": current["complete_period"] and previous["complete_period"],
    }


def baseload_vs_operational(
    site_interval_series: pd.Series, sampling_minutes: int = 5, baseload_percentile: float = 5.0
) -> dict:
    """
    Split consumption into baseload (the "always-on" floor) vs operational
    (activity-driven load on top of it).

    Method: baseload_kw = the given low percentile (default 5th) of the
    interval demand distribution — i.e. "the load level consumption rarely
    drops below", a standard, defensible baseload proxy. Operational energy
    = total energy - (baseload_kw * total_hours).

    sampling_minutes is used to convert interval ENERGY (kWh per interval)
    into DEMAND (kW), since baseload is a power-level concept.
    """
    if site_interval_series.empty:
        return {"available": False, "reason": "no interval data available for this period"}

    demand_kw = site_interval_series / (sampling_minutes / 60.0)  # kWh per interval -> kW
    baseload_kw = float(demand_kw.quantile(baseload_percentile / 100.0))

    total_hours = len(site_interval_series) * (sampling_minutes / 60.0)
    total_kwh = float(site_interval_series.sum())
    baseload_kwh = baseload_kw * total_hours
    operational_kwh = max(total_kwh - baseload_kwh, 0.0)

    return {
        "available": True,
        "baseload_kw": round(baseload_kw, 2),
        "baseload_kwh": round(baseload_kwh, 2),
        "operational_kwh": round(operational_kwh, 2),
        "total_kwh": round(total_kwh, 2),
        "baseload_share_pct": round(baseload_kwh / total_kwh * 100, 1) if total_kwh else None,
        "method": f"baseload = {baseload_percentile}th percentile of interval demand",
    }


def peak_demand(site_interval_series: pd.Series, sampling_minutes: int = 5) -> dict:
    """Peak demand (kW) and when it occurred, over whatever period the series covers."""
    if site_interval_series.empty:
        return {"available": False, "reason": "no interval data available for this period"}

    demand_kw = site_interval_series / (sampling_minutes / 60.0)
    peak_ts = demand_kw.idxmax()
    return {
        "available": True,
        "peak_demand_kw": round(float(demand_kw.max()), 2),
        "peak_timestamp": str(peak_ts),
        "period_start": str(site_interval_series.index.min()),
        "period_end": str(site_interval_series.index.max()),
    }


def load_factor(site_interval_series: pd.Series, sampling_minutes: int = 5) -> dict:
    """
    Load factor = average demand / peak demand (0-1). High load factor means
    flat, efficient usage; low means peaky, spiky usage. The kWh->kW
    conversion factor is constant across a series with uniform sampling, so
    it cancels in this ratio — computing directly on interval energies gives
    the same result as converting to demand first.
    """
    if site_interval_series.empty:
        return {"available": False, "reason": "no interval data available for this period"}

    avg_val = float(site_interval_series.mean())
    peak_val = float(site_interval_series.max())
    if peak_val == 0:
        return {"available": False, "reason": "peak demand is zero, load factor undefined"}

    demand_kw = site_interval_series / (sampling_minutes / 60.0)
    return {
        "available": True,
        "load_factor": round(avg_val / peak_val, 3),
        "average_demand_kw": round(float(demand_kw.mean()), 2),
        "peak_demand_kw": round(float(demand_kw.max()), 2),
    }


def weekday_weekend_profile(daily_df: pd.DataFrame, organization: str, site: str | None, start: date, end: date) -> dict:
    """Average daily consumption split by weekday vs weekend. site=None aggregates the whole organization."""
    df = _filter_date_range(_filter_site(daily_df, organization, site), start, end)
    if df.empty:
        return {"available": False, "reason": "no data in requested range"}

    df["is_weekend"] = df["date"].dt.dayofweek >= 5  # 5=Sat, 6=Sun
    weekday_avg = df.loc[~df["is_weekend"], "kwh"].mean()
    weekend_avg = df.loc[df["is_weekend"], "kwh"].mean()

    return {
        "available": True,
        "organization": organization,
        "site": site if site else "(entire organization)",
        "weekday_avg_kwh": round(float(weekday_avg), 2) if pd.notna(weekday_avg) else None,
        "weekend_avg_kwh": round(float(weekend_avg), 2) if pd.notna(weekend_avg) else None,
        "weekend_to_weekday_ratio": (
            round(float(weekend_avg / weekday_avg), 3) if pd.notna(weekday_avg) and weekday_avg else None
        ),
        "n_weekday_days": int((~df["is_weekend"]).sum()),
        "n_weekend_days": int(df["is_weekend"].sum()),
    }


def rank_sites(
    daily_df: pd.DataFrame,
    start: date,
    end: date,
    normalize_by: dict | None = None,
    normalize_requested: bool = False,
) -> dict:
    """
    Rank all (organization, site) pairs by total consumption over the period.

    normalize_by: optional {(organization, site): denominator_value} to
    divide each site's total by (e.g. kg produced, guest-nights) — needed
    because raw kWh isn't a meaningful cross-industry comparison (a food
    plant and a hotel have fundamentally different load profiles). Without
    this, ranking is raw-kWh only and callers should be warned it's not
    apples-to-apples across organizations.

    normalize_requested: pass True when the caller asked for normalization
    specifically, even if normalize_by ended up empty/unusable — this lets
    the result distinguish "user didn't ask for normalization" from
    "normalization was requested but no denominator data exists", which
    otherwise look identical to a caller (both fall back to raw kWh).
    """
    df = _filter_date_range(daily_df, start, end)
    if df.empty:
        return {"available": False, "reason": "no data in requested range"}

    grouped = df.groupby(["organization", "site"])["kwh"].sum(min_count=1).reset_index()
    grouped["metric"] = grouped["kwh"]
    normalized = bool(normalize_by)  # empty dict or None -> False
    if normalized:
        grouped["metric"] = grouped.apply(
            lambda r: r["kwh"] / normalize_by.get((r["organization"], r["site"]), float("nan")),
            axis=1,
        )

    grouped = grouped.sort_values("metric", ascending=False).reset_index(drop=True)
    grouped["rank"] = grouped.index + 1

    if normalized:
        warning = None
    elif normalize_requested:
        warning = (
            "Normalization was requested, but no activity data (e.g. production volume, "
            "guest-nights) is available in this system for the requested sites/period — "
            "this system's metering does not currently capture that data. Falling back to "
            "raw kWh, which is NOT meaningful across different industries."
        )
    else:
        warning = (
            "Ranking is on raw kWh — not meaningful across different industries "
            "(e.g. a hotel vs a food plant) without a normalization denominator."
        )

    return {
        "available": True,
        "normalized": normalized,
        "warning": warning,
        "rankings": grouped[["rank", "organization", "site", "kwh", "metric"]].to_dict(orient="records"),
    }


def detect_anomalies(
    daily_df: pd.DataFrame,
    organization: str,
    site: str | None,
    start: date,
    end: date,
    lookback_occurrences: int = 8,
    mad_threshold: float = 4.0,
) -> dict:
    """
    Flag days whose consumption is unusual RELATIVE TO THE SAME WEEKDAY'S
    recent history — not relative to the site's overall flat median.
    site=None aggregates the whole organization (all its sites summed)
    before running the same detector on that aggregate series.

    Why: a naive flat-median/z-score check misfires on ordinary weekly
    seasonality (e.g. every Saturday being busier than every Tuesday isn't
    an anomaly). Comparing each day only against its own weekday's recent
    baseline (the last `lookback_occurrences` occurrences of that weekday)
    filters out that seasonality. Uses a MAD-based robust score for the
    same reason as the data-cleaning layer: it isn't distorted by the very
    outliers it's trying to find.

    KNOWN LIMITATION: this does not account for longer cycles (e.g. we
    found a ~28-day recurring pattern in one Beta Resort submeter during
    data-quality checks). A day-of-week baseline won't catch or misfire on
    a monthly cycle either way, since it only compares within the same
    weekday. Longer-cycle-aware detection (e.g. STL decomposition) would be
    a natural next iteration given more time — worth saying exactly this in
    the discussion round.
    """
    df = _filter_site(daily_df, organization, site).sort_values("date").reset_index(drop=True)
    if df.empty:
        return {"available": False, "reason": "no data for this site"}

    df["dow"] = df["date"].dt.dayofweek
    anomalies = []

    period_mask = (df["date"] >= pd.Timestamp(start)) & (df["date"] <= pd.Timestamp(end))
    for idx in df[period_mask].index:
        row = df.loc[idx]
        same_dow_history = df.loc[: idx - 1]
        same_dow_history = same_dow_history[same_dow_history["dow"] == row["dow"]].tail(lookback_occurrences)
        if len(same_dow_history) < 3:
            continue  # not enough history to judge this day yet

        median = same_dow_history["kwh"].median()
        mad = (same_dow_history["kwh"] - median).abs().median()
        if mad == 0 or pd.isna(mad):
            continue
        score = abs(row["kwh"] - median) / (1.4826 * mad)
        if score > mad_threshold:
            anomalies.append(
                {
                    "date": str(row["date"].date()),
                    "kwh": round(float(row["kwh"]), 2),
                    "expected_kwh": round(float(median), 2),
                    "score": round(float(score), 2),
                    "direction": "high" if row["kwh"] > median else "low",
                }
            )

    return {
        "available": True,
        "organization": organization,
        "site": site,
        "method": (
            f"per-weekday MAD score vs trailing {lookback_occurrences} occurrences of the same weekday, "
            f"threshold={mad_threshold}"
        ),
        "known_limitation": "does not detect longer (e.g. monthly) cycles — see docstring",
        "anomalies": anomalies,
        "n_anomalies": len(anomalies),
    }