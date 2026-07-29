"""
Scale test: proves (not just argues) that the processing/analytics layer
holds up at the assessment's stated stress scenario — "a year of 15-minute
data across forty meters" — by generating synthetic data at that exact
scale and running it through our REAL code (data_processing.py,
energy_analytics.py), timing each step.

This does NOT touch the Wattics API — it's pure synthetic data, so it's
fast, free, and safe to run repeatedly. It tests the code paths, not the
real numbers.

Usage:
    python test_scale.py
"""

import os
import sys
import time
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from data_processing import resample  # noqa: E402
import energy_analytics as ea  # noqa: E402

N_METERS = 40
DAYS = 365
INTERVAL_MINUTES = 15
INTERVALS_PER_DAY = 24 * 60 // INTERVAL_MINUTES  # 96


def make_synthetic_detailed_df() -> pd.DataFrame:
    """Generate a fake interval-level dataset at the target scale."""
    start = datetime(2025, 1, 1)
    n_intervals = DAYS * INTERVALS_PER_DAY
    timestamps = [start + timedelta(minutes=INTERVAL_MINUTES * i) for i in range(n_intervals)]

    rng = np.random.default_rng(42)
    frames = []
    for meter_id in range(1, N_METERS + 1):
        # Realistic-ish shape: a daily sine wave (day/night cycle) + noise, always positive.
        hours = np.array([t.hour + t.minute / 60 for t in timestamps])
        base = 5 + 3 * np.sin((hours - 6) / 24 * 2 * np.pi)
        noise = rng.normal(0, 0.5, size=n_intervals)
        values = np.clip(base + noise, 0.1, None)

        frames.append(
            pd.DataFrame(
                {
                    "organization": "Synthetic Org",
                    "site": f"Synthetic Site {meter_id % 4}",  # spread across 4 fake sites
                    "meter_id": meter_id,
                    "meter_name": f"Meter {meter_id}",
                    "meter_type": "electricity",
                    "timestamp": timestamps,
                    "value": values,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def make_synthetic_daily_df() -> pd.DataFrame:
    """Generate a fake daily site-total dataset covering 3 years, 4 sites (matches build_site_total_daily's output shape)."""
    dates = pd.date_range("2023-08-01", periods=3 * 365, freq="D")
    rng = np.random.default_rng(7)
    frames = []
    for site_idx in range(4):
        kwh = rng.normal(1000, 150, size=len(dates)).clip(min=10)
        frames.append(
            pd.DataFrame(
                {
                    "organization": "Synthetic Org",
                    "site": f"Synthetic Site {site_idx}",
                    "date": dates,
                    "kwh": kwh,
                    "meters_reporting": 5,
                    "meters_expected": 5,
                    "is_complete_day": True,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def timed(label, fn, *args, **kwargs):
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed = time.perf_counter() - t0
    print(f"  {label}: {elapsed:.3f}s")
    return result


def main():
    print(f"Generating synthetic detailed data: {N_METERS} meters x {DAYS} days x "
          f"{INTERVALS_PER_DAY} intervals/day...")
    t0 = time.perf_counter()
    detailed_df = make_synthetic_detailed_df()
    print(f"  Generated {len(detailed_df):,} rows in {time.perf_counter() - t0:.2f}s\n")

    print("== Processing layer ==")
    site_series = timed("sum_detailed_to_site_series", ea.sum_detailed_to_site_series, detailed_df)
    timed("resample to hourly", resample, detailed_df, "timestamp", "value", "h", ["meter_id"])
    timed("resample to daily", resample, detailed_df, "timestamp", "value", "D", ["meter_id"])
    timed("resample to monthly", resample, detailed_df, "timestamp", "value", "ME", ["meter_id"])

    print("\n== Analytics layer (interval-based) ==")
    timed("peak_demand", ea.peak_demand, site_series, sampling_minutes=INTERVAL_MINUTES)
    timed("load_factor", ea.load_factor, site_series, sampling_minutes=INTERVAL_MINUTES)
    timed("baseload_vs_operational", ea.baseload_vs_operational, site_series, sampling_minutes=INTERVAL_MINUTES)

    print("\nGenerating synthetic daily data: 4 sites x 3 years...")
    daily_df = make_synthetic_daily_df()
    print(f"  Generated {len(daily_df):,} rows")

    print("\n== Analytics layer (daily-based) ==")
    start, end = date(2025, 1, 1), date(2025, 12, 31)
    timed(
        "total_and_average",
        ea.total_and_average,
        daily_df, "Synthetic Org", "Synthetic Site 0", start, end,
    )
    timed(
        "period_over_period_change",
        ea.period_over_period_change,
        daily_df, "Synthetic Org", "Synthetic Site 0", end, 30,
    )
    timed(
        "detect_anomalies",
        ea.detect_anomalies,
        daily_df, "Synthetic Org", "Synthetic Site 0", start, end,
    )
    timed(
        "rank_sites",
        ea.rank_sites,
        daily_df, start, end,
    )

    print("\n== Memory footprint ==")
    print(f"  detailed_df: {detailed_df.memory_usage(deep=True).sum() / 1e6:.1f} MB")
    print(f"  daily_df: {daily_df.memory_usage(deep=True).sum() / 1e6:.1f} MB")

    print("\nAll steps completed without error at the target scale "
          f"({N_METERS} meters, {DAYS} days, {INTERVAL_MINUTES}-min resolution).")


if __name__ == "__main__":
    main()