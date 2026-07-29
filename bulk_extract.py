"""
Bulk extraction: pulls and caches everything the registry says we need.

  - DAILY totals for the full available range (2023-08 through the latest
    complete month) for every meter in meter_registry.SITE_CONFIGS (total
    meters + normalization meters + weather meters).
  - DETAILED (5-min interval) data for the last 3 months only, for total
    meters only (that's where baseload/peak/load-factor analytics need
    interval-level granularity; normalization/weather meters are daily by
    nature already, e.g. Guest Nights, HDD).

Relies entirely on WatticsClient's existing per-(meter, year, month) cache —
rerunning this script is cheap and idempotent; already-cached months are
read from disk, not re-fetched.

Usage:
    python bulk_extract.py
"""

import os
import sys
from datetime import date

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from wattics_client import WatticsClient  # noqa: E402
from meter_registry import SITE_CONFIGS, resolve_site_meters  # noqa: E402

TOKEN = os.environ.get("WATTICS_API_TOKEN")
ORG_NAMES = ["Food Corp.", "Best Resorts Hotels"]

DAILY_START = (2023, 8)   # from find_data_range.py
DETAILED_MONTHS_BACK = 3  # last N months get 5-min interval data


def month_range(start_year: int, start_month: int, end_year: int, end_month: int):
    y, m = start_year, start_month
    while (y, m) <= (end_year, end_month):
        yield y, m
        m += 1
        if m > 12:
            m = 1
            y += 1


def recent_months(n: int, ref: date):
    y, m = ref.year, ref.month
    out = []
    for _ in range(n):
        out.append((y, m))
        m -= 1
        if m == 0:
            m = 12
            y -= 1
    return sorted(out)


def main():
    if not TOKEN:
        print("ERROR: set WATTICS_API_TOKEN in your .env file first.")
        sys.exit(1)

    client = WatticsClient(api_token=TOKEN)
    today = date.today()

    print("Discovering meters...")
    all_meters = client.discover_meters(organization_names=ORG_NAMES)
    print(f"  {len(all_meters)} meters discovered.\n")

    resolved_sites = [resolve_site_meters(all_meters, cfg) for cfg in SITE_CONFIGS]

    daily_months = list(month_range(DAILY_START[0], DAILY_START[1], today.year, today.month))
    detailed_month_set = set(recent_months(DETAILED_MONTHS_BACK, today))

    total_daily_calls = 0
    total_detailed_calls = 0
    errors = []

    for site in resolved_sites:
        label = f"{site['organization_name']} / {site['site_name']}"
        print(f"== {label} ==")

        # total meters: daily for full range + detailed for last N months
        for m in site["total_meters"]:
            print(f"  [total] {m.name} (id={m.id})")
            for y, mo in daily_months:
                try:
                    client.get_month_consumption(m.id, y, mo, detailed=False)
                    total_daily_calls += 1
                except Exception as e:
                    errors.append((label, m.name, "daily", y, mo, str(e)))
            for y, mo in detailed_month_set:
                try:
                    client.get_month_consumption(m.id, y, mo, detailed=True)
                    total_detailed_calls += 1
                except Exception as e:
                    errors.append((label, m.name, "detailed", y, mo, str(e)))

        # normalization + weather meters: daily only, full range
        context_meters = list(site["normalization_meters"])
        if site["weather_meter"]:
            context_meters.append(site["weather_meter"])

        for m in context_meters:
            print(f"  [context] {m.name} (id={m.id})")
            for y, mo in daily_months:
                try:
                    client.get_month_consumption(m.id, y, mo, detailed=False)
                    total_daily_calls += 1
                except Exception as e:
                    errors.append((label, m.name, "daily", y, mo, str(e)))

        print()

    print("== Extraction complete ==")
    print(f"  Daily pulls attempted:    {total_daily_calls}")
    print(f"  Detailed pulls attempted: {total_detailed_calls}")
    print(f"  Errors: {len(errors)}")
    for e in errors[:20]:
        print(f"    {e}")
    if len(errors) > 20:
        print(f"    ... and {len(errors) - 20} more")

    print("\nAll data is now cached under ./cache/ as JSON files.")
    print("Next: build the pandas processing layer that reads this cache into one unified table.")


if __name__ == "__main__":
    main()