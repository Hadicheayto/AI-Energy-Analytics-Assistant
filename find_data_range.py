"""
Find the actual usable date range for this Wattics test account.

Scans backward month-by-month from the current month using a known-reliable
meter (Alpha Hotel's "Bar", id=757, confirmed to have June 2026 data) until
it hits a month with no data, or a max lookback limit. This tells us the
real window we can safely bulk-pull for the assessment, instead of guessing.

Usage:
    python find_data_range.py
"""

import os
import sys
from datetime import date

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from wattics_client import WatticsClient  # noqa: E402

TOKEN = os.environ.get("WATTICS_API_TOKEN")
PROBE_METER_ID = 757  # Alpha Hotel "Bar" — confirmed to have June 2026 data
MAX_MONTHS_BACK = 36  # don't scan forever


def month_iter_backward(start_year: int, start_month: int, n: int):
    y, m = start_year, start_month
    for _ in range(n):
        yield y, m
        m -= 1
        if m == 0:
            m = 12
            y -= 1


def main():
    if not TOKEN:
        print("ERROR: set WATTICS_API_TOKEN in your .env file first.")
        sys.exit(1)

    client = WatticsClient(api_token=TOKEN)
    today = date.today()

    print(f"Probing meter_id={PROBE_METER_ID} backward from {today.year}-{today.month:02d}...")

    months_with_data = []
    first_gap = None

    for y, m in month_iter_backward(today.year, today.month, MAX_MONTHS_BACK):
        recs = client.get_month_consumption(PROBE_METER_ID, year=y, month=m, detailed=False)
        has_data = bool(recs)
        print(f"  {y}-{m:02d}: {'data (' + str(len(recs)) + ' days)' if has_data else 'EMPTY'}")
        if has_data:
            months_with_data.append((y, m))
        elif first_gap is None:
            first_gap = (y, m)
            # Keep scanning a few more months in case of a one-off gap,
            # rather than stopping at the very first empty month.

    print("\n== Summary ==")
    if not months_with_data:
        print("  No data found at all in the scanned window — something else is wrong.")
        return

    months_with_data.sort()
    print(f"  Earliest month with data: {months_with_data[0][0]}-{months_with_data[0][1]:02d}")
    print(f"  Latest month with data:   {months_with_data[-1][0]}-{months_with_data[-1][1]:02d}")
    print(f"  Total months with data in scanned window: {len(months_with_data)} / {MAX_MONTHS_BACK}")

    # Flag any gaps in the middle of the range (not just leading/trailing).
    contiguous_expected = set()
    y, m = months_with_data[-1]
    for _ in range(len(months_with_data)):
        contiguous_expected.add((y, m))
        m -= 1
        if m == 0:
            m = 12
            y -= 1
    gaps = sorted(contiguous_expected - set(months_with_data))
    if gaps:
        print(f"  NOTE: gaps found within the range: {gaps}")
    else:
        print("  No gaps — data is contiguous across this range.")


if __name__ == "__main__":
    main()