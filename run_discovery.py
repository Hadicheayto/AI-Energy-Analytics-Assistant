


"""
Run this locally (not in the sandbox) to test discovery against the real
Wattics API. Requires:

    export WATTICS_API_TOKEN="your_token_here"

Usage:
    python run_discovery.py
"""

import os
import sys

from dotenv import load_dotenv

load_dotenv()  # reads .env file in the current directory and loads it into os.environ

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from wattics_client import WatticsClient  # noqa: E402

TOKEN = os.environ.get("WATTICS_API_TOKEN")
ORG_NAMES = ["Food Corp.", "Best Resorts Hotels"]


def main():
    if not TOKEN:
        print("ERROR: set WATTICS_API_TOKEN in your environment first.")
        sys.exit(1)

    client = WatticsClient(api_token=TOKEN)

    print("== Organizations ==")
    orgs = client.list_organizations()
    for o in orgs:
        print(f"  id={o['id']:<5} name={o['name']}")

    print("\n== Discovering meters for:", ORG_NAMES, "==")
    meters = client.discover_meters(organization_names=ORG_NAMES)

    if not meters:
        print("  No meters found matching those org names exactly.")
        print("  Raw org names returned by API were:", [o["name"] for o in orgs])
        print("  -> If these don't match ORG_NAMES above verbatim (spacing, punctuation,")
        print("     accents), adjust ORG_NAMES and rerun.")
        return

    by_org: dict[str, list] = {}
    for m in meters:
        by_org.setdefault(m.organization_name, []).append(m)

    for org_name, org_meters in by_org.items():
        print(f"\n{org_name}:")
        by_site: dict[str, list] = {}
        for m in org_meters:
            by_site.setdefault(m.site_name, []).append(m)
        for site_name, site_meters in by_site.items():
            print(f"  Site: {site_name}")
            for m in site_meters:
                print(
                    f"    meter_id={m.id:<6} name={m.name:<20} type={m.type:<15} "
                    f"sampling={m.process_sampling_rate_minutes}min unit={m.unit}"
                )

    print(f"\nTotal meters discovered: {len(meters)}")

    # ---- Targeted check: does Alpha Hotel's MAIN meter equal the sum of its
    # ---- electricity submeters? This tells us whether MAIN is the site-total
    # ---- meter we should treat as authoritative (avoiding double-counting).
    ALPHA_MAIN_ID = 759
    ALPHA_SUBMETER_IDS = {
        "Bar": 757,
        "Conference Area": 756,
        "Floor 1": 761,
        "Floor 2": 762,
        "Floor 3": 758,
        "Landlord Area": 763,
        "Lift 1": 760,
        "MCC 1": 754,
        "MCC 2": 755,
    }

    YEAR, MONTH = 2026, 6

    print(f"\n== MAIN vs submeters check: Alpha Hotel, {YEAR}-{MONTH:02d} ==")

    main_recs = client.get_month_consumption(ALPHA_MAIN_ID, year=YEAR, month=MONTH, detailed=False)
    if not main_recs:
        print(f"  No data for MAIN (id={ALPHA_MAIN_ID}) in {YEAR}-{MONTH:02d}. Try a different month.")
        return

    # Use the first day with data for the comparison.
    sample_day = main_recs[0]["date"]
    main_value = main_recs[0]["total_consumption_value"]
    main_unit = main_recs[0]["total_consumption_unit"]
    print(f"  Day used: {sample_day}")
    print(f"  MAIN total = {main_value} {main_unit}")

    submeter_total = 0.0
    print("  Submeters:")
    for name, mid in ALPHA_SUBMETER_IDS.items():
        recs = client.get_month_consumption(mid, year=YEAR, month=MONTH, detailed=False)
        day_rec = next((r for r in recs if r["date"] == sample_day), None)
        if day_rec is None or day_rec["total_consumption_value"] is None:
            print(f"    {name:<18} (id={mid}): no data for {sample_day}")
            continue
        val = day_rec["total_consumption_value"]
        submeter_total += val
        print(f"    {name:<18} (id={mid}): {val} {day_rec['total_consumption_unit']}")

    print(f"\n  Sum of submeters = {submeter_total}")
    if submeter_total:
        print(f"  Ratio MAIN / sum(submeters) = {main_value / submeter_total:.3f}")
        print("  (near 1.0 => MAIN IS the site total, don't also sum submeters separately)")
        print("  (near 0 or way off => submeters cover only part of the site, or MAIN measures a separate circuit)")

    # ---- Same question for Food Corp: is "x point" the site total, or just
    # ---- another submeter alongside Effluent Area?
    FOOD_XPOINT_ID = 110516
    FOOD_EFFLUENT_ID = 751
    FOOD_YEAR, FOOD_MONTH = 2026, 6

    print(f"\n== x point vs Effluent Area check: Organic Farm, {FOOD_YEAR}-{FOOD_MONTH:02d} ==")

    xpoint_recs = client.get_month_consumption(FOOD_XPOINT_ID, year=FOOD_YEAR, month=FOOD_MONTH, detailed=False)
    effluent_recs = client.get_month_consumption(FOOD_EFFLUENT_ID, year=FOOD_YEAR, month=FOOD_MONTH, detailed=False)

    if not xpoint_recs:
        print(f"  x point (id={FOOD_XPOINT_ID}) returned NO data for {FOOD_YEAR}-{FOOD_MONTH:02d}.")
        print("  Scanning other months for any x point data...")
        found_any = False
        for scan_year in (2024, 2025, 2026):
            for scan_month in range(1, 13):
                if scan_year == FOOD_YEAR and scan_month == FOOD_MONTH:
                    continue
                recs = client.get_month_consumption(
                    FOOD_XPOINT_ID, year=scan_year, month=scan_month, detailed=False
                )
                if recs:
                    print(f"    Found data: {scan_year}-{scan_month:02d} -> {len(recs)} day(s), "
                          f"first={recs[0]['date']}={recs[0]['total_consumption_value']}")
                    found_any = True
        if not found_any:
            print("    No data found for x point in any scanned month (2024-2026).")
            print("    -> x point likely isn't an actively reporting meter in this test account;")
            print("       treat it as unusable and exclude it from analytics.")
    if not effluent_recs:
        print(f"  Effluent Area (id={FOOD_EFFLUENT_ID}) returned NO data for {FOOD_YEAR}-{FOOD_MONTH:02d}"
              f" (unexpected — it had data in an earlier test).")

    if xpoint_recs and effluent_recs:
        xpoint_by_date = {r["date"]: r["total_consumption_value"] for r in xpoint_recs}
        effluent_by_date = {r["date"]: r["total_consumption_value"] for r in effluent_recs}
        common_dates = sorted(set(xpoint_by_date) & set(effluent_by_date))
        if not common_dates:
            print("  No overlapping dates between the two meters.")
        else:
            d = common_dates[0]
            xv, ev = xpoint_by_date[d], effluent_by_date[d]
            print(f"  Day used: {d}")
            print(f"  x point   = {xv}")
            print(f"  Effluent Area = {ev}")
            if ev:
                print(f"  Ratio x point / Effluent Area = {xv / ev:.3f}")
            print("  (x point >> Effluent Area => x point looks like the site total)")
            print("  (x point ~= Effluent Area => they may be duplicates/aliases, needs more digging)")


if __name__ == "__main__":
    main()