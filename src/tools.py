"""
Tool layer: wraps the pure analytics functions (energy_analytics.py) as
LLM-callable tools with explicit JSON schemas.

CRITICAL invariant: the LLM never receives raw rows. Every tool here returns
only the small structured dict that the underlying analytics function
already produces (totals, a peak value + timestamp, a ranking list, etc).
All the actual math happens in energy_analytics.py / data_processing.py —
this file is purely plumbing between "LLM tool call" and "call the right
Python function with the right pre-loaded data."

ToolExecutor loads the unified daily table and per-site detailed series
ONCE at construction time (not per question) — this is the "cache locally,
don't hit the API per question" requirement, one level up: even within a
single running session, we don't rebuild pandas tables on every call.
"""

from __future__ import annotations

import json
import os
import time
from datetime import date, datetime

from wattics_client import WatticsClient
from meter_registry import SITE_CONFIGS, resolve_site_meters
from data_processing import build_site_total_daily, load_detailed_long
import energy_analytics as ea

ORG_NAMES = ["Food Corp.", "Best Resorts Hotels"]


# ------------------------------------------------------------- tool schemas

TOOL_SCHEMAS = [
    {
        "name": "list_organizations_and_sites",
        "description": (
            "List every valid organization and site name available in this system. "
            "Call this FIRST if you are unsure of the exact organization or site name "
            "the user means, rather than guessing — organization/site parameters in "
            "other tools must match these exact names."
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_total_and_average_consumption",
        "description": (
            "Total and average daily electricity consumption (kWh) for a site, or for an "
            "entire organization if site is omitted (sums across all of that organization's "
            "sites, e.g. 'Best Resorts Hotels' as a whole = Alpha Hotel + Beta Resort & Spa)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "organization": {"type": "string"},
                "site": {"type": "string", "description": "Omit to aggregate across the whole organization."},
                "start_date": {"type": "string", "description": "YYYY-MM-DD, inclusive"},
                "end_date": {"type": "string", "description": "YYYY-MM-DD, inclusive"},
            },
            "required": ["organization", "start_date", "end_date"],
        },
    },
    {
        "name": "get_period_over_period_change",
        "description": (
            "Compare a site's (or an entire organization's, if site is omitted) consumption "
            "in a recent period against the immediately preceding period of equal length — "
            "use period_length_days=7 for week-over-week or period_length_days=30 for "
            "month-over-month (a rolling 30-day window, not a calendar month). Returns "
            "absolute and percentage change."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "organization": {"type": "string"},
                "site": {"type": "string", "description": "Omit to aggregate across the whole organization."},
                "period_end_date": {"type": "string", "description": "YYYY-MM-DD, last day of the current period"},
                "period_length_days": {"type": "integer", "description": "7 for week-over-week, 30 for month-over-month"},
            },
            "required": ["organization", "period_end_date", "period_length_days"],
        },
    },
    {
        "name": "get_weekday_weekend_profile",
        "description": (
            "Average daily consumption split by weekday vs weekend, for a site or for an "
            "entire organization if site is omitted."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "organization": {"type": "string"},
                "site": {"type": "string", "description": "Omit to aggregate across the whole organization."},
                "start_date": {"type": "string"},
                "end_date": {"type": "string"},
            },
            "required": ["organization", "start_date", "end_date"],
        },
    },
    {
        "name": "get_site_ranking",
        "description": (
            "Rank ALL sites across both organizations by total consumption over a date range. "
            "By default this ranks raw kWh, which is NOT a fair comparison across different "
            "industries (a hotel vs a food plant) — the result includes an explicit warning "
            "when unnormalized. normalize=true attempts to rank by consumption per unit of "
            "activity (kg produced, guest-nights) instead, BUT this system's metering does not "
            "currently capture that activity data, so normalize=true will currently fall back "
            "to raw kWh with a warning explaining why — mention this limitation if the user asks "
            "for a normalized/efficiency comparison."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "start_date": {"type": "string"},
                "end_date": {"type": "string"},
                "normalize": {"type": "boolean", "description": "default false"},
            },
            "required": ["start_date", "end_date"],
        },
    },
    {
        "name": "get_anomalies",
        "description": (
            "Detect unusual consumption days for a site, or an entire organization if site is "
            "omitted (sums all its sites first). Compares each day only against recent "
            "occurrences of the SAME weekday (not a flat average), so normal weekly patterns "
            "aren't misflagged. Does not detect longer (e.g. monthly) cycles — this limitation "
            "is stated in the tool's own output."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "organization": {"type": "string"},
                "site": {"type": "string", "description": "Omit to aggregate across the whole organization."},
                "start_date": {"type": "string"},
                "end_date": {"type": "string"},
            },
            "required": ["organization", "start_date", "end_date"],
        },
    },
    {
        "name": "get_peak_demand",
        "description": (
            "Peak electricity demand (kW) and when it occurred, for one site. Only available "
            "for the last ~3 months (interval-level data isn't retained further back in this "
            "system); returns available=false with a reason if the range isn't covered."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "organization": {"type": "string"},
                "site": {"type": "string"},
                "start_date": {"type": "string"},
                "end_date": {"type": "string"},
            },
            "required": ["organization", "site", "start_date", "end_date"],
        },
    },
    {
        "name": "get_load_factor",
        "description": (
            "Load factor (average demand / peak demand, 0-1) for one site — higher means "
            "flatter/more efficient usage, lower means peakier/spikier. Same ~3-month data "
            "availability limit as get_peak_demand."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "organization": {"type": "string"},
                "site": {"type": "string"},
                "start_date": {"type": "string"},
                "end_date": {"type": "string"},
            },
            "required": ["organization", "site", "start_date", "end_date"],
        },
    },
    {
        "name": "get_baseload_vs_operational",
        "description": (
            "Split one site's consumption into baseload (the constant 'always-on' floor, "
            "estimated as the 5th percentile of demand) vs operational (activity-driven load "
            "on top of it). Same ~3-month data availability limit as get_peak_demand."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "organization": {"type": "string"},
                "site": {"type": "string"},
                "start_date": {"type": "string"},
                "end_date": {"type": "string"},
            },
            "required": ["organization", "site", "start_date", "end_date"],
        },
    },
]


# --------------------------------------------------------------- executor

def _parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


class ToolExecutor:
    def __init__(self, api_token: str):
        client = WatticsClient(api_token=api_token)

        self.all_meters, discovery_info = client.discover_meters_cached(organization_names=ORG_NAMES)
        if discovery_info["source"] == "cache":
            print(f"[ToolExecutor] discovery loaded from cache (age: {discovery_info['age_hours']}h)")
        else:
            diff = discovery_info["diff"]
            print("[ToolExecutor] discovery refreshed from live API")
            if diff and (diff["added"] or diff["removed"]):
                if diff["added"]:
                    print(f"  NEW meters/sites found since last discovery: {diff['added']}")
                if diff["removed"]:
                    print(f"  Meters/sites no longer present: {diff['removed']}")
                print("  -> New meters are NOT auto-classified — review meter_registry.py "
                      "if any of these should be added as a site-total/normalization meter.")

        self.resolved_sites = {
            (cfg.organization_name, cfg.site_name): resolve_site_meters(self.all_meters, cfg)
            for cfg in SITE_CONFIGS
        }

        # Refresh the CURRENT (in-progress) month's daily data for every
        # meter we actually use, every time the app starts. A month's cache
        # file is otherwise treated as permanent once written, but the
        # current month is never really "done" — yesterday's or today's
        # numbers wouldn't show up without this, potentially for weeks,
        # until the month rolls over and bulk_extract.py is rerun manually.
        self._refresh_current_month(client)

        # Loaded once, reused for every tool call in this session.
        self.daily_df = build_site_total_daily(self.all_meters, SITE_CONFIGS)
        self._detailed_series_cache: dict[tuple[str, str], object] = {}

        # Normalization denominators: total normalization-meter value over
        # the SAME period is computed lazily per ranking call, since it
        # depends on the requested date range (kg produced this month vs
        # this year are different denominators).

    def _refresh_current_month(self, client: WatticsClient, max_age_hours: float = 12.0):
        """
        Only actually re-hits the API if the current month hasn't been
        refreshed recently — same freshness-check pattern as discovery
        caching. Without this, every app startup would force 35 live API
        calls, which is exactly the "don't call the API each time" problem
        this whole project was built to avoid.
        """
        today = date.today()
        marker_path = os.path.join(client.cache_dir, "current_month_refresh_state.json")

        if os.path.exists(marker_path):
            with open(marker_path, "r") as f:
                marker = json.load(f)
            same_month = marker.get("year") == today.year and marker.get("month") == today.month
            age_hours = (time.time() - marker.get("refreshed_at", 0)) / 3600
            if same_month and age_hours <= max_age_hours:
                print(f"[ToolExecutor] current month already refreshed {age_hours:.1f}h ago — skipping")
                return

        all_used_meters = []
        for resolved in self.resolved_sites.values():
            all_used_meters.extend(resolved["total_meters"])
            all_used_meters.extend(resolved["normalization_meters"])
            if resolved["weather_meter"]:
                all_used_meters.append(resolved["weather_meter"])

        print(f"[ToolExecutor] refreshing {today.year}-{today.month:02d} daily data for "
              f"{len(all_used_meters)} meters...")
        for m in all_used_meters:
            try:
                client.get_month_consumption(m.id, today.year, today.month, detailed=False, force_refresh=True)
            except Exception as e:  # noqa: BLE001 — a refresh failure shouldn't block startup
                print(f"  WARNING: failed to refresh {m.name} (id={m.id}): {e}")

        with open(marker_path, "w") as f:
            json.dump({"year": today.year, "month": today.month, "refreshed_at": time.time()}, f)

    def _get_site_interval_series(self, organization: str, site: str):
        key = (organization, site)
        if key not in self.resolved_sites:
            return None
        if key not in self._detailed_series_cache:
            detailed = load_detailed_long(self.resolved_sites[key]["total_meters"])
            self._detailed_series_cache[key] = ea.sum_detailed_to_site_series(detailed)
        return self._detailed_series_cache[key]

    def _normalization_totals(self, start: date, end: date) -> dict:
        """(organization, site) -> total normalization-meter value over [start, end]."""
        totals = {}
        for (org, site), resolved in self.resolved_sites.items():
            norm_meters = resolved["normalization_meters"]
            if not norm_meters:
                continue
            from data_processing import load_daily_long

            norm_df = load_daily_long(norm_meters)
            if norm_df.empty:
                continue
            mask = (norm_df["date"] >= str(start)) & (norm_df["date"] <= str(end))
            totals[(org, site)] = norm_df.loc[mask, "value"].sum(skipna=True)
        return totals

    def list_organizations_and_sites(self) -> dict:
        return {
            "sites": [
                {"organization": cfg.organization_name, "site": cfg.site_name}
                for cfg in SITE_CONFIGS
            ]
        }

    def execute(self, tool_name: str, tool_input: dict) -> dict:
        try:
            if tool_name == "list_organizations_and_sites":
                return self.list_organizations_and_sites()

            if tool_name == "get_total_and_average_consumption":
                return ea.total_and_average(
                    self.daily_df,
                    tool_input["organization"],
                    tool_input.get("site"),
                    _parse_date(tool_input["start_date"]),
                    _parse_date(tool_input["end_date"]),
                )

            if tool_name == "get_period_over_period_change":
                return ea.period_over_period_change(
                    self.daily_df,
                    tool_input["organization"],
                    tool_input.get("site"),
                    _parse_date(tool_input["period_end_date"]),
                    tool_input["period_length_days"],
                )

            if tool_name == "get_weekday_weekend_profile":
                return ea.weekday_weekend_profile(
                    self.daily_df,
                    tool_input["organization"],
                    tool_input.get("site"),
                    _parse_date(tool_input["start_date"]),
                    _parse_date(tool_input["end_date"]),
                )

            if tool_name == "get_site_ranking":
                start = _parse_date(tool_input["start_date"])
                end = _parse_date(tool_input["end_date"])
                normalize = tool_input.get("normalize", False)
                normalize_by = self._normalization_totals(start, end) if normalize else None
                return ea.rank_sites(self.daily_df, start, end, normalize_by=normalize_by, normalize_requested=normalize)

            if tool_name == "get_anomalies":
                return ea.detect_anomalies(
                    self.daily_df,
                    tool_input["organization"],
                    tool_input.get("site"),
                    _parse_date(tool_input["start_date"]),
                    _parse_date(tool_input["end_date"]),
                )

            if tool_name in ("get_peak_demand", "get_load_factor", "get_baseload_vs_operational"):
                series = self._get_site_interval_series(tool_input["organization"], tool_input["site"])
                if series is None or series.empty:
                    return {"available": False, "reason": "no interval-level data for this site/period"}
                # Filter the series to the requested range.
                start = _parse_date(tool_input["start_date"])
                end = _parse_date(tool_input["end_date"])
                import pandas as pd

                mask = (series.index.date >= start) & (series.index.date <= end)
                filtered = series[mask]
                if filtered.empty:
                    return {
                        "available": False,
                        "reason": (
                            "no interval-level data in this range — only the last ~3 months "
                            "have 5-minute data in this system"
                        ),
                    }
                if tool_name == "get_peak_demand":
                    return ea.peak_demand(filtered)
                if tool_name == "get_load_factor":
                    return ea.load_factor(filtered)
                return ea.baseload_vs_operational(filtered)

            return {"error": f"Unknown tool: {tool_name}"}

        except Exception as e:  # noqa: BLE001 — tool errors must come back to the LLM, not crash the session
            return {"error": str(e), "tool": tool_name, "input": tool_input}