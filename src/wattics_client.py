"""
Wattics (arkEMIS) API client.

Responsibilities:
  - Discover organizations, sites, meters (no hardcoded IDs).
  - Pull monthly consumption data per meter, with local JSON caching
    so we never re-hit the API for a (meter, year, month) we already have.
  - Normalize the slightly odd API response shapes (string "X Watt" values,
    per-day nested detailed dicts) into plain Python types.

This module does NOT do resampling/gap-filling/analytics — that's the next
layer (data_processing.py). Keeping this file dumb-and-honest to the API
makes it easy to unit test independently of pandas logic.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, asdict
from datetime import date
from typing import Any

import requests

API_BASE = "https://api.wattics.com/api/v1"
CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "cache")
DISCOVERY_CACHE_FILENAME = "discovery.json"

# Matches strings like "300500.0 Watt" -> 300500.0
_VALUE_RE = re.compile(r"^\s*(-?[\d.]+)\s*([A-Za-z]+)\s*$")


class WatticsAPIError(RuntimeError):
    """Raised for non-2xx responses or malformed payloads from the Wattics API."""


@dataclass(frozen=True)
class Meter:
    id: int
    reference: str
    name: str
    type: str
    process_sampling_rate_minutes: int | None
    unit: str | None
    wh_per_pulse: float | None
    reading: str | None
    organization_id: int
    organization_name: str
    site_id: int
    site_name: str


class WatticsClient:
    def __init__(self, api_token: str, cache_dir: str = CACHE_DIR, request_timeout: int = 30):
        """
        api_token should come from an environment variable (e.g. WATTICS_API_TOKEN),
        never hardcoded in source — see run_discovery.py for the expected pattern.
        """
        self.api_token = api_token
        self.cache_dir = cache_dir
        self.timeout = request_timeout
        os.makedirs(self.cache_dir, exist_ok=True)

    # ---------------------------------------------------------------- HTTP

    def _get(self, path: str, params: dict | None = None) -> Any:
        url = f"{API_BASE}{path}"
        headers = {
            "Content-Type": "application/json",
            "Authorization": self.api_token,
        }
        last_err = None
        for attempt in range(3):  # simple retry for transient failures
            try:
                resp = requests.get(url, headers=headers, params=params, timeout=self.timeout)
            except requests.RequestException as e:
                last_err = e
                time.sleep(1.5 * (attempt + 1))
                continue

            if resp.status_code == 429:
                # rate limited — back off and retry
                last_err = WatticsAPIError(f"429 rate limited on {url}")
                time.sleep(2.0 * (attempt + 1))
                continue
            if not resp.ok:
                raise WatticsAPIError(
                    f"GET {url} params={params} failed: {resp.status_code} {resp.text[:300]}"
                )
            try:
                return resp.json()
            except ValueError as e:
                raise WatticsAPIError(f"Non-JSON response from {url}: {resp.text[:300]}") from e

        raise WatticsAPIError(f"GET {url} failed after retries: {last_err}")

    # ------------------------------------------------------------ discovery

    def list_organizations(self) -> list[dict]:
        return self._get("/organizations")

    def list_sites(self, organization_id: int) -> list[dict]:
        return self._get("/sites", params={"organization_id": organization_id})

    def list_meters(self, organization_id: int, site_id: int) -> list[dict]:
        return self._get("/meters", params={"organization_id": organization_id, "site_id": site_id})

    def discover_meters(self, organization_names: list[str] | None = None) -> list[Meter]:
        """
        Walk organizations -> sites -> meters and return a flat list of Meter
        objects. If organization_names is given, only those orgs (matched
        case-insensitively) are included — everything else is discovered
        dynamically, never hardcoded by ID.
        """
        meters: list[Meter] = []
        orgs = self.list_organizations()

        wanted = {n.lower() for n in organization_names} if organization_names else None

        for org in orgs:
            if wanted is not None and org["name"].lower() not in wanted:
                continue
            sites = self.list_sites(org["id"])
            for site in sites:
                raw_meters = self.list_meters(org["id"], site["id"])
                for m in raw_meters:
                    meters.append(
                        Meter(
                            id=m["id"],
                            reference=m.get("reference", ""),
                            name=m.get("name", ""),
                            type=m.get("type", ""),
                            process_sampling_rate_minutes=m.get("process_sampling_rate_minutes"),
                            unit=m.get("unit"),
                            wh_per_pulse=m.get("wh_per_pulse"),
                            reading=m.get("reading"),
                            organization_id=org["id"],
                            organization_name=org["name"],
                            site_id=site["id"],
                            site_name=site["name"],
                        )
                    )
        return meters

    def discover_meters_cached(
        self,
        organization_names: list[str] | None = None,
        max_age_hours: float = 24.0,
        force_refresh: bool = False,
    ) -> tuple[list[Meter], dict]:
        """
        Same result as discover_meters(), but reads from a local cache file
        (cache/discovery.json) when it's fresh enough, instead of always
        hitting the API. Discovery rarely changes (orgs/sites/meters are
        added occasionally, not every session), so re-querying it on every
        app startup is unnecessary API load.

        Returns (meters, info) where info describes what happened:
          {"source": "cache", "age_hours": ...}  — served from disk
          {"source": "api", "diff": {"added": [...], "removed": [...]}}
            — freshly fetched; diff compares against whatever was cached
            before (by (organization, site, meter) name triples, since IDs
            are stable but names are what a human recognizes). A non-empty
            diff means a real site/meter was added or removed upstream and
            should be reviewed — new meters are never auto-classified into
            site-total/normalization roles, since that's a judgment call
            (see meter_registry.py).
        """
        path = os.path.join(self.cache_dir, DISCOVERY_CACHE_FILENAME)

        old_cached = None
        if os.path.exists(path):
            with open(path, "r") as f:
                old_cached = json.load(f)

            if not force_refresh:
                age_hours = (time.time() - old_cached["fetched_at"]) / 3600
                if age_hours <= max_age_hours:
                    meters = [Meter(**m) for m in old_cached["meters"]]
                    return meters, {"source": "cache", "age_hours": round(age_hours, 2)}

        fresh = self.discover_meters(organization_names=organization_names)

        diff = None
        if old_cached is not None:
            old_keys = {(m["organization_name"], m["site_name"], m["name"]) for m in old_cached["meters"]}
            new_keys = {(m.organization_name, m.site_name, m.name) for m in fresh}
            diff = {
                "added": sorted(new_keys - old_keys),
                "removed": sorted(old_keys - new_keys),
            }

        with open(path, "w") as f:
            json.dump({"fetched_at": time.time(), "meters": [asdict(m) for m in fresh]}, f)

        return fresh, {"source": "api", "diff": diff}

    # ---------------------------------------------------------- consumption

    def _cache_path(self, meter_id: int, year: int, month: int, detailed: bool) -> str:
        suffix = "detailed" if detailed else "daily"
        return os.path.join(self.cache_dir, f"meter{meter_id}_{year}_{month:02d}_{suffix}.json")

    def get_month_consumption(
        self, meter_id: int, year: int, month: int, detailed: bool = True, force_refresh: bool = False
    ) -> list[dict]:
        """
        Return the raw (but value-parsed) list of daily consumption records
        for one meter/month, e.g.:
            [{"date": "2018-06-01", "weekday": "Friday", "total_consumption_value": 300500.0,
              "total_consumption_unit": "Watt",
              "consumption_by_time": {"00h00m": 1024.03, ...}}, ...]

        Cached to disk per (meter_id, year, month, detailed) — never re-fetched
        unless force_refresh=True. This is the ONLY function that hits the
        network for consumption data; everything above this layer should read
        through here.
        """
        cache_path = self._cache_path(meter_id, year, month, detailed)
        if not force_refresh and os.path.exists(cache_path):
            with open(cache_path, "r") as f:
                return json.load(f)

        raw = self._get(
            f"/meters/{meter_id}/consumptions",
            params={"month": month, "year": year, "detailed": str(detailed).lower()},
        )
        if not isinstance(raw, list):
            raise WatticsAPIError(f"Unexpected consumptions payload for meter {meter_id}: {raw!r}")

        parsed = [self._parse_daily_record(rec) for rec in raw]

        with open(cache_path, "w") as f:
            json.dump(parsed, f)

        return parsed

    @staticmethod
    def _parse_value_unit(s: str | None) -> tuple[float | None, str | None]:
        if s is None:
            return None, None
        m = _VALUE_RE.match(s)
        if not m:
            return None, None
        return float(m.group(1)), m.group(2)

    def _parse_daily_record(self, rec: dict) -> dict:
        value, unit = self._parse_value_unit(rec.get("total_consumption"))
        out = {
            "date": rec["date"],
            "weekday": rec.get("weekday"),
            "total_consumption_value": value,
            "total_consumption_unit": unit,  # NB: API labels this "Watt" even though
            # it's a daily total, i.e. an energy quantity — see README note on
            # unit mislabeling. We keep the raw label but treat the value as Wh
            # in the processing layer after a sanity cross-check against the
            # summed detailed intervals.
        }
        if "consumption_by_time" in rec:
            out["consumption_by_time"] = {
                t: self._parse_value_unit(v)[0] for t, v in rec["consumption_by_time"].items()
            }
        return out

    def get_range_consumption(
        self, meter_id: int, start: date, end: date, detailed: bool = True
    ) -> list[dict]:
        """
        Convenience wrapper: walks month-by-month between start and end
        (inclusive) and concatenates daily records, de-duplicating by date
        in case of overlapping fetches. Filters out days outside [start, end].
        """
        records: dict[str, dict] = {}
        y, m = start.year, start.month
        while (y, m) <= (end.year, end.month):
            for rec in self.get_month_consumption(meter_id, y, m, detailed=detailed):
                records[rec["date"]] = rec  # last write wins; cache is stable anyway
            m += 1
            if m > 12:
                m = 1
                y += 1

        start_s, end_s = start.isoformat(), end.isoformat()
        return sorted(
            (r for d, r in records.items() if start_s <= d <= end_s),
            key=lambda r: r["date"],
        )