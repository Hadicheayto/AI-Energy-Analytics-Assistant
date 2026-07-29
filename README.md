# Ark Energy  Natural Language Energy Analytics Assistant

A prototype that lets Ark Energy consultants ask plain-language questions about
electricity consumption at two client organizations  **Food Corp.** (food
production) and **Best Resorts Hotels** (hospitality, two sites: Alpha Hotel
and Beta Resort & Spa)  and get answers computed entirely by Python, with an
LLM handling only question interpretation and response composition.

**Core invariant: the LLM never sees raw data and never computes a number
itself.** Every figure in every answer comes from a pure Python function; the
LLM's job is to pick which function(s) to call and explain the result in
plain language.

---

## Architecture

```
Wattics API
     │  (discovery: orgs → sites → meters; cached, refreshed every 24h)
     ▼
Local JSON cache (cache/)
     │  one file per (meter, year, month, granularity); current month
     │  auto-refreshed if stale (>12h); all past months are permanent
     ▼
data_processing.py + data_cleaning.py
     │  dedup, gap detection, negative-value handling, timezone
     │  documentation, multi-resolution resampling, unified cross-org
     │  daily table (organization, site, date, kwh)
     ▼
energy_analytics.py
     │  pure functions: totals, WoW/MoM, baseload/operational, peak
     │  demand, load factor, weekday/weekend, ranking, anomaly detection
     │   each returns a structured dict, never a string
     ▼
tools.py (ToolExecutor)
     │  wraps each analytics function as an LLM tool with an explicit
     │  JSON schema; loads/caches data ONCE per session, not per question
     ▼
orchestrator.py
     │  OpenAI function-calling loop: the model picks tool(s), the
     │  executor runs them, results loop back until the model has enough
     │  to answer  this is what handles multi-tool questions like
     │  "which org had the bigger WoW increase"
     ▼
app.py (Streamlit UI)
     question box, history, "tools called" expander, token/cost display
```

---

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Copy `.env.example` to `.env` and fill in real credentials:
   ```bash
   cp .env.example .env
   ```
   ```
   WATTICS_API_TOKEN=your_wattics_token
   OPENAI_API_KEY=your_openai_key
   ```
3. (First time only, or to extend historical coverage) Pull the bulk dataset:
   ```bash
   python bulk_extract.py
   ```
   This populates `cache/` with 3 years of daily data and 3 months of 5-minute
   interval data for every meter the system uses. It's idempotent  safe to
   rerun; already-cached months are skipped.
4. Run the app:
   ```bash
   streamlit run app.py
   ```
   Or use the CLI instead of the web UI:
   ```bash
   python orchestrator.py
   ```

**Startup cost:** discovery loads from cache if <24h old; otherwise ~10 quick
API calls. The current month's daily data is refreshed if it hasn't been in
the last 12 hours (~35 calls, ~15–20s)  this is what keeps "today" and
"yesterday" current without re-hitting the API on every single restart.

---

## Project structure

```
ark_assessment/
├── app.py                      # Streamlit UI
├── orchestrator.py             # LLM (OpenAI) tool-calling loop
├── bulk_extract.py             # one-off bulk historical data pull
├── run_discovery.py            # discovery smoke-test script
├── find_data_range.py          # scans for the usable date range
├── check_normalization_data.py # diagnostic: verifies production/occupancy meters
├── test_processing.py          # sanity-checks the unified daily table
├── test_processing_gaps.py     # proves dedup, interval gaps, resampling, timezone
├── test_analytics.py           # exercises every analytics function on real data
├── test_data_quality.py        # data-cleaning / quality report
├── test_questions.py           # end-to-end test suite (see below)
├── test_scale.py                # synthetic scale test (1yr/15-min/40 meters)
├── requirements.txt
├── .env.example
├── .gitignore
├── cache/                      # local JSON cache (gitignored)
└── src/
    ├── wattics_client.py       # API client, caching, cached discovery
    ├── meter_registry.py       # which meters = site total / normalization / weather
    ├── data_processing.py      # loading, gap handling, resampling, unified table
    ├── data_cleaning.py        # negative-value/zero-run/outlier QA layer
    ├── energy_analytics.py     # pure analytics functions
    └── tools.py                # LLM tool schemas + ToolExecutor
```

---

## How the data is organized

Everything downstream  analytics, tools, the LLM  reads from **two unified
tables**, not from 46 separate meter files. Building these two tables is the
whole point of the processing layer: they're what let a Food Corp. day and an
Alpha Hotel day sit side by side in the exact same shape.

**1. The daily table** (`build_site_total_daily()` in `data_processing.py`)

One row per `(organization, site, date)`, columns: `kwh` (summed across that
site's total meters), `meters_reporting`, `meters_expected`,
`is_complete_day`. This covers the full 3-year range and is what almost every
analytics function runs on  totals, week/month-over-week/month, weekday vs.
weekend, ranking, anomaly detection. A site with 9 submeters (like Alpha
Hotel) is reduced to one `kwh` number per day here; which meters get summed
into that number is decided by `meter_registry.py`, not guessed at query
time.

**2. The detailed table** (`load_detailed_long()` in `data_processing.py`)

One row per `(meter, timestamp)` at 5-minute resolution, only covering the
last ~3 months. This feeds the three analytics that need sub-daily
granularity  peak demand, load factor, baseload vs. operational  none of
which mean anything at daily resolution. It's collapsed into a single
site-level series (`sum_detailed_to_site_series()`) before those functions
run on it, the same way the daily table is already pre-summed per site.

Splitting the data this way  one cheap, long-range table for anything
date-based, one expensive, short-range table for anything sub-daily  keeps
the common case (most questions are date-range questions) fast and small,
while still supporting the handful of questions that genuinely need interval
data.

---

## Files inside `src/`

This is the core library  nothing in here runs on its own; every other
script imports from these.

**`wattics_client.py`**  the only file that talks to the Wattics API
directly. Handles authentication, retries/backoff on network errors and rate
limits, and per-(meter, month) JSON caching so the same data is never
fetched twice. `discover_meters_cached()` adds a 24-hour-fresh cache on top
of org/site/meter discovery specifically, diffing against the previous
result so a newly added site or meter gets flagged instead of silently
absorbed.

**`meter_registry.py`**  the business logic of *which* meters count. Maps
each site to its "total" meters (summed for site-level kWh), its
normalization meters (production volume, guest-nights), and its weather
meter (HDD), resolved by name against live discovery rather than hardcoded
IDs. This is where judgment calls like excluding Alpha Hotel's `MAIN` meter
live, as documented, defensible decisions rather than silent choices buried
in code.

**`data_processing.py`**  turns the raw JSON cache into the two tables
described above. Also owns deduplication, interval-level gap detection,
reindexing onto a full expected time grid (gaps become `NaN`, never a
fabricated number), and the `resample()` helper for hourly/daily/weekly/
monthly rollups.

**`data_cleaning.py`**  a data-quality pass that sits between raw loading
and analytics: nulls out physically-impossible negative readings, flags long
runs of exact zeros as "possible meter offline," and flags statistical
outliers using a robust (MAD-based) score. It flags, it doesn't delete  a
real spike might be the actual anomaly the system is supposed to surface.

**`energy_analytics.py`**  the actual math, as pure functions with no LLM
involvement anywhere in this file. Totals, period-over-period change,
baseload/operational split, peak demand, load factor, weekday/weekend
profile, site ranking, and anomaly detection. Every function returns a
structured dict (never a formatted sentence) and is defensive about missing
data  an unavailable answer comes back as `{"available": False, "reason":
...}`, never a crash or a guess.

**`tools.py`**  the bridge between the analytics functions and the LLM.
`TOOL_SCHEMAS` defines each function's name, description, and explicit JSON
parameters; `ToolExecutor` loads and caches the two unified tables once per
session (not per question), resolves organization-only requests into a
full-organization aggregate (summed in Python, never left for the LLM to add
up), and dispatches each tool call to the matching analytics function.

**`orchestrator.py`**  the OpenAI function-calling loop. Sends the
question and tool schemas to the model, executes whatever tools it asks for,
feeds results back, and repeats until the model has enough to answer in
plain language  this is what naturally handles multi-tool questions like
"which org had the bigger WoW increase" (the model just calls the same tool
once per site/org and compares the results itself). Also owns error handling
for the LLM API itself, a hard cap on tool-calling iterations, and per-query
token/cost accounting.

**`app.py`**  the Streamlit UI: a question box, full conversation history,
and a "tools called" expander under each answer showing the exact tool name,
arguments, and result.

---

## Files in the project root

Everything here is a runnable script, not something imported elsewhere 
each one is a distinct step in getting the system built, populated, and
verified.

**`bulk_extract.py`**  the one-off (but rerunnable) historical data pull.
Reads `meter_registry.py` to know which meters matter, then pulls and caches
daily data for the full 3-year range plus 5-minute detailed data for the
last 3 months. Idempotent  safe to rerun any time; already-cached months are
skipped, so it only ever fetches what's missing.

**`run_discovery.py`**  an early smoke-test script: confirms org/site/meter
discovery works and prints the full tree. Also where the MAIN-vs-submeters
and Food Corp "x point" comparisons were run to settle which meters count as
each site's total.

**`find_data_range.py`**  scans backward month-by-month to find the actual
usable date range in the Wattics account (used once, to discover the real
Aug 2023–present window before committing to a bulk-pull scope).

**`check_normalization_data.py`**  a diagnostic that confirmed, via direct
live API calls bypassing the cache, that the production/occupancy meters
have no data in this account  the evidence behind the normalization
limitation.

**`orchestrator.py`**  the OpenAI function-calling loop (see module
walkthrough above); also runnable directly as a CLI chat interface, useful
for quick testing without starting Streamlit.

**`app.py`**  the Streamlit UI (see above); this is the actual deliverable
for section 6 of the assessment.

**Test / verification scripts**  each one exercises a specific layer
against real cached data (or, for `test_scale.py`, synthetic data at a
target scale) rather than being unit tests in the strict sense:
- `test_processing.py`, `test_processing_gaps.py`  the processing layer
  (unified tables, dedup, gap detection, resampling, timezone).
- `test_data_quality.py`  the cleaning layer (negative values, zero-runs,
  outliers).
- `test_analytics.py`  every analytics function.
- `test_questions.py`  end-to-end: the 5 required example questions plus 3
  robustness cases, checking which tool got called and whether results
  matched expectations.
- `test_scale.py`  synthetic data at "a year of 15-minute data across 40
  meters" to prove the architecture holds at that scale, not just argue it
  would.

---

## Known limitations

- **Detailed (5-min) data covers only ~3 months and doesn't auto-extend.**
  It was a one-time pull via `bulk_extract.py`; peak demand, load factor, and
  baseload/operational questions for dates outside that window return
  `available: false` rather than a fabricated number. Extending the window
  requires manually rerunning `bulk_extract.py`.
- **Normalized ranking isn't currently possible.** `get_site_ranking(normalize=true)`
  is built and ready, but the underlying activity meters (`Production data`,
  `Production hotel`, `Guest Nights`) were confirmed  via direct live API
  calls bypassing the cache  to have zero records in this account. It falls
  back to raw kWh with an explicit warning rather than silently misleading.
- **No absolute cross-site intraday time alignment.** The three sites are in
  genuinely different real-world timezones (Dublin, San Diego, Warner
  Robins), confirmed via the Sites API. We treat each site's timestamps as
  already local to that site, which is correct for day-level analytics but
  means we can't say precisely whether one site's peak happened before
  another's in true simultaneous time  the API gives no UTC offset to
  convert with.
- **Alpha Hotel's `MAIN` meter is excluded from the site total**  a
  documented judgment call (see architecture/design notes above), not a
  proven fact, since we can't independently verify what `MAIN` actually
  measures.
- **Food Corp's electricity coverage is thin.** Only `Effluent Area`
  reliably reports data; `HVAC`, `Refrigeration`, and `x point` were
  confirmed to have no data at all. Food Corp's "site total" is therefore
  really just one submeter, not the whole farm's consumption.
- **Anomaly detection doesn't catch cycles longer than a week.** It compares
  each day only to recent occurrences of the same weekday, by design (to
  avoid misflagging ordinary weekly patterns)  but this means a real
  ~28-day recurring pattern we found in one Beta Resort submeter wouldn't be
  caught by site-level anomaly detection built the same way. Stated in the
  tool's own output, not hidden.
- **Month-over-month is a rolling 30-day window**, not a calendar month.
